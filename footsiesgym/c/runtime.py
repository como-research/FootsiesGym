"""``num_envs`` FOOTSIES games on the C backend, on the CPU or a CUDA GPU.

The runtime behind ``footsiesgym.make(backend="c")``: ``reset`` / ``step``
return ``{agent: array}`` dicts of ``tensor_type`` arrays with a leading
``num_envs`` axis, like the vectorized ``FootsiesEnv``. Every call returns
new arrays (the env's own buffers are never handed out), so they can be kept
across steps.

* ``device="cpu"``: ``FootsiesCEnv`` (SIMD + OpenMP); torch outputs share
  memory with fresh numpy copies.
* ``device="cuda[:N]"``: ``CudaEnv``, one kernel per step. Observations and
  rewards stay on the GPU for ``tensor_type="torch"`` (copied device to
  device from the env's buffers, viewed through DLPack); ``"numpy"`` copies
  them to the host. Actions on the same GPU (torch, JAX, CuPy) are read
  where they are.
* ``tensor_type="jax"`` (either device): the step is the XLA FFI call of
  ``jax_env`` (jitted, state donated), so the arrays are JAX's own;
  ``make(functional=True)`` returns those functions for jitted loops.
"""

from typing import Any

import numpy as np

from footsiesgym import tensors

AGENTS = ("p1", "p2")


def parse_device(device: str) -> tuple[str, int]:
    """``"cpu"``, ``"cuda"`` or ``"cuda:N"`` -> ``(kind, index)``."""
    device = str(device)
    if device == "cpu":
        return "cpu", 0
    if device == "cuda":
        return "cuda", 0
    if device.startswith("cuda:") and device[5:].isdigit():
        return "cuda", int(device[5:])
    raise ValueError(f"device must be 'cpu', 'cuda' or 'cuda:N', not {device!r}")


class CRuntime:
    def __init__(
        self,
        num_envs: int,
        config: dict[str, Any] | None = None,
        device: str = "cpu",
        tensor_type: str = "numpy",
        num_threads: int | None = None,
        impl: str = "auto",
    ):
        self.num_envs = num_envs
        self.tensor_type = tensors.check_tensor_type(tensor_type)
        kind, index = parse_device(device)
        self.device = "cpu" if kind == "cpu" else f"cuda:{index}"
        self._jax = self._gpu = None
        if self.tensor_type == "jax":
            self._jax = _JaxRuntime(num_envs, config, kind, index)
        elif kind == "cpu":
            from .env import FootsiesCEnv

            self._env = FootsiesCEnv(
                num_envs, config, num_threads=num_threads, impl=impl
            )
        else:
            from .cuda import CudaEnv

            self._env = CudaEnv(num_envs, config, device=index)
            self._gpu = _GpuOutputs(self._env, self.tensor_type)

    def reset(self) -> dict:
        if self._jax is not None:
            return self._jax.reset()
        self._env.reset()
        if self._gpu is not None:
            return self._gpu.obs()
        return self._cpu_outputs(with_rewards=False)[0]

    def step(self, actions, autoreset: bool = True):
        """``(obs, rewards, terminateds, truncateds)``, dicts keyed by agent.

        :param actions: ``{"p1": (N,), "p2": (N,)}`` or an ``(N, 2)`` array
            of ints: numpy, torch, JAX or a list.
        """
        if self._jax is not None:
            return self._jax.step(actions, autoreset)
        if self._gpu is not None:
            self._gpu.write_actions(actions)
            self._env.step(None, autoreset)
            return self._gpu.outputs()
        p1, p2 = _split_actions(actions, tensors.to_numpy)
        self._env.step({"p1": p1, "p2": p2}, autoreset=autoreset)
        return self._cpu_outputs(with_rewards=True)

    def _cpu_outputs(self, with_rewards: bool):
        """Per-agent arrays with the env's outputs (copied in parallel), from a
        set nobody else references any more, or a new one."""
        from .lib import ptr

        env = self._env
        arrays = self._free_output_set()
        obs = {"p1": arrays[0], "p2": arrays[1]}
        rew = {"p1": arrays[2], "p2": arrays[3]}
        term, trunc = arrays[4], arrays[5]
        env.lib.split_outputs(
            ptr(env._obs),
            ptr(env._rew),
            ptr(env._term),
            ptr(env._trunc),
            ptr(obs["p1"]),
            ptr(obs["p2"]),
            ptr(rew["p1"]) if with_rewards else None,
            ptr(rew["p2"]),
            ptr(term),
            ptr(trunc),
            self.num_envs,
        )
        out = [obs]
        if with_rewards:
            out += [rew, {a: term for a in AGENTS}, {a: trunc for a in AGENTS}]
        return tuple(self._convert(v) for v in out)

    _MAX_OUTPUT_SETS = 4

    def _free_output_set(self) -> list[np.ndarray]:
        # An array the caller still holds (directly, through a view or a
        # torch tensor sharing it) has more references than the pool's own.
        # sys.getrefcount counts the pool's list and its own argument.
        import sys

        pool = self.__dict__.setdefault("_output_sets", [])
        for arrays in pool:
            if all(sys.getrefcount(a) == 3 for a in arrays):
                return arrays
        n, d = self.num_envs, self._env.obs_dim
        arrays = [np.empty((n, d), np.float32), np.empty((n, d), np.float32)]
        arrays += [np.empty(n, np.float32), np.empty(n, np.float32)]
        arrays += [np.empty(n, np.bool_), np.empty(n, np.bool_)]
        if len(pool) < self._MAX_OUTPUT_SETS:
            pool.append(arrays)
        return arrays

    def _convert(self, values: dict) -> dict:
        out = tensors.convert(values, self.tensor_type)
        # One array for both agents' done flags, as the gRPC env returns.
        if values["p1"] is values["p2"]:
            out["p2"] = out["p1"]
        return out


def _split_actions(actions, convert):
    if isinstance(actions, dict):
        return convert(actions["p1"]), convert(actions["p2"])
    a = convert(actions)
    return a[:, 0], a[:, 1]


class _GpuOutputs:
    """Hands out ``CudaEnv``'s results as ``tensor_type`` arrays."""

    def __init__(self, env, tensor_type: str):
        from .dlpack import DLPackBuffer

        self.env, self.tensor_type = env, tensor_type
        n = env.num_envs
        view = lambda buf, shape: DLPackBuffer(  # noqa: E731
            buf.ptr, shape, buf.dtype, env.device, buf, env.order_before
        )
        self._buffers = {
            "obs": view(env.obs, (n, 2, 88)),
            "rew": view(env.rew, (n, 2)),
            "term": view(env.term, (n,)),
            "trunc": view(env.trunc, (n,)),
        }
        if tensor_type == "torch":
            import torch

            self._torch = torch
            self._dev = torch.device("cuda", env.device)
            # Aliases of the env's buffers, made once; outputs are clones.
            self._alias = {k: torch.from_dlpack(b) for k, b in self._buffers.items()}

    # ── outputs ──────────────────────────────────────────────────────────

    def obs(self) -> dict:
        return self._fetch(("obs",))["obs"]

    def outputs(self):
        out = self._fetch(("obs", "rew", "term", "trunc"))
        return out["obs"], out["rew"], out["term"], out["trunc"]

    def _fetch(self, keys) -> dict:
        if self.tensor_type == "torch":
            return self._fetch_torch(keys)
        host = {k: getattr(self.env, k).download() for k in keys}
        if "obs" in host:
            host["obs"] = {a: host["obs"][:, i].copy() for i, a in enumerate(AGENTS)}
        if "rew" in host:
            host["rew"] = {a: host["rew"][:, i].copy() for i, a in enumerate(AGENTS)}
        for k in ("term", "trunc"):
            if k in host:
                done = host[k].view(bool)
                host[k] = {a: done for a in AGENTS}
        return host

    def _fetch_torch(self, keys) -> dict:
        torch = self._torch
        user = torch.cuda.current_stream(self._dev)
        default = torch.cuda.default_stream(self._dev)
        out = {}
        # Copy on the env's (legacy default) stream: after this step's kernel,
        # before the next one overwrites the buffers.
        with torch.cuda.stream(default):
            for k in keys:
                a = self._alias[k]
                if k in ("obs", "rew"):
                    out[k] = {
                        p: a[:, i].clone(memory_format=torch.contiguous_format)
                        for i, p in enumerate(AGENTS)
                    }
                else:
                    done = a.to(torch.bool)
                    out[k] = {p: done for p in AGENTS}
        if user.cuda_stream != default.cuda_stream:
            user.wait_stream(default)
            for v in out.values():
                for t in v.values():
                    t.record_stream(user)
        return out

    # ── actions ──────────────────────────────────────────────────────────

    def write_actions(self, actions):
        p1, p2 = _split_actions(actions, lambda x: x)
        dev_p1, dev_p2 = self._on_device(p1), self._on_device(p2)
        if dev_p1 is not None and dev_p2 is not None:
            (ptr1, bytes1, keep1), (ptr2, bytes2, keep2) = dev_p1, dev_p2
            if bytes1 == bytes2:
                self.env.pack_actions(ptr1, ptr2, bytes1)
                self._keep = (keep1, keep2)  # converted copies: alive until reused
                return
        host = np.stack([tensors.to_numpy(p1), tensors.to_numpy(p2)], axis=1).astype(
            np.int32
        )
        self.env.actions.upload(host)

    def _on_device(self, x):
        """``(pointer, element bytes, keepalive)`` of a contiguous (N,) int32 /
        int64 array on this GPU, or None (then it goes through the host)."""
        n, device = self.env.num_envs, self.env.device
        module = type(x).__module__.split(".")[0]
        if module == "torch":
            import torch

            if x.device.type != "cuda" or x.device.index != device:
                return None
            if x.dtype not in (torch.int32, torch.int64):
                x = x.to(torch.int64)
            x = x.contiguous()
            if x.shape != (n,):
                return None
            stream = torch.cuda.current_stream(x.device)
            if stream.cuda_stream != torch.cuda.default_stream(x.device).cuda_stream:
                self.env.wait_for(stream.cuda_stream)
            return x.data_ptr(), x.element_size(), x
        if module in ("jax", "jaxlib"):
            import jax.numpy as jnp

            if {d.platform for d in x.devices()} != {"gpu"} or [
                d.id for d in x.devices()
            ] != [device]:
                return None
            if x.dtype not in (jnp.int32, jnp.int64):
                x = x.astype(jnp.int32)
            if x.shape != (n,):
                return None
            x.block_until_ready()
            return x.unsafe_buffer_pointer(), x.dtype.itemsize, x
        iface = getattr(x, "__cuda_array_interface__", None)  # CuPy, Numba
        if iface is None or iface.get("strides") is not None:
            return None
        if iface["typestr"] not in ("<i4", "<i8") or tuple(iface["shape"]) != (n,):
            return None
        if getattr(getattr(x, "device", None), "id", device) != device:
            return None
        if iface.get("stream") not in (None, 1):
            self.env.wait_for(iface["stream"])  # 2 is cudaStreamPerThread
        return iface["data"][0], int(iface["typestr"][2]), x


class _JaxRuntime:
    """The env as jitted XLA FFI calls (``jax_env``), state held here."""

    def __init__(self, num_envs: int, config, kind: str, index: int):
        import jax

        from .jax_env import FootsiesFFIEnv

        if kind == "cpu":
            self.device = jax.devices("cpu")[0]
        else:
            try:
                gpus = jax.devices("gpu")
            except RuntimeError:
                gpus = []
            match = [d for d in gpus if d.id == index]
            if not match:
                raise ValueError(
                    f"device='cuda:{index}' with tensor_type='jax' needs JAX with "
                    "CUDA support (pip install 'jax[cuda12]')"
                )
            self.device = match[0]
        self.env = FootsiesFFIEnv(config).vectorize(num_envs)
        self._jax = jax
        self._reset = jax.jit(self.env.reset)
        self._step = jax.jit(self.env.step, donate_argnums=1)
        self._step_env = jax.jit(self.env.step_env, donate_argnums=1)
        self._state = None

    def reset(self) -> dict:
        with self._jax.default_device(self.device):
            obs, self._state, _ = self._reset(None)
        return obs

    def step(self, actions, autoreset: bool):
        p1, p2 = _split_actions(actions, self._to_device)
        step = self._step if autoreset else self._step_env
        obs, self._state, rew, term, trunc, _ = step(
            None, self._state, {"p1": p1, "p2": p2}
        )
        return obs, rew, term, trunc

    def _to_device(self, x):
        jax = self._jax
        if type(x).__module__.split(".")[0] == "torch" and x.device.type == "cuda":
            x = jax.dlpack.from_dlpack(x.contiguous())  # stays on the GPU
        elif not isinstance(x, jax.Array):
            x = tensors.to_numpy(x)
        return jax.device_put(x, self.device)
