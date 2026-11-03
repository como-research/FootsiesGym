"""The CUDA build through its plain C API (ctypes), without JAX.

``CudaEnv`` is the GPU runtime behind ``footsiesgym.make(backend="c",
device="cuda")``; the tests also use it (and ``CudaGames``) to check the CUDA
build against the CPU build byte for byte.
"""

import ctypes
import functools

import numpy as np

from .build import build_cuda
from .env import env_config, make_config
from .lib import load, ptr

_P = ctypes.c_void_p
_I = ctypes.c_int
_SIGNATURES = {
    "footsies_cuda_sizeof_env_state": ([], _I),
    "footsies_cuda_sizeof_game": ([], _I),
    "footsies_cuda_sizeof_env_config": ([], _I),
    "footsies_cuda_max_delay_steps": ([], _I),
    "footsies_cuda_device_count": ([], _I),
    "footsies_cuda_malloc": ([ctypes.c_size_t], _P),
    "footsies_cuda_free": ([_P], _I),
    "footsies_cuda_h2d": ([_P, _P, ctypes.c_size_t], _I),
    "footsies_cuda_d2h": ([_P, _P, ctypes.c_size_t], _I),
    "footsies_cuda_synchronize": ([], _I),
    "footsies_cuda_set_device": ([_I], _I),
    "footsies_cuda_d2d": ([_P, _P, ctypes.c_size_t], _I),
    "footsies_cuda_order_before": ([_P], _I),
    "footsies_cuda_wait_for": ([_P], _I),
    "footsies_cuda_pack_actions": ([_P, _P, _I, _P, _I], _I),
    "footsies_cuda_env_reset": ([_P, _P, _I], _I),
    "footsies_cuda_env_step": ([_P, _P, _P, _P, _P, _P, _P, _I, _I], _I),
    "footsies_cuda_game_new": ([_P, _I], _I),
    "footsies_cuda_game_step": ([_P, _P, _P, _I, _I], _I),
}


class CudaLibrary:
    def __init__(self, path):
        self.lib = ctypes.CDLL(str(path))
        for name, (args, res) in _SIGNATURES.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, res
            setattr(self, name.removeprefix("footsies_cuda_"), fn)
        cpu = load()
        if (self.sizeof_env_state(), self.sizeof_game()) != (
            cpu.env_state_size,
            cpu.game_size,
        ):
            raise RuntimeError("struct layouts differ between the CPU and CUDA builds")
        self.max_delay_steps = self.max_delay_steps()

    def check(self, err: int):
        if err:
            raise RuntimeError(f"CUDA error {err}")


@functools.cache
def load_cuda() -> CudaLibrary:
    return CudaLibrary(build_cuda())


def available() -> bool:
    """Whether a CUDA device and nvcc are available."""
    import glob

    from .build import nvcc_path

    if nvcc_path() is None or not glob.glob("/dev/nvidia[0-9]*"):
        return False
    try:
        return load_cuda().device_count() > 0
    except (RuntimeError, OSError):
        return False


class DeviceArray:
    """A device buffer with the shape and dtype of a host array."""

    def __init__(self, lib: CudaLibrary, like: np.ndarray):
        self.lib, self.shape, self.dtype = lib, like.shape, like.dtype
        self.nbytes = like.nbytes
        self.ptr = lib.malloc(max(self.nbytes, 1))
        if not self.ptr:
            raise MemoryError(f"cudaMalloc of {self.nbytes} bytes failed")

    def upload(self, host: np.ndarray):
        host = np.ascontiguousarray(host, self.dtype)
        assert host.nbytes == self.nbytes
        self.lib.check(self.lib.h2d(self.ptr, ptr(host), self.nbytes))

    def download(self) -> np.ndarray:
        out = np.empty(self.shape, self.dtype)
        self.lib.check(self.lib.d2h(ptr(out), self.ptr, self.nbytes))
        return out

    def __del__(self):
        if getattr(self, "ptr", None):
            self.lib.free(self.ptr)
            self.ptr = None


class CudaEnv:
    """``FootsiesCEnv`` on the GPU: the same states, byte for byte.

    Every buffer lives on CUDA device ``device``; ``step`` runs one kernel on
    the legacy default stream.
    """

    def __init__(self, num_envs: int, config=None, device: int = 0, **kwargs):
        self.lib = load_cuda()
        self.device = int(device)
        if not 0 <= self.device < self.lib.device_count():
            raise ValueError(f"no CUDA device {self.device}")
        self.lib.check(self.lib.set_device(self.device))
        self.num_envs = n = num_envs
        self.config = make_config(config, **kwargs)
        self._config = env_config(self.config, self.lib.max_delay_steps)
        size = self.lib.sizeof_env_state()
        new = lambda like: DeviceArray(self.lib, like)
        self.state = new(np.zeros((n, size), np.uint8))
        self.actions = new(np.zeros((n, 2), np.int32))
        self.obs = new(np.zeros((n, 2, 88), np.float32))
        self.rew = new(np.zeros((n, 2), np.float32))
        self.term = new(np.zeros(n, np.uint8))
        self.trunc = new(np.zeros(n, np.uint8))

    def reset(self):
        self.lib.check(self.lib.set_device(self.device))
        self.lib.check(self.lib.env_reset(self.state.ptr, self.obs.ptr, self.num_envs))

    def step(self, actions: np.ndarray | None = None, autoreset: bool = True):
        """One step; ``actions`` (N, 2) are uploaded first unless None."""
        self.lib.check(self.lib.set_device(self.device))
        if actions is not None:
            self.actions.upload(actions)
        self.lib.check(
            self.lib.env_step(
                ptr(self._config),
                self.state.ptr,
                self.actions.ptr,
                self.obs.ptr,
                self.rew.ptr,
                self.term.ptr,
                self.trunc.ptr,
                int(autoreset),
                self.num_envs,
            )
        )

    def pack_actions(self, p1_ptr: int, p2_ptr: int, elem_bytes: int):
        """Write ``actions`` from two (N,) int32 / int64 device arrays."""
        self.lib.check(self.lib.set_device(self.device))
        self.lib.check(
            self.lib.pack_actions(
                p1_ptr, p2_ptr, elem_bytes, self.actions.ptr, self.num_envs
            )
        )

    def order_before(self, stream):
        """Make ``stream`` (a cudaStream_t, or None: the host) wait for the
        work launched so far."""
        self.lib.check(self.lib.set_device(self.device))
        self.lib.check(self.lib.order_before(stream))

    def wait_for(self, stream):
        """Make the env's stream wait for the work launched on ``stream``."""
        self.lib.check(self.lib.set_device(self.device))
        self.lib.check(self.lib.wait_for(stream))

    def synchronize(self):
        self.lib.check(self.lib.synchronize())

    def outputs(self) -> dict[str, np.ndarray]:
        return {
            "state": self.state.download(),
            "obs": self.obs.download(),
            "rew": self.rew.download(),
            "term": self.term.download(),
            "trunc": self.trunc.download(),
        }


class CudaGames:
    """``CGames`` on the GPU (frame stepping), for byte comparisons."""

    def __init__(self, num_games: int):
        self.lib = load_cuda()
        self.num_games = n = num_games
        self.state = DeviceArray(
            self.lib, np.zeros((n, self.lib.sizeof_game()), np.uint8)
        )
        self.p1 = DeviceArray(self.lib, np.zeros(n, np.int32))
        self.p2 = DeviceArray(self.lib, np.zeros(n, np.int32))
        self.lib.check(self.lib.game_new(self.state.ptr, n))

    def step(self, inputs: np.ndarray, n_frames: int = 1):
        self.p1.upload(inputs[:, 0])
        self.p2.upload(inputs[:, 1])
        self.lib.check(
            self.lib.game_step(
                self.state.ptr, self.p1.ptr, self.p2.ptr, n_frames, self.num_games
            )
        )

    def state_bytes(self) -> np.ndarray:
        return self.state.download()
