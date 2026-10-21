"""Compile ``csrc/`` into the shared libraries the C backend loads.

The compiler flags are part of the byte-identity spec (see ``csrc/footsies.h``):
floating-point contraction must stay off, and fast-math must never be on.
Libraries are cached by a hash of the sources, flags and compiler, in
``$FOOTSIES_C_CACHE`` (default ``~/.cache/footsiesgym``).

    python -m footsiesgym.c.build          # build the CPU libraries, print their paths
    python -m footsiesgym.c.build --cuda   # also the CUDA library (needs nvcc)
"""

import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile


def _find_csrc() -> pathlib.Path:
    """``$FOOTSIES_CSRC``, else ``csrc/`` at the repo root (a checkout), else
    the copy installed inside this package (a wheel)."""
    if os.environ.get("FOOTSIES_CSRC"):
        return pathlib.Path(os.environ["FOOTSIES_CSRC"])
    here = pathlib.Path(__file__).resolve().parent
    repo = here.parents[1] / "csrc"
    return repo if (repo / "footsies.h").exists() else here / "csrc"


CSRC = _find_csrc()
HEADERS = ("footsies.h", "footsies_batch.h", "footsies_avx512.h", "frame_data.gen.h")

# Part of the spec: never -ffast-math / -Ofast, and no FMA contraction.
FP_FLAGS = ["-O3", "-ffp-contract=off", "-fno-fast-math"]
CFLAGS = FP_FLAGS + ["-fPIC", "-shared", "-std=c11", "-fopenmp-simd"]
CXXFLAGS = FP_FLAGS + ["-fPIC", "-shared", "-std=c++17"]
OPENMP = ["-fopenmp"]
# nvcc fuses a*b+c into an FMA by default: -fmad=false is required.
NVCCFLAGS = [
    "-O3",
    "-fmad=false",
    "-prec-div=true",
    "-prec-sqrt=true",
    "-ftz=false",
    "-std=c++17",
    "-Xcompiler",
    "-fPIC,-ffp-contract=off",
    "-shared",
    "-diag-suppress",
    "940,2473",  # warnings inside the XLA FFI headers
]


def cache_dir() -> pathlib.Path:
    default = pathlib.Path.home() / ".cache" / "footsiesgym"
    path = pathlib.Path(os.environ.get("FOOTSIES_C_CACHE", default))
    path.mkdir(parents=True, exist_ok=True)
    return path


def _compiler(cxx: bool = False) -> str:
    if cxx:
        cc = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    else:
        cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        raise RuntimeError(f"footsiesgym.c needs a {'C++' if cxx else 'C'} compiler")
    return cc


def nvcc_path() -> str | None:
    """nvcc from $NVCC, $CUDA_HOME/bin, PATH or /usr/local/cuda/bin."""
    if os.environ.get("NVCC"):
        return os.environ["NVCC"]
    for home in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")):
        if home and (pathlib.Path(home) / "bin" / "nvcc").exists():
            return str(pathlib.Path(home) / "bin" / "nvcc")
    default = pathlib.Path("/usr/local/cuda/bin/nvcc")
    return shutil.which("nvcc") or (str(default) if default.exists() else None)


def ffi_include_dir() -> str | None:
    try:
        import jax.ffi
    except ImportError:
        return None
    return jax.ffi.include_dir()


def march_flags() -> list[str]:
    """``-march`` for the SIMD batches: ``$FOOTSIES_MARCH`` (default ``native``;
    empty for the compiler's default target)."""
    march = os.environ.get("FOOTSIES_MARCH", "native")
    return [f"-march={march}"] if march else []


def _cpu_features() -> str:
    """The CPU's feature flags, so a -march=native build is never reused on
    another CPU sharing the cache."""
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith(("flags", "Features")):
                return line
    except OSError:
        pass
    import platform

    return platform.processor() or platform.machine()


def _digest(sources, flags, compiler) -> str:
    h = hashlib.sha256()
    if any(f == "-march=native" for f in flags):
        h.update(_cpu_features().encode())
    for name in sorted(set(sources) | set(HEADERS)):
        h.update(name.encode())
        h.update((CSRC / name).read_bytes())
    h.update(" ".join(flags).encode())
    h.update(compiler.encode())
    return h.hexdigest()[:16]


def _compile(cmd: list[str], out: pathlib.Path) -> None:
    # Build into a temp file and rename, so concurrent builds never see a
    # partial library.
    fd, tmp = tempfile.mkstemp(suffix=out.suffix, dir=out.parent)
    os.close(fd)
    try:
        result = subprocess.run(cmd + ["-o", tmp], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"build failed: {' '.join(cmd)}\n{result.stderr}")
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _cpu_toolchains(compiler: str | None, openmp: bool):
    """(compiler, OpenMP flags) to try, best first: clang vectorizes the SIMD
    batches 10-15% better, but only if its OpenMP runtime (libomp) is
    installed; multi-threading matters more, so gcc with OpenMP comes next."""
    if compiler or os.environ.get("CC"):
        cc = compiler or os.environ["CC"]
        return [(cc, ["-fopenmp"])] * openmp + [(cc, [])]
    chain = []
    clang, gcc = shutil.which("clang"), shutil.which("gcc") or shutil.which("cc")
    if openmp:
        chain += [(c, ["-fopenmp"]) for c in (clang, gcc) if c]
    chain += [(c, []) for c in (clang, gcc) if c]
    if not chain:
        raise RuntimeError("footsiesgym.c needs a C compiler (set CC)")
    return chain


def build_cpu(
    openmp: bool = True, extra_flags: tuple[str, ...] = (), compiler: str | None = None
) -> pathlib.Path:
    """``libfootsies`` (``footsies_cpu.c``); returns its path."""
    errors = []
    for cc, omp in _cpu_toolchains(compiler, openmp):
        flags = CFLAGS + march_flags() + omp + list(extra_flags)
        out = cache_dir() / f"libfootsies-{_digest(['footsies_cpu.c'], flags, cc)}.so"
        if out.exists():
            return out
        try:
            _compile([cc, *flags, str(CSRC / "footsies_cpu.c")], out)
            return out
        except RuntimeError as e:
            errors.append(str(e))
    raise RuntimeError("could not build libfootsies:\n" + "\n".join(errors))


def build_ffi_cpu(openmp: bool = True) -> pathlib.Path:
    """``libfootsies_ffi`` (``footsies_ffi_cpu.cc``): XLA FFI handlers on the host."""
    include = ffi_include_dir()
    if include is None:
        raise RuntimeError("the XLA FFI build needs JAX (jax.ffi.include_dir())")
    cxx = _compiler(cxx=True)
    flags = CXXFLAGS + (OPENMP if openmp else []) + [f"-I{include}"]
    out = (
        cache_dir()
        / f"libfootsies_ffi-{_digest(['footsies_ffi_cpu.cc'], flags, cxx)}.so"
    )
    if not out.exists():
        try:
            _compile([cxx, *flags, str(CSRC / "footsies_ffi_cpu.cc")], out)
        except RuntimeError:
            if not openmp:
                raise
            return build_ffi_cpu(openmp=False)
    return out


def build_cuda(arch: str | None = None, ffi: bool | None = None) -> pathlib.Path:
    """``libfootsies_cuda`` (``footsies_cuda.cu``): kernels, a C API on device
    pointers and, if JAX is installed, XLA FFI handlers.

    :param arch: nvcc ``-arch`` (default ``$FOOTSIES_CUDA_ARCH`` or ``native``).
    """
    nvcc = nvcc_path()
    if nvcc is None:
        raise RuntimeError(
            "the CUDA build needs nvcc (set NVCC or CUDA_HOME); to run the C "
            "backend on the CPU instead, use jax.default_device(jax.devices('cpu')[0])"
        )
    arch = arch or os.environ.get("FOOTSIES_CUDA_ARCH", "native")
    include = ffi_include_dir() if ffi is not False else None
    flags = NVCCFLAGS + [f"-arch={arch}"]
    flags += [f"-I{include}"] if include else ["-DFOOTSIES_NO_FFI"]
    out = (
        cache_dir()
        / f"libfootsies_cuda-{_digest(['footsies_cuda.cu'], flags, nvcc)}.so"
    )
    if not out.exists():
        _compile([nvcc, *flags, str(CSRC / "footsies_cuda.cu")], out)
    return out


if __name__ == "__main__":
    print(build_cpu())
    if ffi_include_dir():
        print(build_ffi_cpu())
    if "--cuda" in sys.argv:
        print(build_cuda())
    sys.exit(0)
