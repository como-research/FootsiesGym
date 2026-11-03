"""Expose device buffers through DLPack, without depending on torch or JAX.

``DLPackBuffer`` wraps a pointer to contiguous device memory and implements
``__dlpack__`` / ``__dlpack_device__``, so ``torch.from_dlpack``,
``jax.dlpack.from_dlpack`` and ``cupy.from_dlpack`` view it without a copy.
The capsule keeps ``owner`` (whatever frees the memory) alive until the
consumer's tensor is gone.
"""

import ctypes

import numpy as np

KDL_CUDA = 2
_DTYPE_CODES = {"i": 0, "u": 1, "f": 2, "b": 6}  # kDLInt, kDLUInt, kDLFloat, kDLBool


class _Device(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _Tensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _Device),
        ("ndim", ctypes.c_int32),
        ("dtype", _DataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _ManagedTensor(ctypes.Structure):
    pass


_DELETER = ctypes.CFUNCTYPE(None, ctypes.POINTER(_ManagedTensor))
_ManagedTensor._fields_ = [
    ("dl_tensor", _Tensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DELETER),
]

_CAPSULE_DESTRUCTOR = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_capsule_new = ctypes.pythonapi.PyCapsule_New
_capsule_new.restype = ctypes.py_object
_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _CAPSULE_DESTRUCTOR]
_capsule_is_valid = ctypes.pythonapi.PyCapsule_IsValid
_capsule_is_valid.restype = ctypes.c_int
_capsule_is_valid.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_capsule_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_pointer.restype = ctypes.c_void_p
_capsule_pointer.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

# Managed tensors handed out and not yet deleted: address -> (struct, shape, owner).
_LIVE: dict[int, tuple] = {}


@_DELETER
def _delete(managed):
    _LIVE.pop(ctypes.addressof(managed.contents), None)


@_CAPSULE_DESTRUCTOR
def _destroy_capsule(capsule):
    # A consumer renames the capsule ("used_dltensor") and calls the deleter
    # itself; an unconsumed one is still "dltensor" and is deleted here.
    if _capsule_is_valid(capsule, b"dltensor"):
        _LIVE.pop(_capsule_pointer(capsule, b"dltensor"), None)


class DLPackBuffer:
    """Contiguous device memory with a shape and dtype, for DLPack consumers.

    :param ptr: device pointer to the first element.
    :param shape: the array shape (C order).
    :param dtype: a numpy dtype.
    :param device_id: the CUDA device.
    :param owner: kept alive while any consumer views the memory.
    :param order_before: ``f(stream)`` that makes ``stream`` (a pointer, or
        ``None`` for the host) wait for the work that writes the memory.
    """

    def __init__(self, ptr, shape, dtype, device_id, owner, order_before):
        self.ptr, self.shape = int(ptr), tuple(int(s) for s in shape)
        self.dtype, self.device_id = np.dtype(dtype), int(device_id)
        self.owner, self._order_before = owner, order_before

    def __dlpack_device__(self):
        return (KDL_CUDA, self.device_id)

    def __dlpack__(self, stream=None, **kwargs):
        # DLPack stream values for CUDA: 1 is the legacy default stream (ours:
        # already ordered), 2 cudaStreamPerThread, else a cudaStream_t; with
        # None (no stream given) the host waits.
        if stream != 1:
            self._order_before(stream)
        ndim = len(self.shape)
        shape = (ctypes.c_int64 * max(ndim, 1))(*self.shape)
        managed = _ManagedTensor()
        managed.dl_tensor.data = self.ptr
        managed.dl_tensor.device = _Device(KDL_CUDA, self.device_id)
        managed.dl_tensor.ndim = ndim
        managed.dl_tensor.dtype = _DataType(
            _DTYPE_CODES[self.dtype.kind], self.dtype.itemsize * 8, 1
        )
        managed.dl_tensor.shape = shape
        managed.dl_tensor.strides = None  # compact row-major
        managed.dl_tensor.byte_offset = 0
        managed.deleter = _delete
        address = ctypes.addressof(managed)
        _LIVE[address] = (managed, shape, self.owner)
        return _capsule_new(address, b"dltensor", _destroy_capsule)
