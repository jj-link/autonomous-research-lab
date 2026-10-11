"""Small CUDA Driver API binding; all arithmetic runs on the selected device."""
import ctypes as C
import os
import uuid


class Cuda:
    def __init__(self):
        self.lib = C.CDLL("libcuda.so.1")
        self.context = C.c_void_p()
        self.allocations = []
        self.modules = []
        self.events = []
        ptr = C.c_void_p
        u64 = C.c_uint64
        signatures = {
            "cuInit": [C.c_uint], "cuDeviceGetCount": [C.POINTER(C.c_int)],
            "cuDeviceGet": [C.POINTER(C.c_int), C.c_int],
            "cuDeviceGetName": [ptr, C.c_int, C.c_int],
            "cuDeviceGetUuid_v2": [ptr, C.c_int],
            "cuDeviceTotalMem_v2": [C.POINTER(C.c_size_t), C.c_int],
            "cuDriverGetVersion": [C.POINTER(C.c_int)],
            "cuCtxCreate_v2": [C.POINTER(ptr), C.c_uint, C.c_int], "cuCtxDestroy_v2": [ptr],
            "cuMemAlloc_v2": [C.POINTER(u64), C.c_size_t], "cuMemFree_v2": [u64],
            "cuMemcpyHtoD_v2": [u64, ptr, C.c_size_t], "cuMemcpyDtoH_v2": [ptr, u64, C.c_size_t],
            "cuModuleLoadData": [C.POINTER(ptr), ptr], "cuModuleUnload": [ptr],
            "cuModuleGetFunction": [C.POINTER(ptr), ptr, C.c_char_p],
            "cuLaunchKernel": [ptr, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, ptr, C.POINTER(ptr), ptr],
            "cuCtxSynchronize": [], "cuEventCreate": [C.POINTER(ptr), C.c_uint],
            "cuEventRecord": [ptr, ptr], "cuEventSynchronize": [ptr],
            "cuEventElapsedTime": [C.POINTER(C.c_float), ptr, ptr], "cuEventDestroy_v2": [ptr],
            "cuGetErrorName": [C.c_int, C.POINTER(C.c_char_p)],
            "cuGetErrorString": [C.c_int, C.POINTER(C.c_char_p)],
        }
        for name, args in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes = args
            function.restype = C.c_int
        self.call("cuInit", 0)
        count = C.c_int()
        self.call("cuDeviceGetCount", C.byref(count))
        if count.value != 1:
            raise RuntimeError(f"Expected exactly the allocated CUDA device; got {count.value}")
        device = C.c_int()
        self.call("cuDeviceGet", C.byref(device), 0)
        name = C.create_string_buffer(256)
        raw_uuid = (C.c_ubyte * 16)()
        memory = C.c_size_t()
        version = C.c_int()
        self.call("cuDeviceGetName", name, len(name), device)
        self.call("cuDeviceGetUuid_v2", raw_uuid, device)
        self.call("cuDeviceTotalMem_v2", C.byref(memory), device)
        self.call("cuDriverGetVersion", C.byref(version))
        gpu_uuid = "GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid)))
        if gpu_uuid.lower() != os.environ.get("CUDA_VISIBLE_DEVICES", "").lower():
            raise RuntimeError(f"CUDA selected an unexpected device: {gpu_uuid}")
        self.device = {"name": name.value.decode(), "uuid": gpu_uuid, "memory_bytes": memory.value, "cuda_driver_version": version.value}
        self.call("cuCtxCreate_v2", C.byref(self.context), 0, device)
        for _ in range(2):
            event = C.c_void_p()
            self.call("cuEventCreate", C.byref(event), 0)
            self.events.append(event)

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            label, description = C.c_char_p(), C.c_char_p()
            self.lib.cuGetErrorName(status, C.byref(label))
            self.lib.cuGetErrorString(status, C.byref(description))
            raise RuntimeError(f"{name}: {(label.value or b'CUDA error').decode()} ({status}): {(description.value or b'').decode()}")

    def allocate(self, size):
        pointer = C.c_uint64()
        self.call("cuMemAlloc_v2", C.byref(pointer), size)
        self.allocations.append(pointer)
        return pointer

    def upload(self, pointer, values):
        self.call("cuMemcpyHtoD_v2", pointer, C.c_void_p(values.buffer_info()[0]), len(values) * values.itemsize)

    def download(self, values, pointer):
        self.call("cuMemcpyDtoH_v2", C.c_void_p(values.buffer_info()[0]), pointer, len(values) * values.itemsize)

    def module(self, ptx):
        module = C.c_void_p()
        source = C.create_string_buffer(ptx.encode())
        self.call("cuModuleLoadData", C.byref(module), source)
        self.modules.append(module)
        return module

    def function(self, module, name):
        function = C.c_void_p()
        self.call("cuModuleGetFunction", C.byref(function), module, name.encode())
        return function

    def launch(self, function, args, grid, block):
        parameters = (C.c_void_p * len(args))(*(C.cast(C.pointer(arg), C.c_void_p) for arg in args))
        self.call("cuLaunchKernel", function, grid, 1, 1, block, 1, 1, 0, None, parameters, None)

    def time(self, operation):
        self.call("cuEventRecord", self.events[0], None)
        operation()
        self.call("cuEventRecord", self.events[1], None)
        self.call("cuEventSynchronize", self.events[1])
        elapsed = C.c_float()
        self.call("cuEventElapsedTime", C.byref(elapsed), *self.events)
        return elapsed.value

    def close(self):
        # Context destruction also releases allocations if an earlier CUDA call failed.
        if self.context.value:
            self.call("cuCtxDestroy_v2", self.context)
            self.context = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, trace):
        self.close()
