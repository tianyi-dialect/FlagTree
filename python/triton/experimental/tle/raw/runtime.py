from .cuda import CUDAJITFunction
from .mlir import MLIRJITFunction
from triton._flagtree_backend import FLAGTREE_BACKEND

registry = {"cuda": CUDAJITFunction, "mlir": MLIRJITFunction}

try:
    from .tops import TOPSJITFunction, TOPSMLIRJITFunction
    registry["tops"] = TOPSJITFunction
    registry["tops_mlir"] = TOPSMLIRJITFunction
except ImportError:
    pass

try:
    from .ppu import PPUJITFunction
    registry["ppu"] = PPUJITFunction
    if FLAGTREE_BACKEND == "ppu":
        registry["cuda"] = PPUJITFunction
except ImportError:
    pass


def dialect(
    *,
    name: str,
    **kwargs,
):

    def decorator(fn):
        edsl = registry[name](fn, **kwargs)
        return edsl

    return decorator
