from __future__ import annotations
import os
from pathlib import Path
import subprocess
from typing import Any, Final

from triton._C.libtriton import llvm  # pyright: ignore[reportMissingImports]
from triton._C.libtriton.tle.llvm import parse_llvm_ir  # pyright: ignore[reportMissingImports]

# TODO: We use cli tools to compile PPU code temporarily, and plan to replace it with LLVM components Python bindings in the future.
CLANG = os.getenv("CLANG", "clang")

# PPU target intrinsics (``llvm.ppu.*``) are emitted by the PPU SDK clang fork but
# are unknown to the LLVM that Triton links at build time. If they reach MLIR's
# ``llvm.to_module`` as ``llvm.call_intrinsic`` ops, intrinsic-id lookup fails. To let the
# foreign IR round-trip through the host LLVM untouched, rewrite the ``llvm.ppu.``
# prefix to an ordinary (non-``llvm.``) external-symbol prefix here, so MLIR imports
# and re-exports them as plain calls.
PPU_INTRINSIC_PASSTHROUGH_SENTINEL: Final[str] = "__flagtree_ppu_intrinsic__"


def _disguise_ppu_intrinsics(ir_text: str) -> str:
    # Anchored on ``@`` to touch symbol references only.
    return ir_text.replace("@llvm.ppu.", "@" + PPU_INTRINSIC_PASSTHROUGH_SENTINEL)


class PPUJITFunction(object):

    def __init__(self, fn: Any, file: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn: Final[Any] = fn
        self.code: Final[str] = file.read_text()
        self.__triton_builtin__: Final[bool] = True

    def make_llvm(self, mlir_context) -> str:
        build = subprocess.run(
            [
                CLANG,
                "-x",
                "hggc",
                "--hggc-device-only",
                "-emit-llvm",
                "-O2",
                "-S",
                "-",
                "-o",
                "-",
            ],
            input=self.code.encode(),
            capture_output=True,
        )
        assert build.returncode == 0, (f"clang failed\nstderr:\n{build.stderr.decode()}")
        ir_text = _disguise_ppu_intrinsics(build.stdout.decode())
        llvm_context = llvm.context()
        module = parse_llvm_ir(ir_text, llvm_context, mlir_context)
        return f"{module}"
