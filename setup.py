import os
import sys
from pathlib import Path

# Required on Windows when vcvars is already activated (PyTorch CUDA extensions).
os.environ.setdefault("DISTUTILS_USE_SDK", "1")

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
include_dirs = [
    str(ROOT / "csrc" / "forward"),
    str(ROOT / "csrc" / "backward"),
]

# GLM is header-only. Prefer a vendored copy, then the active env include path.
glm_candidates = [
    ROOT / "third_party" / "glm",
    Path(sys.prefix) / "Library" / "include",  # conda on Windows
    Path(sys.prefix) / "include",              # conda/venv on Linux
]
for candidate in glm_candidates:
    if (candidate / "glm" / "glm.hpp").exists() or (candidate / "glm.hpp").exists():
        include_dirs.append(str(candidate))
        break

setup(
    name="custom_rasterizer_cuda",
    packages=find_packages(include=["gaussian_splatting", "gaussian_splatting.*"]),
    ext_modules=[
        CUDAExtension(
            name="custom_rasterizer_cuda",
            sources=[
                "csrc/bindings.cpp",
                "csrc/forward/projection.cu",
                "csrc/forward/rasterize.cu",
                "csrc/forward/SH.cu",
                "csrc/backward/rasterize_backward.cu",
                "csrc/backward/projection_backward.cu",
                "csrc/backward/delight_backward.cu",
            ],
            include_dirs=include_dirs,
        )
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
)
