import os
import sys
from pathlib import Path

# Required on Windows when vcvars is already activated (PyTorch CUDA extensions).
os.environ.setdefault("DISTUTILS_USE_SDK", "1")

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).resolve().parent
include_dirs = [
    str(ROOT / "csrc"),
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


def _install_requires() -> list[str]:
    """Runtime packages from requirements.txt. Torch is imported above and must already be installed."""
    requirements = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("torch"):
            continue
        requirements.append(line)
    return requirements


def _verify_pybind11() -> None:
    """Fail before compilation if this PyTorch build does not ship PyBind11 headers."""
    from torch.utils.cpp_extension import include_paths

    header = Path("pybind11") / "pybind11.h"
    searched = [str(path) for path in include_paths(cuda=False)]
    if any((Path(path) / header).is_file() for path in searched):
        return
    looked_in = ", ".join(searched) or "(no include paths)"
    raise SystemExit(
        "PyBind11 headers were not found in the active PyTorch install "
        f"({looked_in}). Install PyTorch before building this extension."
    )


_verify_pybind11()

setup(
    name="custom_rasterizer_cuda",
    description="CUDA 3D Gaussian Splatting rasterizer (PyBind11 extension)",
    python_requires=">=3.10",
    install_requires=_install_requires(),
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
