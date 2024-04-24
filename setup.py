import os
from pathlib import Path
from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if os.environ.get("TORCH_CUDA_ARCH_LIST"):
    # Let PyTorch builder to choose device to target for.
    device_capability = ""
else:
    device_capability = torch.cuda.get_device_capability()
    device_capability = f"{device_capability[0]}{device_capability[1]}"

cwd = Path(os.path.dirname(os.path.abspath(__file__)))

nvcc_flags = [
    "-std=c++17",  # NOTE: CUTLASS requires c++17
]

if device_capability:
    nvcc_flags.extend([
        f"--generate-code=arch=compute_{device_capability},code=sm_{device_capability}",
        f"-DGROUPED_GEMM_DEVICE_CAPABILITY={device_capability}",
    ])

ext_modules = [
    CUDAExtension(
        "grouped_gemm_backend",
        ["csrc/ops.cu", "csrc/grouped_gemm.cu", "csrc/sinkhorn.cu", "csrc/permute.cu"],
        include_dirs = [
            f"{cwd}/third_party/cutlass/include/"
        ],
        extra_compile_args={
            "cxx": [
                "-fopenmp", "-fPIC", "-Wno-strict-aliasing"
            ],
            "nvcc": nvcc_flags,
        }
    )
]

setup(
    name="grouped_gemm",
    version="1.1.1",
    author="Trevor Gale, Jiang Shao, Shiqing Fan",
    author_email="tgale@stanford.edu, jiangs@nvidia.com, shiqingf@nvidia.com",
    description="GEMM Grouped",
    url="https://github.com/fanshiqing/grouped_gemm",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["absl-py", "numpy", "torch"],
)
