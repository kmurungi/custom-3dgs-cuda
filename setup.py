import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_rasterizer_cuda',
    ext_modules=[
        CUDAExtension(
            name='custom_rasterizer_cuda', 
            sources=[
                'csrc/bindings.cpp',
                'csrc/forward/projection.cu',
                'csrc/forward/rasterize.cu',
                'csrc/forward/SH.cu',
            ], 
            include_dirs=["csrc/forward", "csrc/backward"]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension       
    }, 
    
)