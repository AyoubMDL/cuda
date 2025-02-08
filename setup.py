from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='custom_cuda',
    version="0.0.1",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='custom_cuda',
            sources=['day005/relu.cu']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
