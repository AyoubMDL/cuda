from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='cross_entropy_cuda',
    ext_modules=[
        CUDAExtension(
            'cross_entropy_cuda',
            ['cross_entropy.cu']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
