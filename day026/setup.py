from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='tanh_cuda',
    ext_modules=[
        CUDAExtension(
            'tanh_cuda',
            ['tanh.cu']
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
