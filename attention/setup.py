import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp') + glob.glob('*.cu')

setup(
    name="cppcuda_tutorial",
    version="1.0",
    author="Youming Zhao",
    author_email="youming0.zhao@gmail.com",
    description="cppcuda example",
    long_description="cppcuda example for Pytorch",
    ext_modules=[
        CUDAExtension(
            name="cppcuda_tutorial",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'nvcc': ['-code=sm_80', '-arch=compute_80']}
            # extra_compile_args={'nvcc': ['-rdc=true']}  # (optional) code optimization
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)