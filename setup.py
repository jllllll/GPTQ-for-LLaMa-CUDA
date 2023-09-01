import copy
import os
import platform
import re
import subprocess
import torch

from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

def get_cuda_version(cuda_home=os.environ.get('CUDA_PATH', os.environ.get('CUDA_HOME', ''))):
    if cuda_home == '' or not os.path.exists(os.path.join(cuda_home,"bin","nvcc.exe" if platform.system() == "Windows" else "nvcc")):
        return ''
    version_str = subprocess.check_output([os.path.join(cuda_home,"bin","nvcc"),"--version"]).decode('utf-8')
    idx = version_str.find("release")
    return version_str[idx+len("release "):idx+len("release ")+4]
    
CUDA_VERSION = "".join(get_cuda_version().split(".")) if not torch.version.hip else False
ROCM_VERSION = os.environ.get('ROCM_VERSION', torch.version.hip) if torch.version.hip else False

extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": ["-O3"],
}
extra_compile_args_faster = copy.deepcopy(extra_compile_args)
    
quant_cuda = [ "quant_cuda/quant_cuda.cpp", "quant_cuda/quant_cuda_kernel.cu" ]
quant_cuda_faster = [ "quant_cuda_faster/quant_cuda.cpp", "quant_cuda_faster/quant_cuda_kernel.cu" ]

if torch.version.hip:
    quant_cuda = [ "quant_cuda/quant_hip.cpp", "quant_cuda/quant_hip_kernel.hip" ]
    quant_cuda_faster = [ "quant_cuda_faster/quant_hip.cpp", "quant_cuda_faster/quant_hip_kernel.hip" ]

# quant_cuda_faster requires minimum compute 6.0
elif torch.version.cuda:
    flags_faster = [flag for flag in cpp_extension._get_cuda_arch_flags() if int(re.sub('\D', '', flag.split(',')[0])) >= 60]
    if not flags_faster:
        flags_faster = ['-gencode=arch=compute_60,code=compute_60', '-gencode=arch=compute_60,code=sm_60']
    extra_compile_args_faster["nvcc"] = extra_compile_args_faster["nvcc"] + flags_faster

version = "0.1.0" + (f"+cu{CUDA_VERSION}" if CUDA_VERSION else f"+rocm{ROCM_VERSION}" if ROCM_VERSION else "")
setup(
    name="gptq_for_llama",
    version=version,
    packages=find_packages(),
    ext_modules=[
        cpp_extension.CUDAExtension(
            "quant_cuda", quant_cuda,
            extra_compile_args=extra_compile_args,
        ),
        cpp_extension.CUDAExtension(
            "quant_cuda_faster", quant_cuda_faster,
            extra_compile_args=extra_compile_args_faster,
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
