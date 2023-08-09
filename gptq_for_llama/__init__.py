import importlib.util
import os
import torch
import sys


def redirect_imports(name='.', package=__name__, version='old'):
    spec = importlib.util.find_spec(f".gptq_{version}.{name}", package=package)
    loader = importlib.util.LazyLoader(spec.loader)
    spec.loader = loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[package+'.'+name] = module
    loader.exec_module(module)
    
def switch_gptq(version):
    versions = ['old', 'new']
    if version.lower() in versions:
        for module in module_list[version.lower()]:
            redirect_imports(name=module, version=version.lower())
    else:
        print(f"WARNING: '{version}' is not one of: ({', '.join(versions)})")
  
def get_compute_capability():
    if torch.version.hip:
        return 9999
    compute_array = []
    for i in range(torch.cuda.device_count()):
        compute_major, compute_minor = torch.cuda.get_device_capability(torch.cuda.device(i))
        compute_array.append(int(f"{compute_major}{compute_minor}"))
    compute_array.sort()
    highest_compute = compute_array[-1]
    return highest_compute
    
module_list = {
    'old': ['datautils', 'gptq', 'llama_inference', 'llama_inference_offload',
            'modelutils', 'opt', 'quant', 'test_kernel'],
    'new': ['datautils', 'fused_attn', 'gptq', 'llama',
            'llama_inference', 'llama_inference_dmapauto', 'llama_inference_offload', 'modelutils',
            'opt', 'quant', 'share_tensors_across_processes', 'test_kernel']
}

if torch.cuda.is_available() and os.environ.get('QUANT_CUDA_OVERRIDE', 'old').strip().lower() != 'new' and get_compute_capability() >= 60:
    switch_gptq('old')
elif torch.cuda.is_available() and os.environ.get('QUANT_CUDA_OVERRIDE', 'new').strip().lower() != 'old':
    switch_gptq('new')
