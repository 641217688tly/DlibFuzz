import importlib
import inspect
import json

from utils import map_module2lib, inspect_api_info, validate_api_existence
# api_path = 'torch.ao.quantization.qconfig.float16_dynamic_qconfig'
# print(validate_api_existence('torch.ao.quantization.qconfig', 'float16_dynamic_qconfig'))
# info = inspect_api_info('torch.ao.quantization.qconfig', 'float16_dynamic_qconfig')
# for key, value in info.items():
#     print(f'{key}: {value}')
# description = "hahaha"
# prompt = f"{'- Function Description: '+ description if description else ''}"
# print(prompt)
# var = "tensorflow"
# module_alias_mapper = {
#     "tf": "tensorflow",
#     "ms": "mindspore",
#     "np": "numpy",
#     "pd": "pandas",
#     "jt": "jittor",
#     "pytorch": "torch",
# }
# print(module_alias_mapper.get(var,var))

# var1 = """```json
# {
#     "Pytorch": [
#         ["torch.abs"],
#         ["torch.Tensor.abs"]
#     ],
#     "MindSpore": [
#         ["mindspore.ops.Abs"],
#         ["mindspore.Tensor.abs"]
#     ],
#     "JAX": [
#         ["jax.numpy.abs"]
#     ],
#     "Jittor": [
#         ["jittor.abs"],
#         ["jittor.Tensor.abs"]
#     ]
# }
# ```"""
# print(var1.split('\n', 1)[1])
# print('\n'.join(var1.split('\n')[:-1]))

json_data = """{
    "Pytorch": [
        ["torch.absolute"],
        ["torch.abs"],
        ["torch.Tensor.abs"],
    ],
    "MindSpore": [
        ["mindspore.ops.Abs"],
        ["mindspore.Tensor.abs"],
    ],
    "Jittor": [
        ["jittor.abs"],
        ["jittor.Tensor.abs"],
    ],
    "JAX": [
        ["jax.numpy.abs"],
        ["jax.numpy.fabs"],
    ]
}"""
