import importlib
import inspect
import json

# from utils import map_module2lib, inspect_api_info, validate_api_existence
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
    "Code": "import torch; import torch.nn.functional as F; logits = torch.randn(4, 5, requires_grad=True); target = torch.tensor([1, 4, 3, 0]); loss = F.cross_entropy(input=logits, target=target, weight=torch.tensor([1.0, 2.0, 0.5, 0.8, 1.2]), ignore_index=-1, reduction='mean', label_smoothing=0.1); loss.backward(); print(loss.item())",
    "APIs": ["torch.nn.functional.F.cross_entropy", "torch.randn", "torch.tensor"]
}"""

dict_var = json.loads(json_data)
print(dict_var)