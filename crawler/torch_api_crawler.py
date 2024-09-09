import json
from functools import _lru_cache_wrapper
from typing import _UnionGenericAlias
import torch
import inspect

# 定义严格的模块名称，包括 torch 顶层
strict_modules = [
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.Tensor',
    'torch.amp',
    'torch.autograd',
    'torch.cuda',
    'torch.backends',
    'torch.distributed',
    'torch.distributions',
    'torch.fft',
    'torch.fx',
    'torch.hub',
    'torch.jit',
    'torch.linalg',
    'torch.special',
    'torch.package',
    'torch.profiler',
    'torch.onnx',
    'torch.optim',
    'torch.utils'
]


def get_full_api_names(module, prefix=''):
    apis = []
    stack = [(module, prefix)]
    visited = set()

    while stack:
        current_module, current_prefix = stack.pop()
        if current_module in visited:
            continue
        visited.add(current_module)

        try:
            members = inspect.getmembers(current_module)
        except (ModuleNotFoundError, ImportError) as e:
            print(f"Skipping module {current_prefix} due to import error: {e}")
            continue
        except Exception as e:
            print(f"Skipping module {current_prefix} due to unexpected error: {e}")
            continue

        for name, member in members:
            full_name = current_prefix + '.' + name if current_prefix else name

            if inspect.ismodule(member):
                if name.startswith('_') or member in visited:
                    continue
                # 确保 torch 顶层 API 被捕捉
                if full_name == 'torch' or any(
                        mod == full_name or full_name.startswith(mod + '.') for mod in strict_modules):
                    stack.append((member, full_name))
            elif name.startswith('_') or isinstance(member, type):
                continue
            elif (inspect.isclass(member) or
                  inspect.isfunction(member) or
                  isinstance(member, _lru_cache_wrapper) or
                  isinstance(member, _UnionGenericAlias)):
                # 过滤掉不需要的 API
                if ('._' in full_name or  # 忽略所有私有模块
                        'torch._' in full_name or  # 忽略torch下划线开头的API
                        'torch._C' in full_name or  # 忽略C扩展相关API
                        'torch.testing' in full_name or
                        'torch.__config__' in full_name):
                    continue

                try:
                    signature = str(inspect.signature(member))
                except ValueError:
                    signature = "N/A"

                doc = inspect.getdoc(member) or "No description available."

                # 进一步过滤不常用的API
                if "deprecated" in doc.lower() or "experimental" in doc.lower():
                    continue

                apis.append({
                    "name": name,
                    "module": current_prefix,
                    "fullName": full_name,
                    "signature": signature,
                    "description": doc.split('\n')[0]  # 只取docstring的第一行
                })

    return apis


# 获取 PyTorch 的 API，包括 torch 顶层的 API
apis = get_full_api_names(torch, 'torch')

# 打印收集到的 API 数量
print(f"Total APIs collected: {len(apis)}")

# 将 API 写入 JSON 文件
apis_dict = {str(index + 1): api for index, api in enumerate(apis)}
with open('torch_api_list.json', 'w') as f:
    json.dump(apis_dict, f, indent=2)
