import json
import os
from functools import _lru_cache_wrapper
from typing import _UnionGenericAlias
import torch
import torch.monitor
import torch.signal
import torch.onnx
import inspect

import torch.nn
import torch.nn.functional
import torch.amp
import torch.autograd
import torch.distributed
import torch.distributions
import torch.fft
import torch.fx
import torch.hub
import torch.jit
import torch.linalg
import torch.monitor
import torch.signal
import torch.special
import torch.package
import torch.profiler
import torch.onnx
import torch.optim

# 定义严格的模块名称，包括 torch 顶层
strict_modules = [
    'torch.nn',
    'torch.nn.functional',
    'torch.Tensor',
    'torch.amp',
    'torch.autograd',
    'torch.distributed',
    'torch.distributions',
    'torch.fft',
    'torch.fx',
    'torch.hub',
    'torch.jit',
    'torch.linalg',
    'torch.monitor',
    'torch.signal',
    'torch.special',
    'torch.package',
    'torch.optim',
]

excluded_modules = [
    'torch.cuda',
    'torch.backends',
    'torch.utils',
    'torch.nn.modules',
    'torch.profiler',
    'torch.onnx',  # 数量过多，暂时不包含
    'torch.xpu',
    'torch.testing',
    'torch.windows',
]

MAX_DEPTH = 4

def is_allowed_module(full_name: str) -> bool:
    """
    判断 full_name 是否在白名单范围
     """
    if is_excluded_module(full_name):
        return False

    if full_name == 'torch':
        return True

    for mod in strict_modules:
        if full_name == mod or full_name.startswith(mod + '.'):
            return True
    return False


def is_excluded_module(full_name: str):
    """
    判断 full_name 是否在黑名单范围。
    """
    for ex in excluded_modules:
        if full_name == ex or full_name.startswith(ex + '.'):
            return True
    return False


def get_full_api_names(module, prefix='', depth=0):
    apis = []
    stack = [(module, prefix, depth)]
    visited = set()

    while stack:
        current_module, current_prefix, current_depth = stack.pop()
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

            # 遇到子模块跳过
            if inspect.ismodule(member):
                if not is_allowed_module(full_name):
                    continue
                if member not in visited and current_depth < MAX_DEPTH:
                    stack.append((member, full_name, current_depth + 1))
                continue

            # 跳过下划线开头
            if name.startswith('_'):
                continue

            # 过滤掉不需要的 API
            if (
                    '._' in full_name or
                    'torch._' in full_name or
                    'torch._C' in full_name or
                    'torch.testing' in full_name or
                    'torch.__config__' in full_name
            ):
                continue

            # 类/函数/泛型别名等
            if (
                    inspect.isclass(member) or
                    inspect.isfunction(member) or
                    isinstance(member, _lru_cache_wrapper) or
                    isinstance(member, _UnionGenericAlias)
            ):
                if not is_allowed_module(current_prefix):
                    continue

                real_module = getattr(member, '__module__', '')
                # 若实际属于非白名单模块, 跳过
                if not is_allowed_module(real_module):
                    continue

                try:
                    signature = str(inspect.signature(member))
                except ValueError:
                    signature = "N/A"

                doc = inspect.getdoc(member) or "No description available."
                # 过滤被标记为不常用的API
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


apis = get_full_api_names(torch, 'torch')
print(f"Total APIs collected: {len(apis)}")

# 将 API 写入 JSON 文件
apis_dir = 'api_list'
os.makedirs(apis_dir, exist_ok=True)
apis_dict = {str(index + 1): api for index, api in enumerate(apis)}
with open(os.path.join(apis_dir, 'torch_api_list.json'), 'w') as f:
    json.dump(apis_dict, f, indent=2)
