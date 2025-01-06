import json
import os
from functools import _lru_cache_wrapper
from typing import _UnionGenericAlias
import jittor
import jittor.einops
import jittor.weightnorm
import jittor.models
import inspect

# 定义严格的模块名称，包括 jittor 顶层
strict_modules = [
    'jittor',
    'jittor.nn',
    'jittor.models',
    'jittor.init',
    'jittor.contrib',
    'jittor.dataset',
    'jittor.linalg',
    'jittor.distributions',
    'jittor.attention',
    'jittor.einops',
    'jittor.sparse',
    'jittor.weightnorm',
    'jittor.optim',
]


def is_allowed_module(full_name: str) -> bool:
    """
    判断 full_name 是否处于 strict_modules 白名单范围。如果任意 strict_modules == full_name
    或 strict_modules 是它的前缀(子模块)，则判定为允许的模块。
     """
    # 特殊情况：顶层 'jittor'
    if full_name == 'jittor':
        return True

    for mod in strict_modules:
        if full_name == mod or full_name.startswith(mod + '.'):
            return True
    return False


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

            # 遇到子模块，若子模块不在白名单范围，或已访问过，则跳过
            if inspect.ismodule(member):
                if is_allowed_module(full_name) and member not in visited:
                    stack.append((member, full_name))
                continue

            # 跳过下划线开头
            if name.startswith('_'):
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
                # 若实际属于非白名单模块，比如 jittor.compile_extern, 跳过
                if not is_allowed_module(real_module):
                    continue

                # 过滤掉不需要的 API
                if (
                        '._' in full_name or
                        'jittor._' in full_name or
                        'jittor._C' in full_name or
                        'jittor.testing' in full_name or
                        'jittor.__config__' in full_name
                ):
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
                    "description": ' '.join(doc.replace('\n', ' ').split())
                })

    return apis


apis = get_full_api_names(jittor, 'jittor')
print(f"Total APIs collected: {len(apis)}")

# 将 API 写入 JSON 文件
apis_dir = 'api_list'
os.makedirs(apis_dir, exist_ok=True)
apis_dict = {str(index + 1): api for index, api in enumerate(apis)}
with open(os.path.join(apis_dir, 'jt_api_lists.json'), 'w') as f:
    json.dump(apis_dict, f, indent=2)
