import json
from functools import _lru_cache_wrapper
from typing import _UnionGenericAlias
import jax
import inspect
from jaxlib.xla_extension import PjitFunction


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
                stack.append((member, full_name))
            elif name.startswith('_') or isinstance(member, type):
                continue
            # 当 member 为类或函数时，记录 API 信息
            elif (inspect.isclass(member) or
                  inspect.isfunction(member) or
                  isinstance(member, PjitFunction) or
                  isinstance(member, _lru_cache_wrapper) or
                  isinstance(member, _UnionGenericAlias)):
                # 过滤掉不需要的 API
                if ('version' in full_name.split('.')
                        or 'interpreters' in full_name.split('.')
                        or 'core' in full_name.split('.')
                        or 'api_util' in full_name.split('.')
                        or 'custom_derivatives' in full_name.split('.')
                        or 'custom_batching' in full_name.split('.')
                        or 'np' in full_name.split('.')):
                    continue
                try:
                    signature = str(inspect.signature(member))
                except ValueError:
                    signature = "N/A"

                doc = inspect.getdoc(member) or "No description available."
                apis.append({
                    "name": name,
                    "module": current_prefix,
                    "fullName": full_name,
                    "signature": signature,
                    "description": doc.split('\n')[0]  # Take the first line of the docstring
                })

    return apis


# 获取 JAX 的 API
apis = get_full_api_names(jax, 'jax')

# 打印收集到的 API 数量
print(f"Total APIs collected: {len(apis)}")

# 将 API 写入 JSON 文件
apis_dict = {str(index + 1): api for index, api in enumerate(apis)}
with open('jax_api_list.json', 'w') as f:
    json.dump(apis_dict, f, indent=2)
