import tensorflow as tf
import inspect
import json


def get_tf_full_api_names(module, prefix=''):
    apis = []
    stack = [(module, prefix)]
    visited = set()  # 记录已访问的模块，避免重复访问

    while stack:
        current_module, current_prefix = stack.pop()
        if current_module in visited:
            continue
        visited.add(current_module)

        try:
            # 获取当前模块的所有成员
            members = inspect.getmembers(current_module)
        except ModuleNotFoundError as e:
            print(f"Skipping module {current_prefix} due to import error: {e}")
            continue
        except ImportError as e:
            print(f"Skipping module {current_prefix} due to import error: {e}")
            continue
        except Exception as e:
            print(f"Skipping module {current_prefix} due to unexpected error: {e}")
            continue

        for name, member in members:
            full_name = current_prefix + name
            if inspect.ismodule(member):
                # 过滤掉私有模块和已访问的模块
                if name.startswith('_') or member in visited:
                    continue
                stack.append((member, full_name + '.'))
            elif inspect.isclass(member) or inspect.isfunction(member):
                # 过滤掉包含 "dtensor" 的 API，该类 API 通常是内部使用的
                # 过滤掉包含 "v1", "v2" 的 API，v1是旧版本API，v2是新版本API。默认只保留最新版本的API
                if 'dtensor' in full_name.split('.') or 'v1' in full_name.split('.') or 'v2' in full_name.split('.'):
                    continue

                # 检查文档字符串是否包含 "deprecated"，如果包含则认为是废弃的 API
                doc = inspect.getdoc(member)
                if doc and "deprecated" in doc.lower():
                    continue

                try:
                    signature = str(inspect.signature(member))
                except ValueError:
                    signature = "N/A"

                apis.append({
                    "name": name,
                    "module": current_prefix[:-1],
                    "fullName": full_name,
                    "signature": signature,
                    "description": doc.split('\n')[0] if doc else "No description available."  # 只取docstring的第一行
                })

                # 输出当前API，调试时可以帮助查看进度
                if len(apis) % 100 == 0:
                    print(f'Collected {len(apis)} APIs...')

    return apis


apis = get_tf_full_api_names(tf, 'tf.')

# 将 API 写入 JSON 文件
apis_dict = {str(index + 1): api for index, api in enumerate(apis)}
with open('tensorflow_api_list.json', 'w') as f:
    json.dump(apis_dict, f, indent=2)
