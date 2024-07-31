import tensorflow as tf
import inspect


def get_tf_full_api_names(module, prefix=''):
    apis = []
    deprecated_apis = []
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
            # 捕获其他异常，通常是因为模块中包含无法处理的代码
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
                    deprecated_apis.append(full_name)
                else:
                    apis.append(full_name)

                # 输出当前API，调试时可以帮助查看进度
                if len(apis) % 100 == 0 or len(deprecated_apis) % 100 == 0:
                    print(f'Collected {len(apis)} APIs and {len(deprecated_apis)} deprecated APIs...')

    return apis, deprecated_apis


apis, deprecated_apis = get_tf_full_api_names(tf, 'tf.')

with open('tensorflow_api_list.txt', 'w') as f:
    for api in apis:
        f.write(f"{api}\n")

with open('deprecated_api_list.txt', 'w') as f:
    for api in deprecated_apis:
        f.write(f"{api}\n")
