import json
import inspect
import mindspore

modules_to_include = [
    'mindspore',
    'mindspore.nn',
    'mindspore.nn.functional',
    'mindspore.ops',
    'mindspore.ops.primitive',
    'mindspore.mint',
    'mindspore.amp',
    'mindspore.train',
    'mindspore.communication',
    'mindspore.communication.comm_func',
    'mindspore.common.initializer',
    'mindspore.hal',
    'mindspore.dataset',
    'mindspore.dataset.transforms',
    'mindspore.mindrecord',
    'mindspore.nn.probability',
    'mindspore.rewrite',
    'mindspore.multiprocessing',
    'mindspore.boost',
    'mindspore.numpy',
    'mindspore.scipy',
    'mindspore.utils',
    'mindspore.experimental',
]

def get_mindspore_full_api_names():
    apis = []
    visited_modules = set()

    for module_name in modules_to_include:
        if module_name in visited_modules:
            continue
        visited_modules.add(module_name)

        try:
            module = __import__(module_name, fromlist=[''])
        except ImportError as e:
            print(f"Cannot import module {module_name}: {e}")
            continue

        # 将模块本身添加到API列表中
        apis.append({
            "name": module_name.split('.')[-1],
            "module": '.'.join(module_name.split('.')[:-1]),
            "fullName": module_name,
            "signature": "",
            "description": inspect.getdoc(module).split('\n')[0] if inspect.getdoc(module) else "No description available."
        })

        try:
            members = inspect.getmembers(module)
        except Exception as e:
            print(f"Skipping module {module_name} due to error: {e}")
            continue

        for name, member in members:
            if name.startswith('_'):
                continue

            full_name = module_name + '.' + name

            if inspect.isclass(member) or inspect.isfunction(member):
                # 检查文档字符串是否包含 "deprecated"
                doc = inspect.getdoc(member) or "No description available."
                if "deprecated" in doc.lower():
                    continue

                try:
                    signature = str(inspect.signature(member))
                except (ValueError, TypeError):
                    signature = "N/A"

                apis.append({
                    "name": name,
                    "module": module_name,
                    "fullName": full_name,
                    "signature": signature,
                    "description": doc.split('\n')[0]
                })

            elif inspect.ismodule(member):
                # 如果子模块在 modules_to_include 中，才继续处理
                if full_name in modules_to_include and full_name not in visited_modules:
                    visited_modules.add(full_name)
                    apis.append({
                        "name": name,
                        "module": module_name,
                        "fullName": full_name,
                        "signature": "",
                        "description": inspect.getdoc(member).split('\n')[0] if inspect.getdoc(member) else "No description available."
                    })

                    # 获取子模块的成员（不再递归，防止深入未指定的子模块）
                    try:
                        sub_members = inspect.getmembers(member)
                    except Exception as e:
                        print(f"Skipping module {full_name} due to error: {e}")
                        continue

                    for sub_name, sub_member in sub_members:
                        if sub_name.startswith('_'):
                            continue

                        sub_full_name = full_name + '.' + sub_name

                        if inspect.isclass(sub_member) or inspect.isfunction(sub_member):
                            doc = inspect.getdoc(sub_member) or "No description available."
                            if "deprecated" in doc.lower():
                                continue

                            try:
                                signature = str(inspect.signature(sub_member))
                            except (ValueError, TypeError):
                                signature = "N/A"

                            apis.append({
                                "name": sub_name,
                                "module": full_name,
                                "fullName": sub_full_name,
                                "signature": signature,
                                "description": doc.split('\n')[0]
                            })
                else:
                    # 子模块不在 modules_to_include 中，忽略
                    continue

    return apis

# 获取 MindSpore 的 API
apis = get_mindspore_full_api_names()

# 打印收集到的 API 数量
print(f"Total APIs collected: {len(apis)}")

# 将 API 写入 JSON 文件
with open('mindspore_apis.json', 'w') as f:
    json.dump(apis, f, indent=2, ensure_ascii=False)
