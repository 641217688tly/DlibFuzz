'''
尝试优化后的api_extracter.py代码:

import importlib
import torch
import inspect
from collections import deque
import warnings

def check_if_deprecated(function):
    """检查函数是否被标记为弃用."""
    # 检查文档字符串中的弃用信息(可过滤近200个API)
    docstring = inspect.getdoc(function)
    if docstring and ('deprecated' or 'removed') in docstring.lower():
        return True

    # 捕获DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            # 尝试导入function
            importlib.import_module(function.__module__)
            pass
        except Exception as e:
            print(f"Error calling function {function.__name__}: {e}")
            return True
        for warning in w:
            if warning.category == DeprecationWarning or UserWarning:
                return True
    return False


def extract_apis(module):
    queue = deque([module])  # 存储模块
    visited = set()
    count = 0  # 初始化API计数器

    while queue:
        mod = queue.popleft()
        # 避免重复访问
        if mod in visited:
            continue
        visited.add(mod)

        try:
            members = inspect.getmembers(mod)  # 获取模块的成员
        except (TypeError, ModuleNotFoundError) as e:
            print(f'Error with module {mod.__name__}: {e}')
            continue

        for name, obj in members:
            if inspect.ismodule(obj):
                queue.append(obj)  # 只有模块才加入队列继续遍历
            elif inspect.isfunction(obj):
                if not check_if_deprecated(obj):  # 检查函数是否被弃用
                    count += 1  # 非弃用函数计数
                    print(f'Function: {name}')

    print(f'Total non-deprecated function APIs found: {count}')  # 打印非弃用函数的总数

# 提取PyTorch中的API
extract_apis(torch)
'''


'''
inspect的使用样例:

import inspect
import warnings
import jax.debug

# 获取函数对象
function = jax.debug.visualize_sharding1

# 提取API名称
api_name = function.__name__
print(f"API Name: {api_name}")

# 提取API所属模块
api_module = function.__module__
print(f"API Module: {api_module}")

# 提取完整的API名称
full_api_name = f"{api_module}.{api_name}"
print(f"Full API Name: {full_api_name}")

# 获取函数签名
signature = inspect.signature(function)
print(f"Function Signature: {signature}")

# 获取函数文档字符串
docstring = inspect.getdoc(function)
print(f"Function Docstring: {docstring}")


def check_if_deprecated(function):
    # 查看文档字符串中的弃用信息
    docstring = inspect.getdoc(function)
    if docstring and 'deprecated' in docstring.lower():
        return True

    # 捕获DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        try:
            # 尝试调用函数
            function()
        except Exception:
            pass
        return any(item.category == DeprecationWarning for item in w)

is_deprecated = check_if_deprecated(function)
print(f"{function.__name__} is {'deprecated' if is_deprecated else 'not deprecated'}")

# 检查对象类型
if inspect.isfunction(function):
    print(f"{full_api_name} is a function.")
elif inspect.isclass(function):
    print(f"{full_api_name} is a class.")
else:
    print(f"{full_api_name} is neither a function nor a class.")

'''

import jax
import torch
import tensorflow as tf
import inspect
import types
import sys
import os


# 定义一个函数来获取指定模块及其子模块的所有API签名
def get_apis(module, module_name, max_depth=10):
    signatures = {}  # 存储API签名的字典
    visited = set()  # 记录已访问模块的集合

    # 内部函数，递归获取模块成员的签名
    def _get_info(mod, name, depth):
        # 如果当前递归深度超过最大深度或模块已访问，则返回
        if depth > max_depth or mod in visited:
            return
        visited.add(mod)  # 将当前模块标记为已访问

        try:
            members = inspect.getmembers(mod)  # 获取模块的所有成员
        except Exception as e:
            # 如果获取成员时出错，记录错误并返回
            signatures[name] = f"Error retrieving members: {e}"
            return

        # 遍历所有成员
        for subname, obj in members:
            full_name = f"{name}.{subname}"  # 构建完整成员名称
            if inspect.isfunction(obj) or inspect.isclass(obj):
                try:
                    # 获取函数或类的签名
                    signatures[full_name] = str(inspect.signature(obj))
                except (TypeError, ValueError):
                    # 如果获取签名时出错，记录“签名不可用”
                    signatures[full_name] = "Signature not available"
            elif isinstance(obj, types.ModuleType) and obj.__name__.startswith(module_name):
                # 如果成员是模块且名称以指定模块名称开头，递归获取子模块的签名
                _get_info(obj, full_name, depth + 1)

    _get_info(module, module_name, 0)  # 从根模块开始获取签名
    return signatures  # 返回包含所有签名的字典


def save_apis_to_file(signatures, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for name, sig in signatures.items():
            f.write(f"{name}: {sig}\n")


if __name__ == "__main__":
    # 增加递归限制
    sys.setrecursionlimit(2000)

    # 获取torch模块的所有API签名
    torch_signatures = get_apis(torch, "torch")
    # 打印PyTorch签名总数
    print(f"Total PyTorch APIs: {len(torch_signatures)}")  # 5105
    # 保存PyTorch签名到文件
    save_apis_to_file(torch_signatures, "../cluster/apis/pytorch/torch_apis")

    # 获取tensorflow模块的所有API签名
    tf_signatures = get_apis(tf, "tensorflow")
    # 打印TensorFlow签名总数
    print(f"Total TensorFlow APIs: {len(tf_signatures)}")  # 30562
    # 保存TensorFlow签名到文件
    save_apis_to_file(tf_signatures, "../cluster/apis/tensorflow/tf_apis")

    # 获取jax模块的所有API签名
    jax_signatures = get_apis(jax, "jax")
    # 打印JAX签名总数
    print(f"Total JAX APIs: {len(jax_signatures)}")
    # 保存JAX签名到文件
    save_apis_to_file(jax_signatures, "../cluster/apis/jax/jax_apis")
