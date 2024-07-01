import jax
import torch
import tensorflow as tf
import inspect
import types
import sys
import os


# 定义一个函数来获取指定模块及其子模块的所有API签名
def get_api_signatures(module, module_name, max_depth=10):
    signatures = {}  # 存储API签名的字典
    visited = set()  # 记录已访问模块的集合

    # 内部函数，递归获取模块成员的签名
    def _get_signatures(mod, name, depth):
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
                _get_signatures(obj, full_name, depth + 1)

    _get_signatures(module, module_name, 0)  # 从根模块开始获取签名
    return signatures  # 返回包含所有签名的字典


def save_signatures_to_file(signatures, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for name, sig in signatures.items():
            f.write(f"{name}: {sig}\n")


if __name__ == "__main__":
    # 增加递归限制
    sys.setrecursionlimit(2000)

    # # 获取torch模块的所有API签名
    # torch_signatures = get_api_signatures(torch, "torch")
    # # 打印PyTorch签名总数
    # print(f"Total PyTorch API signatures: {len(torch_signatures)}")  # 5105
    # # 保存PyTorch签名到文件
    # save_signatures_to_file(torch_signatures, "api_signatures/pytorch/torch_apis")
    #
    # # 获取tensorflow模块的所有API签名
    # tf_signatures = get_api_signatures(tf, "tensorflow")
    # # 打印TensorFlow签名总数
    # print(f"Total TensorFlow API signatures: {len(tf_signatures)}")  # 30562
    # # 保存TensorFlow签名到文件
    # save_signatures_to_file(tf_signatures, "api_signatures/tensorflow/tf_apis")

    # 获取jax模块的所有API签名
    jax_signatures = get_api_signatures(jax, "jax")
    # 打印JAX签名总数
    print(f"Total JAX API signatures: {len(jax_signatures)}")
    # 保存JAX签名到文件
    save_signatures_to_file(jax_signatures, "api_signatures/jax/jax_apis")
