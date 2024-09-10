import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import time
import tensorflow as tf
import jax.numpy as jnp
import multiprocessing
import traceback


def execute_code_snippets(file_path: str, code_lines: list):
    # 分离公共部分和各库的代码片段
    print(f"Executing code snippets in {file_path}")
    common_code = []
    pytorch_code = []
    tensorflow_code = []
    jax_code = []

    current_section = None

    results = {
        "file": file_path,
        "results": []
    }

    for line in code_lines:
        if re.search(r'#\s*pytorch', line, re.IGNORECASE):
            current_section = "pytorch"
            continue
        elif re.search(r'#\s*tensorflow', line, re.IGNORECASE):
            current_section = "tensorflow"
            continue
        elif re.search(r'#\s*jax', line, re.IGNORECASE):
            current_section = "jax"
            continue

        if current_section == "pytorch":
            pytorch_code.append(line)
        elif current_section == "tensorflow":
            tensorflow_code.append(line)
        elif current_section == "jax":
            jax_code.append(line)
        else:
            common_code.append(line)

    # 执行 PyTorch 代码
    try:
        pytorch_result = run_pytorch_code(common_code, pytorch_code)
        results["results"].append({"framework": "PyTorch", "result": pytorch_result})
    except Exception as e:
        print(f"PyTorch code failed: {e}")
        results["results"].append({"framework": "PyTorch", "error": str(e)})

    # 执行 TensorFlow 代码
    try:
        tensorflow_result = run_tensorflow_code(common_code, tensorflow_code)
        results["results"].append({"framework": "TensorFlow", "result": tensorflow_result})
    except Exception as e:
        print(f"TensorFlow code failed: {e}")
        results["results"].append({"framework": "TensorFlow", "error": str(e)})

    # 执行 JAX 代码
    try:
        jax_result = run_jax_code(common_code, jax_code)
        results["results"].append({"framework": "JAX", "result": jax_result})
    except Exception as e:
        print(f"JAX code failed: {e}")
        results["results"].append({"framework": "JAX", "error": str(e)})

    return results

def traverse_and_execute_seeds(base_dir):
    """
    并行化处理文件，并执行其中的代码片段
    """
    results = []
    file_paths = []

    # 获取所有待处理的文件路径
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.readlines()
                file_paths.append((file_path, content))

    # 使用 ThreadPoolExecutor 处理并发
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(execute_code_snippets, path, content): path for path, content in file_paths}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error executing {futures[future]}: {str(e)}")

    return results


# def traverse_and_execute_seeds(base_dir):
#     """
#     遍历Python 文件，并执行其中的代码片段
#     """
#     results = []
#     for root, dirs, files in os.walk(base_dir):
#         for file in files:
#             if file.endswith(".py"):
#                 file_path = os.path.join(root, file)
#                 print(f"Executing seed file: {file_path}")
#                 result = execute_code_snippets(file_path)
#                 results.append(result)
#     return results


def run_pytorch_code(common_code, pytorch_code):
    code = "\n".join(common_code + pytorch_code)
    exec_locals = {}
    exec(code, {}, exec_locals)

    # 尝试获取多个可能的输出变量
    output = exec_locals.get('output_pt')

    if output is None:
        return None
    # 如果是 Tensor，尝试转为数值或列表
    elif isinstance(output, torch.Tensor):
        return output.item() if output.numel() == 1 else output.tolist()
    else:
        try:
            # 尝试直接返回对象，如果不可序列化会抛出异常
            json.dumps(output)
            return output
        except TypeError:
            # 如果不可序列化，返回错误信息
            return f"Unserializable output of type {type(output).__name__}"


def run_tensorflow_code(common_code, tensorflow_code):
    def target(return_dict):
        try:
            code = "\n".join(common_code + tensorflow_code)
            exec_locals = {}
            exec(code, {}, exec_locals)

            output = exec_locals.get('output_tf')

            if output is None:
                return_dict["result"] = None
            elif isinstance(output, tf.Tensor):
                # 如果输出是一个形状为 () 的标量张量，直接返回标量值
                if output.shape == ():
                    return_dict["result"] = output.numpy().item()
                # 如果输出是形状为 (1, 1) 的张量，简化为标量
                elif output.shape == (1, 1):
                    return_dict["result"] = output.numpy().item()
                else:
                    # 否则，将张量转化为列表
                    return_dict["result"] = output.numpy().tolist()
            else:
                # 尝试将其他类型也转换为可序列化格式
                try:
                    json.dumps(output)
                    return_dict["result"] = output
                except TypeError:
                    # 如果不可序列化，返回错误信息
                    return_dict["result"] = f"Unserializable output of type {type(output).__name__}"
        except Exception as e:
            return_dict["error"] = f"{traceback.format_exc()}"

    # 使用共享字典来获取子进程的结果
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # 启动子进程
    p = multiprocessing.Process(target=target, args=(return_dict,))
    p.start()

    # 设置超时时间
    p.join(timeout=6)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"error": "TensorFlow execution timed out"}

    # 如果子进程有错误或结果，返回相应信息
    if "error" in return_dict:
        return {"result": f"TensorFlow execution failed: {return_dict['error']}"}
    return return_dict.get("result", None)


# def run_tensorflow_code(common_code, tensorflow_code):
#     code = "\n".join(common_code + tensorflow_code)
#     exec_globals = globals().copy()
#     exec(code, exec_globals)
#
#     # 尝试获取多个可能的输出变量
#     output = exec_globals.get('output_tf')
#
#     if output is None:
#         return None
#     elif isinstance(output, tf.Tensor):
#         # 如果输出是一个形状为 () 的标量张量，直接返回标量值
#         if output.shape == ():
#             return output.numpy().item()
#         # 如果输出是形状为 (1, 1) 的张量，简化为标量
#         elif output.shape == (1, 1):
#             return output.numpy().item()
#         else:
#             # 否则，将张量转化为列表
#             return output.numpy().tolist()
#     else:
#         # 返回原始对象
#         return output


def run_jax_code(common_code, jax_code):
    code = "\n".join(common_code + jax_code)
    exec_locals = {}
    exec(code, {}, exec_locals)

    output = exec_locals.get('output_jax')

    if output is None:
        return None
    elif isinstance(output, jnp.ndarray):
        # 如果是 JAX 数组，尝试转为列表或标量
        try:
            return output.tolist() if output.size > 1 else float(output)
        except Exception as e:
            print(f"Error converting JAX output: {e}")
            return f"Error converting JAX output: {e}"
    else:
        # 尝试将其他类型也转换为可序列化格式
        try:
            json.dumps(output)
            return output
        except TypeError:
            # 如果不可序列化，返回错误信息
            return f"Unserializable output of type {type(output).__name__}"


def convert_ndarray_to_list(obj):
    # 将 TensorFlow Tensor 转换为列表
    if isinstance(obj, tf.Tensor):
        return obj.numpy().tolist()
    # 将 PyTorch Tensor 转换为 NumPy 并转为列表
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    # 处理 NumPy 数组
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # 处理字典
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    # 处理列表
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    # 对于其他类型，直接返回
    return obj


def analyze_results(all_results):
    analysis = {
        "consistent_results": [],
        "approximate_results": [],
        "divergent_results": [],
        "error_results": [],
        "none_outputs": []
    }

    def are_results_close(result1, result2, tolerance=1e-5):
        if isinstance(result1, (int, float)) and isinstance(result2, (int, float)):
            return abs(result1 - result2) < tolerance
        elif isinstance(result1, tuple) and isinstance(result2, tuple):
            return all(are_results_close(r1, r2, tolerance) for r1, r2 in zip(result1, result2))
        else:
            return result1 == result2

    for result in all_results:
        try:
            file_path = result["file"]
            results = result["results"]

            # 检查是否所有结果都是错误
            if all("error" in r for r in results):
                analysis["error_results"].append({
                    "file": file_path,
                    "error": "All frameworks failed",
                    "details": results
                })
                continue

            # 检查是否有任何结果为 None
            if any(r.get("result") is None for r in results):
                analysis["none_outputs"].append({
                    "file": file_path,
                    "details": results
                })
                continue

            # 检查是否有任何结果为非数值
            numeric_results = [r["result"] for r in results if "result" in r]
            if len(numeric_results) < 3:
                analysis["error_results"].append({
                    "file": file_path,
                    "error": "Partial failure",
                    "details": results
                })
                continue

            # 尝试将结果标准化为可哈希的格式
            try:
                normalized_results = [convert_to_hashable(res) for res in numeric_results]
            except TypeError:
                analysis["error_results"].append({
                    "file": file_path,
                    "error": "Failed to normalize results",
                    "details": results
                })
                continue

            # 检查结果是否一致
            if len(set(normalized_results)) == 1:
                analysis["consistent_results"].append({
                    "file": file_path,
                    "result": normalized_results[0]
                })
            # 检查结果是否近似一致
            elif all(are_results_close(normalized_results[0], res) for res in normalized_results[1:]):
                analysis["approximate_results"].append({
                    "file": file_path,
                    "results": normalized_results
                })
            # 结果不一致
            else:
                analysis["divergent_results"].append({
                    "file": file_path,
                    "results": normalized_results
                })
        except Exception as e:
            analysis["error_results"].append({
                "file": file_path if 'file_path' in locals() else "Unknown file",
                "error": f"Failed during analysis: {str(e)}"
            })

    return analysis


def convert_to_hashable(item):
    if isinstance(item, list):
        return tuple(convert_to_hashable(sub_item) for sub_item in item)
    return item


def get_next_available_filename(directory, base_name, extension):
    """
    生成新的文件名
    """
    files = os.listdir(directory)
    existing_files = [f for f in files if re.match(rf'{base_name}_(\d+){re.escape(extension)}', f)]

    if not existing_files:
        return f"{base_name}_1{extension}"
    else:
        numbers = [int(re.search(rf'{base_name}_(\d+){re.escape(extension)}', f).group(1)) for f in existing_files]
        next_number = max(numbers) + 1
        return f"{base_name}_{next_number}{extension}"


def write_safe_to_file(results, output_file):
    """
    将结果逐条写入文件，如果某条记录无法写入，记录错误并继续写入后续内容。
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        if isinstance(results, dict):
            f.write("{\n")  # 开始写入 JSON 对象
            first_key = True
            for key, value in results.items():
                if not first_key:
                    f.write(",\n")
                f.write(f'"{key}": ')

                if isinstance(value, list):
                    f.write("[\n")
                    first_item = True
                    for item in value:
                        try:
                            if not first_item:
                                f.write(",\n")
                            # 尝试序列化正常的 item
                            json.dump(item, f, indent=4)
                            first_item = False
                        except Exception as e:
                            # 捕获序列化错误并记录错误信息作为列表中的一部分
                            error_msg = f"Failed to serialize result: {str(e)}"
                            if not first_item:
                                f.write(",\n")
                            f.write(json.dumps(error_msg, indent=4))
                            first_item = False
                    f.write("\n]")
                else:
                    try:
                        # 如果是其他类型，直接序列化
                        json.dump(value, f, indent=4)
                    except Exception as e:
                        # 处理单个值的序列化错误
                        error_msg = f"Failed to serialize key '{key}': {str(e)}"
                        f.write(json.dumps(error_msg, indent=4))
                first_key = False
            f.write("\n}")

        elif isinstance(results, list):
            f.write("[\n")
            first = True
            for result in results:
                try:
                    if not first:
                        f.write(",\n")
                    json.dump(result, f, indent=4)
                    first = False
                except Exception as e:
                    error_msg = {"error": f"Failed to serialize result: {str(e)}"}
                    if not first:
                        f.write(",\n")
                    f.write(json.dumps(error_msg, indent=4))
                    first = False
            f.write("\n]")


if __name__ == "__main__":
    start_time = time.time()

    project_root = os.path.dirname(os.path.dirname(__file__))
    # seeds_dir = os.path.join(project_root, 'fuzzer/seeds/test_seeds/test')  # 测试换文件夹用
    seeds_dir = os.path.join(project_root, 'fuzzer/seeds/test_seeds/zero-shot')
    output_dir = os.path.join(project_root, 'oracle/outputs')

    results_file = get_next_available_filename(output_dir, "results", ".json")
    analysis_file = get_next_available_filename(output_dir, "analysis", ".json")

    all_results = traverse_and_execute_seeds(seeds_dir)
    # 确保所有 Tensor 和 NumPy 对象都被转换为可序列化格式
    all_results = convert_ndarray_to_list(all_results)
    results_path = os.path.join(output_dir, results_file)
    write_safe_to_file(all_results, results_path)

    # 分析结果并写入分析文件
    analysis = analyze_results(all_results)
    # 确保所有 Tensor 和 NumPy 对象都被转换为可序列化格式
    analysis = convert_ndarray_to_list(analysis)
    analysis_path = os.path.join(output_dir, analysis_file)
    write_safe_to_file(analysis, analysis_path)

    end_time = time.time()
    print(f"Execution completed in {end_time - start_time:.2f} seconds.")
