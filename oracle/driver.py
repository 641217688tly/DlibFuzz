import ast
import os
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import jax
import torch
import numpy as np
import tensorflow as tf
import jax.numpy as jnp
import multiprocessing
import traceback


# 设置随机种子函数
def set_seed(seed=42):
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    jax_key = jax.random.PRNGKey(seed)
    return jax_key


def traverse_and_execute_seeds(seed_dir):
    """
    并行化处理文件，并执行其中的代码片段
    """
    results = []
    file_paths = []

    # 获取所有待处理的文件路径
    for root, dirs, files in os.walk(seed_dir):
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


def execute_code_snippets(file_path: str, code_lines: list):
    """
    拆解出 PyTorch、TensorFlow 和 JAX 代码片段，执行后返回结果
    """
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
        if "error" in pytorch_result:
            results["results"].append({"framework": "PyTorch", "error": pytorch_result["error"]})
        else:
            results["results"].append({"framework": "PyTorch", "result": pytorch_result})
    except Exception as e:
        print(f"PyTorch code failed: {e}")
        results["results"].append({"framework": "PyTorch", "error": str(e)})

    # 执行 TensorFlow 代码
    try:
        tensorflow_result = run_tensorflow_code(common_code, tensorflow_code)
        if "error" in tensorflow_result:
            results["results"].append({"framework": "TensorFlow", "error": tensorflow_result["error"]})
        else:
            results["results"].append({"framework": "TensorFlow", "result": tensorflow_result})
    except Exception as e:
        print(f"TensorFlow code failed: {e}")
        results["results"].append({"framework": "TensorFlow", "error": str(e)})

    # 执行 JAX 代码
    try:
        jax_result = run_jax_code(common_code, jax_code)
        if "error" in jax_result:
            results["results"].append({"framework": "JAX", "error": jax_result["error"]})
        else:
            results["results"].append({"framework": "JAX", "result": jax_result})
    except Exception as e:
        print(f"JAX code failed: {e}")
        results["results"].append({"framework": "JAX", "error": str(e)})

    return results


def run_pytorch_code(common_code, pytorch_code):
    # 设置随机种子，确保每次运行一致
    set_seed(42)
    # 先对 PyTorch 代码进行 print 提取和修改
    modified_pytorch_code, output_vars = extract_and_modify_print_statements(pytorch_code, "output_pt")
    code = "\n".join(common_code + modified_pytorch_code)
    exec_locals = {}

    try:
        # 执行代码并打印调试信息
        print("Executing PyTorch code...")
        exec(code, {}, exec_locals)
        print("Execution completed.")

        outputs = {}
        for var in output_vars:
            if var in exec_locals:
                try:
                    output = exec_locals[var]
                    if isinstance(output, torch.Tensor):
                        outputs[var] = output.tolist()  # 转为列表
                    elif isinstance(output, np.ndarray):
                        outputs[var] = output.tolist()  # 转为列表
                    else:
                        json.dumps(output)  # 测试是否可序列化
                        outputs[var] = output
                except Exception as e:
                    outputs[var] = f"Unserializable output of type {type(output).__name__}: {str(e)}"
        return outputs if outputs else {"error": "No valid output found"}
    except Exception as e:
        return {"error": f"PyTorch code execution failed: {str(e)}"}


def run_tensorflow_code(common_code, tensorflow_code):
    def target(return_dict):
        try:
            # 设置随机种子，确保每次运行一致
            set_seed(42)
            # 先对 TensorFlow 代码进行 print 提取和修改
            modified_tensorflow_code, output_vars = extract_and_modify_print_statements(tensorflow_code, "output_tf")
            code = "\n".join(common_code + modified_tensorflow_code)
            exec_locals = {}

            # 执行代码并打印调试信息
            print("Executing TensorFlow code...")
            exec(code, {}, exec_locals)
            print("Execution completed.")

            outputs = {}
            for var in output_vars:
                if var in exec_locals:
                    try:
                        output = exec_locals[var]
                        # 检查类型并进行转换
                        if isinstance(output, tf.Tensor):
                            outputs[var] = output.numpy().tolist()  # 将 TensorFlow 张量转为列表
                        elif isinstance(output, np.ndarray):
                            outputs[var] = output.tolist()  # 将 NumPy 数组转为列表
                        elif isinstance(output, list):
                            outputs[var] = output  # 已经是标准 Python 列表，直接使用
                        elif isinstance(output, (np.float32, tf.float32, tf.float64, np.float64)):
                            outputs[var] = float(output)  # 转换为 Python 的 float 类型
                        else:
                            try:
                                json.dumps(output)  # 测试是否可以序列化
                                outputs[var] = output
                            except TypeError as e:
                                outputs[var] = f"Unserializable output of type {type(output).__name__}: {str(e)}"
                    except Exception as e:
                        print(f"Failed to process output: {str(e)}")
                        outputs[var] = f"Unserializable output of type {type(output).__name__}: {str(e)}"

            return_dict["result"] = outputs if outputs else {"error": "No valid output found"}
        except Exception as e:
            return_dict["error"] = f"TensorFlow execution failed: {traceback.format_exc()}"

    # 使用共享字典来获取子进程的结果
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # 启动子进程
    p = multiprocessing.Process(target=target, args=(return_dict,))
    p.start()

    # 设置超时时间
    p.join(timeout=32)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"error": "TensorFlow execution timed out"}

    if "error" in return_dict:
        return {"error": return_dict["error"]}
    return return_dict.get("result", None)


def run_jax_code(common_code, jax_code):
    # 设置随机种子，确保每次运行一致
    set_seed(42)
    # 先对 JAX 代码进行 print 提取和修改
    modified_jax_code, output_vars = extract_and_modify_print_statements(jax_code, "output_jax")
    code = "\n".join(common_code + modified_jax_code)
    exec_locals = {}

    try:
        # 执行代码并打印调试信息
        print("Executing JAX code...")
        exec(code, {}, exec_locals)
        print("Execution completed.")

        outputs = {}
        for var in output_vars:
            if var in exec_locals:
                try:
                    output = exec_locals[var]
                    if isinstance(output, jnp.ndarray):
                        outputs[var] = output.tolist()
                    elif isinstance(output, np.ndarray):
                        outputs[var] = output.tolist()
                    else:
                        json.dumps(output)
                        outputs[var] = output
                except Exception as e:
                    outputs[var] = f"Unserializable output of type {type(output).__name__}: {str(e)}"
        return outputs if outputs else {"error": "No valid output found"}
    except Exception as e:
        return {"error": f"JAX code execution failed: {str(e)}"}


def extract_and_modify_print_statements(code_lines, prefix):
    """
    从代码中提取 `print` 语句中的变量和表达式。返回修改后的代码和有效输出变量列表。
    """
    modified_code_lines = []
    output_vars = []
    output_var_counter = 1

    for line in code_lines:
        try:
            tree = ast.parse(line)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'print':
                    print(f"Found print call in line: {line.strip()}")
                    new_assignments = []
                    # 遍历 `print` 语句中的所有参数
                    for arg in node.args:
                        # 忽略常量字符串
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            continue

                        # 创建新变量名，确保每个print输出表达式都有独立的变量
                        var_name = f"{prefix}_{output_var_counter}"
                        new_assignments.append(var_name)
                        output_vars.append(var_name)

                        # 处理可能的属性访问或下标访问，并将其转为新变量的赋值语句
                        full_expr = ast.unparse(arg)
                        modified_code_lines.append(f"{var_name} = {full_expr}")
                        output_var_counter += 1

                    # 重构 `print` 语句，使用新的变量来替代表达式
                    new_print_stmt = f"print({', '.join(new_assignments)})"
                    modified_code_lines.append(new_print_stmt)
                    break
            else:
                # 非 print 语句，直接添加原代码
                modified_code_lines.append(line.strip())

        except SyntaxError as e:
            print(f"Syntax error while parsing line: {line.strip()} - {e}")
            modified_code_lines.append(line.strip())

    return modified_code_lines, output_vars


def convert_ndarray_to_list(obj):
    # 将 TensorFlow Tensor 转换为列表
    if isinstance(obj, tf.Tensor):
        return obj.numpy().tolist()
    # 将 PyTorch Tensor 转换为 NumPy 并转为列表
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    # 处理 JAX 数组
    elif isinstance(obj, jnp.ndarray):
        return obj.tolist()
    # 处理 NumPy 数组
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # 处理 TensorFlow 数据类型 (如 tf.float32, tf.float64, tf.int32)
    elif isinstance(obj, (tf.dtypes.DType, np.float32, np.float64, np.int32, np.int64)):
        # 转换为 Python 的基本类型
        return float(obj)
    elif isinstance(obj, tf.Variable):
        return obj.numpy().tolist()  # 处理 tf.Variable
    # 处理字典类型
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    # 处理列表类型
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
        "timeout_results": [],  # 新增timeout分类
        "none_outputs": []
    }

    def convert_to_comparable_format(result):
        # 如果是 PyTorch 张量，转换为 numpy
        if isinstance(result, torch.Tensor):
            return result.detach().cpu().numpy().tolist()
        # 如果是 TensorFlow 张量，转换为 numpy
        elif isinstance(result, tf.Tensor):
            return result.numpy().tolist()
        # 如果是 JAX ，转换为 numpy
        elif isinstance(result, jnp.ndarray):
            return result.tolist()
        # 如果是 numpy 数组，转换为列表
        elif isinstance(result, np.ndarray):
            return result.tolist()
        return result

    def are_results_close(result1, result2, tolerance=1e-5):
        result1 = convert_to_comparable_format(result1)
        result2 = convert_to_comparable_format(result2)

        # 递归地比较数值、列表、字典是否相近
        if isinstance(result1, (int, float)) and isinstance(result2, (int, float)):
            return abs(result1 - result2) < tolerance
        elif isinstance(result1, list) and isinstance(result2, list):
            return len(result1) == len(result2) and all(
                are_results_close(r1, r2, tolerance) for r1, r2 in zip(result1, result2))
        elif isinstance(result1, dict) and isinstance(result2, dict):
            result1_values = list(result1.values())
            result2_values = list(result2.values())
            return are_results_close(result1_values, result2_values, tolerance)
        return result1 == result2

    def are_results_exactly_equal(result1, result2):
        result1 = convert_to_comparable_format(result1)
        result2 = convert_to_comparable_format(result2)

        if isinstance(result1, (int, float)) and isinstance(result2, (int, float)):
            return result1 == result2
        elif isinstance(result1, list) and isinstance(result2, list):
            return len(result1) == len(result2) and all(
                are_results_exactly_equal(r1, r2) for r1, r2 in zip(result1, result2))
        elif isinstance(result1, dict) and isinstance(result2, dict):
            result1_values = list(result1.values())
            result2_values = list(result2.values())
            return are_results_exactly_equal(result1_values, result2_values)
        return result1 == result2

    for result in all_results:
        try:
            file_path = result["file"]
            results = result["results"]

            # 分离 timeout 错误
            timeout_found = False
            for res in results:
                if "error" in res and "TensorFlow execution timed out" in res["error"]:
                    analysis["timeout_results"].append({
                        "file": file_path,
                        "error": "TensorFlow execution timed out",
                        "details": results
                    })
                    timeout_found = True
                    break

            if timeout_found:
                continue

            # 检查是否存在其他错误
            if any("error" in r for r in results):
                analysis["error_results"].append({
                    "file": file_path,
                    "error": "Error in one or more frameworks",
                    "details": results
                })
                continue

            # 提取每个框架的输出 (多个变量可能存在)
            outputs = [r.get("result") for r in results if "result" in r]

            # 检查是否所有输出都为 None
            if all(output is None for output in outputs):
                analysis["none_outputs"].append({
                    "file": file_path,
                    "details": results
                })
                continue

            # 将所有输出标准化为可比较的格式
            try:
                normalized_outputs = [convert_ndarray_to_list(output) for output in outputs]
            except TypeError as e:
                analysis["error_results"].append({
                    "file": file_path,
                    "error": f"Failed to normalize outputs: {str(e)}",
                    "details": results
                })
                continue

            # 比较每个输出中的变量
            first_framework_output = normalized_outputs[0]
            consistent = True
            approximate = True

            for output in normalized_outputs[1:]:
                if not are_results_exactly_equal(first_framework_output, output):
                    consistent = False
                if not are_results_close(first_framework_output, output):
                    approximate = False
                    break

            # 检查输出是否一致
            if consistent:
                print(f"Consistent results for {file_path}")
                analysis["consistent_results"].append({
                    "file": file_path,
                    "result": normalized_outputs[0]
                })
            elif approximate:
                analysis["approximate_results"].append({
                    "file": file_path,
                    "results": normalized_outputs
                })
            else:
                analysis["divergent_results"].append({
                    "file": file_path,
                    "results": normalized_outputs
                })

        except Exception as e:
            analysis["error_results"].append({
                "file": file_path if 'file_path' in locals() else "Unknown file",
                "error": f"Failed during analysis: {str(e)}"
            })

    return analysis


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
                            # 序列化每个结果项
                            json.dump(item, f, indent=4)
                            first_item = False
                        except Exception as e:
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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽 INFO 和 WARNING 消息

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
