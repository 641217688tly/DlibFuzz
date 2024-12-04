import ast
import os
import json
import re
import time
import sys
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import jax
import torch
import numpy as np
import tensorflow as tf
import jax.numpy as jnp
import multiprocessing


# 设置随机种子函数
def set_seed(seed=42):
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    jax_key = jax.random.PRNGKey(seed)
    return jax_key


def traverse_and_execute_seeds(seed_dir):
    """
    Traverse the seed directory and execute code snippets concurrently.
    """
    results = []
    file_paths = []

    for file_dir, dirs, files in os.walk(seed_dir):
        # Only proceed if it's a seed directory containing library code files
        if any(file.endswith(('.py')) for file in files):
            library_files = {
                'pytorch': None,
                'tensorflow': None,
                'jax': None
            }

            for file in files:
                file_path = os.path.join(file_dir, file)
                if 'torch_seed' in file.lower():
                    library_files['pytorch'] = file_path
                elif 'tf_seed' in file.lower():
                    library_files['tensorflow'] = file_path
                elif 'jax_seed' in file.lower():
                    library_files['jax'] = file_path

            if any(library_files.values()):
                file_paths.append((file_dir, library_files))

    # Concurrent execution
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        try:
            futures = {executor.submit(execute_code_snippets, seed_dir, file_dir, library_files): file_dir for
                       file_dir, library_files in file_paths}
            for future in as_completed(futures):
                results.append(future.result())
        except Exception as e:
            print(f"Execution error: {e}")
        finally:
            executor.shutdown(wait=False)  # 强制关闭

    return results


def execute_code_snippets(seed_dir: str, file_dir: str, library_files: dict):
    """
    Execute PyTorch, TensorFlow, and JAX code snippets for each library in the seed directory.
    """

    relative_path = os.path.relpath(file_dir, seed_dir)

    pytorch_code, tensorflow_code, jax_code = [], [], []
    results = {
        "seed": relative_path,
        "results": []
    }

    if library_files['pytorch']:
        pytorch_code_path = library_files['pytorch']
    if library_files['tensorflow']:
        tensorflow_code_path = library_files['tensorflow']
    if library_files['jax']:
        jax_code_path = library_files['jax']

    # Execute PyTorch code
    try:
        pytorch_result = run_pytorch_code(pytorch_code_path)
        if "error" in pytorch_result:
            results["results"].append({"framework": "PyTorch", "error": pytorch_result["error"]})
        else:
            results["results"].append({"framework": "PyTorch", "result": pytorch_result})
    except Exception as e:
        print(f"PyTorch code failed: {e}")
        results["results"].append({"framework": "PyTorch", "error": str(e)})

    # Execute TensorFlow code
    try:
        tensorflow_result = run_tensorflow_code(tensorflow_code_path)
        if "error" in tensorflow_result:
            results["results"].append({"framework": "TensorFlow", "error": tensorflow_result["error"]})
        else:
            results["results"].append({"framework": "TensorFlow", "result": tensorflow_result})
    except Exception as e:
        print(f"TensorFlow code failed: {e}")
        results["results"].append({"framework": "TensorFlow", "error": str(e)})

    # Execute JAX code
    try:
        jax_result = run_jax_code(jax_code_path)
        if "error" in jax_result:
            results["results"].append({"framework": "JAX", "error": jax_result["error"]})
        else:
            results["results"].append({"framework": "JAX", "result": jax_result})
    except Exception as e:
        print(f"JAX code failed: {e}")
        results["results"].append({"framework": "JAX", "error": str(e)})

    return results


def run_pytorch_code(pytorch_code_path):
    set_seed(42)
    with open(pytorch_code_path, 'r') as f:
        code_lines = f.readlines()
    modified_pytorch_code, output_vars = extract_and_modify_print_statements(code_lines, "output_pt")
    exec_locals = {}

    try:
        # 执行代码并打印调试信息
        exec('\n'.join(modified_pytorch_code), globals(), exec_locals)

        outputs = {}
        for var in output_vars:
            if var in exec_locals:
                try:
                    output = exec_locals[var]
                    outputs[var] = convert_ndarray_to_list(output)
                except Exception as e:
                    outputs[var] = f"Unserializable output of type {type(output).__name__}: {str(e)}"
        return outputs if outputs else {"error": "No valid output found"}
    except Exception as e:
        # 只捕获错误消息的简短形式，而非完整的堆栈信息
        exc_type, exc_value, exc_traceback = sys.exc_info()
        return {"error": f"Pytorch code execution failed: {exc_type.__name__}: {exc_value}"}


def run_tensorflow_code(tensorflow_code_path):
    def target(return_dict):
        try:
            set_seed(42)
            with open(tensorflow_code_path, 'r') as f:
                code_lines = f.readlines()
            modified_tensorflow_code, output_vars = extract_and_modify_print_statements(code_lines, "output_tf")
            exec_locals = {}

            # Execute the modified TensorFlow code
            exec('\n'.join(modified_tensorflow_code), globals(), exec_locals)
            outputs = {}
            for var in output_vars:
                if var in exec_locals:
                    try:
                        output = exec_locals[var]
                        # Use convert_ndarray_to_list to process the output
                        outputs[var] = convert_ndarray_to_list(output)
                    except Exception as e:
                        outputs[var] = f"Unserializable output of type {type(output).__name__}: {str(e)}"

            return_dict["result"] = outputs if outputs else {"error": "No valid output found"}
        except Exception as e:
            # Capture and return the error message
            exc_type, exc_value, exc_traceback = sys.exc_info()
            return_dict["error"] = f"TensorFlow code execution failed: {exc_type.__name__}: {exc_value}"

    # Use a shared dictionary to get results from the child process
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # Start the child process
    p = multiprocessing.Process(target=target, args=(return_dict,))
    p.start()

    # Set a timeout
    p.join(timeout=6)
    if p.is_alive():
        p.terminate()
        p.join()
        return {"error": "TensorFlow execution timed out"}

    # Ensure return_dict is converted to a regular dict
    return_dict = dict(return_dict)

    if not return_dict:
        return {"error": "TensorFlow code execution failed: NoValidOutput: This might be due to an internal error."}
    elif "error" in return_dict:
        return {"error": return_dict["error"]}
    elif "result" in return_dict:
        return return_dict["result"]
    else:
        return {"error": "TensorFlow code execution failed: Unknown error"}


def run_jax_code(jax_code_path):
    set_seed(42)
    with open(jax_code_path, 'r') as f:
        code_lines = f.readlines()
    modified_jax_code, output_vars = extract_and_modify_print_statements(code_lines, "output_jax")
    exec_locals = {}

    try:
        exec('\n'.join(modified_jax_code), globals(), exec_locals)
        outputs = {}
        for var in output_vars:
            if var in exec_locals:
                try:
                    output = exec_locals[var]
                    outputs[var] = convert_ndarray_to_list(output)
                except Exception as e:
                    outputs[var] = f"Unserializable output of type {type(output).__name__}: {str(e)}"
        return outputs if outputs else {"error": "No valid output found"}
    except Exception as e:
        # 只捕获错误消息的简短形式，而非完整的堆栈信息
        exc_type, exc_value, exc_traceback = sys.exc_info()
        return {"error": f"Jax code execution failed: {exc_type.__name__}: {exc_value}"}


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
            has_print_statement = False  # 标记是否有print语句

            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'print':
                    has_print_statement = True
                    new_assignments = []
                    # 遍历 `print` 语句中的所有参数
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            continue

                        # 创建新变量名，确保每个 print 输出表达式都有独立的变量
                        var_name = f"{prefix}_{output_var_counter}"
                        new_assignments.append(var_name)
                        output_vars.append(var_name)

                        try:
                            full_expr = ast.unparse(arg)  # 转换表达式为代码行
                            modified_code_lines.append(f"{var_name} = {full_expr}")
                        except Exception as e:
                            print(f"Error unparsing argument in line: {line.strip()} - {e}")
                            modified_code_lines.append(line.strip())  # 保留原始行
                        output_var_counter += 1

                    # 重构 `print` 语句，使用新的变量来替代表达式
                    new_print_stmt = f"print({', '.join(new_assignments)})"
                    modified_code_lines.append(new_print_stmt)
                    break

            if not has_print_statement:
                modified_code_lines.append(line.rstrip('\n'))
        except SyntaxError as e:
            print(f"Syntax error while parsing line: {line.strip()} - {e}")
            modified_code_lines.append(line.rstrip('\n'))

    return modified_code_lines, output_vars


def convert_ndarray_to_list(obj):
    if isinstance(obj, tf.Tensor):
        array = obj.numpy()
        if array.ndim == 0:
            return array.item()
        else:
            return array.tolist()
    elif isinstance(obj, tf.Variable):
        array = obj.numpy()
        if array.ndim == 0:
            return array.item()
        else:
            return array.tolist()
    elif isinstance(obj, torch.Tensor):
        array = obj.detach().cpu().numpy()
        if array.ndim == 0:
            return array.item()
        else:
            return array.tolist()
    elif isinstance(obj, jnp.ndarray):
        array = np.array(obj)
        if array.ndim == 0:
            return array.item()
        else:
            return array.tolist()
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return obj.item()
        else:
            return obj.tolist()
    elif isinstance(obj, (np.generic, np.number)):
        return obj.item()
    elif isinstance(obj, (float, int, str, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_ndarray_to_list(item) for item in obj)
    else:
        return str(obj)  # Convert other types to string


def analyze_results(all_results):
    analysis = {
        "consistent_results": [],
        "approximate_results": [],
        "divergent_results": [],
        "error_results": [],
        "timeout_results": [],  # Timeout classification
        "none_outputs": []
    }

    def convert_to_comparable_format(result):
        # Convert PyTorch tensors to lists
        if isinstance(result, torch.Tensor):
            return result.detach().cpu().numpy().tolist()
        # Convert TensorFlow tensors to lists
        elif isinstance(result, tf.Tensor):
            array = result.numpy()
            if array.ndim == 0:
                return array.item()
            else:
                return array.tolist()
        # Convert JAX arrays to lists
        elif isinstance(result, jnp.ndarray):
            return result.tolist()
        # Convert NumPy arrays to lists
        elif isinstance(result, np.ndarray):
            return result.tolist()
        # Handle NumPy scalars
        elif isinstance(result, np.generic):
            return result.item()
        return result

    def normalize_output_keys(output_dict):
        normalized_dict = {}
        for key, value in output_dict.items():
            # Remove framework-specific prefixes
            if key.startswith("output_pt_"):
                new_key = key.replace("output_pt_", "output_")
            elif key.startswith("output_tf_"):
                new_key = key.replace("output_tf_", "output_")
            elif key.startswith("output_jax_"):
                new_key = key.replace("output_jax_", "output_")
            else:
                new_key = key  # Keep key as is if no known prefix
            normalized_dict[new_key] = value
        return normalized_dict

    def normalize_structure(output):
        if isinstance(output, list):
            if len(output) == 1:
                # Recursively normalize the single element
                return normalize_structure(output[0])
            else:
                # Normalize each element in the list
                return [normalize_structure(item) for item in output]
        elif isinstance(output, dict):
            # Normalize each value in the dictionary
            return {k: normalize_structure(v) for k, v in output.items()}
        else:
            return output  # Return the output as is if it's not a list or dict

    def are_results_close(result1, result2, tolerance=1e-5):
        result1 = convert_to_comparable_format(result1)
        result2 = convert_to_comparable_format(result2)

        # Recursively compare numbers, lists, and dictionaries
        if isinstance(result1, (int, float)) and isinstance(result2, (int, float)):
            if math.isinf(result1) and math.isinf(result2):
                return (result1 > 0) == (result2 > 0)
            if math.isnan(result1) and math.isnan(result2):
                return True
            return abs(result1 - result2) < tolerance
        elif isinstance(result1, list) and isinstance(result2, list):
            if len(result1) != len(result2):
                return False
            return all(are_results_close(r1, r2, tolerance) for r1, r2 in zip(result1, result2))
        elif isinstance(result1, dict) and isinstance(result2, dict):
            if result1.keys() != result2.keys():
                return False
            return all(are_results_close(result1[k], result2[k], tolerance) for k in result1)
        else:
            return result1 == result2

    def are_results_exactly_equal(result1, result2):
        result1 = convert_to_comparable_format(result1)
        result2 = convert_to_comparable_format(result2)

        if isinstance(result1, (int, float, str, bool, type(None))) and isinstance(result2,
                                                                                   (int, float, str, bool, type(None))):
            if isinstance(result1, float) and isinstance(result2, float):
                if math.isinf(result1) and math.isinf(result2):
                    return (result1 > 0) == (result2 > 0)
                if math.isnan(result1) and math.isnan(result2):
                    return True
            return result1 == result2
        elif isinstance(result1, list) and isinstance(result2, list):
            if len(result1) != len(result2):
                # Try to normalize structures by flattening singleton lists
                normalized_r1 = normalize_structure(result1)
                normalized_r2 = normalize_structure(result2)
                return are_results_exactly_equal(normalized_r1, normalized_r2)
            return all(are_results_exactly_equal(r1, r2) for r1, r2 in zip(result1, result2))
        elif isinstance(result1, dict) and isinstance(result2, dict):
            if result1.keys() != result2.keys():
                return False
            return all(are_results_exactly_equal(result1[k], result2[k]) for k in result1)
        else:
            return result1 == result2

    for result in all_results:
        try:
            seed = result["seed"]
            results = result["results"]

            # Handle timeout errors
            timeout_found = False
            for res in results:
                if "error" in res and "execution timed out" in res["error"].lower():
                    analysis["timeout_results"].append({
                        "seed": seed,
                        "error": res["error"],
                        "details": results
                    })
                    timeout_found = True
                    break

            if timeout_found:
                continue

            # Check for other errors
            if any("error" in r for r in results):
                analysis["error_results"].append({
                    "seed": seed,
                    "error": "Error in one or more frameworks",
                    "details": results
                })
                continue

            # Extract and normalize outputs
            outputs = [r.get("result") for r in results if "result" in r]
            if all(output is None for output in outputs):
                analysis["none_outputs"].append({
                    "seed": seed,
                    "details": results
                })
                continue

            try:
                # Convert outputs to serializable formats and normalize keys
                normalized_outputs = []
                for output in outputs:
                    converted_output = convert_ndarray_to_list(output)
                    normalized_keys_output = normalize_output_keys(converted_output)
                    normalized_structure_output = normalize_structure(normalized_keys_output)
                    normalized_outputs.append(normalized_structure_output)
            except TypeError as e:
                analysis["error_results"].append({
                    "seed": seed,
                    "error": f"Failed to normalize outputs: {str(e)}",
                    "details": results
                })
                continue

            # Compare outputs
            first_framework_output = normalized_outputs[0]
            consistent = True
            approximate = True

            for output in normalized_outputs[1:]:
                if not are_results_exactly_equal(first_framework_output, output):
                    consistent = False
                if not are_results_close(first_framework_output, output):
                    approximate = False
                    break  # No need to continue if not approximate

            # Classify results
            if consistent:
                analysis["consistent_results"].append({
                    "seed": seed,
                    "details": first_framework_output
                })
            elif approximate:
                analysis["approximate_results"].append({
                    "seed": seed,
                    "details": normalized_outputs
                })
            else:
                analysis["divergent_results"].append({
                    "seed": seed,
                    "details": normalized_outputs
                })

        except Exception as e:
            analysis["error_results"].append({
                "seed": seed if 'seed' in locals() else "Unknown seed",
                "error": f"Failed during analysis: {str(e)}"
            })

    return analysis


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


def get_next_filename(directory, base_name, extension):
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


def get_next_run_dir(output_root):
    """
    生成新的运行文件夹
    """
    existing_dirs = [f for f in os.listdir(output_root) if re.match(r'run_(\d+)', f)]
    if not existing_dirs:
        return os.path.join(output_root, "run_1")
    else:
        numbers = [int(re.search(r'run_(\d+)', f).group(1)) for f in existing_dirs]
        next_number = max(numbers) + 1
        return os.path.join(output_root, f"run_{next_number}")


if __name__ == "__main__":
    start_time = time.time()

    project_root = os.path.dirname(os.path.dirname(__file__))
    seeds_dir = os.path.join(project_root, 'fuzzer/seeds/test_seeds')  # 测试换文件夹用
    # seeds_dir = os.path.join(project_root, 'fuzzer/seeds/initial_seeds')
    output_dir = os.path.join(project_root, 'oracle/outputs')

    # 创建一个新的运行文件夹
    run_dir = get_next_run_dir(output_dir)
    os.makedirs(run_dir, exist_ok=True)

    for i in range(1, 2):  # TODO 修改迭代次数
        # 创建当前迭代的文件夹
        iteration_dir = os.path.join(run_dir, f"iteration_{i}")
        os.makedirs(iteration_dir, exist_ok=True)

        # 生成当前迭代的结果文件名和分析文件名
        results_file = get_next_filename(iteration_dir, "results", ".json")
        analysis_file = get_next_filename(iteration_dir, "analysis", ".json")

        all_results = traverse_and_execute_seeds(seeds_dir)
        # 确保所有 Tensor 和 NumPy 对象都被转换为可序列化格式
        all_results = convert_ndarray_to_list(all_results)
        results_path = os.path.join(iteration_dir, results_file)
        write_safe_to_file(all_results, results_path)

        # 分析结果并写入分析文件
        analysis = analyze_results(all_results)
        # 确保所有 Tensor 和 NumPy 对象都被转换为可序列化格式
        analysis = convert_ndarray_to_list(analysis)
        analysis_path = os.path.join(iteration_dir, analysis_file)
        write_safe_to_file(analysis, analysis_path)

    end_time = time.time()
    print(f"Execution completed in {end_time - start_time:.2f} seconds.")
