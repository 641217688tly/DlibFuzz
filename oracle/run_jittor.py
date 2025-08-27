import os
import random
import re
import psutil
import signal
import resource

import numpy as np

try:
    import jittor as jt
    import jittor.nn as nn

    JITTOR_AVAILABLE = True
except ImportError:
    JITTOR_AVAILABLE = False
    print("Warning: Jittor not available, will return errors for Jittor code execution")

import json
import sys
import ast
import tempfile
import time
import multiprocessing

# 禁用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 设置内存限制 (4GB)
MEMORY_LIMIT_GB = 4
MEMORY_LIMIT_BYTES = MEMORY_LIMIT_GB * 1024 * 1024 * 1024


def set_memory_limit():
    """设置进程内存限制"""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (OSError, ValueError) as e:
        print(f"Warning: Could not set memory limit: {e}")


def get_memory_usage():
    """获取当前内存使用情况（MB）"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0


def timeout_handler(signum, frame):
    """超时处理函数"""
    raise TimeoutError("Execution timed out")


def set_seed(seed=42):
    """
    仅用 Python 自带和 numpy 来设定随机种子，
    避免在子进程中调用 jittor 相关随机函数而导致错误。
    """
    random.seed(seed)
    np.random.seed(seed)
    if JITTOR_AVAILABLE:
        jt.set_global_seed(seed)


def execute_jittor_code(code_path):
    """
    在临时目录执行种子脚本，并捕获输出/错误。
    """
    if not JITTOR_AVAILABLE:
        return {"error": "Jittor not available in this environment"}

    # 暂时注释掉 CUDA 设置，看看是否是这个导致的问题
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    set_seed(42)

    # 设置内存限制
    set_memory_limit()

    # 监控内存使用
    initial_memory = get_memory_usage()

    # 设置超时 (60秒)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)

    try:
        # 读取原始脚本
        with open(code_path, "r", encoding="utf-8") as f:
            original_code = f.read()

        # 通过在临时目录创建一个脚本文件，再 exec 到本进程中的方式来执行
        with tempfile.TemporaryDirectory() as temp_dir:
            old_cwd = os.getcwd()
            os.chdir(temp_dir)
            try:
                exec_locals = {}
                exec(original_code, globals(), exec_locals)

                # 查找所有以 output 为前缀的变量
                outputs = {}
                for var_name, var_value in exec_locals.items():
                    if var_name.startswith("output"):
                        try:
                            outputs[var_name] = serialize_output(var_value)
                        except Exception as e:
                            outputs[var_name] = f"Output serialization failed: {str(e)}"

                # 检查内存使用情况
                final_memory = get_memory_usage()
                memory_increase = final_memory - initial_memory

                if memory_increase > 1000:
                    outputs["_memory_warning"] = f"High memory usage: {memory_increase:.1f}MB increase"

                return outputs if outputs else {"error": "No valid output variables found"}

            except TimeoutError:
                return {"error": "Execution timed out (60 seconds)"}
            except MemoryError:
                return {"error": "Memory limit exceeded during execution"}
            except Exception as e:
                exc_type, exc_value, _ = sys.exc_info()
                error_msg = f"Jittor code execution failed: {exc_type.__name__ if exc_type else 'Unknown'}: {exc_value}"
                return {"error": error_msg}
            finally:
                os.chdir(old_cwd)
    finally:
        signal.alarm(0)  # 取消超时


def run_seed_in_subprocess(code_path):
    """
    在子进程中执行 execute_jittor_code，避免主进程因 OOM 等被 kill。
    针对Windows环境做了简化处理。
    """
    if not JITTOR_AVAILABLE:
        return {"error": "Jittor not available in this environment"}

    try:
        # 在Windows环境下，直接在当前进程中执行，避免多进程问题
        if os.name == "nt":  # Windows
            return execute_jittor_code(code_path)
        else:
            # Linux/Unix 系统使用多进程
            def target(return_dict, code_path):
                """
                子进程的目标函数，执行 Jittor 代码并将结果存入 return_dict。
                """
                try:
                    # 在子进程中设置内存限制
                    set_memory_limit()

                    try:
                        print(f"DEBUG: About to execute {code_path}")
                        result = execute_jittor_code(code_path)
                        print(f"DEBUG: Execution result: {result}")
                        return_dict["result"] = result
                    except Exception as e:
                        print(f"DEBUG: Exception in subprocess: {e}")
                        return_dict["result"] = {"error": f"Subprocess execution failed: {str(e)}"}
                except MemoryError:
                    return_dict["result"] = {"error": "Memory limit exceeded in subprocess"}
                except Exception as e:
                    return_dict["result"] = {"error": f"Subprocess execution failed: {str(e)}"}

            # 创建子进程，执行目标函数
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            p = multiprocessing.Process(target=target, args=(return_dict, code_path))

            # 监控主进程内存
            initial_memory = get_memory_usage()

            p.start()
            p.join(timeout=30)  # 降低超时时间到 30 秒

            if p.is_alive():
                print(f"    Warning: Process timed out, terminating...")
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                p.join()
                return {"error": "Subprocess timed out or hung"}

            if p.exitcode is not None and p.exitcode != 0:
                return {"error": f"Subprocess exited with code {p.exitcode}"}

            result = dict(return_dict).get("result", {"error": "No result returned from subprocess"})

            # 检查内存使用情况
            final_memory = get_memory_usage()
            memory_increase = final_memory - initial_memory
            if memory_increase > 500:  # 超过500MB增长
                if isinstance(result, dict):
                    result["_memory_warning"] = f"High memory usage: {memory_increase:.1f}MB increase"

            return result
    except Exception as e:
        return {"error": f"Failed to execute: {str(e)}"}


def serialize_output(output, depth=1, sample_num=5):
    """
    将输出结果(包含 jittor.Var、复数、函数对象等)序列化成可JSON化的对象。
    超过递归层数或超大结构，改用 summarize_output 做简要概述。
    :param output: 要序列化的对象
    :param depth: 递归展开的层数。若 depth >=5 则直接 summarize
    :param sample_num: 在 summarize 中的采样数
    """
    # TODO 将元素数量阈值、递归深度、采样数等参数作为可选参数传递

    if depth >= 5:
        return summarize_output(output, sample_num)

    try:
        # 处理 np.ndarray
        if isinstance(output, np.ndarray):
            # 若数组很大，直接做摘要
            if output.size > 1000:
                return summarize_output(output, sample_num)
            # 否则转成 Python 列表，递归处理
            return serialize_output(output.tolist(), depth=depth + 1, sample_num=sample_num)

        # 处理 jittor.Var
        if JITTOR_AVAILABLE and isinstance(output, jt.Var):
            element_num = output.size
            if element_num > 1000:
                return summarize_output(output, sample_num)
            else:
                array_data = output.numpy()
                return serialize_output(array_data, depth=depth + 1, sample_num=sample_num)

        # 处理复数
        elif isinstance(output, complex):
            return {"real": output.real, "imag": output.imag}

        # 处理可调用对象(函数等)
        elif callable(output):
            if hasattr(output, "__name__"):
                return f"<function {output.__name__}>"
            return f"<callable {repr(output)}>"

        # TODO 处理 OpOverloadPacket
        elif type(output).__name__ == "OpOverloadPacket":
            return {"type": "OpOverloadPacket", "desc": str(output)}

        # 处理基础类型
        elif isinstance(output, (int, float, str, bool, type(None))):
            return output

        # 处理 list/tuple
        elif isinstance(output, (list, tuple)):
            length = len(output)
            # 如果长度特别大，直接 summarize
            if length > 1000:
                return summarize_output(output, sample_num)
            else:
                # 只取前 sample_num 个元素做递归序列化，避免爆炸
                snippet_count = min(sample_num, length)
                snippet = output[:snippet_count]
                items_serialized = [serialize_output(item, depth=depth + 1, sample_num=sample_num) for item in snippet]
                # 若有剩余，则添加省略提示
                if snippet_count < length:
                    omitted = length - snippet_count
                    items_serialized.append(f"... omitted {omitted} elements ...")
                return items_serialized

        # 处理 dict
        elif isinstance(output, dict):
            if len(output) > 1000:
                return summarize_output(output, sample_num)
            else:
                new_dict = {}
                for k, v in output.items():
                    if not isinstance(k, str):
                        k = repr(k)  # 保证 JSON key 是字符串
                    new_dict[k] = serialize_output(v, depth=depth + 1, sample_num=sample_num)
                return new_dict

        # 其他未知类型：做摘要
        else:
            return summarize_output(output, sample_num)

    except Exception as e:
        exc_info = sys.exc_info()
        tb_lineno = exc_info[2].tb_lineno if exc_info[2] is not None else "Unknown"
        print(f"[serialize_output] Error at line {tb_lineno}: {e}")
        return f"Error serializing output of type {type(output).__name__}: {str(e)}"


def summarize_output(output, sample_num=5):
    """
    对大型/复杂对象进行摘要，而不是直接输出所有元素。
    只记录 shape/dtype/min/max/mean，外加极少数行/列做采样，不做进一步递归。
    """
    summary = {}

    try:
        if isinstance(output, np.ndarray):
            size = output.size
            ndim = output.ndim

            summary["type"] = "np.ndarray"
            summary["shape"] = list(output.shape)
            summary["dtype"] = str(output.dtype)
            summary["ndim"] = ndim
            summary["size"] = size

            # 若维度大于 2，或元素数量过大，不统计详细信息
            if ndim > 2:
                summary["desc"] = "Dimension > 2, skipped detailed stats"
                return summary
            elif size > 50_000:
                summary["desc"] = "Size too large, skipped detailed stats"
                return summary
            else:
                # 计算统计信息
                if size > 0:
                    arr_min = output.min()
                    arr_max = output.max()
                    arr_mean = output.mean()
                    summary["min"] = _safe_float(arr_min)
                    summary["max"] = _safe_float(arr_max)
                    summary["mean"] = _safe_float(arr_mean)

            # 采样前 sample_num 行的前 sample_num 个元素
            rows = min(sample_num, output.shape[0])
            cols = min(sample_num, output.shape[1] if output.ndim > 1 else 1)
            samples = []
            for i in range(rows):
                # 对该行做切片
                if output.ndim == 1:
                    row_slice = [output[i]]
                else:
                    row_slice = output[i, :cols]

                # 转成纯 Python 数值
                row_data = []
                for val in np.ravel(row_slice):
                    if np.iscomplexobj(val):
                        row_data.append((val.real.item(), val.imag.item()))
                    else:
                        row_data.append(float(val))
                samples.append(row_data)
            summary["samples"] = samples
            return summary

        elif JITTOR_AVAILABLE and isinstance(output, jt.Var):
            var_array = output.numpy()
            ndim = var_array.ndim
            size = var_array.size

            summary["type"] = "jt.Var"
            summary["shape"] = list(output.shape)
            summary["dtype"] = str(output.dtype)
            summary["ndim"] = ndim
            summary["size"] = size

            # 多于 2 维的 Var 不展开
            if ndim > 2:
                summary["desc"] = f"Dimension > 2, skipped detailed stats"
                return summary
            # 太大的 Var 不展开
            if size > 50_000:
                summary["desc"] = f"Size too large, skipped detailed stats"
                return summary
            else:
                if size > 0:
                    v_min = var_array.min()
                    v_max = var_array.max()
                    v_mean = var_array.mean()
                    summary["min"] = _safe_float(v_min)
                    summary["max"] = _safe_float(v_max)
                    summary["mean"] = _safe_float(v_mean)

            # 采样前 sample_num 行、每行前 sample_num 列
            rows = min(sample_num, var_array.shape[0] if var_array.ndim > 0 else 1)
            cols = min(sample_num, var_array.shape[1] if var_array.ndim > 1 else 1)
            samples = []
            for i in range(rows):
                if var_array.ndim == 1:
                    row_slice = [var_array[i]]
                else:
                    row_slice = var_array[i, :cols]

                row_data = []
                for val in row_slice:
                    if np.iscomplexobj(val):
                        row_data.append((val.real, val.imag))
                    else:
                        row_data.append(float(val))
                samples.append(row_data)
            summary["samples"] = samples

        elif isinstance(output, (list, tuple)):
            summary["type"] = "list" if isinstance(output, list) else "tuple"
            length = len(output)
            summary["length"] = length

            # 仅保留前 sample_num 个元素的"字符串形式"，不再递归
            snippet_count = min(sample_num, length)
            snippet = output[:snippet_count]
            snippet_str = [str(x) for x in snippet]
            if snippet_count < length:
                snippet_str.append(f"... omitted {length - snippet_count} elements ...")
            summary["samples"] = snippet_str

        # dict
        elif isinstance(output, dict):
            summary["type"] = "dict"
            length = len(output)
            summary["shape"] = length
            sample_count = min(sample_num, length)
            keys = list(output.keys())[:sample_count]
            summary["samples"] = [str(k) for k in keys]

        elif isinstance(output, (int, float, str, bool, type(None))):
            return _safe_str(output)

        # 其他类型
        else:
            summary["type"] = str(type(output))
            summary["desc"] = _safe_str(output)

        return summary

    except Exception as e:
        exc_info = sys.exc_info()
        tb_lineno = exc_info[2].tb_lineno if exc_info[2] is not None else "Unknown"
        print(f"[summarize_output] Unserializable {type(output).__name__} at line {tb_lineno}: {e}")
        summary["desc"] = f"Unserializable output of type {type(output).__name__}: {str(e)}"
        return summary


def _safe_float(val):
    """将可能是 inf / nan 的浮点数转换成字符串，或者普通 float。"""
    if np.isnan(val):
        return "NaN"
    elif np.isinf(val):
        return "Infinity" if val > 0 else "-Infinity"
    else:
        return float(val)


def _safe_str(obj):
    """
    对象转字符串时，若是浮点 inf/nan，也转换成可写入 JSON 的形式。
    """
    if isinstance(obj, float):
        if np.isnan(obj):
            return "NaN"
        elif np.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
    return str(obj)


def get_iteration_num(output_dir):
    """
    获取下一个迭代的编号。
    """
    os.makedirs(output_dir, exist_ok=True)
    existing_dirs = [d for d in os.listdir(output_dir) if re.match(r"^iteration_\d+$", d)]
    if not existing_dirs:
        return 1
    numbers = []
    for d in existing_dirs:
        match_obj = re.match(r"^iteration_(\d+)$", d)
        if match_obj:
            numbers.append(int(match_obj.group(1)))
    return max(numbers) + 1 if numbers else 1


def save_batch_results(batch_results, output_dir, iteration_num, batch_num):
    """
    将当前批次的结果保存到指定的批次文件中。
    """
    iteration_dir = os.path.join(output_dir, f"iteration_{iteration_num}")
    batch_dir = os.path.join(iteration_dir, "jittor")
    os.makedirs(batch_dir, exist_ok=True)

    batch_file = os.path.join(batch_dir, f"batch_{batch_num}.json")

    # 保存当前批次结果
    try:
        with open(batch_file, "w", encoding="utf-8") as f:
            json.dump(batch_results, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving batch {batch_num} results: {e}")

    print(f"Batch {batch_num} results saved to {batch_file}")


def merge_results(output_dir, iteration_num):
    """
    合并所有批次的结果并保存到一个单独的 JSON 文件中。
    """
    iteration_dir = os.path.join(output_dir, f"iteration_{iteration_num}")
    batch_dir = os.path.join(iteration_dir, "jittor")
    merged_results = {}
    # 遍历所有批次文件
    for batch_file in sorted(os.listdir(batch_dir)):
        print(f"Merging results from {batch_file}")
        if batch_file.endswith(".json"):
            batch_file_path = os.path.join(batch_dir, batch_file)
            with open(batch_file_path, "r", encoding="utf-8") as f:
                batch_results = json.load(f)
                merged_results.update(batch_results)

    # 保存合并后的结果
    final_output_file = os.path.join(iteration_dir, "jittor_results.json")
    with open(final_output_file, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=4, ensure_ascii=False)

    print(f"Final merged results saved to {final_output_file}")


def collect_seed_directories(value_equivalent_dir):
    """
    收集所有的种子目录，返回格式为 [(cluster_name, seed_name, seed_dir_path), ...]
    """
    seed_dirs = []
    for cluster_dir in os.listdir(value_equivalent_dir):
        cluster_path = os.path.join(value_equivalent_dir, cluster_dir)
        if os.path.isdir(cluster_path) and cluster_dir.startswith("valid_Cluster_"):
            for seed_dir in os.listdir(cluster_path):
                seed_path = os.path.join(cluster_path, seed_dir)
                if os.path.isdir(seed_path) and seed_dir.startswith("seed_"):
                    seed_dirs.append((cluster_dir, seed_dir, seed_path))
    return seed_dirs


def find_jittor_files(seed_dir):
    """
    在种子目录中查找所有 jittor 相关的 Python 文件
    """
    jittor_files = []
    for file in os.listdir(seed_dir):
        if (
            file.endswith(".py")
            and ("jittor" in file.lower() or "jt" in file.lower())
            and not file.startswith("invalid.")
        ):
            jittor_files.append(os.path.join(seed_dir, file))
    return jittor_files


def process_seed_directory(cluster_name, seed_name, seed_dir_path, output_dir, problematic_files=None):
    """
    处理单个种子目录，运行其中的所有 jittor 文件，并将结果保存到对应的 JSON 文件中
    """
    print(f"Processing {cluster_name}/{seed_name}")

    # 创建输出目录结构
    output_cluster_dir = os.path.join(output_dir, "ValueEquivalent", cluster_name)
    # output_cluster_dir = os.path.join(output_dir, "StateEquivalent", cluster_name)
    os.makedirs(output_cluster_dir, exist_ok=True)

    # 输出 JSON 文件路径
    output_json_path = os.path.join(output_cluster_dir, f"{seed_name}.json")

    # 读取现有的输出文件（如果存在）
    existing_results = {}
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            print(f"  Found existing results with {len(existing_results)} entries")
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: Could not read existing results: {e}")
            existing_results = {}

    # 查找所有 jittor 相关文件
    jittor_files = find_jittor_files(seed_dir_path)

    if not jittor_files:
        print(f"No jittor files found in {seed_dir_path}")
        # 如果没有 jittor 文件但有现有结果，保持现有结果不变
        if not existing_results:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=4, ensure_ascii=False)
        return

    # 运行所有 jittor 文件并收集结果
    jittor_results = {}
    for jittor_file in jittor_files:
        file_name = os.path.basename(jittor_file)
        file_path = f"{cluster_name}/{seed_name}/{file_name}"

        # 检查是否是已知的问题文件
        if problematic_files and file_path in problematic_files:
            print(f"  Skipping {file_name} (marked as problematic)")
            jittor_results[file_name] = {"error": "Skipped due to previous memory issues"}
            continue

        print(f"  Executing {file_name}")
        memory_before = get_memory_usage()

        try:
            result = run_seed_in_subprocess(jittor_file)

            # 检查是否包含内存警告
            if isinstance(result, dict) and "_memory_warning" in result:
                print(f"    Warning: {result['_memory_warning']}")
                # 标记为问题文件
                if problematic_files is not None:
                    problematic_files.add(file_path)

            jittor_results[file_name] = result

        except Exception as e:
            print(f"    Error: {str(e)}")
            jittor_results[file_name] = {"error": f"Failed to execute: {str(e)}"}
            # 标记为问题文件
            if problematic_files is not None:
                problematic_files.add(file_path)

        memory_after = get_memory_usage()
        memory_diff = memory_after - memory_before
        if memory_diff > 100:  # 超过100MB增长
            print(f"    Memory increased by {memory_diff:.1f}MB")

    # 合并现有结果和新的 jittor 结果
    merged_results = existing_results.copy()
    merged_results.update(jittor_results)

    # 保存合并后的结果
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(merged_results, f, indent=4, ensure_ascii=False)

    print(f"  Results saved to {output_json_path} (total {len(merged_results)} entries)")


def save_problematic_files(problematic_files, output_dir):
    """保存问题文件列表"""
    if not problematic_files:
        return

    problematic_file_path = os.path.join(output_dir, "problematic_files.json")
    problematic_list = list(problematic_files)

    with open(problematic_file_path, "w", encoding="utf-8") as f:
        json.dump(problematic_list, f, indent=4, ensure_ascii=False)

    print(f"Problematic files saved to {problematic_file_path}")


def load_problematic_files(output_dir):
    """加载已知的问题文件列表"""
    problematic_file_path = os.path.join(output_dir, "problematic_files.json")

    if os.path.exists(problematic_file_path):
        try:
            with open(problematic_file_path, "r", encoding="utf-8") as f:
                problematic_list = json.load(f)
            return set(problematic_list)
        except (json.JSONDecodeError, IOError):
            pass

    return set()


def main():
    """
    主函数，遍历 ValueEquivalent 文件夹结构，运行 jittor 相关代码文件
    """
    start_time = time.time()

    if len(sys.argv) != 3:
        print("Usage: python run_jittor.py <value_equivalent_dir> <output_dir>")
        print("Example: python run_jittor.py fuzzer/seed/ValueEquivalent oracle/outputs")
        # print("Example: python run_jittor.py fuzzer/seed/StateEquivalent oracle/outputs")
        sys.exit(1)

    value_equivalent_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(value_equivalent_dir):
        print(f"Error: ValueEquivalent directory {value_equivalent_dir} does not exist")
        # print(f"Error: StateEquivalent directory {value_equivalent_dir} does not exist")
        sys.exit(1)

    # 加载已知的问题文件
    problematic_files = load_problematic_files(output_dir)
    if problematic_files:
        print(f"Loaded {len(problematic_files)} known problematic files")

    # 收集所有种子目录
    seed_dirs = collect_seed_directories(value_equivalent_dir)
    print(f"Found {len(seed_dirs)} seed directories to process")

    # 处理每个种子目录
    for i, (cluster_name, seed_name, seed_dir_path) in enumerate(seed_dirs, 1):
        try:
            current_memory = get_memory_usage()
            print(f"[{i}/{len(seed_dirs)}] Processing {cluster_name}/{seed_name} (Memory: {current_memory:.1f} MB)")

            process_seed_directory(cluster_name, seed_name, seed_dir_path, output_dir, problematic_files)

        except Exception as e:
            print(f"Error processing {cluster_name}/{seed_name}: {e}")
            continue

    # 保存问题文件列表
    save_problematic_files(problematic_files, output_dir)

    end_time = time.time()
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    print(f"Results saved to {output_dir}/ValueEquivalent/")
    # print(f"Results saved to {output_dir}/StateEquivalent/")

    if problematic_files:
        print(f"Found {len(problematic_files)} problematic files that were skipped")


if __name__ == "__main__":
    # 命令：python oracle/run_jittor.py fuzzer/seed/ValueEquivalent oracle/outputs
    main()
