import os
import re

import numpy as np
import torch
import json
import sys
import ast
import tempfile
import time
import multiprocessing

# 禁用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def set_seed(seed=42):
    torch.manual_seed(seed)


def execute_pytorch_code(code_path):
    """
    在临时目录执行种子脚本，并捕获输出/错误。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    set_seed(42)

    # 读取原始脚本，并进行修改（将 print 的内容捕获到变量中）
    with open(code_path, 'r', encoding='utf-8') as f:
        code_lines = f.readlines()
    modified_pytorch_code, output_vars = extract_and_modify_print_statements(code_lines, "output_pt")
    # 输出修改后的代码，保存至output中


    # 通过在临时目录创建一个脚本文件，再 exec 到本进程中的方式来执行
    with tempfile.TemporaryDirectory() as temp_dir:
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            exec_locals = {}
            exec('\n'.join(modified_pytorch_code), globals(), exec_locals)

            outputs = {}
            for var in output_vars:
                if var in exec_locals:
                    try:
                        output = exec_locals[var]
                        outputs[var] = serialize_output(output)
                    except Exception as e:
                        outputs[var] = f"Output serialization failed: {str(e)}"
            return outputs if outputs else {"error": "No valid output found"}
        except Exception as e:
            # 捕获异常，返回错误信息
            exc_type, exc_value, _ = sys.exc_info()
            return {"error": f"Pytorch code execution failed: {exc_type.__name__}: {exc_value}"}
        finally:
            os.chdir(old_cwd)


def run_seed_in_subprocess(code_path):
    """
    在子进程中执行 execute_pytorch_code，避免主进程因 OOM 等被 kill。
    """

    def target(return_dict, code_path):
        """
        子进程的目标函数，执行 PyTorch 代码并将结果存入 return_dict。
        """
        # 重定向 stdout/stderr 到 /dev/null，实现静默执行，避免控制台输出过多
        old_stdout, old_stderr = sys.stdout, sys.stderr
        with open(os.devnull, 'w') as devnull:
            sys.stdout, sys.stderr = devnull, devnull
            try:
                result = execute_pytorch_code(code_path)
                return_dict["result"] = result
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
        # 正常打印种子执行结果
        # result = execute_pytorch_code(code_path)
        # return_dict["result"] = result

    # 创建子进程，执行目标函数
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=target, args=(return_dict, code_path))
    p.start()
    p.join(timeout=10)  # 单个种子执行超时时间为 10 秒

    if p.is_alive():
        p.terminate()
        p.join()
        return {"error": "Subprocess timed out or hung"}

    if p.exitcode is not None and p.exitcode != 0:
        return {"error": f"Subprocess exited with code {p.exitcode}"}

    return dict(return_dict).get("result", {})


def extract_and_modify_print_statements(code_lines, prefix):
    modified_code_lines = []
    output_vars = []
    output_var_counter = 1
    for line in code_lines:
        try:
            tree = ast.parse(line)
            has_print_statement = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'print':
                    has_print_statement = True
                    new_assignments = []
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            continue
                        var_name = f"{prefix}_{output_var_counter}"
                        new_assignments.append(var_name)
                        output_vars.append(var_name)
                        try:
                            full_expr = ast.unparse(arg)
                            modified_code_lines.append(f"{var_name} = {full_expr}")
                        except Exception as e:
                            print(f"Error unparsing argument in line: {line.strip()} - {e}")
                            modified_code_lines.append(line.strip())
                        output_var_counter += 1

                    new_print_stmt = f"print({', '.join(new_assignments)})"
                    modified_code_lines.append(new_print_stmt)
                    break
            if not has_print_statement:
                modified_code_lines.append(line.rstrip('\n'))
        except SyntaxError as e:
            print(f"Syntax error while parsing line: {line.strip()} - {e}")
            modified_code_lines.append(line.rstrip('\n'))
    return modified_code_lines, output_vars


def serialize_output(output, depth=1, sample_num=5):
    """
    将输出结果(包含 PyTorch Tensor、复数、函数对象等)序列化成可JSON化的对象。
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

        # 处理 PyTorch Tensor
        if isinstance(output, torch.Tensor):
            element_num = output.numel()
            if element_num > 1000:
                return summarize_output(output, sample_num)
            else:
                array_data = output.detach().cpu().numpy()
                return serialize_output(array_data, depth=depth + 1, sample_num=sample_num)

        # 处理复数
        elif isinstance(output, complex):
            return {"real": output.real, "imag": output.imag}

        # 处理可调用对象(函数等)
        elif callable(output):
            if hasattr(output, '__name__'):
                return f"<function {output.__name__}>"
            return f"<callable {repr(output)}>"

        # 处理 OpOverloadPacket
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
                items_serialized = [
                    serialize_output(item, depth=depth + 1, sample_num=sample_num)
                    for item in snippet
                ]
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
        print(f"[serialize_output] Error at line {sys.exc_info()[-1].tb_lineno}: {e}")
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

        elif isinstance(output, torch.Tensor):
            cpu_tensor = output.detach().cpu()
            size = cpu_tensor.numel()  # 元素数量 (包括所有维度)

            summary["type"] = "torch.Tensor"
            summary["shape"] = list(output.shape)
            summary["dtype"] = str(output.dtype)
            summary["ndim"] = cpu_tensor.ndim
            summary["size"] = size

            # 多于 2 维的 Tensor 不展开
            if cpu_tensor.ndim > 2:
                summary["desc"] = f"Dimension > 2, skipped detailed stats"
                return summary
            # 太大的 Tensor 不展开
            if size > 50_000:
                summary["desc"] = f"Size too large, skipped detailed stats"
                return summary
            else:
                if size > 0:
                    t_min = cpu_tensor.min()
                    t_max = cpu_tensor.max()
                    t_mean = cpu_tensor.float().mean()
                    summary["min"] = _safe_float(t_min.item())
                    summary["max"] = _safe_float(t_max.item())
                    summary["mean"] = _safe_float(t_mean.item())

            # 采样前 sample_num 行、每行前 10 列
            rows = min(sample_num, cpu_tensor.shape[0] if cpu_tensor.ndim > 0 else 1)
            cols = min(sample_num, cpu_tensor.shape[1] if cpu_tensor.ndim > 1 else 1)
            samples = []
            for i in range(rows):
                if cpu_tensor.ndim == 1:
                    row_slice = [cpu_tensor[i]]
                else:
                    row_slice = cpu_tensor[i, :cols]

                row_data = []
                for val in row_slice:
                    if torch.is_complex(val):
                        row_data.append((val.real.item(), val.imag.item()))
                    else:
                        row_data.append(float(val.item()))
                samples.append(row_data)
            summary["samples"] = samples

        elif isinstance(output, (list, tuple)):
            summary["type"] = "list" if isinstance(output, list) else "tuple"
            length = len(output)
            summary["length"] = length

            # 仅保留前 sample_num 个元素的“字符串形式”，不再递归
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
        print(f"[summarize_output] Unserializable {type(output).__name__} at line {sys.exc_info()[-1].tb_lineno}: {e}")
        summary["desc"] = f"Unserializable output of type {type(output).__name__}: {str(e)}"
        return summary


def _safe_float(val):
    """ 将可能是 inf / nan 的浮点数转换成字符串，或者普通 float。 """
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
    existing_dirs = [d for d in os.listdir(output_dir) if re.match(r'^iteration_\d+$', d)]
    if not existing_dirs:
        return 1
    numbers = []
    for d in existing_dirs:
        match_obj = re.match(r'^iteration_(\d+)$', d)
        if match_obj:
            numbers.append(int(match_obj.group(1)))
    return max(numbers) + 1 if numbers else 1


def save_batch_results(batch_results, output_dir, iteration_num, batch_num):
    """
    将当前批次的结果保存到指定的批次文件中。
    """
    iteration_dir = os.path.join(output_dir, f"iteration_{iteration_num}")
    batch_dir = os.path.join(iteration_dir, "pytorch")
    os.makedirs(batch_dir, exist_ok=True)

    batch_file = os.path.join(batch_dir, f"batch_{batch_num}.json")

    # 保存当前批次结果
    try:
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving batch {batch_num} results: {e}")

    print(f"Batch {batch_num} results saved to {batch_file}")


def merge_results(output_dir, iteration_num):
    """
    合并所有批次的结果并保存到一个单独的 JSON 文件中。
    """
    iteration_dir = os.path.join(output_dir, f"iteration_{iteration_num}")
    batch_dir = os.path.join(iteration_dir, "pytorch")
    merged_results = {}
    # 遍历所有批次文件
    for batch_file in sorted(os.listdir(batch_dir)):
        print(f"Merging results from {batch_file}")
        if batch_file.endswith(".json"):
            batch_file_path = os.path.join(batch_dir, batch_file)
            with open(batch_file_path, 'r', encoding='utf-8') as f:
                batch_results = json.load(f)
                merged_results.update(batch_results)

    # 保存合并后的结果
    final_output_file = os.path.join(iteration_dir, "pytorch_results.json")
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=4, ensure_ascii=False)

    print(f"Final merged results saved to {final_output_file}")


def main():
    """
    主函数，分批处理种子，逐批写入结果，最后统一合并。
    """
    start_time = time.time()
    if len(sys.argv) != 3:
        print("Usage: python run_pytorch.py <seed_dir> <output_dir>")
        sys.exit(1)

    seed_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # 搜集所有种子
    seeds_to_run = []
    for root, _, files in os.walk(seed_dir):
        for file in files:
            if 'torch_seed' in file.lower():
                code_path = os.path.join(root, file)
                seed_name = os.path.relpath(code_path, seed_dir)
                seeds_to_run.append((seed_name, code_path))

    # 分批执行种子
    batch_size = 50  # 每批次处理的种子数量
    iteration_num = get_iteration_num(output_dir)
    total_seeds = len(seeds_to_run)

    print(f"Total seeds to process: {total_seeds}")
    batch_num = 1
    for batch_start in range(0, total_seeds, batch_size):
        batch_end = min(batch_start + batch_size, total_seeds)
        batch = seeds_to_run[batch_start:batch_end]
        print(f"Processing batch {batch_num}: seeds {batch_start + 1} to {batch_end}")

        batch_results = {}

        # 用于生成指定batch的结果
        # if batch_num != 47 and batch_num != 54 and batch_num != 74:
        #     batch_num += 1
        #     print(f"Skip batch {batch_num}")
        #     continue

        # # 单线程版本
        for seed_name, code_path in batch:
            print(f"Executing Pytorch code in {code_path}")
            batch_results[seed_name] = run_seed_in_subprocess(code_path)
        #
        # # TODO 多线程版本(暂时不可用)
        # # with ThreadPoolExecutor(max_workers=4) as executor:
        # #     future_to_seed = {
        # #         executor.submit(run_seed_in_subprocess, code_path): seed_name
        # #         for seed_name, code_path in batch
        # #     }
        # #     for future in as_completed(future_to_seed):
        # #         seed_name = future_to_seed[future]
        # #         try:
        # #             batch_results[seed_name] = future.result()
        # #         except Exception as e:
        # #             batch_results[seed_name] = {"error": str(e)}
        #
        # 保存当前批次结果
        save_batch_results(batch_results, output_dir, iteration_num, batch_num)
        print(f"Executed batch {batch_num} at {time.time() - start_time:.2f} seconds")
        batch_num += 1

    # 合并所有批次结果
    merge_results(output_dir, iteration_num)

    print(f"\nAll batches processed. Total seeds: {total_seeds}")
    print(f"Execution time: {time.time() - start_time:.2f} seconds")

    # 用于重复合并测试
    # merge_results(output_dir, 1)


if __name__ == "__main__":
    # 启动命令：python run_pytorch.py /path/to/seeds /path/to/output
    # 例如：python run_pytorch.py ../fuzzer/seeds/test_seeds ../oracle/outputs
    main()
