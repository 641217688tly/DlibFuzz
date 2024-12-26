import os
import re
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
                        outputs[var] = convert_to_serializable(output)
                    except Exception as e:
                        outputs[var] = f"Unserializable output of type {type(output).__name__}: {str(e)}"
            return outputs if outputs else {"error": "No valid output found"}
        except Exception as e:
            exc_type, exc_value, _ = sys.exc_info()
            return {"error": f"Pytorch code execution failed: {exc_type.__name__}: {exc_value}"}
        finally:
            os.chdir(old_cwd)

    # 创建临时文件，写入修改后的脚本
    # with tempfile.TemporaryDirectory() as temp_dir:
    #     temp_script_path = os.path.join(temp_dir, "temp_script.py")
    #     # 写入修改后的脚本
    #     with open(temp_script_path, "w", encoding="utf-8") as script_file:
    #         script_file.write('\n'.join(modified_pytorch_code))
    #
    #     try:
    #         # 读取刚才写入的脚本，然后 exec 执行
    #         with open(temp_script_path, "r", encoding="utf-8") as script_file:
    #             script_content = script_file.read()
    #
    #         exec_locals = {}
    #         exec(script_content, globals(), exec_locals)
    #
    #         outputs = {}
    #         for var in output_vars:
    #             if var in exec_locals:
    #                 try:
    #                     output = exec_locals[var]
    #                     outputs[var] = convert_to_serializable(output)
    #                 except Exception as e:
    #                     outputs[var] = f"Unserializable output of type {type(output).__name__}: {str(e)}"
    #         return outputs if outputs else {"error": "No valid output found"}
    #
    #     except Exception as e:
    #         exc_type, exc_value, _ = sys.exc_info()
    #         return {"error": f"Pytorch code execution failed: {exc_type.__name__}: {exc_value}"}


def run_seed_in_subprocess(code_path):
    """
    在子进程中执行 execute_pytorch_code，避免主进程因 OOM 等被 kill。
    """
    def target(return_dict, code_path):
        old_stdout, old_stderr = sys.stdout, sys.stderr
        with open(os.devnull, 'w') as devnull:
            sys.stdout, sys.stderr = devnull, devnull
            try:
                result = execute_pytorch_code(code_path)
                return_dict["result"] = result
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
        # result = execute_pytorch_code(code_path)
        # return_dict["result"] = result

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    p = multiprocessing.Process(target=target, args=(return_dict, code_path))
    p.start()
    p.join(timeout=10)

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


def convert_to_serializable(output):
    LARGE_TENSOR_SIZE_THRESHOLD = 1000

    # 处理 PyTorch Tensor
    if isinstance(output, torch.Tensor):
        numel = output.numel()
        if numel > LARGE_TENSOR_SIZE_THRESHOLD:
            return summarize_tensor(output)
        else:
            # 如果是复数 Tensor，numpy().tolist() 也可能包含复数，需要再递归处理
            array_data = output.detach().cpu().numpy()
            return _recursive_serialize(array_data)

    # 处理复数类型
    elif isinstance(output, complex):
        # 可以转成字符串，也可以转成 [real, imag]，看你需求
        return {
            "real": output.real,
            "imag": output.imag
        }

    # 其他基础类型
    elif isinstance(output, (int, float, str, bool, type(None))):
        return output

    # 万一传进来 list/dict，需要再递归处理（以防其中有复数或其他复杂对象）
    elif isinstance(output, (list, tuple)):
        return [_recursive_serialize(item) for item in output]
    elif isinstance(output, dict):
        return {key: _recursive_serialize(val) for key, val in output.items()}

    else:
        # 默认兜底
        return str(output)


def _recursive_serialize(item):
    """
    辅助函数，用于递归处理 list/dict 中的“复数”、“函数对象”、“OpOverloadPacket”等。
    """
    if isinstance(item, complex):
        return {"real": item.real, "imag": item.imag}

    elif callable(item):
        if hasattr(item, '__name__'):
            return f"<function {item.__name__}>"
        else:
            return f"<callable {repr(item)}>"

    elif type(item).__name__ == "OpOverloadPacket":
        return {"type": "OpOverloadPacket", "desc": str(item)}

    elif isinstance(item, (int, float, str, bool, type(None))):
        return item

    elif isinstance(item, (list, tuple)):
        return [_recursive_serialize(x) for x in item]

    elif isinstance(item, dict):
        new_dict = {}
        for k, v in item.items():
            # 确保 key 是字符串
            if not isinstance(k, str):
                k = repr(k)
            new_dict[k] = _recursive_serialize(v)
        return new_dict

    else:
        # 如果是 numpy array / Tensor / 其他自定义类型
        if hasattr(item, 'tolist'):
            return _recursive_serialize(item.tolist())
        return str(item)


def summarize_tensor(tensor):
    summary = {}
    if isinstance(tensor, torch.Tensor):
        summary['type'] = 'torch.Tensor'
        summary['shape'] = list(tensor.shape)
        summary['dtype'] = str(tensor.dtype)
    else:
        summary['type'] = str(type(tensor))
        summary['shape'] = str(tensor.shape)
        summary['dtype'] = str(tensor.dtype)
    return summary


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
        # # 多线程版本（可能会导致大量并发进程），如果数量很大，可改用进程池
        # with ThreadPoolExecutor(max_workers=4) as executor:
        #     future_to_seed = {
        #         executor.submit(run_seed_in_subprocess, code_path): seed_name
        #         for seed_name, code_path in batch
        #     }
        #     for future in as_completed(future_to_seed):
        #         seed_name = future_to_seed[future]
        #         try:
        #             batch_results[seed_name] = future.result()
        #         except Exception as e:
        #             batch_results[seed_name] = {"error": str(e)}

        # 单线程版本
        for seed_name, code_path in batch:
            print(f"Executing Pytorch code in {code_path}")
            batch_results[seed_name] = run_seed_in_subprocess(code_path)
            #
            # if batch_num == 11:
            #     batch_results[seed_name] = run_seed_in_subprocess(code_path)
            # else:
            #     batch_results[seed_name] = {"error": "skip"}

        # 保存当前批次结果
        save_batch_results(batch_results, output_dir, iteration_num, batch_num)
        print(f"Executed batch {batch_num} at {time.time() - start_time:.2f} seconds")
        batch_num += 1

    # 合并所有批次结果
    merge_results(output_dir, iteration_num)

    print(f"\nAll batches processed. Total seeds: {total_seeds}")
    print(f"Execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    # 启动命令：python run_pytorch.py /path/to/seeds /path/to/output
    # 例如：python run_pytorch.py /path/to/seeds /path/to/output
    main()
