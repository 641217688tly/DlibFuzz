import importlib


def validate_api(module_name, api_name):  # 验证API是否存在的函数
    # module_name, api_name = full_api_name.rsplit('.', 1)
    try:
        # 先检查模块是否存在
        module = importlib.import_module(module_name)
        # 再检查API是否存在
        getattr(module, api_name)
        return True
    except (ModuleNotFoundError, AttributeError):
        return False


def check_api_list(file_path):  # 检查每个API是否存在
    with open(file_path, 'r') as file:
        api_names = [line.strip() for line in file.readlines()]
    results = {}
    exists_count = 0
    not_exists_count = 0
    for full_api_name in api_names:
        module_name, api_name = full_api_name.rsplit('.', 1)
        try:
            existence = validate_api(module_name, api_name)
            results[full_api_name] = existence
            if existence:
                exists_count += 1
            else:
                not_exists_count += 1
        except ModuleNotFoundError:
            results[full_api_name] = False
            not_exists_count += 1
    return results, exists_count, not_exists_count


def check_all_api_lists():
    file_path1 = './api_signatures/jax/jax_valid_apis.txt'
    file_path2 = './api_signatures/pytorch/torch_valid_apis.txt'
    file_path3 = './api_signatures/tensorflow/tf_valid_apis.txt'
    api_check_results1, exists_count1, not_exists_count1 = check_api_list(file_path1)
    # for api, exists in api_check_results.items():
    #     print(f"{api}: {'Exists' if exists else 'Does not exist'}")
    print(f"\nNumber of JAX APIs that exist: {exists_count1}")
    print(f"Number of JAX APIs that do not exist: {not_exists_count1}")
    api_check_results2, exists_count2, not_exists_count2 = check_api_list(file_path2)
    print(f"\nNumber of Pytorch APIs that exist: {exists_count2}")
    print(f"Number of Pytorch APIs that do not exist: {not_exists_count2}")
    api_check_results3, exists_count3, not_exists_count3 = check_api_list(file_path3)
    print(f"\nNumber of Tensorflow APIs that exist: {exists_count3}")
    print(f"Number of Tensorflow APIs that do not exist: {not_exists_count3}")
