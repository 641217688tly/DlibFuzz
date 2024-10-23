import importlib
import inspect
import os
import warnings
import httpx
from openai import OpenAI
from sqlalchemy.orm import sessionmaker
from orm import *
import inspect
import importlib

TORCH_VERSION = "1.12"
TF_VERSION = "2.10"
JAX_VERSION = "0.4.13"


def get_session():
    with open('config.yml', 'r', encoding='utf-8') as file:  # 读取config.yml文件
        config = yaml.safe_load(file)
        # 从配置中提取数据库连接信息
        db_config = config['db']['mysql']
        host = db_config['host']
        user = db_config['user']
        password = db_config['password']
        database = db_config['database']
        db_url = f"mysql+pymysql://{user}:{password}@{host}/{database}"  # 创建数据库连接字符串
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        session = Session()
        return session


def get_openai_client():
    with open('config.yml', 'r', encoding='utf-8') as file:  # 读取config.yml文件
        config = yaml.safe_load(file)
        # 设置代理
        proxy = httpx.Client(proxies={
            "http://": "http://127.0.0.1:7890",
            "https://": "http://127.0.0.1:7890"
        })
        openai_client = OpenAI(api_key=config['openai']['api_key'], http_client=proxy)
        # openai_client = OpenAI(base_url="https://api.gptsapi.net/v1", api_key="sk-lBR9ab45cb8a12646896f37fe57070f6e1b7b05e8a3N9xPt")  # WildCard API + 转发, 无需代理
        return openai_client


def get_library_version():
    # 创建一个字典，用于存储各个库的版本
    library_version = {
        "pytorch": TORCH_VERSION,
        "tensorflow": TF_VERSION,
        "jax": JAX_VERSION
    }
    return library_version


def validate_api_existence(module_name, api_name):  # 验证API是否存在的函数
    # module_name, api_name = full_api_name.rsplit('.', 1)
    try:
        # 先检查模块是否存在
        module = importlib.import_module(module_name)
        # 再检查API是否存在
        func = getattr(module, api_name, None)
        if inspect.ismodule(func):
            return False
        if inspect.isclass(func):
            return False
        return True
    except (ModuleNotFoundError, AttributeError, ImportError, ValueError, Exception):
        return False


def validate_api_availability(function):  # 验证API是否为被弃用的函数
    """Check if the function is deprecated."""
    docstring = inspect.getdoc(function)
    if docstring and ('deprecated' and 'removed') in docstring.lower():
        return True

    # Capture DeprecationWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', DeprecationWarning)
        try:
            function()  # Attempt to call the function
        except Exception:
            pass
        return any(item.category == DeprecationWarning for item in w)


def check_api_list(file_path):  # 检查每个API是否存在
    with open(file_path, 'r') as file:
        api_names = [line.strip() for line in file.readlines()]
    results = {}
    exists_count = 0
    not_exists_count = 0
    for full_api_name in api_names:
        module_name, api_name = full_api_name.rsplit('.', 1)
        try:
            existence = validate_api_existence(module_name, api_name)
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
    file_path1 = 'cluster/apis/jax/jax_valid_apis.txt'
    file_path2 = 'cluster/apis/pytorch/torch_valid_apis.txt'
    file_path3 = 'cluster/apis/tensorflow/tf_valid_apis.txt'
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


def get_api_signature(full_api_name):
    """
    根据 API 的全名获取其函数签名
    :param full_name: API的全名（例如 'torch.nn.Conv2d'）
    :return: 该API的函数签名或者当无法获取时返回错误信息
    """

    def process_signature(full_api_name, raw_signature):
        # 将raw_signature按照最后一个'->'分割为输入参数和输出参数
        parts = raw_signature.strip().rsplit('->', 1)
        if len(parts) == 2:
            input_params, output_params = parts
            input_params = input_params.strip()
            output_params = output_params.strip()
            # 如果input_params没有被"()"包围，则为其添加括号
            if not input_params.startswith('(') and not input_params.endswith(')'):
                input_params = f"({input_params})"
            # 如果output_params没有被"()"包围，则为其添加括号
            if not output_params.startswith('(') and not output_params.endswith(')'):
                output_params = f"({output_params})"
            signature = f"{full_api_name}{input_params} -> {output_params}"
            return signature
        else:  # 如果函数没有输出值
            input_params = raw_signature.strip()
            output_params = "()"
            # 如果signature没有被"()"包围，则添加括号
            if not input_params.startswith('(') and not input_params.endswith(')'):
                input_params = f"({input_params})"
            signature = f"{full_api_name}{input_params} -> {output_params}"
            return signature

    try:
        # 分割全名以获得模块和属性名
        module_name, attribute_name = full_api_name.rsplit('.', 1)
        # 动态导入模块
        module = importlib.import_module(module_name)
        # 获取属性（函数、类等）
        attribute = getattr(module, attribute_name)
        # 获取签名
        raw_signature = str(inspect.signature(attribute))
        # 处理签名
        signature = process_signature(full_api_name, raw_signature)
        return signature
    except ImportError as e:
        print(f"get_api_signature() encounters an ImportError: {e}")
        return f"{full_api_name}()"
    except AttributeError as e:
        print(f"get_api_signature() encounters an AttributeError: {e}")
        return f"{full_api_name}()"
    except ValueError as e:
        print(f"get_api_signature() encounters a ValueError: {e}")
        return f"{full_api_name}()"
    except Exception as e:
        print(f"get_api_signature() encounters an unexpected error: {e}")
        return f"{full_api_name}()"


def export_all_validated_seeds():  # 从数据库中将所有验证过的seed导出为Python文件
    session = get_session()
    # 获取所有is_validated == True的种子
    seeds = session.query(ClusterTestSeed).filter(ClusterTestSeed.is_validated == True).all()
    for seed in seeds:
        print(f"Exporting validated seed: {seed.valid_folder_path}")
        if not os.path.exists(seed.valid_folder_path):
            os.makedirs(os.path.dirname(seed.valid_folder_path), exist_ok=True)
        # TODO 保存valid_code到该文件夹下


if __name__ == '__main__':
    # export_all_validated_seeds()
    # 示例使用
    signature = get_api_signature("tensorflow.nn.softmax_cross_entropy_with_logits")
    print(signature)
