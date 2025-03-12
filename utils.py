import importlib
import inspect
import warnings
import httpx
import jax
import mindspore
import jittor
import numpy as np
import torch
from openai import OpenAI
from sqlalchemy.orm import sessionmaker
from orm import *
from rag.rag_client import RagClient


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


def get_llm_client(llm='gpt4o-mini', proxy_url="http://127.0.0.1:7890"):
    # 设置代理
    proxy = httpx.Client(proxies={
        "http://": proxy_url,
        "https://": proxy_url
    })
    # 根据llm的名称返回对应的客户端
    if llm == 'gpt4o-mini':
        with open('config.yml', 'r', encoding='utf-8') as file:  # 读取config.yml文件
            config = yaml.safe_load(file)
            openai_client = OpenAI(api_key=config['openai']['api_key'], http_client=proxy)
            return openai_client
    elif llm == 'QianWen':
        return None
    elif llm == 'gpt4o-mini-with-rag':
        with open('config.yml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            rag_client = RagClient(base_url="http://localhost:8000", api_key=config['openai']['api_key'])
            return rag_client
    else:
        return None


def get_libs_info():  # 该函数将返回数据库中待测试的深度学习库的名称和版本, 比如[('Pytorch', '1.12'), ('JAX', '0.4.13'), ('MindSpore', '2.4.0')]
    db_session = get_session()
    try:
        results = db_session.query(API.lib, API.version).distinct().all()
        return results
    except Exception as e:
        print(f"An error occurred while getting tested libraries: {str(e)}")
        return []
    finally:
        db_session.close()


def validate_api_existence(module_name: str, api_name: str):  # 验证API是否存在的函数
    # module_name, api_name = full_api_name.rsplit('.', 1)
    try:
        # 先检查来源库是否为Pytorch, JAX, MindSpore或Jittor中的任意一个
        api_lib = module_name.split('.')[0]  # 用"."分割module_name, 然后取第一个部分作为库名
        lib = map_module2lib(api_lib)
        if lib == 'Unknown':
            return False
        # 然后检查模块是否存在
        module = importlib.import_module(module_name)
        # 再检查API是否存在
        func = getattr(module, api_name, None)
        if func is None:
            return False
        if inspect.ismodule(func):
            return False
        # if inspect.isclass(func): # 诸如torch.nn.CrossEntropyLoss等用类封装的API将无法被测试, 因此选择注释掉
        #     return True
        return True
    except (ModuleNotFoundError, AttributeError, ImportError, ValueError, Exception) as e:
        print(f"validate_api_existence() encounters an error: {e}")
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


def map_module2lib(module_name):
    lib_map = {
        'Pytorch': 'Pytorch',
        'torch': 'Pytorch',
        'JAX': 'JAX',
        'jax': 'JAX',
        'jaxlib': 'JAX',
        'MindSpore': 'MindSpore',
        'ms': 'MindSpore',
        'mindspore': 'MindSpore',
        'Jittor': 'Jittor',
        'jittor': 'Jittor',
    }
    return lib_map.get(module_name, 'Unknown')


def inspect_api_info(module_name, api_name):
    if validate_api_existence(module_name, api_name) is False:  # 验证API是否存在
        print(f"API {api_name} does not exist.")
        return None

    module = importlib.import_module(module_name)  # 动态导入模块
    func = getattr(module, api_name)  # 从模块中获取函数对象

    # if validate_api_availability(func) is True:  # 验证API是否为被弃用的函数
    #     print(f"API {api_name} is deprecated.")
    #     return None

    # 获取函数签名
    signature = get_api_signature(f"{module_name}.{api_name}")

    # 获取函数文档
    description = ""
    try:
        description = inspect.getdoc(func)
    except Exception as e:
        print(f"inspect_api_info(): Error getting doc for {module_name}.{api_name} due to: {e}")

    # 获取API所属的库
    api_lib = module_name.split('.')[0]  # 用"."分割module_name, 然后取第一个部分作为库名
    lib = map_module2lib(api_lib)

    # 获取API的版本
    version = ""
    lib_version_list = get_libs_info()  # [('Pytorch', '1.12'), ('JAX', '0.4.13'), ('MindSpore', '2.4.0')]
    for lib_name, lib_version in lib_version_list:
        if lib_name.lower() == lib.lower():
            version = lib_version
            break

    api_info = {
        "module": module_name,
        "name": api_name,
        "full_name": f"{module_name}.{api_name}",
        "signature": signature,
        "description": description,
        "lib": lib,
        "version": version
    }
    return api_info

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
            # output_params = "()"
            # 如果signature没有被"()"包围，则添加括号
            if not input_params.startswith('(') and not input_params.endswith(')'):
                input_params = f"({input_params})"
            # signature = f"{full_api_name}{input_params} -> {output_params}"
            signature = f"{full_api_name}{input_params}"
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


def count_fuzz_time():
    session = get_session()
    try:
        seeds = session.query(ClusterTestSeed).filter(
            ClusterTestSeed.start_test != None,
            ClusterTestSeed.end_test != None
        ).all()

        total_duration = sum(
            (seed.end_test - seed.start_test).total_seconds() for seed in seeds if seed.end_test and seed.start_test)
        total_seeds = len(seeds)

        if total_seeds > 0:
            average_duration = total_duration / total_seeds
            formatted_total_duration = format(total_duration, ".3f")
            formatted_average_duration = format(average_duration, ".3f")
        else:
            formatted_total_duration = "0.000"
            formatted_average_duration = "0.000"

        print(f"Total time spent on generating {total_seeds} seeds: {formatted_total_duration} seconds")
        print(f"Average time per seed: {formatted_average_duration} seconds")

    except Exception as e:
        print(f"An error occurred while calculating fuzzing times: {str(e)}")
    finally:
        session.close()


def get_cluster_api_group(cluster_id: int):
    session = get_session()
    cluster = session.query(Cluster).filter(Cluster.id == cluster_id).first()
    api_groups = cluster.api_groups
    for api_group in api_groups:
        print("-" * 60)
        apis = api_group.apis
        for api in apis:
            print(f"(API ID: {api.id}, Full Name: {api.full_name})", end=", ")


def convert2numpy(vector):
    # 先检查x是否是张量
    if isinstance(vector, torch.Tensor):
        return vector.numpy()
    elif isinstance(vector, jax.Array):
        return np.array(vector)
    elif isinstance(vector, mindspore.Tensor):
        return vector.asnumpy()
    elif isinstance(vector, jittor.Var):
        return vector.numpy()
    elif isinstance(vector, list):
        return np.array(vector)
    elif isinstance(vector, np.ndarray):
        return vector
    else:
        try:
            return np.array(vector)
        except Exception as e:
            print(f"Failed to convert {type(vector)} to numpy array: {e}")
            return vector  # 如果无法转换，返回原始输出


def cosine_similarity(x, y):
    x = convert2numpy(x)
    y = convert2numpy(y)
    if x.shape != y.shape:  # 查看维度是否相同
        raise ValueError(f"Shape mismatch when calculating cosine similarity: {x.shape} vs {y.shape}")
    return np.dot(x.flatten(), y.flatten()) / (np.linalg.norm(x.flatten()) * np.linalg.norm(y.flatten()))


def count_api_nums_with_history_errors(lib):
    session = get_session()
    try:
        apis = session.query(API).filter_by(lib=lib).all()
        count = 0
        for api in apis:
            if api.history_errors:
                count += 1
        print(f"Number of {lib} APIs with history errors: {count} / {len(apis)}")
    except Exception as e:
        print(f"An error occurred while counting APIs with history errors: {str(e)}")
    finally:
        session.close()


def retrieve_api_issues(full_api_name):
    session = get_session()
    try:
        api = session.query(API).filter_by(full_name=full_api_name).first()
        if api:
            issues = api.history_errors
            for issue in issues:
                print("-" * 60)
                print(f"ID:{issue.id}")
                print(f"URL:{issue.issue_url}")
                print(f"Title: {issue.title}")
                print(f"Description: {issue.description}")
                print(f"Code:")
                print(f"{issue.code}")
        else:
            print(f"API {full_api_name} does not exist.")
    except Exception as e:
        print(f"An error occurred while retrieving API issues: {str(e)}")
    finally:
        session.close()


if __name__ == '__main__':
    # list = [
    #    'torch.nn.functional.relu',
    #    'torch.nn.ReLU',
    #    'mindspore.ops.relu',
    #    'mindspore.nn.ReLU',
    #    'jax.nn.relu',
    #    'jittor.init.calculate_gain',
    # ]
    # list = [
    #     'torch.nn.functional.cross_entropy',
    #     'torch.nn.CrossEntropyLoss',
    #     'mindspore.nn.CrossEntropyLoss',
    #     "mindspore.ops.cross_entropy",
    # ]
    # for api in list:
    #     module_name, api_name = api.rsplit('.', 1)
    #     print(validate_api_existence(module_name, api_name))

    # count_api_nums_with_history_errors('Pytorch') # 85/890(旧); 461/1201(旧); 597/1335(新)
    # count_api_nums_with_history_errors('JAX') # 269/961(旧); 306/974(旧); 463/1015(新)
    # count_api_nums_with_history_errors('MindSpore') # 248/2378(新)
    # session = get_session()
    ## 从数据库中获取API中version为""的api
    # apis = session.query(API).filter_by(version="").all()
    # for api in apis:
    #    print(api.full_name)
    retrieve_api_issues('torch.nn.functional.cross_entropy')
