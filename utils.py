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


def get_libs_info():  # 该函数将返回数据库中待测试的深度学习库的名称和版本, 比如[('Pytorch', '2.4.1'), ('JAX', '0.4.33'), ('MindSpore', '2.5.0'), ('Jittor', '1.3.9.14')]
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
    module_alias_mapper = {
        "tf": "tensorflow",
        "ms": "mindspore",
        "np": "numpy",
        "pd": "pandas",
        "jt": "jittor",
        "pytorch": "torch",
    }
    try:
        module_list = module_name.split('.')
        # 先检查来源库是否为Pytorch, JAX, MindSpore或Jittor中的任意一个
        api_lib = module_list[0]
        if map_module2lib(api_lib) == 'Unknown':
            return False
        module = importlib.import_module(module_alias_mapper.get(api_lib, api_lib))
        if len(module_list) > 1:
            # 将module_name_list进行切片, 只保留除第一个元素以外的部分
            for submodule_name in module_list[1:]:
                module = getattr(module, submodule_name, None)
                if module is None:
                    return False
        api = getattr(module, api_name, None)
        if api is None:
            return False
        else:
            return True
    except (ModuleNotFoundError, AttributeError, ImportError, ValueError, Exception) as e:
        print(f"validate_api_existence({module_name}, {api_name}) Error: {e}")
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
        'pytorch': "Pytorch",
        'torch': 'Pytorch',
        'JAX': 'JAX',
        'jax': 'JAX',
        'jaxlib': 'JAX',
        'MindSpore': 'MindSpore',
        'ms': 'MindSpore',
        'mindspore': 'MindSpore',
        'Jittor': 'Jittor',
        'jittor': 'Jittor',
        'jt': 'Jittor',
    }
    return lib_map.get(module_name, 'Unknown')


def inspect_api_info(module_name, api_name):
    module_alias_mapper = {
        "tf": "tensorflow",
        "ms": "mindspore",
        "np": "numpy",
        "pd": "pandas",
        "jt": "jittor",
        "jnp": "jax.numpy",
        "pytorch": "torch",
    }
    # 将module_name中的lib别名转换为实际的库名
    module_list = module_name.split('.')
    module_list[0] = module_alias_mapper.get(module_list[0], module_list[0])
    module_name = '.'.join(module_list)
    module_list = module_name.split('.') # 防止"jax.numpy"这种情况
    if validate_api_existence(module_name, api_name) is False:  # 验证API是否存在
        print(f"inspect_api_info({module_name}, {api_name}) Error: API {api_name} does not exist.")
        return None

    module = importlib.import_module(module_list[0])
    if len(module_list) > 1:
        for submodule_name in module_list[1:]:
            module = getattr(module, submodule_name, None)
    api = getattr(module, api_name, None)

    # 获取API所属的库
    api_lib = module_list[0]  # 用"."分割module_name, 然后取第一个部分作为库名
    lib = map_module2lib(api_lib)

    # 获取函数签名
    signature = get_api_signature(f"{module_name}.{api_name}")

    # 获取函数文档
    description = ""
    try:
        description = inspect.getdoc(api)
    except Exception as e:
        print(f"inspect_api_info({module_name}, {api_name}) Warning: inspect.getdoc({api_name}) encountered '{e}'")

    # 获取API的版本
    version = ""
    lib_version_list = get_libs_info()  # [('Pytorch', '2.4.1'), ('JAX', '0.4.33'), ('MindSpore', '2.5.0'), ('Jittor', '1.3.9.14')]
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
        module_name, api_name = full_api_name.rsplit('.', 1)
        module_list = module_name.split('.')
        module = importlib.import_module(module_list[0])
        if len(module_list) > 1:
            for submodule_name in module_list[1:]:
                module = getattr(module, submodule_name, None)
        api = getattr(module, api_name, None)
        # 获取签名
        raw_signature = ""
        if inspect.isbuiltin(api) is False and callable(api):
            str(inspect.signature(api))
        # 处理签名
        signature = process_signature(full_api_name, raw_signature)
        return signature
    except ImportError as e:
        print(f"get_api_signature({full_api_name}) ImportError: {e}")
        return f"{full_api_name}()"
    except AttributeError as e:
        print(f"get_api_signature({full_api_name}) AttributeError: {e}")
        return f"{full_api_name}()"
    except ValueError as e:
        print(f"get_api_signature({full_api_name}) ValueError: {e}")
        return f"{full_api_name}()"
    except Exception as e:
        print(f"get_api_signature({full_api_name}) Error: {e}")
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


def count_api_nums_with_history_errors(lib): # 计算指定库中有多少API存在历史错误
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

def get_api_info(full_api_name='torch.nn.functional.cross_entropy'):
    # 从数据库中获取API信息
    session = get_session()
    api = session.query(API).filter_by(full_name=full_api_name).first()
    if api:
        print(f"id: {api.id}\n\n"
              f"name: {api.name}\n\n"
              f"lib: {api.lib}\n\n"
              f"version: {api.version}\n\n"
              f"module: {api.module}\n\n"
              f"full_name: {api.full_name}\n\n"
              f"signature: {api.signature}\n\n"
              f"parameters: {api.parameters}\n\n"
              f"attributes: {api.attributes}\n\n"
              f"output: {api.output}\n\n"
              f"description: {api.description}\n\n"
              f"example: {api.example}\n\n")

def list_clusters(cluster_type='ValueEquivalent'):
    """
    列出指定类型的所有cluster及其包含的API
    
    Args:
        cluster_type (str): 聚类类型，可以是 'ValueEquivalent' 或 'StateEquivalent'
    """
    if cluster_type not in ['ValueEquivalent', 'StateEquivalent']:
        print(f"Error: Invalid cluster_type '{cluster_type}'. Must be 'ValueEquivalent' or 'StateEquivalent'.")
        return
    
    session = get_session()
    try:
        # 查询指定类型的所有cluster
        clusters = session.query(Cluster).filter_by(type=cluster_type).all()
        
        if not clusters:
            print(f"No {cluster_type} clusters found.")
            return
        
        print(f"=== {cluster_type} Clusters ===")
        print(f"Total {cluster_type} clusters: {len(clusters)}")
        print("=" * 80)
        
        for i, cluster in enumerate(clusters, 1):
            print(f"\nCluster #{i} (ID: {cluster.id})")
            print("-" * 40)
            
            # 获取该cluster下的所有API组
            api_groups = cluster.api_groups
            if not api_groups:
                print("  No API groups found in this cluster.")
                continue
            
            for j, api_group in enumerate(api_groups, 1):
                if len(api_group.apis) == 1:
                    print(f"  Group {j}: Single API")
                else:
                    print(f"  Group {j}: API Group ({len(api_group.apis)} APIs)")
                
                # 打印该组中的所有API
                for api in api_group.apis:
                    print(f"    - {api.full_name} (ID: {api.id}, Lib: {api.lib})")
            
            print(f"  Total API groups in this cluster: {len(api_groups)}")
            total_apis = sum(len(group.apis) for group in api_groups)
            print(f"  Total APIs in this cluster: {total_apis}")
        
        # 统计信息
        total_api_groups = sum(len(cluster.api_groups) for cluster in clusters)
        total_apis = sum(sum(len(group.apis) for group in cluster.api_groups) for cluster in clusters)
        
        print("=" * 80)
        print(f"Summary for {cluster_type} clusters:")
        print(f"  Total clusters: {len(clusters)}")
        print(f"  Total API groups: {total_api_groups}")
        print(f"  Total APIs: {total_apis}")
        
    except Exception as e:
        print(f"An error occurred while listing clusters: {str(e)}")
    finally:
        session.close()


def clean_invalid_clusters():
    """
    清理无效的cluster
    无效cluster包括：
    1. 只包含一个API组的cluster
    2. 包含重复APIGroup的cluster（APIGroup下的API集合完全相同）
    """
    session = get_session()
    try:
        invalid_clusters = []
        # 查询所有cluster
        clusters = session.query(Cluster).all()
        for cluster in clusters:
            is_invalid = False
            invalid_reason = ""
            
            # 检查条件1：只有一个或没有API组
            if len(cluster.api_groups) <= 1:
                is_invalid = True
                invalid_reason = f"Only {len(cluster.api_groups)} API group(s)"
            
            # 检查条件2：存在重复的APIGroup（API集合相同）
            elif len(cluster.api_groups) > 1:
                api_group_signatures = []
                for api_group in cluster.api_groups:
                    # 为每个APIGroup创建签名：按API的full_name排序后组成的元组
                    api_names = sorted([api.full_name for api in api_group.apis])
                    signature = tuple(api_names)
                    api_group_signatures.append(signature)
                
                # 检查是否所有APIGroup都是重复的（即只有一种唯一的API集合）
                unique_signatures = set(api_group_signatures)
                if len(unique_signatures) == 1:
                    is_invalid = True
                    invalid_reason = f"All API groups are identical (API set: {list(unique_signatures)[0]})"
            
            if is_invalid:
                invalid_clusters.append((cluster, invalid_reason))
        
        if not invalid_clusters:
            print("No invalid clusters found.")
            return
        
        print(f"Found {len(invalid_clusters)} invalid clusters:")
        for cluster, reason in invalid_clusters:
            print(f"  Cluster ID: {cluster.id}, Type: {cluster.type}, Reason: {reason}")
            print(f"    API groups: {len(cluster.api_groups)}")
            if cluster.api_groups:
                for i, api_group in enumerate(cluster.api_groups, 1):
                    api_names = [api.full_name for api in api_group.apis]
                    print(f"      Group {i}: {api_names}")
        
        # 询问是否删除
        response = input(f"\nDo you want to delete these {len(invalid_clusters)} invalid clusters? (y/N): ")
        if response.lower() in ['y', 'yes']:
            for cluster, reason in invalid_clusters:
                print(f"Deleting cluster {cluster.id} (Reason: {reason})...")
                
                # 1. 重置相关API的is_clustered状态
                for api_group in cluster.api_groups:
                    for api in api_group.apis:
                        api.is_clustered = False
                        print(f"  Reset API {api.full_name} is_clustered to False")
                
                # 2. 显式删除相关的APITestSeed
                for api_group in cluster.api_groups:
                    api_seeds = api_group.api_seeds
                    for api_seed in api_seeds:
                        print(f"  Deleting APITestSeed {api_seed.id}")
                        session.delete(api_seed)
                
                # 3. 显式删除相关的ClusterTestSeed
                cluster_seeds = cluster.cluster_seeds
                for cluster_seed in cluster_seeds:
                    print(f"  Deleting ClusterTestSeed {cluster_seed.id}")
                    # 先删除cluster_seed下的所有api_seeds
                    for api_seed in cluster_seed.api_seeds:
                        print(f"    Deleting APITestSeed {api_seed.id} from ClusterTestSeed")
                        session.delete(api_seed)
                    session.delete(cluster_seed)
                
                # 4. 显式删除相关的APIGroup
                api_groups = list(cluster.api_groups)  # 创建副本避免迭代时修改
                for api_group in api_groups:
                    print(f"  Deleting APIGroup {api_group.id}")
                    session.delete(api_group)
                
                # 5. 最后删除cluster
                print(f"  Deleting Cluster {cluster.id}")
                session.delete(cluster)
            
            session.commit()
            print(f"Successfully deleted {len(invalid_clusters)} invalid clusters with cascade deletion.\n\n")
        else:
            print("No clusters were deleted.\n\n")
            
    except Exception as e:
        session.rollback()
        print(f"An error occurred while cleaning invalid clusters: {str(e)}\n\n")
    finally:
        session.close()
        
def count_api_without_cluster():
    """
    统计已完成聚类但没有匹配到任何等价API的API数量
    即统计 is_clustered=True 但不属于任何APIGroup的API
    """
    session = get_session()
    try:
        # 查询所有已聚类的API
        clustered_apis = session.query(API).filter(API.is_clustered == True).all()
        
        # 查询所有在APIGroup中的API ID
        apis_in_groups = session.query(api_group_association.c.api_id).all()
        apis_in_groups_ids = {api_id[0] for api_id in apis_in_groups}
        
        # 统计已聚类但不在任何APIGroup中的API
        apis_without_cluster = []
        for api in clustered_apis:
            if api.id not in apis_in_groups_ids:
                apis_without_cluster.append(api)
        
        # 按库分类统计
        lib_counts = {}
        for api in apis_without_cluster:
            lib_counts[api.lib] = lib_counts.get(api.lib, 0) + 1
        
        print(f"已完成聚类但没有匹配到任何等价API的API统计:")
        print("-" * 60)
        total_count = len(apis_without_cluster)
        for lib, count in lib_counts.items():
            print(f"{lib}: {count} APIs")
        print("-" * 60)
        print(f"总计: {total_count} APIs")
        
        print("\n详细列表:")
        for api in apis_without_cluster:
            print(f"  {api.full_name} (ID: {api.id}, Lib: {api.lib})")
        
        return total_count
        
    except Exception as e:
        print(f"统计无聚类API时发生错误: {str(e)}")
        return 0
    finally:
        session.close()


if __name__ == '__main__':
    # print(get_libs_info())
    # count_api_nums_with_history_errors('Pytorch')
    # count_api_nums_with_history_errors('MindSpore')
    # count_api_nums_with_history_errors('JAX')
    # count_api_nums_with_history_errors('Jittor')

    # list_clusters('ValueEquivalent')
    # print("\n")
    # list_clusters('StateEquivalent')
    # print("\n")

    count_api_without_cluster()

    # clean_invalid_clusters()

    # full_api_name = "jax.jit"
    # retrieve_api_issues(full_api_name)
    # print("="*60)
    # get_api_info(full_api_name)

    # session = get_session()
    # # 检查哪个API的history_errors最多
    # apis = session.query(API).all()
    # max_history_errors = 0
    # max_history_errors_api = None
    # for api in apis:
    #     if len(api.history_errors) > max_history_errors:
    #         max_history_errors = len(api.history_errors)
    #         max_history_errors_api = api
    # print(f"API with the most history errors: {max_history_errors_api.full_name} ({max_history_errors})")
    # session.close()