import importlib
import inspect
import warnings
import httpx
import jax
import mindspore
import jittor
import numpy as np
import os
import torch
from openai import OpenAI
from sqlalchemy.orm import sessionmaker
from orm import *
from rag.rag_client import RagClient
import ast
import re

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
    elif llm == 'gpt4o-mini-with-rag':
        with open('config.yml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            rag_client = RagClient(base_url="http://localhost:8000", api_key=config['openai']['api_key'])
            return rag_client
    elif llm == 'gpt4o-mini-bianxie':
        with open('config.yml', 'r', encoding='utf-8') as file:  # 读取config.yml文件
            config = yaml.safe_load(file)
            openai_client = OpenAI(api_key=config['openai']['bianxie_api_key'], http_client=proxy, base_url="https://api.bianxie.ai/v1")
            return openai_client
    elif llm == 'gpt4.1-mini-bianxie':
        with open('config.yml', 'r', encoding='utf-8') as file:  # 读取config.yml文件
            config = yaml.safe_load(file)
            openai_client = OpenAI(api_key=config['openai']['bianxie_api_key'], http_client=proxy, base_url="https://api.bianxie.ai/v1")
            return openai_client
    elif llm == 'QianWen':
        return None
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
    统计API的聚类状态:
    1. 已完成聚类但没有匹配到任何等价API的API数量 (is_clustered=True 但不属于任何APIGroup)
    2. 未完成聚类的API数量 (is_clustered=False)
    """
    session = get_session()
    try:
        # 查询所有已聚类的API
        clustered_apis = session.query(API).filter(API.is_clustered == True).all()
        
        # 查询所有未聚类的API
        unclustered_apis = session.query(API).filter(API.is_clustered == False).all()
        
        # 查询所有在APIGroup中的API ID
        apis_in_groups = session.query(api_group_association.c.api_id).all()
        apis_in_groups_ids = {api_id[0] for api_id in apis_in_groups}
        
        # 统计已聚类但不在任何APIGroup中的API
        apis_without_cluster = []
        for api in clustered_apis:
            if api.id not in apis_in_groups_ids:
                apis_without_cluster.append(api)
        
        # 按库分类统计已聚类但没有匹配的API
        lib_counts_without_cluster = {}
        for api in apis_without_cluster:
            lib_counts_without_cluster[api.lib] = lib_counts_without_cluster.get(api.lib, 0) + 1
        
        # 按库分类统计未聚类的API
        lib_counts_unclustered = {}
        for api in unclustered_apis:
            lib_counts_unclustered[api.lib] = lib_counts_unclustered.get(api.lib, 0) + 1
        
        # 打印已聚类但没有匹配的API统计
        print(f"已完成聚类但没有匹配到任何等价API的API统计:")
        print("-" * 60)
        total_without_cluster = len(apis_without_cluster)
        for lib, count in lib_counts_without_cluster.items():
            print(f"{lib}: {count} APIs")
        print("-" * 60)
        print(f"总计: {total_without_cluster} APIs")
        
        # 打印未聚类的API统计
        print(f"\n未完成聚类的API统计:")
        print("-" * 60)
        total_unclustered = len(unclustered_apis)
        for lib, count in lib_counts_unclustered.items():
            print(f"{lib}: {count} APIs")
        print("-" * 60)
        print(f"总计: {total_unclustered} APIs")
        
        # 打印总体统计
        print(f"\n总体统计:")
        print("-" * 60)
        print(f"已完成聚类但没有匹配的API: {total_without_cluster}")
        print(f"未完成聚类的API: {total_unclustered}")
        print(f"需要处理的API总数: {total_without_cluster + total_unclustered}")
        
        print("\n已聚类但没有匹配的API详细列表:")
        for api in apis_without_cluster:
            print(f"  {api.full_name} (ID: {api.id}, Lib: {api.lib})")
        
        print("\n未聚类的API详细列表:")
        for api in unclustered_apis:
            print(f"  {api.full_name} (ID: {api.id}, Lib: {api.lib})")
        
        return {
            'without_cluster': total_without_cluster,
            'unclustered': total_unclustered,
            'total_need_processing': total_without_cluster + total_unclustered
        }
        
    except Exception as e:
        print(f"统计API聚类状态时发生错误: {str(e)}")
        return {
            'without_cluster': 0,
            'unclustered': 0,
            'total_need_processing': 0
        }
    finally:
        session.close()


def count_cluster_test_status(cluster_type='ValueEquivalent'):
    """
    统计指定类型的Cluster的测试完成情况
    
    Args:
        cluster_type (str): 聚类类型，可以是 'ValueEquivalent' 或 'StateEquivalent'
    """
    if cluster_type not in ['ValueEquivalent', 'StateEquivalent']:
        print(f"错误: 无效的cluster_type '{cluster_type}'. 必须是 'ValueEquivalent' 或 'StateEquivalent'.")
        return None
    
    session = get_session()
    try:
        # 查询指定类型的所有cluster
        clusters = session.query(Cluster).filter_by(type=cluster_type).all()
        
        if not clusters:
            print(f"未找到 {cluster_type} 类型的cluster.")
            return {'tested': 0, 'untested': 0, 'total': 0}
        
        # 统计已测试和未测试的cluster数量
        tested_clusters = []
        untested_clusters = []
        
        for cluster in clusters:
            if cluster.is_tested:
                tested_clusters.append(cluster)
            else:
                untested_clusters.append(cluster)
        
        tested_count = len(tested_clusters)
        untested_count = len(untested_clusters)
        total_count = len(clusters)
        
        print(f"=== {cluster_type} Cluster 测试状态统计 ===")
        print(f"已完成测试的cluster数量: {tested_count}")
        print(f"未完成测试的cluster数量: {untested_count}")
        print(f"总cluster数量: {total_count}")
        
        if total_count > 0:
            completion_rate = (tested_count / total_count) * 100
            print(f"测试完成率: {completion_rate:.2f}%")
        return {
            'tested': tested_count,
            'untested': untested_count,
            'total': total_count,
            'completion_rate': (tested_count / total_count) * 100 if total_count > 0 else 0
        }
    except Exception as e:
        print(f"统计cluster测试状态时发生错误: {str(e)}")
        return None
    finally:
        session.close()

def count_invalid_cluster_seeds(clusters_folder_path):
    """
    统计指定路径下无效的cluster种子数量
    
    Args:
        clusters_folder_path: cluster文件夹的路径
        
    Returns:
        dict: 包含统计信息的字典
    """
    if not os.path.exists(clusters_folder_path):
        print(f"错误: 路径 {clusters_folder_path} 不存在")
        return {'invalid_files_count': 0, 'clusters_with_invalid_files': 0}
    
    invalid_files_num = 0  # 统计所有无效py文件总数
    total_files_num = 0  # 统计所有py文件总数
    clusters_with_invalid_files_num = 0  # 统计包含无效文件的cluster数量
    
    # 获取所有cluster文件夹（包括带valid前缀的）
    all_folders = [f for f in os.listdir(clusters_folder_path) if os.path.isdir(os.path.join(clusters_folder_path, f))]
    
    # 筛选出cluster文件夹（Cluster_开头或valid_Cluster_开头）
    cluster_folders = []
    for folder in all_folders:
        if folder.startswith('Cluster_') or folder.startswith('valid_Cluster_'):
            cluster_folders.append(folder)
    
    print(f"在{clusters_folder_path}路径下找到 {len(cluster_folders)} 个cluster文件夹")
    print("-" * 60)
    
    for cluster_folder in cluster_folders:
        cluster_path = os.path.join(clusters_folder_path, cluster_folder)
        cluster_invalid_count = 0  # 当前cluster中的无效文件数量
        
        # 获取该cluster下的所有子文件夹（seed文件夹）
        try:
            sub_folders = [f for f in os.listdir(cluster_path) if os.path.isdir(os.path.join(cluster_path, f))]
        except Exception as e:
            print(f"访问 {cluster_path} 时出错: {e}")
            continue
        
        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(cluster_path, sub_folder)
            
            # 获取该子文件夹下的所有文件
            try:
                files = [f for f in os.listdir(sub_folder_path) if os.path.isfile(os.path.join(sub_folder_path, f))]
            except Exception as e:
                print(f"访问 {sub_folder_path} 时出错: {e}")
                continue
            
            # 统计py文件
            for file in files:
                if file.endswith('.py'):
                    total_files_num += 1  # 统计所有py文件
                    if file.startswith('invalid.'):
                        cluster_invalid_count += 1
                        invalid_files_num += 1
                        print(f"  找到无效文件: {cluster_folder}/{sub_folder}/{file}")
        
        # 如果当前cluster有无效文件，则增加cluster计数
        if cluster_invalid_count > 0:
            clusters_with_invalid_files_num += 1
            print(f"{cluster_folder}文件夹下有 {cluster_invalid_count} 个无效文件")
        else:
            print(f"{cluster_folder}文件夹下没有无效文件")
    
    print("-" * 60)
    print(f"统计结果:")
    print(f"  无效py文件总数: {invalid_files_num} / {total_files_num}")
    print(f"  包含无效文件的cluster数量: {clusters_with_invalid_files_num} / {len(cluster_folders)}")
    
    return {
        'invalid_files_count': invalid_files_num,
        'clusters_with_invalid_files': clusters_with_invalid_files_num,
    }

def count_syntax_error_cluster_seeds():
    """
    统计fuzzer/seeds/validated_seeds/下的py文件中存在语法问题的文件数量
    
    检查的语法问题包括：
    1. 使用分号(;)连接代码行而不是换行符
    2. 缺少适当的换行符和缩进
    3. 一行代码过长（超过合理长度）
    
    Returns:
        dict: 包含统计信息的字典
    """

    def check_python_file_syntax(file_path):
        """
        检查单个Python文件是否存在语法问题

        Args:
            file_path: Python文件路径

        Returns:
            tuple: (是否存在语法问题, 问题列表)
        """
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查1: 是否使用分号连接代码而不是换行符
            if ';' in content:
                # 排除字符串中的分号和注释中的分号
                lines = content.split('\n')
                for i, line in enumerate(lines, 1):
                    stripped_line = line.strip()
                    if not stripped_line or stripped_line.startswith('#'):
                        continue

                    # 简单检查：如果一行中有多个分号且不在字符串中
                    semicolon_count = line.count(';')
                    if semicolon_count > 0:
                        # 检查是否在字符串中
                        in_string = False
                        quote_char = None
                        actual_semicolons = 0

                        for j, char in enumerate(line):
                            if char in ['"', "'"] and (j == 0 or line[j - 1] != '\\'):
                                if not in_string:
                                    in_string = True
                                    quote_char = char
                                elif char == quote_char:
                                    in_string = False
                                    quote_char = None
                            elif char == ';' and not in_string:
                                actual_semicolons += 1

                        if actual_semicolons > 0:
                            issues.append(f"第{i}行使用分号连接代码: {actual_semicolons}个分号")

            # 检查2: 检查是否存在过长的单行代码（可能是缺少换行符的标志）
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if len(line.strip()) > 200:  # 超过200字符认为过长
                    issues.append(f"第{i}行代码过长({len(line)}字符)，可能缺少换行符")

            # 检查3: 尝试用AST解析，检查语法是否正确
            try:
                ast.parse(content)
            except SyntaxError as e:
                issues.append(f"Python语法错误: {e.msg} (行 {e.lineno})")

            # 检查4: 检查缩进问题 - 寻找明显的缩进错误模式
            # 例如类或函数定义后没有正确缩进
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.endswith(':') and (
                        'def ' in stripped or 'class ' in stripped or 'if ' in stripped or 'for ' in stripped or 'while ' in stripped):
                    # 检查下一行是否正确缩进
                    if i < len(lines):
                        next_line = lines[i]
                        if next_line.strip() and not next_line.startswith('    ') and not next_line.startswith('\t'):
                            # 但是要排除空行和注释行
                            if not next_line.strip().startswith('#'):
                                issues.append(f"第{i + 1}行可能缺少正确的缩进")

            return len(issues) > 0, issues

        except Exception as e:
            return True, [f"读取文件时出错: {e}"]
    
    # 定义要检查的路径
    paths_to_check = [
        'fuzzer/seeds/validated_seeds/StateEquivalent',
        'fuzzer/seeds/validated_seeds/ValueEquivalent'
    ]
    
    total_files = 0
    syntax_error_files = 0
    detailed_results = []
    
    print("=== 检查 Python 文件语法问题 ===")
    print("-" * 80)
    
    for base_path in paths_to_check:
        if not os.path.exists(base_path):
            print(f"路径不存在: {base_path}")
            continue
            
        print(f"\n检查路径: {base_path}")
        print("-" * 60)
        
        path_total_files = 0
        path_syntax_error_files = 0
        
        # 遍历所有cluster文件夹
        cluster_folders = [f for f in os.listdir(base_path) 
                          if os.path.isdir(os.path.join(base_path, f)) and 
                          (f.startswith('Cluster_') or f.startswith('valid_Cluster_'))]
        
        for cluster_folder in cluster_folders:
            cluster_path = os.path.join(base_path, cluster_folder)
            cluster_error_files = 0
            
            try:
                # 遍历cluster下的所有子文件夹
                sub_folders = [f for f in os.listdir(cluster_path) 
                              if os.path.isdir(os.path.join(cluster_path, f))]
                
                for sub_folder in sub_folders:
                    sub_folder_path = os.path.join(cluster_path, sub_folder)
                    
                    try:
                        # 检查所有Python文件
                        files = [f for f in os.listdir(sub_folder_path) 
                                if f.endswith('.py') and os.path.isfile(os.path.join(sub_folder_path, f))]
                        
                        for file in files:
                            file_path = os.path.join(sub_folder_path, file)
                            total_files += 1
                            path_total_files += 1
                            
                            # 检查文件是否存在语法问题
                            has_syntax_issues, issues = check_python_file_syntax(file_path)
                            
                            if has_syntax_issues:
                                syntax_error_files += 1
                                path_syntax_error_files += 1
                                cluster_error_files += 1
                                
                                detailed_results.append({
                                    'file_path': file_path,
                                    'issues': issues
                                })
                                
                    except Exception as e:
                        print(f"    访问子文件夹 {sub_folder_path} 时出错: {e}")
                        
            except Exception as e:
                print(f"  访问cluster文件夹 {cluster_path} 时出错: {e}")
                continue
            
            # 如果cluster有语法错误文件，显示统计信息
            if cluster_error_files > 0:
                print(f"  {cluster_folder}: {cluster_error_files} 个文件存在语法问题")
        
        print(f"\n{base_path} 统计结果:")
        print(f"  总文件数: {path_total_files}")
        print(f"  存在语法问题的文件数: {path_syntax_error_files}")
        if path_total_files > 0:
            error_rate = (path_syntax_error_files / path_total_files) * 100
            print(f"  语法错误率: {error_rate:.2f}%")
    
    print("\n" + "=" * 80)
    print("总体统计结果:")
    print(f"  检查的总文件数: {total_files}")
    print(f"  存在语法问题的文件数: {syntax_error_files}")
    if total_files > 0:
        overall_error_rate = (syntax_error_files / total_files) * 100
        print(f"  总体语法错误率: {overall_error_rate:.2f}%")
    
    # 显示详细的问题分类统计
    if not detailed_results:
        return

    print("\n详细问题分类统计:")
    print("-" * 60)

    issue_categories = {
        '使用分号': 0,
        '代码过长': 0,
        'Python语法错误': 0,
        '缩进问题': 0,
        '其他问题': 0
    }

    for result in detailed_results:
        for issue in result['issues']:
            if '分号' in issue:
                issue_categories['使用分号'] += 1
            elif '过长' in issue:
                issue_categories['代码过长'] += 1
            elif 'Python语法错误' in issue:
                issue_categories['Python语法错误'] += 1
            elif '缩进' in issue:
                issue_categories['缩进问题'] += 1
            else:
                issue_categories['其他问题'] += 1

    for category, count in issue_categories.items():
        if count > 0:
            print(f"  {category}: {count} 个问题")
    return {
        'total_files': total_files,
        'syntax_error_files': syntax_error_files,
        'error_rate': (syntax_error_files / total_files) * 100 if total_files > 0 else 0,
        'detailed_results': detailed_results
    }

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

    # 统计cluster测试状态
    # count_cluster_test_status('ValueEquivalent')
    # print("\n")
    # count_cluster_test_status('StateEquivalent')
    # print("\n")

    # count_api_without_cluster()
    #count_invalid_cluster_seeds('fuzzer/seeds/validated_seeds/ValueEquivalent') # 无效py文件总数: 858 / 16423; 包含无效文件的cluster数量: 297 / 1006
    #count_invalid_cluster_seeds('fuzzer/seeds/validated_seeds/StateEquivalent') # 无效py文件总数: 2609 / 28825; 包含无效文件的cluster数量: 754 / 1918
    # clean_invalid_clusters()
    
    # 统计语法错误的cluster种子文件
    count_syntax_error_cluster_seeds() # Total: 19187/45248(错误率42.40%) | StateEquivalent: 12398/28825(错误率43.01%) | ValueEquivalent: 6789/16423(错误率41.34%)

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

    # retrieve_api_issues('jax.jit')