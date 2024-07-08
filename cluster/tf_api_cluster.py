import json
import httpx
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, joinedload
import yaml
from orm import OutputEquivalenceCluster, Tensorflow, Pytorch, JAX


def tensorflow_apis_output_equivalence_cluster(session, openai_client, tensorflow_api):
    # 利用clusterer进行聚类
    clusterer_example_1 = """
    {
        "Pytorch" : {
            "1" : "torch.nn.ReLU",
        },
        "JAX" : {
            "1" : "jax.nn.softmax_cross_entropy"
        }
    }
    """
    clusterer_example_2 = """
    {
        "Pytorch" : {}, # When there are no Pytorch APIs whose combined output with Tensorflow's API has the same value, output an empty json
        "JAX" : {}
    }
    """
    clusterer_prompt = f"""
    Which APIs in PyTorch (v2.10) and JAX (v0.4.13) have the same functionality or return values as the {tensorflow_api.name} in TensorFlow (v2.10)? 
    Note: "The same functionality" means that these APIs are responsible for performing the same tasks, such as PyTorch's torch.scatter_, TensorFlow's tf.scatter_update, and JAX's jax.ops.index_update all have the functionality to update tensors. "The same return values" mean that when the input values are the same or equivalent, the output values of the API are equal or equivalent, such as PyTorch's torch.nn.ReLU, TensorFlow's tf.nn.relu or tf.keras.layers.ReLU, and JAX's jax.nn.relu all have the same output values. 
    Please output the function names in Pytorch and JAX that meet the above conditions in JSON format, with some examples shown below:
    Example 1: {clusterer_example_1}
    Example 2: {clusterer_example_2}
    """
    clusterer_response_data = None
    clusterer_json_data = None
    clusterer_attempt_num = 0
    clusterer_max_attempts_num = 5  # 设置最大尝试次数以避免无限循环
    while clusterer_attempt_num < clusterer_max_attempts_num:
        try:  # 假如返回的数据不符合JSON格式, 则重新调用OpenAI API, 直到返回的数据符合JSON格式为止
            # 调用 OpenAI API
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": clusterer_prompt}
                ],
                temperature=0.0,
            )
            clusterer_response_data = response.choices[0].message.content
            print(f"Clustered Tensorflow API: {tensorflow_api.name}")
            print(clusterer_response_data)
            clusterer_json_data = json.loads(clusterer_response_data)
            break  # 成功解析 JSON,跳出循环
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON due to: {e}.\nCurrent attempt: {clusterer_attempt_num + 1}. Retrying...")
            clusterer_attempt_num += 1
        except Exception as e:
            session.rollback()  # 回滚在异常中的任何数据库更改
            print(f"An unexpected error occurred: {e}")
            break
    if clusterer_attempt_num == clusterer_max_attempts_num:
        print("Max attempts reached. Unable to get valid JSON data.")
        return

    try:
        # 1. 解析返回的 JSON 数据, 如果Pytorch表或JAX表中没有对应的API Name, 则先在Pytorch表或JAX表中创建对应的数据
        pytorch_apis = {}
        for torch_id, torch_name in clusterer_json_data['Pytorch'].items():
            if torch_name not in pytorch_apis:
                torch_api = session.query(Pytorch).filter_by(name=torch_name).first()
                if not torch_api:
                    torch_api = Pytorch(name=torch_name)
                    session.add(torch_api)
                    session.commit()
                pytorch_apis[torch_name] = torch_api

        jax_apis = {}
        for jax_key, jax_name in clusterer_json_data['JAX'].items():
            if jax_name not in jax_apis:
                jax_api = session.query(JAX).filter_by(name=jax_name).first()
                if not jax_api:
                    jax_api = JAX(name=jax_name)
                    session.add(jax_api)
                    session.commit()
                jax_apis[jax_name] = jax_api

        # 2. 创建OutputEquivalenceCluster对象, 并将其与Pytorch对象, Tensorflow对象和JAX对象关联
        tensorflow_api.is_clustered = True
        if tensorflow_api.output_clusters:  # 先检查pytorch_api是否已经出现在一个聚类中
            # 如果该tensorflow_apis已经在一个输出等价集群中,则检测pytorch_api和jax_apis是否已经在该集群中
            cluster = tensorflow_api.output_clusters[0]
            for torch_api in pytorch_apis.values():
                if torch_api not in cluster.tensorflows:
                    cluster.tensorflows.append(torch_api)
            for jax_api in jax_apis.values():
                if jax_api not in cluster.jaxes:
                    cluster.jaxes.append(jax_api)
        else:  # 如果该pytorch_api还没有出现在一个聚类中,则创建一个新的输出等价集群
            cluster = OutputEquivalenceCluster()
            cluster.tensorflows.append(tensorflow_api)
            cluster.pytorches.extend(pytorch_apis.values())
            cluster.jaxes.extend(jax_apis.values())
            session.add(cluster)
        session.commit()
    except Exception as e:
        session.rollback()  # 回滚在异常中的任何数据库更改
        print(f"An error occurred: {e}")


def run():
    with open('config.yml', 'r', encoding='utf-8') as file:  # 读取config.yml文件
        config = yaml.safe_load(file)
    # 从配置中提取数据库连接信息
    db_config = config['db']['mysql']
    host = db_config['host']
    user = db_config['user']
    password = db_config['password']
    database = db_config['database']
    db_url = f"mysql+pymysql://{user}:{password}@{host}/{database}"  # 创建数据库连接字符串
    # 创建数据库连接
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    session = Session()

    # 设置代理
    # proxy = httpx.Client(proxies={
    #     "http://": "http://127.0.0.1:7890",
    #     "https://": "http://127.0.0.1:7890"
    # })
    # openai_client = OpenAI(api_key=config['openai']['api_key'], http_client=proxy)

    openai_client = OpenAI(base_url="https://api.gptsapi.net/v1/",
                           api_key="sk-4Yg7f4b436b8fb189fc0f426d378e395adf93f7ba45pT6Os")  # WildCard API + 转发, 无需代理

    uncluttered_torch_apis = session.query(Tensorflow).filter_by(is_clustered=False).all()
    while uncluttered_torch_apis:
        print("----------------------------------------------------------------------------------")
        tensorflow_apis_output_equivalence_cluster(session, openai_client, uncluttered_torch_apis[0])
        uncluttered_torch_apis = session.query(Tensorflow).filter_by(is_clustered=False).all()
        total_apis_num = session.query(Tensorflow).count()
        unclustered_torch_apis_num = len(uncluttered_torch_apis)
        print(f"Unclustered / Total: {unclustered_torch_apis_num} / {total_apis_num}")


if __name__ == '__main__':
    run()
