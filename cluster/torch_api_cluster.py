import json

import httpx
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, joinedload
import yaml
from orm import OutputEquivalenceCluster, Tensorflow, Pytorch, JAX


def pytorch_apis_output_equivalence_cluster(session, openai_client, pytorch_api):
    # 构造Prompt
    example_1 = """
    {
        "Tensorflow" : {
            1 : "tf.keras.losses.CategoricalCrossentropy",
            2 : "tf.nn.softmax_cross_entropy_with_logits"
        },
        "JAX" : {
            1 : "jax.nn.softmax_cross_entropy"
        }
    }
    """
    example_2 = """
    {
        "Tensorflow" : {}, # When there are no TensorFlow APIs whose combined output with Pytorch's API has the same value, output an empty json
        "JAX" : {}
    }
    """
    prompt = f"""
    Which apis in Tensorflow and JAX (or combinations of these and other apis) have the same return value as Pytorch's {pytorch_api.name}? Please output the function names in JSON format following these examples:
    Example 1: {example_1}
    Example 2: {example_2}
    """

    json_data = None
    attempt = 0
    max_attempts = 5  # 设置最大尝试次数以避免无限循环
    while attempt < max_attempts:
        try:  # 假如返回的数据不符合JSON格式, 则重新调用OpenAI API, 直到返回的数据符合JSON格式为止
            # 调用 OpenAI API
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
            )
            data = response.choices[0].message.content
            json_data = json.loads(data)
            print(json_data)  # 打印解析后的 JSON 数据
            break  # 成功解析 JSON,跳出循环
        except json.JSONDecodeError:
            print(f"Failed to decode JSON on attempt {attempt + 1}. Retrying...")
            attempt += 1
        except Exception as e:
            session.rollback()  # 回滚在异常中的任何数据库更改
            print(f"An unexpected error occurred: {e}")
            break
    if attempt == max_attempts:
        print("Max attempts reached. Unable to get valid JSON data.")
        return

    try:
        # 1. 解析返回的 JSON 数据, 如果Tensorflow表或JAX表中没有对应的API Name, 则先在Tensorflow表或JAX表中创建对应的数据
        tensorflow_apis = []
        for tf_id, tf_name in json_data['Tensorflow'].items():
            tensorflow_api = session.query(Tensorflow).filter_by(name=tf_name).first()
            if not tensorflow_api:
                tensorflow_api = Tensorflow(name=tf_name)
                session.add(tensorflow_api)
                session.commit()
            tensorflow_apis.append(tensorflow_api)

        jax_apis = []
        for jax_id, jax_name in json_data['JAX'].items():
            jax_api = session.query(JAX).filter_by(name=jax_name).first()
            if not jax_api:
                jax_api = JAX(name=jax_name)
                session.add(jax_api)
                session.commit()
            jax_apis.append(jax_api)

        # 2. 创建OutputEquivalenceCluster对象, 并将其与Pytorch对象,Tensorflow对象和JAX对象关联
        pytorch_api.is_clustered = True
        if pytorch_api.output_clusters:  # 先检查pytorch_api是否已经出现在一个聚类中
            # 如果该pytorch_api已经在一个输出等价集群中,则检测tensorflow_apis和jax_apis是否已经在该集群中
            cluster = pytorch_api.output_clusters[0]
            for tensorflow_api in tensorflow_apis:
                if tensorflow_api not in cluster.tensorflows:
                    cluster.tensorflows.append(tensorflow_api)
            for jax_api in jax_apis:
                if jax_api not in cluster.jaxes:
                    cluster.jaxes.append(jax_api)
        else:  # 如果该pytorch_api还没有出现在一个聚类中,则创建一个新的输出等价集群
            cluster = OutputEquivalenceCluster()
            cluster.pytorches.append(pytorch_api)
            cluster.tensorflows.extend(tensorflow_apis)
            cluster.jaxes.extend(jax_apis)
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
    proxy = httpx.Client(proxies={
        "http://": "http://127.0.0.1:7890",
        "https://": "http://127.0.0.1:7890"
    })
    openai_client = OpenAI(api_key=config['openai']['api_key'], http_client=proxy)

    uncluttered_torch_apis = session.query(Pytorch).filter_by(is_clustered=False).all()
    while uncluttered_torch_apis:
        print("----------------------------------------------------------------------------------")
        pytorch_apis_output_equivalence_cluster(session, openai_client, uncluttered_torch_apis[0])
        uncluttered_torch_apis = session.query(Pytorch).filter_by(is_clustered=False).all()
        total_apis_num = session.query(Pytorch).count()
        unclustered_torch_apis_num = len(uncluttered_torch_apis)
        print(f"Unclustered / Total: {unclustered_torch_apis_num} / {total_apis_num}")


if __name__ == '__main__':
    run()
