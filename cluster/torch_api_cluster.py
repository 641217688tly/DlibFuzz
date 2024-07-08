import json
import httpx
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, joinedload
import yaml
from orm import OutputEquivalenceCluster, Tensorflow, Pytorch, JAX


def pytorch_apis_output_equivalence_cluster(session, openai_client, pytorch_api):
    # 利用clusterer进行聚类
    clusterer_example_1 = """
    {
        "Tensorflow" : {
            "1" : "tf.keras.losses.CategoricalCrossentropy",
            "2" : "tf.nn.softmax_cross_entropy_with_logits"
        },
        "JAX" : {
            "1" : "jax.nn.softmax_cross_entropy"
        }
    }
    """
    clusterer_example_2 = """
    {
        "Tensorflow" : {}, # When there are no TensorFlow APIs whose combined output with Pytorch's API has the same value, output an empty json
        "JAX" : {}
    }
    """
    clusterer_prompt = f"""
    Which APIs in TensorFlow (v2.10) and JAX (v0.4.13) have the same functionality or return values as the {pytorch_api.name} in PyTorch (v2.10)? 
    Note: "The same functionality" means that these APIs are responsible for performing the same tasks, such as PyTorch's torch.scatter_, TensorFlow's tf.scatter_update, and JAX's jax.ops.index_update all have the functionality to update tensors. "The same return values" mean that when the input values are the same or equivalent, the output values of the API are equal or equivalent, such as PyTorch's torch.nn.ReLU, TensorFlow's tf.nn.relu or tf.keras.layers.ReLU, and JAX's jax.nn.relu all have the same output values. 
    Please output the function names in TensorFlow and JAX that meet the above conditions in JSON format, with some examples shown below:
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
            print(clusterer_response_data)
            clusterer_json_data = json.loads(clusterer_response_data)
            print(clusterer_json_data)  # 打印解析后的 JSON 数据
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

   # # 利用Validator对Clusterer返回的数据进行验证
   # validator_prompt = f"""
   # Please check that you answered the previous question correctly. I expect the Pytorch API in the question and the Tensorflow or JAX API given in the response to have the same or equivalent output values when the input values are consistent.
   # If the above answer is not correct, give the correct answer in JSON format.
   # If the answer is correct, return the previous answer in JSON format.
   # """
   # validator_response_data = None
   # validator_json_data = None
   # validator_attempt_num = 0
   # validator_max_attempts_num = 5  # 设置最大尝试次数以避免无限循环
   # while validator_attempt_num < validator_max_attempts_num:
   #     try:  # 假如返回的数据不符合JSON格式, 则重新调用OpenAI API, 直到返回的数据符合JSON格式为止
   #         # 调用 OpenAI API
   #         response = openai_client.chat.completions.create(
   #             model="gpt-3.5-turbo",
   #             response_format={"type": "json_object"},
   #             messages=[
   #                 {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
   #                 {"role": "user", "content": clusterer_prompt},
   #                 {"role": "assistant", "content": clusterer_response_data},
   #                 {"role": "user", "content": validator_prompt},
   #             ],
   #             temperature=0.0,
   #         )
   #         validator_response_data = response.choices[0].message.content
   #         print(validator_response_data)
   #         validator_json_data = json.loads(validator_response_data)
   #         print(validator_json_data)  # 打印解析后的 JSON 数据
   #         break  # 成功解析 JSON,跳出循环
   #     except json.JSONDecodeError as e:
   #         print(f"Failed to decode JSON due to: {e}.\nCurrent attempt: {validator_attempt_num + 1}. Retrying...")
   #         validator_attempt_num += 1
   #     except Exception as e:
   #         session.rollback()  # 回滚在异常中的任何数据库更改
   #         print(f"An unexpected error occurred: {e}")
   #         break
   # if validator_attempt_num == validator_max_attempts_num:
   #     print("Max attempts reached. Unable to get valid JSON data.")
   #     return

    try:
        # 1. 解析返回的 JSON 数据, 如果Tensorflow表或JAX表中没有对应的API Name, 则先在Tensorflow表或JAX表中创建对应的数据
        tensorflow_apis = {}
        for tf_id, tf_name in clusterer_json_data['Tensorflow'].items():
            if tf_name not in tensorflow_apis:
                tf_api = session.query(Tensorflow).filter_by(name=tf_name).first()
                if not tf_api:
                    tf_api = Tensorflow(name=tf_name)
                    session.add(tf_api)
                    session.commit()
                tensorflow_apis[tf_name] = tf_api

        jax_apis = {}
        for jax_key, jax_name in clusterer_json_data['JAX'].items():
            if jax_name not in jax_apis:
                jax_api = session.query(JAX).filter_by(name=jax_name).first()
                if not jax_api:
                    jax_api = JAX(name=jax_name)
                    session.add(jax_api)
                    session.commit()
                jax_apis[jax_name] = jax_api

        # 2. 创建OutputEquivalenceCluster对象, 并将其与Pytorch对象,Tensorflow对象和JAX对象关联
        pytorch_api.is_clustered = True
        if pytorch_api.output_clusters:  # 先检查pytorch_api是否已经出现在一个聚类中
            # 如果该pytorch_api已经在一个输出等价集群中,则检测tensorflow_apis和jax_apis是否已经在该集群中
            cluster = pytorch_api.output_clusters[0]
            for tf_api in tensorflow_apis.values():
                if tf_api not in cluster.tensorflows:
                    cluster.tensorflows.append(tf_api)
            for jax_api in jax_apis.values():
                if jax_api not in cluster.jaxes:
                    cluster.jaxes.append(jax_api)
        else:  # 如果该pytorch_api还没有出现在一个聚类中,则创建一个新的输出等价集群
            cluster = OutputEquivalenceCluster()
            cluster.pytorches.append(pytorch_api)
            cluster.tensorflows.extend(tensorflow_apis.values())
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
