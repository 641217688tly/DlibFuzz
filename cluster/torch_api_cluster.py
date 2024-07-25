import json
import httpx
from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, joinedload
import yaml
from orm import Cluster, TensorflowAPI, PytorchAPI, JaxAPI, PytorchAPICombination, TensorflowAPICombination, \
    JaxAPICombination


def process_apis(session, api_data, api_class):
    api_objects = {}
    for api_id, api_names in api_data.items():
        for api_name in api_names:
            api_obj = session.query(api_class).filter_by(name=api_name).first()
            if not api_obj:
                api_obj = api_class(name=api_name)
                session.add(api_obj)
                session.commit()
            api_objects[api_id] = api_obj
    return api_objects

def associate_api_combinations_to_cluster(session, cluster, api_objects, combination_class):
    for api_id, api_obj in api_objects.items():
        combination = combination_class(apis=[api_obj], cluster=[cluster])
        session.add(combination)
        session.commit()

def pytorch_apis_cluster(session, openai_client, pytorch_api):
    # 利用clusterer进行聚类
    clusterer_example_1 = """json
    {   
        "Pytorch" : {
            "1" : ["torch.tensor", "torch.nn.CrossEntropyLoss()"],
        },
        "Tensorflow" : {
            "1" : ["tf.keras.losses.CategoricalCrossentropy"], // tf.keras.losses.CategoricalCrossentropy internal will automatically array into Tensorflow tensor, so there is no need to be used with tf.constant
            "2" : ["tf.constant","tf.nn.softmax_cross_entropy_with_logits"] // Before using tf.nn.softmax_cross_entropy_with_logits, it needs to use tf.constant to convert the input value into a tensor
        },
        "JAX" : {
            "1" : ["jnp.array","jax.nn.log_softmax","jnp.sum"] // Before using jax.nn.softmax_cross_entropy, it needs to use jnp.array to convert the input value into a tensor. After using jax.nn.softmax_cross_entropy, it needs to use jnp.sum to calculate the sum of the cross entropy loss
        }
    }
    """
    clusterer_example_2 = """json
    {   
        // When there are no TensorFlow APIs whose combined output with Pytorch's API has the same value, output an empty json
        "Pytorch" : {}, 
        "Tensorflow" : {}, 
        "JAX" : {}
    }
    """
    clusterer_prompt = f"""
    Which apis or combinations of api calls in TensorFlow (v2.10) and JAX (v0.4.13) have the exact same functionality as {pytorch_api.name} in PyTorch (v2.10)?
    Note: "The same functionality" means that these APIs are responsible for performing exactly the same tasks. When these APIs have no return value, using these APIs to perform the same operations on inputs with the same structure or element values (such as tensors) should result in consistent changes to the original input. For example, PyTorch's torch.scatter_, TensorFlow's tf.scatter_update, and JAX's jax.ops.index_update all have the functionality to update tensors, and when the tensors being updated and the update strategies are the same, the updated tensors should be consistent. When these APIs have return values, PyTorch's torch.nn.ReLU, TensorFlow's tf.nn.relu or tf.keras.layers.ReLU, and JAX's jax.nn.relu all produce the same output values when given the same input values.
    Please output the function names or combinations of function names in PyTorch, TensorFlow, and JAX that meet the above conditions in JSON format, with an example shown below:
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
            print(f"Clustered Pytorch API: {pytorch_api.name}")
            print(clusterer_response_data)
            clusterer_json_data = json.loads(clusterer_response_data)
            #TODO 在此处需要检查返回的是API的函数名而非函数签名, 同时必须检查所有的API函数名是否有效(不是虚构的)
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
        pytorch_api.is_clustered = True
        #1. 解析返回的JSON数据并检查Pytorch,Tensorflow和Jax中的所有API名,如果PytorchAPI表或TensorflowAPI表或JAX表中没有对应的API,则先在对应表中创建对应的数据
        pytorch_apis = process_apis(session, clusterer_json_data['Pytorch'], PytorchAPI)
        tensorflow_apis = process_apis(session, clusterer_json_data['Tensorflow'], TensorflowAPI)
        jax_apis = process_apis(session, clusterer_json_data['JAX'], JaxAPI)

        #2. 创建Cluster对象
        new_cluster = Cluster()
        session.add(new_cluster)
        session.commit()

        #3. 为Pytorch, Tensorflow和Jax的每个API组合创建对应的PytorchAPICombination, TensorflowAPICombination和JaxAPICombination对象, 之后将它们与新创建的Cluster对象关联
        associate_api_combinations_to_cluster(session, new_cluster, pytorch_apis, PytorchAPICombination)
        associate_api_combinations_to_cluster(session, new_cluster, tensorflow_apis, TensorflowAPICombination)
        associate_api_combinations_to_cluster(session, new_cluster, jax_apis, JaxAPICombination)
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

    # openai_client = OpenAI(base_url="https://api.gptsapi.net/v1/",
    #                        api_key="sk-4Yg7f4b436b8fb189fc0f426d378e395adf93f7ba45pT6Os")  # WildCard API + 转发, 无需代理

    uncluttered_torch_apis = session.query(PytorchAPI).filter_by(is_clustered=False).all()
    while uncluttered_torch_apis:
        print("----------------------------------------------------------------------------------")
        pytorch_apis_cluster(session, openai_client, uncluttered_torch_apis[0])
        uncluttered_torch_apis = session.query(PytorchAPI).filter_by(is_clustered=False).all()
        total_apis_num = session.query(PytorchAPI).count()
        unclustered_torch_apis_num = len(uncluttered_torch_apis)
        print(f"Unclustered / Total: {unclustered_torch_apis_num} / {total_apis_num}")


if __name__ == '__main__':
    run()
