from utils import *
from sqlalchemy.orm import joinedload
from tqdm.contrib import itertools
import os
from orm import *


def generate_seeds(session, openai_client, cluster, seeds_num=5):
    # 在fuzzer/seeds/unverified_seeds/下创建一个新的文件夹, 文件夹名为cluster.id
    if not os.path.exists(f'seeds/unverified_seeds/{cluster.id}'):
        os.makedirs(f'seeds/unverified_seeds/{cluster.id}')

    # 先从cluster中获取所有的Pytorch对象, Tensorflow对象和JAX对象的组合
    # 比如当前有一个cluster, 其中包含了Pytorch对象A1和A2, Tensorflow对象B1和JAX对象C1, 则有以下组合: (A1,B1,C1), (A2,B1,C1)
    # 再比如当前有一个cluster, 其中包含了Pytorch对象A1, Tensorflow对象为null, JAX对象C1和C2, 则有以下组合: (A1,C1), (A1,C2)
    all_combinations = list(itertools.product(
        cluster.pytorches if cluster.pytorches else [None],
        cluster.tensorflows if cluster.tensorflows else [None],
        cluster.jaxes if cluster.jaxes else [None]
    ))

    for combination in all_combinations:  # 对于每种组合都生成5个seed
        # 在seeds/unverified_seeds/{cluster.id}/下创建一个新的文件夹, 文件夹名为combination中Pytorch, Tensorflow和JAX对象的name组合:
        # 比如当前的combination为(A1,B1,C1), 则创建的文件夹名为A1_B1_C1
        seed_folder_name = "_".join([obj.name for obj in combination if obj])
        if not os.path.exists(f'seeds/unverified_seeds/{cluster.id}/{seed_folder_name}'):
            os.makedirs(f'seeds/unverified_seeds/{cluster.id}/{seed_folder_name}')

        output_format_example = """
        {
            Pytorch : {"code" : "code snippet..."},
            TensorFlow : {"code" : "code snippet..."},
            JAX : {"code" : "code snippet..."},
        }
        """
        torch_api = combination[0].api_signature if combination[0] else "Pytorch"
        tf_api = combination[1].api_signature if combination[1] else "TensorFlow"
        jax_api = combination[2].api_signature if combination[2] else "JAX"
        prompt = f"""
        Given the {torch_api} of the Pytorch library, the {tf_api} of the TensorFlow library and the {jax_api} of the JAX library all have the same functionality.

        Generate test code snippets for the apis of the different libraries mentioned above.

        Requirements:
        1. The output values of test code snippets generated for different apis need to remain the same
        2. Your answers should be in JSON format:
        {output_format_example}
        """
        for i in range(seeds_num):  # 生成5个seed
            response_data = None
            json_data = None
            attempt_num = 0
            max_attempts_num = 5  # 设置最大尝试次数以避免无限循环
            while attempt_num < max_attempts_num:
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
                    response_data = response.choices[0].message.content
                    print(response_data)
                    json_data = json.loads(response_data)
                    break  # 成功解析 JSON,跳出循环
                except json.JSONDecodeError as e:
                    print(
                        f"Failed to decode JSON due to: {e}.\nCurrent attempt: {attempt_num + 1}. Retrying...")
                    attempt_num += 1
                except Exception as e:
                    session.rollback()  # 回滚在异常中的任何数据库更改
                    print(f"An unexpected error occurred: {e}")
                    break
            if attempt_num == max_attempts_num:
                print("Max attempts reached. Unable to get valid JSON data.")
                return
            # 在seed_folder_name文件夹下创建一个新的json文件, 文件名为seed_{i}.json, 文件内容为json_data
            with open(f'seeds/unverified_seeds/{cluster.id}/{seed_folder_name}/seed_{i}.json', 'w') as file:
                json.dump(json_data, file, indent=4)

def run():
    session = get_session()
    openai_client = get_openai_client()

    # 获得所有的OutputEquivalenceCluster
    clusters = session.query(Cluster).options(joinedload(Cluster.tensorflows),
                                                               joinedload(Cluster.pytorches),
                                                               joinedload(Cluster.jaxes)).all()

    # 取前20个cluster进行处理
    clusters = clusters[:1]
    for cluster in clusters:
        print("----------------------------------------------------------------------------------")
        generate_seeds(session, openai_client, cluster)


if __name__ == '__main__':
    run()
