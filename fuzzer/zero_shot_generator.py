from utils import *
from tqdm.contrib import itertools
from orm import *
import os


class SeedGenerator:
    def __init__(self, session, openai_client, output_folder="seeds/unverified_seeds/zero-shot"):
        self.session = session
        self.openai_client = openai_client
        self.output_folder = output_folder

    def generate_seeds(self, cluster: Cluster):
        # 在folder_path下创建一个新的文件夹, 文件夹名为cluster.id, 后续该Cluster的所有种子都会放在这个文件夹下
        cluster_folder_path = f'{self.output_folder}/{cluster.id}'
        if not os.path.exists(cluster_folder_path):
            os.makedirs(cluster_folder_path)

        # 先从cluster中获取所有的PytorchAPI对象, TensorflowAPI对象和JAXAPI对象的组合.
        # 比如当前有一个cluster, 其中包含了PytorchAPI的对象A1和A2, TensorflowAPI对象B1和JAXAPI对象C1, 则有以下组合: (A1,B1,C1), (A2,B1,C1);
        # 再比如当前有一个cluster, 其中包含了PytorchAPI对象A1, TensorflowAPI对象为null, JAXAPI对象C1和C2, 则有以下组合: (A1,C1), (A1,C2)
        pytorch_apis = cluster.pytorch_apis if cluster.pytorch_apis else [None]
        tensorflow_apis = cluster.tensorflow_apis if cluster.tensorflow_apis else [None]
        jax_apis = cluster.jax_apis if cluster.jax_apis else [None]
        combinations = list(itertools.product(pytorch_apis, tensorflow_apis, jax_apis))  # 生成所有可能的 API 组合

        # 对每种组合都生成(cluster.energy - 已生成的种子个数)个种子
        for combination in combinations:
            pytorch_api = combination[0] if combination[0] else None
            tensorflow_api = combination[1] if combination[1] else None
            jax_api = combination[2] if combination[2] else None
            # 先查询该组合已经生成了几个种子
            seeds_num = self.session.query(ClusterTestSeed).filter_by(cluster_id=cluster.id,
                                                                      pytorch_api_id=pytorch_api.id,
                                                                      tensorflow_api_id=tensorflow_api.id,
                                                                      jax_api_id=jax_api.id).count()
            remaining_energy = cluster.energy - seeds_num
            if remaining_energy <= 0:
                continue

            # 在cluster_folder_path下创建一个新的文件夹, 文件夹名为combination中PytorchAPI, TensorflowAPI和JAXAPI对象的id和api_name的组合:
            combination_folder_name = "_".join(
                [f"{api.__class__.__name__}({api.api_name})" for api in combination if api])
            combination_folder_path = f'{cluster_folder_path}/{combination_folder_name}'
            if not os.path.exists(combination_folder_path):
                os.makedirs(combination_folder_path)

            # 为当前组合生成remaining_energy个种子
            for i in range(cluster.energy - remaining_energy + 1,
                           cluster.energy + 1):  # 生成remaining_energy个种子, 种子的序号为[1, energy]
                seed_folder_path = f'{combination_folder_path}/{i}'
                if not os.path.exists(seed_folder_path):
                    os.makedirs(seed_folder_path)
                    try:
                        self.generate_seed(cluster, combination, seed_folder_path)
                    except Exception as e:
                        print(f"Error in generating seed for cluster {cluster.id}: {e}")
                        self.session.rollback()
                        break

    def generate_seed(self, cluster: Cluster, combination, seed_folder_path):
        pytorch_api = combination[0] if combination[0] else None
        tensorflow_api = combination[1] if combination[1] else None
        jax_api = combination[2] if combination[2] else None
        base_api = pytorch_api if cluster.base == 'Pytorch' else tensorflow_api if cluster.base == 'Tensorflow' else jax_api
        seed = ClusterTestSeed(
            cluster_id=cluster.id,
            pytorch_api_id=pytorch_api.id if pytorch_api else None,
            tensorflow_api_id=tensorflow_api.id if tensorflow_api else None,
            jax_api_id=jax_api.id if jax_api else None,
            raw_folder_path=seed_folder_path
        )
        self.session.add(seed)

        # 先为基底API生成测试用例
        base_seed_code = self.generate_seed4base(seed, base_api)

        # 随后对基底API进行修复
        # TODO

        # 然后参考基底API的测试用例生成其他库中的孪生API的测试用例
        # TODO

        # 最后对孪生API进行修复
        # TODO

    def generate_seed4base(self, seed: ClusterTestSeed, base_api):  # 生成基底API的测试用例
        prompt = f"""
Generate test cases for the {base_api.full_name} in {base_api.__class__.__name__.replace("API", "")} (ver{base_api.version})     
"""
        base_seed_code = self.query_openai(prompt)
        if base_seed_code is None:
            print(f"During generate seed for base API {base_api.full_name}, some error occurred.")
            return None
        # 为seed对象赋值
        if isinstance(base_api, PytorchAPI):
            seed.raw_pytorch_code = base_seed_code
        elif isinstance(base_api, TensorflowAPI):
            seed.raw_tensorflow_code = base_seed_code
        else:
            seed.raw_jax_code = base_seed_code
        self.session.commit()
        return base_seed_code

    def generate_seed4twin(self, twin_api, base_seed, seed_folder_path):  # 生成孪生API的测试用例

        pass
        return None

    def query_openai(self, prompt, retry_num=5, model="gpt-4o-mini"):
        attempt_num = 0
        while attempt_num < retry_num:  # 设置最大尝试次数以避免无限循环
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,  # gpt-4o-mini  gpt-3.5-turbo
                    messages=[
                        {"role": "system",
                         "content": "You're an AI assistant adept at using multiple deep learning libraries"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1,
                )
                response_data = response.choices[0].message.content
                print(response_data)
                return response_data
            except Exception as e:
                print(f"Failed to get response due to: \n{e} \nRetrying(Current attempt: {attempt_num + 1})...")
                attempt_num += 1
                self.session.rollback()  # 回滚在异常中的任何数据库更改
        if attempt_num >= 5:  # 设置最大尝试次数以避免无限循环
            print("Max attempts reached. Unable to get valid JSON data.")
            return None
