from tqdm.contrib import itertools
from validator import SeedValidator
from utils import *


class SeedGenerator:
    def __init__(self, session, openai_client):
        self.session = session
        self.openai_client = openai_client

    def generate_seeds4cluster(self, cluster: Cluster):
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
            print(
                "+" * 40 + f"Generate Seeds for Combination(Pytorch[{pytorch_api.name if pytorch_api else 'Null'}]-Tensorflow[{tensorflow_api.name if tensorflow_api else 'Null'}]-JAX[{jax_api.name if jax_api else 'Null'}]) " + "+" * 40)
            # 先查询该组合已经生成了几个种子
            seeds_num = self.session.query(ClusterTestSeed).filter_by(
                cluster_id=cluster.id,
                pytorch_api_id=pytorch_api.id if pytorch_api else None,
                tensorflow_api_id=tensorflow_api.id if tensorflow_api else None,
                jax_api_id=jax_api.id if jax_api else None
            ).count()

            remaining_energy = cluster.energy - seeds_num
            if remaining_energy <= 0:
                continue

            # 为当前组合生成remaining_energy个种子, 种子的序号为[1, energy]
            for i in range(cluster.energy - remaining_energy + 1, cluster.energy + 1):
                try:
                    print("*" * 30 + f"Generate Seed{i}" + "*" * 30)
                    base_api_seed, twin_apis_seeds = self.generate_seed4combination(cluster, combination)
                    print("-" * 20 + f"Base API Seed" + "-" * 20 + f"\n{base_api_seed}")
                    for twin_apis_seed in twin_apis_seeds:
                        print("-" * 20 + f"Twin API Seed" + "-" * 20 + f"\n{twin_apis_seed}")
                except Exception as e:
                    print(f"Error in generating seed for cluster {cluster.id}: {e}")
                    self.session.rollback()
                    break

        # 检查是否所有的种子都已经生成完毕
        seeds_num = self.session.query(ClusterTestSeed).filter_by(cluster_id=cluster.id).count()
        if seeds_num >= (cluster.energy * len(combinations)):
            cluster.is_tested = True
            self.session.commit()

    def generate_seed4combination(self, cluster: Cluster, combination):
        pytorch_api = combination[0] if combination[0] else None
        tensorflow_api = combination[1] if combination[1] else None
        jax_api = combination[2] if combination[2] else None
        seed = ClusterTestSeed(
            cluster_id=cluster.id,
            pytorch_api_id=pytorch_api.id if pytorch_api else None,
            tensorflow_api_id=tensorflow_api.id if tensorflow_api else None,
            jax_api_id=jax_api.id if jax_api else None,
        )
        self.session.add(seed)
        self.session.commit()

        # 1.先为基底API生成测试用例
        base_api = pytorch_api if cluster.base == 'Pytorch' else tensorflow_api if cluster.base == 'Tensorflow' else jax_api
        base_seed = self.generate_seed4base(seed, base_api)
        # 2.随后尝试对基底API进行修复
        base_seed_validator = SeedValidator(self.session, self.openai_client, seed, base_seed, cluster.base)
        validated_base_seed = base_seed_validator.validate()
        if validated_base_seed is None:
            # 如果修复失败, 依旧使用修复前的代码
            validated_base_seed = base_seed
        # 3.参考基底API的测试用例生成其他库中的孪生API的测试用例
        twin_apis = [api for api in combination if api != base_api]
        twin_apis_seeds = []
        for twin_api in twin_apis:
            if twin_api is None:
                continue
            twin_api_seed = self.generate_seed4twin(seed, twin_api, base_api, validated_base_seed)
            twin_apis_seeds.append(twin_api_seed)
        return validated_base_seed, twin_apis_seeds

    def generate_seed4base(self, seed: ClusterTestSeed, base_api):  # 生成基底API的测试用例
        prompt = f"""
Tasks:
1.Import Required Modules
2.Call {base_api.signature} in {base_api.__class__.__name__.replace("API", "")} (ver{base_api.version} to perform the necessary computations or actions.
3.Generate input data that is likely to trigger an edge case or boundary condition (such as high values, nulls, extreme dimensions...) and pass it to the API function.
4.4.If the ({base_api.signature}) has a return value, print its output. If it does not have a return value, print the value of the variables affected by ({base_api.signature}).

Requirements:
1.Imports: Ensure that all necessary modules or APIs are imported.
2.Code-Only Format: Only output code and comments in the required format, avoiding any additional text or Markdown syntax.
3.Correctness: Ensure the generated code does not contain syntax errors (e.g., SyntaxError, NameError) or invalid input errors (e.g., ValueError, InvalidArgumentError).
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

    def generate_seed4twin(self, seed: ClusterTestSeed, twin_api, base_api, base_seed):  # 生成孪生API的测试用例
        prompt = f"""
Task:
It is known that the API ({twin_api.signature}) in {twin_api.__class__.__name__.replace("API", "")} (ver{twin_api.version}) has the same functionality as the API ({base_api.signature}) in {base_api.__class__.__name__.replace("API", "")} (ver{base_api.version}). Please imitate the logic of the usage of ({base_api.signature}) in {base_api.__class__.__name__.replace("API", "")} (ver{base_api.version}) shown in the code example below and generate an equivalent code snippet using the API ({twin_api.signature}) in {twin_api.__class__.__name__.replace("API", "")} (ver{twin_api.version}).
```python
{base_seed}
```

Requirements:
1.Imports: Ensure that all necessary modules or APIs are imported.
2.Consistency in Input: The input parameters for the API ({twin_api.signature}) in {twin_api.__class__.__name__.replace("API", "")} (ver{twin_api.version}) in your generated code should be the same as the input parameters for the API ({base_api.signature}) in {base_api.__class__.__name__.replace("API", "")} (ver{base_api.version}) in the sample code.
3.Consistency in Output: The example code prints the return value or the affected variables from the call to ({base_api.signature}). Ensure that your generated code also prints the return value or the affected variables from the call to ({twin_api.signature}), and that this output is consistent with the output of the sample code. This requires that your invocation of ({twin_api.signature}) is consistent with the invocation of ({base_api.signature}).
4.Code-Only Format: Only output code and comments in the required format, avoiding any additional text or Markdown syntax.
5.Correctness: Ensure the generated code does not contain syntax errors (e.g., SyntaxError, NameError) or invalid input errors (e.g., ValueError, InvalidArgumentError).
"""
        twin_seed_code = self.query_openai(prompt)
        if twin_seed_code is None:
            print(f"During generate seed for base API {twin_api.full_name}, some error occurred.")
            return None
            # 为seed对象赋值
        if isinstance(twin_api, PytorchAPI):
            seed.raw_pytorch_code = twin_seed_code
        elif isinstance(twin_api, TensorflowAPI):
            seed.raw_tensorflow_code = twin_seed_code
        else:
            seed.raw_jax_code = twin_seed_code
        self.session.commit()
        return twin_seed_code

    def query_openai(self, prompt, max_retry_limit=5, model="gpt-4o-mini"):
        attempt_num = 0
        while attempt_num < max_retry_limit:  # 设置最大尝试次数以避免无限循环
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
                return response_data
            except Exception as e:
                print(f"Failed to get response due to: \n{e} \nRetrying(Current attempt: {attempt_num + 1})...")
                attempt_num += 1
                self.session.rollback()  # 回滚在异常中的任何数据库更改
        if attempt_num >= 5:  # 设置最大尝试次数以避免无限循环
            print("Max attempts reached. Unable to get valid JSON data.")
            return None


def run():
    session = get_session()
    openai_client = get_openai_client()

    # 获得所有的cluster未测试的cluster
    untested_clusters = session.query(Cluster).filter(Cluster.is_tested == False).all()
    while untested_clusters:
        print("=" * 80 + f"Generate Seeds for Cluster({untested_clusters[0].id})" + "=" * 80)
        SeedGenerator(session, openai_client).generate_seeds4cluster(untested_clusters[0])

        untested_clusters = session.query(Cluster).filter(Cluster.is_tested == False).all()
        total_clusters_num = session.query(Cluster).count()
        print(f"Untested / Total: {len(untested_clusters)} / {total_clusters_num}")


def run_linearly():
    # 创建数据库连接
    session = get_session()
    openai_client = get_openai_client()

    # 对未聚类的TensorflowAPI进行聚类
    untested_clusters = session.query(Cluster).filter(Cluster.is_tested == False).all()
    for i, untested_cluster in enumerate(untested_clusters):
        print("=" * 80 + f"Generates Seed for Cluster({untested_cluster.id})" + "=" * 80)
        SeedGenerator(session, openai_client).generate_seeds4cluster(untested_clusters[0])
        print(f"Untested / Total: {len(untested_clusters) - i - 1} / {len(untested_clusters)}" + "\n")


def clear_cache():
    session = get_session()
    session.query(ClusterTestSeed).delete()
    session.commit()
    # 将所有cluster的is_tested字段设置为False
    clusters = session.query(Cluster).all()
    for cluster in clusters:
        cluster.is_tested = False
    # 将所有的ClusterTestSeed中的数据都删除
    session.query(ClusterTestSeed).delete()
    session.commit()


if __name__ == '__main__':
    run()
