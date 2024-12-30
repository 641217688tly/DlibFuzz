import json
import random
from json import JSONDecodeError
from sqlalchemy import func
from utils import *

EXAMPLE1 = """json
{
    "apis" : [
        'torch.nn.functional.max_pool1d', 
        'torch.nn.functional.avg_pool1d', 
        'torch.nn.functional.adaptive_max_pool1d', 
        'torch.nn.functional.lp_pool1d', 
        'torch.nn.functional.adaptive_avg_pool1d',
    ]
}
"""

EXAMPLE2 = """json
{
    "apis" : [
        'jax.nn.relu',
        'jax.nn.leaky_relu',
        'jax.nn.sigmoid',
        'jax.nn.tanh',
        'jax.nn.gelu',
        'jax.nn.softplus',
        'jax.nn.elu',
        'jax.nn.selu',
        'jax.nn.softsign',
        'jax.nn.swish',
    ]
}
"""

EXAMPLE3 = """json
{

}
"""


# ----------------------------------------------Cluster----------------------------------------------
class StateEquivalentCluster:
    def __init__(self, api, session, openai_client):
        self.api = api
        self.session = session
        self.openai_client = openai_client
        self.messages = self.initialize_message(api)
        self.responses = []
        self.errors = []
        self.module_alias_mapper = {
            "tf": "tensorflow",
            "ms": "mindspore",
            "np": "numpy",
            "pd": "pandas",
        }

    def handle_module_alias(self, module_name):
        module_parts = module_name.split('.')
        # 如果模块名的第一个部分是别名, 则将其替换为完整模块名
        if module_parts[0] in self.module_alias_mapper:
            module_parts[0] = self.module_alias_mapper[module_parts[0]]
        # 将替换后的模块名重新组合为完整模块名
        new_module_name = '.'.join(module_parts)
        return new_module_name

    def validate_api(self, full_api_name):
        module_name = ""
        api_name = ""
        try:
            module_name, api_name = full_api_name.rsplit('.', 1)
            module_name = self.handle_module_alias(module_name)
            module = importlib.import_module(module_name)
            func = getattr(module, api_name, None)
            # if func is None or not callable(func):
            #    self.errors.append(f"{full_api_name} is not callable or does not exist.")
            #    return False
            if inspect.ismodule(func):
                self.errors.append(f"{full_api_name} is a module, not a function.")
                return False
            if inspect.isclass(func):
                self.errors.append(f"{full_api_name} is a class, not a function.")
                return False
            # if validate_api_availability(func):
            #    self.errors.append(f"{full_api_name} is deprecated.")
            #    return False
            return True
        except ModuleNotFoundError as e:
            self.errors.append(f"Module {module_name} not found: {str(e)}")
            return False
        except ImportError as e:
            self.errors.append(f"Module {module_name} not found: {str(e)}")
            return False
        except AttributeError:
            self.errors.append(f"{api_name} does not exist in {module_name}.")
            return False
        except Exception as e:
            self.errors.append(str(e))
            return False

    def validate_apis(self, response):
        """
        验证JSON中API的格式是完整函数名(完整函数名 = 模块名.API名)而非函数签名
        所有的API函数名必须有效(有效的定义为: JSON数据中的API为函数全名(函数全名 = 模块.函数名)而非函数签名, 该API不是被弃用的, 该API必须是函数而非模块或类, 该API可以被导入)
        """
        try:
            is_valid = True
            json_data = json.loads(response)
            for dl_lib, api_groups in json_data.items():  # 逐个访问Pytorch, Tensorflow和Jax
                for api_group_id, api_group in api_groups.items():  # 逐个访问Pytorch, Tensorflow和Jax下的各个API组合
                    for full_api_name in api_group:  # 逐个访问API组合下的各个API
                        if not self.validate_api(full_api_name):
                            is_valid = False
            return is_valid
        except JSONDecodeError:
            self.errors.append("The response data has an invalid JSON format.")
            return False
        except Exception as e:
            self.errors.append(str(e))
            return False

    def initialize_message(self, api):  # 构建cluster的初始提词并返回对话消息
        base_lib = api.lib
        base_lib_version = api.version
        twin_libs = []  # [('JAX', '0.4.13'), ('MindSpore', '2.4.0')]
        libs_info = get_libs_info()  # [('Pytorch', '1.12'), ('JAX', '0.4.13'), ('MindSpore', '2.4.0')]
        for lib_name, lib_version in libs_info:
            if lib_name.lower() == base_lib.lower() and lib_version == base_lib_version:
                continue
            if lib_name.lower() == 'mindspore':
                continue  # 使用MindSpore官方提供的API对应关系来完成值等价聚类
            twin_libs.append((lib_name, lib_version))

        # 为twin_libs中的所有库生成一个通用的提示词, 比如[('JAX', '0.4.13')]的提示词为: "JAX (v0.4.13)"; 再比如[('JAX', '0.4.13'), ('Pytorch', '1.12')]的提示词为: "JAX (v0.4.13) and Pytorch (v1.12)"
        twin_libs_prompt = " and ".join([f"{lib} (v{ver})" for lib, ver in twin_libs])
        # 拼接获取所有的twin_libs内库的名称
        twin_libs_name = " and ".join([f"\"{lib}\"" for lib, ver in twin_libs])
        cluster_prompt = f"""
TODO
    """
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": cluster_prompt}
        ]
        return messages

    def conduct_cluster(self):  # 生成并检验JSON数据, 在检验完成或尝试次数达到上限后返回JSON数据或空值
        attempt_num = 0
        while attempt_num < 5:  # 设置最大尝试次数以避免无限循环
            try:  # 假如返回的数据不符合JSON格式, 则重新调用OpenAI API, 直到返回的数据符合JSON格式为止
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # gpt-4o-mini  gpt-3.5-turbo
                    response_format={"type": "json_object"},
                    messages=self.messages,
                    temperature=0,
                )
                response = response.choices[0].message.content
                self.responses.append(response)
                self.messages.append({"role": "assistant", "content": response})
                print(f"Clustered Pytorch API: {self.api.name}\nResponse:\n{response}")
                # 在此处需要检查: 1.响应的数据是否遵循JSON格式; 2.返回的是API的完整函数名(完整函数名 = 模块名.API名)而非函数签名 3.所有的API函数名必须有效(不是虚构的, 也不是被弃用的)
                if self.validate_apis(response):  # 经验证证明返回的数据是有效的
                    self.errors = []  # 清空错误列表
                    return json.loads(response)
                else:
                    attempt_num = attempt_num + 1
                    self.messages.append({"role": "user",
                                          "content": f"The JSON data you generated has the following errors: \n{self.errors} \n Please try again."})
                    print(
                        f"Incorrect JSON format or invalid API.\n Error Details: \n {self.errors} \nRetrying(Current attempt: {attempt_num})...")
                    self.errors = []  # 清空错误列表
            except Exception as e:
                attempt_num = attempt_num + 1
                self.session.rollback()  # 回滚在异常中的任何数据库更改
                print(f"An unexpected error occurred: {e}")
        self.errors = []  # 清空错误列表
        print("Max attempts reached. Unable to get valid JSON data.")
        return None

    # --------------------------------------save API groups and Cluster into database--------------------------------------
    def supplement_apis(self, api_groups):  # 将JAX/Pytorch的API组合内不在数据库中的API添加到数据库中
        """
        以下列数据为例:
        api_groups = "Tensorflow" : {
            "1" : ["tensorflow.keras.losses.CategoricalCrossentropy"],
            "2" : ["tensorflow.constant", "tensorflow.nn.softmax_cross_entropy_with_logits"]
        }
        api_class = Tensorflow
        """
        api_group_objects = {}
        for api_count, api_group in api_groups.items():  # 逐个访问每个API组合
            api_group_objects[api_count] = []  # api_group_objects = { "1" : [], "2" : [] }
            for full_api_name in api_group:  # 获取某个API组合中的每个API
                api = self.session.query(API).filter_by(full_name=full_api_name).first()
                if not api:
                    module_name, api_name = full_api_name.rsplit('.', 1)
                    api_info = inspect_api_info(module_name, api_name)
                    api = API(
                        name=api_name,
                        module=module_name,
                        full_name=full_api_name,
                        lib=api_info['lib'],
                        description=api_info['description'],
                        signature=api_info['signature'],
                        version=api_info['version'],
                    )
                    self.session.add(api)
                    self.session.commit()
                api_group_objects[api_count].append(
                    api)  # { "1" : [CategoricalCrossentropy], "2" : [constant, softmax_cross_entropy_with_logits] }
        return list(
            api_group_objects.values())  # [['CategoricalCrossentropy'], ['constant', 'softmax_cross_entropy_with_logits']]

    def save_cluster(self, json_data):
        """
            接收并处理clusterer的响应结果, 创建cluster聚类和关联的API组合
        """
        try:
            self.api.is_clustered = True
            # 1. 解析返回的JSON数据并检查Pytorch和JAX中的所有API名,如果API表中没有对应的条目,则先在对应表中创建对应的数据
            libs_apis_group_objects = {}  # {"Pytorch" : [[API1],[API2, API3]], "JAX" : [[API1],[API2, API3]], ...}
            for lib, dict_api_groups in json_data.items():
                apis_group_objects = self.supplement_apis(dict_api_groups)
                libs_apis_group_objects[lib] = apis_group_objects

            # 2. 如果libs_apis_group_objects中至少有2个库的apis_group_objects不为空, 那么创建Cluster对象:
            if len([lib for lib, apis_group_objects in libs_apis_group_objects.items() if
                    apis_group_objects]) >= 2:
                # 从{"Pytorch": [["API1"], ["API2", "API3"]], "JAX": [["API4"], ["API5", "API6"]], ...}中获取由单独的API组成的API组合:[["API1"], ["API4"]]
                single_api_groups = [sublist for dictionary in libs_apis_group_objects.values() for sublist
                                           in dictionary if len(sublist) == 1]
                if len(single_api_groups) > 0:
                    cluster_dict = {}
                    for single_api_group in single_api_groups:
                        api = single_api_group[0]
                        api_obj_groups = (self.session.query(APIGroup)
                                                .join(APIGroup.apis)
                                                .filter(Cluster.type == 'StateEquivalent')
                                                .group_by(APIGroup.id)
                                                .having(func.count(API.id) == 1,  # 确保每个组合只有一个API
                                                        func.min(API.id) == api.id)
                                                .all())
                        for api_obj_group in api_obj_groups:
                            cluster = api_obj_group.cluster
                            cluster_dict[cluster] = cluster_dict.get(cluster, 0) + 1
                    if cluster_dict:
                        # Case 2.1
                        value_equivalent_cluster = max(cluster_dict, key=cluster_dict.get)
                    else:
                        # Case 2.2
                        value_equivalent_cluster = Cluster(
                            type='StateEquivalent',
                            energy=5,
                        )
                        self.session.add(value_equivalent_cluster)
                        self.session.commit()
                else:
                    # Case 2.2
                    value_equivalent_cluster = Cluster(
                        type='StateEquivalent',
                        energy=5,
                    )
                    self.session.add(value_equivalent_cluster)
                    self.session.commit()

                # 3. 为每个API组合创建对应的APIgroup对象, 之后将它们与新创建的Cluster对象关联
                for lib, apis_group_objects in libs_apis_group_objects.items():
                    for api_group in apis_group_objects:  # 逐个访问每个API组合
                        group = APIGroup(
                            apis=api_group,
                            cluster=value_equivalent_cluster
                        )
                        self.session.add(group)
                        self.session.commit()
            self.session.commit()
        except Exception as e:
            self.session.rollback()  # 回滚在异常中的任何数据库更改
            print(f"An error occurred: {e}")

    # ----------------------------------------------run()----------------------------------------------
    def cluster_api(self):
        json_data = self.conduct_cluster()
        new_cluster = None
        if json_data:
            new_cluster = self.save_cluster(json_data)
        return new_cluster


def run_randomly():  # 随机挑选未聚类的API进行聚类
    # 创建数据库连接
    session = get_session()
    openai_client = get_llm_client()

    # 对未聚类的PytorchAPI进行聚类
    uncluttered_torch_apis = session.query(API).filter_by(is_clustered=False).all()
    while uncluttered_torch_apis:
        print("----------------------------------------------------------------------------------")
        # 随机选择一个未聚类的API
        uncluttered_torch_api = random.choice(uncluttered_torch_apis)
        cluster = StateEquivalentCluster(uncluttered_torch_api, session, openai_client)
        cluster.cluster_api()

        uncluttered_torch_apis = session.query(API).filter_by(is_clustered=False).all()
        total_apis_num = session.query(API).count()
        unclustered_torch_apis_num = len(uncluttered_torch_apis)
        print(f"Unclustered / Total: {unclustered_torch_apis_num} / {total_apis_num}")


def run_linearly():  # 线性地对未聚类的API进行聚类
    # 创建数据库连接
    session = get_session()
    openai_client = get_llm_client()

    # 对未聚类的API进行聚类
    uncluttered_torch_apis = session.query(API).filter_by(is_clustered=False).all()
    for i, uncluttered_torch_api in enumerate(uncluttered_torch_apis):
        print("----------------------------------------------------------------------------------")
        # 选择一个未聚类的TensorflowAPI
        cluster = StateEquivalentCluster(uncluttered_torch_api, session, openai_client)
        cluster.cluster_api()
        print(f"Unclustered / Total: {len(uncluttered_torch_apis) - i - 1} / {len(uncluttered_torch_apis)}" + "\n")


if __name__ == '__main__':
    # run_randomly()
    run_linearly()
