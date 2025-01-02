import json
import random
from json import JSONDecodeError
from sqlalchemy import func
from utils import *


# ----------------------------------------------Cluster----------------------------------------------
class ValueEquivalentCluster:
    def __init__(self, api: API, session, llm_client):
        self.api = api
        self.session = session
        self.llm_client = llm_client
        self.messages = self.construct_query_message(api)
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
            # 先检查来源库是否为Pytorch, JAX, MindSpore或Jittor中的任意一个
            lib_name = module_name.split('.')[0]  # 用"."分割module_name, 然后取第一个部分作为库名
            lib = map_module2lib(lib_name)
            if lib == 'Unknown':
                raise Exception(f"{full_api_name} does not belong to Pytorch, JAX, MindSpore or Jittor.")
            module = importlib.import_module(module_name)
            func = getattr(module, api_name, None)
            if func is None:
               self.errors.append(f"{full_api_name} does not exist.")
               return False
            if func is None:
                return False
            if inspect.ismodule(func):
                self.errors.append(f"{full_api_name} is a module, not a function.")
                return False
            # if inspect.isclass(func): # 诸如torch.nn.CrossEntropyLoss等用类封装的API将无法被测试, 因此选择注释掉
            #     self.errors.append(f"{full_api_name} is a class, not a function.")
            #     return False
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
            for dl_lib, api_groups in json_data.items():  # 逐个访问Pytorch, JAX, MindSpore和Jittor下的API二维数组
                for api_group in api_groups:  # 逐个访问Pytorch, Tensorflow和Jax下的各个API组合
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

    def construct_query_message(self, api):  # 构建cluster的初始提词并返回对话消息
        system_prompt = f"""
(1) Role Definition
You are an AI assistant specialized in deep learning framework APIs (e.g., PyTorch, JAX, MindSpore). Your primary task is to help users find equivalent APIs (or equivalent API groups) across different deep learning libraries.
(2) Background Knowledge
Definition of Equivalent API:
- API with a Return Value: When an API has a return value, that API and its equivalent API should always produce the same output results given the same parameter inputs.
- API without a Return Value: If an API does not have a return value, that API and its equivalent API should perform the same in-place operations on the initial input (e.g., a tensor) under the same parameter inputs.
- API that Modifies the Global or Context Environment: For APIs that modify the global or context environment (such as setting a random seed, configuring GPUs, configuring logging, etc.), we assume by default that there is no equivalent API.
Definition of an Equivalent API Group:
- If the functionality of an API can be achieved by calling a group of APIs, then that group of APIs is defined as an “Equivalent API Group.”
(3) Output Format
Provide the answer in JSON format:
- Key: The name of the deep learning library where the target API resides.
- Value: A two-dimensional array, where each element is a one-dimensional array containing one or more fully qualified API names (i.e., module name + API name).
For the target API whose equivalent API (or group) you want to match, the JSON must include a key-value pair for that target API. The key should be the name of the deep learning library where the target API resides, and the value should be the two-dimensional array that encapsulates that API.
"""
        # Example 1
        context_query_prompt1 = f"""
API Information:
- API Name: jittor.nn.CrossEntropyLoss
- Source Library: Jittor
- Version: 1.3.9.10
- API Signature: jittor.nn.CrossEntropyLoss(weight=None, ignore_index=None)
- Function Description: This class is used to compute the cross-entropy loss between the output values and the target values. Cross-entropy loss is a commonly used loss function for classification tasks, especially when dealing with multi-class problems.
- Usage Example:
m = nn.CrossEntropyLoss()
output = jt.array([[1.5, 2.3, 0.7], [1.8, 0.5, 2.2]])
target = jt.array([1, 2])
loss_var = m(output, target) # jt.Var([0.5591628], dtype=float32)

Task:
Search for an API or API group in the following deep learning libraries that is functionally equivalent to jittor.nn.CrossEntropyLoss.
Target Libraries: 
Jittor (v1.3.9.10)
Pytorch (v1.12.0)
JAX (v0.4.13)
MindSpore (v2.4.0)
"""

        context_answer_prompt1 = """
{
    "Jittor": [
        ["jittor.nn.CrossEntropyLoss"],
        ["jittor.nn.cross_entropy_loss"],
    ],
    "Pytorch": [
         ["torch.nn.CrossEntropyLoss"],
         ["torch.nn.functional.cross_entropy"],
    ],
    "JAX": [
        ["jax.nn.log_softmax", "jax.numpy.sum", "jax.numpy.mean"]
    ],
    "MindSpore": [
        ["mindspore.nn.CrossEntropyLoss"],
        ["mindspore.ops.cross_entropy"],
    ]
}
"""
        # Example 2
        context_query_prompt2 = f"""
API Information:
- API Name: mindspore.ops.relu
- Source Library: MindSpore
- Version: 2.4.0
- API Signature: mindspore.ops.relu(input)
- Function Description: Computes the Rectified Linear Unit (ReLU) activation function on each element of the input tensor.
- Usage Example:
import mindspore
import numpy as np
from mindspore import Tensor, ops
input = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
output = ops.relu(input)
print(output) # [[0. 4. 0.], [2. 0. 9.]]

Task:
Search for an API or API group in the following deep learning libraries that is functionally equivalent to mindspore.ops.relu.
Target Libraries: 
MindSpore (v2.4.0)
Pytorch (v1.12.0)
JAX (v0.4.13)
Jittor (v1.3.9.10)
"""
        context_answer_prompt2 = """
{
    "MindSpore": [
        ["mindspore.ops.relu"],
        ["mindspore.nn.ReLU"],
    ],
    "Pytorch": [
        ["torch.nn.ReLU"],
        ["torch.nn.functional.relu"],
    ],
    "JAX": [
        ["jax.nn.relu"],
    ],
    "Jittor": [
        ["jittor.nn.relu"],
    ],
}
"""
        # query
        twin_libs = [
            (lib_name, lib_version)
            for lib_name, lib_version in get_libs_info()
            if not (lib_name.lower() == api.lib.lower() and lib_version == api.version)
        ]  # [('JAX', '0.4.13'), ('MindSpore', '2.4.0')]
        twin_libs_list = "\n".join([f"{lib} (v{ver})" for lib, ver in twin_libs])
        query_prompt = f"""
API Information:
- API Name: {api.full_name}
- Source Library: {api.lib}
- Version: {api.version}
- API Signature: {api.signature}
{'- Function Description: ' + api.description if api.description else ''}
{'- Usage Example:' + api.example if api.example else ''}

Task:
Search for an API or API group in the following deep learning libraries that is functionally equivalent to {api.full_name}.
Target Libraries: 
{api.lib} (v{api.version})
{twin_libs_list}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_query_prompt1},
            {"role": "assistant", "content": context_answer_prompt1},
            {"role": "user", "content": context_query_prompt2},
            {"role": "assistant", "content": context_answer_prompt2},
            # {"role": "user", "content": context_query_prompt3},
            # {"role": "assistant", "content": context_answer_prompt3},
            {"role": "user", "content": query_prompt},
        ]
        return messages

    def conduct_cluster(self):  # 生成并检验JSON数据, 在检验完成或尝试次数达到上限后返回JSON数据或空值
        attempt_num = 0
        while attempt_num < 5:  # 设置最大尝试次数以避免无限循环
            try:  # 假如返回的数据不符合JSON格式, 则重新调用OpenAI API, 直到返回的数据符合JSON格式为止
                response = self.llm_client.chat.completions.create(
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
                    self.messages.append({"role": "user", "content": f"The JSON data you generated has the following errors: \n{self.errors} \n Please try again."})
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
        api_groups = "Tensorflow" : [
            ["tensorflow.keras.losses.CategoricalCrossentropy"],
            ["tensorflow.constant", "tensorflow.nn.softmax_cross_entropy_with_logits"]
        ]
        """
        api_group_objects = []  # [[CategoricalCrossentropy], [constant, softmax_cross_entropy_with_logits]]
        for api_group in api_groups:  # 逐个访问每个API组合
            api_list = []
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
                    self.session.flush()
                api_list.append(api)  # [constant, softmax_cross_entropy_with_logits]
            api_group_objects.append(api_list)
        return api_group_objects  # [['CategoricalCrossentropy'], ['constant', 'softmax_cross_entropy_with_logits']]

    def save_cluster(self, json_data):
        """
            接收并处理clusterer的响应结果, 创建cluster聚类和关联的API组合
        """
        try:
            # 1. 解析返回的JSON数据并检查Pytorch和JAX中的所有API名,如果API表中没有对应的条目,则先在对应表中创建对应的数据
            libs_apis_group_objects = {}  # {"Pytorch" : [[API1],[API2, API3]], "JAX" : [[API1],[API2, API3]], ...}
            for lib, dict_api_groups in json_data.items():
                apis_group_objects = self.supplement_apis(dict_api_groups)
                libs_apis_group_objects[lib] = apis_group_objects

            # 2. 如果libs_apis_group_objects中至少有2个库的apis_group_objects不为空, 那么查找已经存在的Cluster对象或创建Cluster对象:
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
                                          .filter(
                            Cluster.type == 'ValueEquivalent')  # group.cluster.type == 'ValueEquivalent'
                                          .group_by(APIGroup.id)
                                          .having(func.count(API.id) == 1,  # 确保当前Group内只包含一个API
                                                  func.min(API.id) == api.id)  # 确保当前Group内包含的API是api
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
                            type='ValueEquivalent',
                            energy=5,
                        )
                        self.session.add(value_equivalent_cluster)
                        self.session.flush()
                else:
                    # Case 2.2
                    value_equivalent_cluster = Cluster(
                        type='ValueEquivalent',
                        energy=5,
                    )
                    self.session.add(value_equivalent_cluster)
                    self.session.flush()

                # 3. 为每个API组合创建对应的APIgroup对象, 之后将它们与新创建的Cluster对象关联
                for lib, apis_group_objects in libs_apis_group_objects.items():
                    for api_group in apis_group_objects:  # 逐个访问每个API组合
                        group = APIGroup(
                            apis=api_group,
                            cluster=value_equivalent_cluster
                        )
                        self.session.add(group)
                        self.session.flush()
            self.api.is_clustered_by_value = True
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
    llm_client = get_llm_client('gpt4o-mini-with-rag')

    # 对未聚类的PytorchAPI进行聚类
    uncluttered_torch_apis = session.query(API).filter_by(is_clustered_by_value=False).all()
    while uncluttered_torch_apis:
        print("----------------------------------------------------------------------------------")
        # 随机选择一个未聚类的API
        uncluttered_torch_api = random.choice(uncluttered_torch_apis)
        cluster = ValueEquivalentCluster(uncluttered_torch_api, session, llm_client)
        cluster.cluster_api()

        uncluttered_torch_apis = session.query(API).filter_by(is_clustered_by_value=False).all()
        total_apis_num = session.query(API).count()
        unclustered_torch_apis_num = len(uncluttered_torch_apis)
        print(f"Unclustered / Total: {unclustered_torch_apis_num} / {total_apis_num}")


def run_linearly():  # 线性地对未聚类的API进行聚类
    # 创建数据库连接
    session = get_session()
    llm_client = get_llm_client('gpt4o-mini-with-rag')

    # 对未聚类的API进行聚类
    uncluttered_torch_apis = session.query(API).filter_by(is_clustered_by_value=False).all()
    for i, uncluttered_torch_api in enumerate(uncluttered_torch_apis):
        print("----------------------------------------------------------------------------------")
        # 选择一个未聚类的TensorflowAPI
        cluster = ValueEquivalentCluster(uncluttered_torch_api, session, llm_client)
        cluster.cluster_api()
        print(f"Unclustered / Total: {len(uncluttered_torch_apis) - i - 1} / {len(uncluttered_torch_apis)}" + "\n")


if __name__ == '__main__':
    # run_randomly()
    run_linearly()
