import json
import random
import traceback
from json import JSONDecodeError
from sqlalchemy import func
from utils import *
from fuzzer import validator


# ----------------------------------------------Cluster----------------------------------------------
class EquivalentCluster:
    def __init__(self, api: API, session, llm_client):
        self.api = api
        self.session = session
        self.llm_client = llm_client
        self.error_log = []
        self.module_alias_mapper = {
            "tf": "tensorflow",
            "ms": "mindspore",
            "np": "numpy",
            "pd": "pandas",
            "jt": "jittor"
        }

    def construct_cluster_messages(self, api):  # 构建cluster的初始提词并返回对话消息
        system_prompt = f"""
(1) Role Definition
You are an AI assistant specialized in deep learning framework APIs (e.g., PyTorch, JAX, MindSpore and Jittor). Your primary task is to help users find equivalent APIs (or API groups) across different deep learning libraries.
(2) Background Knowledge
Definition of Equivalent API:
- Value Equivalent: Given the same parameter inputs, when an API has a return value, calling this API and its value-equivalent API should always produce identical output results. If an API does not have a return value, then calling this API and its value-equivalent API should perform the same in-place operations on the initial input (e.g., tensors).
- State Equivalent: Given the same parameter inputs, calling an API and its state-equivalent API may produce different return values or perform different in-place operations on the initial input. However, their runtime states after being called should be identical (for example, both should execute normally or both should encounter a crash).
- Non-Equivalent: For APIs that modify the global or contextual environment (such as setting random seeds, configuring GPUs, configuring logs, etc.), we define that these APIs do not have equivalent APIs.
Definition of Equivalent API Groups:
- If the functionality of an API can be achieved by calling a group of APIs, then this group of APIs is defined as an "Equivalent API Group".
(3) Output Format
Provide the answer in JSON format:
- Key: The name of the deep learning library where the target API resides.
- Value: A two-dimensional array, where each element is a one-dimensional array containing one or more fully qualified API names (i.e., module name + API name).
Additionally, the JSON must include at least the target API for which equivalent APIs (or API groups) are being sought.
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
>>> m = nn.CrossEntropyLoss()
>>> output = jt.array([[1.5, 2.3, 0.7], [1.8, 0.5, 2.2]])
>>> target = jt.array([1, 2])
>>> loss_var = m(output, target)
>>> loss_var
jt.Var([0.5591628], dtype=float32)

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
- Source Library: {api.lib} (version{api.version})
- API Signature: {api.signature}
{'- Function Description: ' + api.description if api.description else ''}
{'- Usage Example:' + api.example if api.example else ''}

Task:
Search for the APIs or API groups in the following deep learning libraries that is functionally equivalent or similar to {api.full_name}.
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
            {"role": "user", "content": query_prompt},
        ]
        return messages

    def query_llm4cluster(self, messages, max_try=5):  # 生成并检验JSON数据, 在检验完成或尝试次数达到上限后返回JSON数据或空值
        attempt_num = 0
        while attempt_num < max_try:  # 设置最大尝试次数以避免无限循环
            try:  # 假如返回的数据不符合JSON格式, 则重新调用OpenAI API, 直到返回的数据符合JSON格式为止
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",  # gpt-4o-mini  gpt-3.5-turbo
                    response_format={"type": "json_object"},
                    messages=messages,
                    temperature=0.4,
                )
                response = response.choices[0].message.content
                messages.append({"role": "assistant", "content": response})
                print(f"Clustered API: {self.api.full_name}\nResponse:\n{response}")
                # 在此处需要检查: 1.响应的数据是否遵循JSON格式; 2.返回的是API的完整函数名(完整函数名 = 模块名.API名)而非函数签名 3.所有的API函数名必须有效(不是虚构的, 也不是被弃用的)
                if self.validate_apis(response):  # 经验证证明返回的数据是有效的
                    self.error_log = []  # 清空错误列表
                    return json.loads(response)
                else:
                    attempt_num = attempt_num + 1
                    messages.append({"role": "user",
                                     "content": f"The JSON data you generated has the following errors: \n{self.error_log} \n Please try again."})
                    print(
                        f"Incorrect JSON format or invalid API.\n Error Details: \n {self.error_log} \nRetrying(Current attempt: {attempt_num})...")
                    self.error_log = []  # 清空错误列表
            except Exception as e:
                attempt_num = attempt_num + 1
                self.session.rollback()  # 回滚在异常中的任何数据库更改
                print(f"An unexpected error occurred: {e}")
        self.error_log = []  # 清空错误列表
        print("Max attempts reached. Unable to get valid JSON data.")
        return None

    def construct_verify_messages(self, base_api, twin_api_group): # twin_api_group = [API1, API2, ...]
        # twin_api_group 的详情
        if len(twin_api_group) == 1:
            twin_api = twin_api_group[0]
            api_group_info_prompt = f"""
- API Name: {twin_api.full_name}
- API Library: {twin_api.lib} (version{twin_api.version})
- API Signature: {twin_api.signature}
{'- Function Description: ' + twin_api.description if twin_api.description else ''}
{'- Usage Example:' + twin_api.example if twin_api.example else ''}
"""
        else:
            api_group_info_prompt = ""
            for count, twin_api in enumerate(twin_api_group):
                api_group_info_prompt = api_group_info_prompt + f"""
Member{count + 1} of API Group:
- API Name: {twin_api.full_name}
- API Signature: {twin_api.signature}
{'- Function Description: ' + twin_api.description if twin_api.description else ''}
{'- Usage Example:' + twin_api.example if twin_api.example else ''}
"""

        # twin_api_group brief info
        if len(twin_api_group) == 1:
            api_group_brief_info = f"{twin_api_group[0].signature}"
        else:
            api_group_brief_info = f"({', '.join([api.signature for api in twin_api_group])})"

        # 背景知识
        if len(twin_api_group) == 1:
            twin_api = twin_api_group[0]
            background_knowledge_prompt = f"""
The API ({twin_api.signature}) from library {twin_api.lib}(v{twin_api.version}) has the similar function as the API ({base_api.signature}) from library {base_api.lib}(v{base_api.version}).
The detail of API ({base_api.signature}) is as follows:
- API Name: {base_api.full_name}
- Source Library: {base_api.lib} (version{base_api.version})
- API Signature: {base_api.signature}
{'- Function Description: ' + base_api.description if base_api.description else ''}
"""
        else:
            background_knowledge_prompt = f"""
By combining the APIs in {api_group_brief_info}, it can achieve the similar functionality as the API {base_api.signature} from library {base_api.lib}(v{base_api.version}).
The detail of API ({base_api.signature}) is as follows:
- API Name: {base_api.full_name}
- Source Library: {base_api.lib} (version{base_api.version})
- API Signature: {base_api.signature}
{'- Function Description: ' + base_api.description if base_api.description else ''}
"""

        # 构建最终提示词
        prompt = f"""
Information of the API {'group' if len(twin_api_group) > 1 else ''} to be called:
{api_group_info_prompt}        

Background knowledge:
{background_knowledge_prompt}

Task requirements:
Below is a code snippet calling ({base_api.signature}). Please generate a code snippet that replaces ({base_api.full_name}) with {api_group_brief_info}, ensuring that the input parameters remain unchanged. At the end of the code, print the return value of the ({base_api.full_name}) or the variable modified by the ({base_api.full_name}) in-place operations.
{base_api.example}
"""

        messages = [
            {"role": "system",
             "content": "You are an AI assistant specialized in deep learning framework APIs (e.g., PyTorch, JAX, MindSpore and Jittor)."},
            {"role": "user", "content": prompt},
        ]
        return messages

    def query_llm4verify(self, messages, max_retry=5):
        attempt_num = 0
        llm_client = get_llm_client('gpt4o-mini')
        while attempt_num < max_retry:  # 设置最大尝试次数以避免无限循环
            try:
                response = llm_client.chat.completions.create(
                    model="gpt-4o-mini",  # gpt-4o-mini  gpt-3.5-turbo
                    messages=messages,
                    temperature=0.4,
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

    # --------------------------------------save API groups and Cluster into database--------------------------------------

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
                self.error_log.append(f"{full_api_name} does not exist.")
                return False
            if func is None:
                return False
            if inspect.ismodule(func):
                self.error_log.append(f"{full_api_name} is a module, not a function.")
                return False
            # if inspect.isclass(func): # 诸如torch.nn.CrossEntropyLoss等用类封装的API将无法被测试, 因此选择注释掉
            #     self.cluster_errors.append(f"{full_api_name} is a class, not a function.")
            #     return False
            # if validate_api_availability(func):
            #    self.cluster_errors.append(f"{full_api_name} is deprecated.")
            #    return False
            return True
        except ModuleNotFoundError as e:
            self.error_log.append(f"Module {module_name} not found: {str(e)}")
            return False
        except ImportError as e:
            self.error_log.append(f"Module {module_name} not found: {str(e)}")
            return False
        except AttributeError:
            self.error_log.append(f"{api_name} does not exist in {module_name}.")
            return False
        except Exception as e:
            self.error_log.append(str(e))
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
            self.error_log.append("The response data has an invalid JSON format.")
            return False
        except Exception as e:
            self.error_log.append(str(e))
            return False

    def identify_apis(self, api_groups, whether_supplement_api=False):
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
            if_api_group_complete = True
            for full_api_name in api_group:  # 获取某个API组合中的每个API
                api = self.session.query(API).filter_by(full_name=full_api_name).first()
                if api is None and whether_supplement_api:
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
                elif api is None and not whether_supplement_api:
                    if_api_group_complete = False
                    break
                api_list.append(api)  # [constant, softmax_cross_entropy_with_logits]
            if if_api_group_complete:
                api_group_objects.append(tuple(api_list))
        return api_group_objects  # [('CategoricalCrossentropy'), ('constant', 'softmax_cross_entropy_with_logits')]

    def judge_equivalence(self, api_groups, execute_results, threshold=0.95):  # 根据测试用例的运行结果判断API组合的等价关系
        # Example: api_groups = [("jittor.nn.CrossEntropyLoss"), ("jax.nn.log_softmax", "jax.numpy.sum", "jax.numpy.mean"), ...]
        # Example: execute_results = {("jittor.nn.CrossEntropyLoss"):[result1, result2], ), ("jax.nn.log_softmax", "jax.numpy.sum", "jax.numpy.mean"):[result1, result2, ..], ...}
        target_api_group = [api_group for api_group in api_groups if len(api_group) == 1 and api_group[0] == self.api][
            0]
        value_equivalent_api_groups = [list(target_api_group), ]
        state_equivalent_api_groups = [list(target_api_group), ]
        target_api_group_results = execute_results[target_api_group]
        for api_group, results in execute_results.items():
            if api_group == target_api_group:
                continue
            # 先检查target_api_group_results和results中的每个result的状态是否一致
            is_state_equivalent = True
            for i in range(len(target_api_group_results)):
                if target_api_group_results[i]['status'] != results[i]['status']:
                    is_state_equivalent = False
                    break
            if is_state_equivalent is False:  # 当前API组合的状态不等价
                continue
            # 再检查target_api_group_results和results中的每个result的值是否一致
            is_value_equivalent = True
            for i in range(len(target_api_group_results)):
                target_output = target_api_group_results[i]['output']
                compare_output = results[i]['output']
                if isinstance(target_output, str) and isinstance(compare_output, str):  # 输出值为字符串
                    # 字符串完全相同
                    if target_output != compare_output:
                        is_value_equivalent = False
                        break
                else:
                    # 判断向量的维度是否一致
                    if len(target_output) != len(compare_output):
                        is_value_equivalent = False
                        break
                    # 计算余弦相似度
                    try:
                        similarity = cosine_similarity(target_output, compare_output)
                        if similarity < threshold:
                            is_value_equivalent = False
                            break
                    except Exception as e:
                        print(f"Error computing similarity between vectors: {e}")
                        is_value_equivalent = False
                        break
            if is_value_equivalent == True:
                value_equivalent_api_groups.append(list(api_group))
            else:  # 值不等价但状态等价
                state_equivalent_api_groups.append(list(api_group))
        return value_equivalent_api_groups, state_equivalent_api_groups  # [[API], [API, API], ...]

    def verify_equivalence(self, json_data):  # 验证API组合的等价关系是否成立(值等价(1)/状态等价(2)/无等价关系(0))
        libs_apis_group_objects = {}  # {"Pytorch" : [(API1), (API2, API3)], "JAX" : [(API1),(API2, API3)], ...}
        for lib, dict_api_groups in json_data.items():
            apis_group_objects = self.identify_apis(api_groups=dict_api_groups, whether_supplement_api=False)
            libs_apis_group_objects[lib] = apis_group_objects
        api_groups = [api_group for api_groups in libs_apis_group_objects.values() for api_group in api_groups]
        if tuple([self.api]) not in api_groups:  # 检查api_group_objects最终是否有包含(self.api,), 如果没有则手动添加
            api_groups.append(tuple([self.api]))

        # 如果没有匹配到任何等价API, 在判定为该API无等价关系
        if len(api_groups) < 2:
            if len(api_groups[0]) == 1 and api_groups[0][0] == self.api:  # 没有匹配到等价API
                self.api.is_clustered = True
                self.session.commit()
                return None, None

        # 使用等价API的文档中API的调用样例作为测试输入
        single_api_groups = [api_group for api_group in api_groups if len(api_group) == 1]
        # 检查是否single_api_groups中的所有APIGroup的api.example都为空
        if all([api_group[0].example is None for api_group in single_api_groups]):
            # TODO 考虑根据self.api的文档为self.api生成一个example作为测试输入
            self.api.is_clustered = True
            self.session.commit()
            return None, None

        execute_results = {api_group: [] for api_group in api_groups}
        base_api_groups = [api_group for api_group in single_api_groups if
                           api_group[0].example is not None and api_group[0].example != ""]
        while base_api_groups:
            base_api_group = base_api_groups[0]

            # 生成测试用例
            test_cases = {api_group: (base_api_group[0].example if api_group == base_api_group else "") for api_group in
                          api_groups}
            try:
                for twin_api_group in api_groups:
                    if twin_api_group == base_api_group:
                        continue
                    messages = self.construct_verify_messages(base_api_group[0], twin_api_group)
                    test_case = self.query_llm4verify(messages)
                    validate_test_case = validator.APITestSeedValidator(llm_client=self.llm_client,
                                                                        raw_code=test_case).validate4code()
                    test_cases[twin_api_group] = validate_test_case
            except Exception as e:
                print(f"An error occurred when verify equivalence: {e}")
                continue

            # 执行测试用例
            try:
                for api_group, test_case in test_cases.items():
                    if not test_case:
                        execute_results[api_group].append(
                            {"status": "Skipped", "output": "Error: No test case provided."})
                        continue

                    # 执行代码片段
                    exec_namespace = {}  # 使用独立的命名空间来隔离执行环境
                    try:
                        exec(test_case, {}, exec_namespace)
                        # 假设代码片段定义了一个变量 `result` 来表示执行结果
                        output = exec_namespace.get('output', 'No result returned.')  # 此处默认变量名为"output"
                        execute_results[api_group].append({"status": "Success", "output": output})
                    except Exception as exec_e:
                        # 捕获执行中的异常
                        error_trace = traceback.format_exc()
                        execute_results[api_group].append({"status": "Failed", "output": error_trace})
            except Exception as e:
                print(f"An error occurred when execute test cases: {e}")
                continue
            base_api_groups.pop(0)
        # 根据execute_results判断值等价/状态等价/无等价关系
        value_equivalent_api_groups, state_equivalent_api_groups = self.judge_equivalence(api_groups, execute_results)
        return value_equivalent_api_groups, state_equivalent_api_groups

    def save_cluster(self, equivalence_type, api_groups):
        """
            接收并处理clusterer的响应结果, 创建cluster聚类和关联的API组合
        """
        try:
            # 1. 判断是否有等价关系
            if api_groups is None:  # 如果没有匹配到等价API, 则直接返回
                return None

            # 2. apis_group_objects中目前至少有2个API Group, 查找已经存在的Cluster对象或创建Cluster对象:
            single_api_groups = [api_group for api_group in api_groups if len(api_group) == 1]
            if len(single_api_groups) > 0:  # 如果存在由单独的API组成的API组合
                cluster_dict = {}
                for single_api_group in single_api_groups:
                    single_api = single_api_group[0]
                    api_obj_groups = (self.session.query(APIGroup)
                                      .join(APIGroup.apis)
                                      .filter(Cluster.type == equivalence_type)
                                      .group_by(APIGroup.id)
                                      .having(func.count(API.id) == 1,  # 确保当前Group内只包含一个API
                                              func.min(API.id) == single_api.id)  # 确保当前Group内包含的API是api
                                      .all())
                    for api_obj_group in api_obj_groups:
                        cluster = api_obj_group.cluster
                        cluster_dict[cluster] = cluster_dict.get(cluster, 0) + 1
                if cluster_dict:  # 选择已有的值等价簇加入
                    # Case 2.1
                    value_equivalent_cluster = max(cluster_dict, key=cluster_dict.get)
                else:  # 创建一个新的值等价簇并加入
                    # Case 2.2
                    value_equivalent_cluster = Cluster(
                        type=equivalence_type,
                        energy=5,
                    )
                    self.session.add(value_equivalent_cluster)
                    self.session.flush()
            else:  # 创建一个新的值等价簇并加入
                # Case 2.2
                value_equivalent_cluster = Cluster(
                    type=equivalence_type,
                    energy=5,
                )
                self.session.add(value_equivalent_cluster)
                self.session.flush()

            # 3. 为每个API组合创建对应的APIGroup对象, 之后将它们与新创建的Cluster对象关联
            for api_group in api_groups:  # 逐个访问每个API组合
                group = APIGroup(
                    apis=list(api_group),
                    cluster=value_equivalent_cluster
                )
                self.session.add(group)
                self.session.flush()

            # 4. 将single_api_groups中的API标记为已经被聚类
            # for single_api_group in single_api_groups:
            #     single_api_group[0].is_clustered = True
            # self.session.commit()
        except Exception as e:
            self.session.rollback()  # 回滚在异常中的任何数据库更改
            print(f"An error occurred: {e}")

    # ----------------------------------------------run()----------------------------------------------
    def cluster_api(self):
        cluster_query_messages = self.construct_cluster_messages(self.api)
        cluster_json_data = self.query_llm4cluster(cluster_query_messages)
        if cluster_json_data:
            value_equivalent_api_groups, state_equivalent_api_groups = self.verify_equivalence(cluster_json_data)
            self.save_cluster('ValueEquivalent', value_equivalent_api_groups)
            self.save_cluster('StateEquivalent', state_equivalent_api_groups)
            self.api.is_clustered = True
            self.session.commit()


def run_randomly():  # 随机挑选未聚类的API进行聚类
    # 创建数据库连接
    session = get_session()
    llm_client = get_llm_client('gpt4o-mini-with-rag')

    # 对未聚类的PytorchAPI进行聚类
    uncluttered_torch_apis = session.query(API).filter_by(is_clustered=False).all()
    while uncluttered_torch_apis:
        print("----------------------------------------------------------------------------------")
        # 随机选择一个未聚类的API
        uncluttered_torch_api = random.choice(uncluttered_torch_apis)
        cluster = EquivalentCluster(uncluttered_torch_api, session, llm_client)
        cluster.cluster_api()

        uncluttered_torch_apis = session.query(API).filter_by(is_clustered=False).all()
        total_apis_num = session.query(API).count()
        unclustered_torch_apis_num = len(uncluttered_torch_apis)
        print(f"Unclustered / Total: {unclustered_torch_apis_num} / {total_apis_num}")


def run_linearly():  # 线性地对未聚类的API进行聚类
    # 创建数据库连接
    session = get_session()
    llm_client = get_llm_client('gpt4o-mini-with-rag')

    # 对未聚类的API进行聚类
    uncluttered_torch_apis = session.query(API).filter_by(is_clustered=False).all()
    for i, uncluttered_torch_api in enumerate(uncluttered_torch_apis):
        print("----------------------------------------------------------------------------------")
        # 选择一个未聚类的TensorflowAPI
        cluster = EquivalentCluster(uncluttered_torch_api, session, llm_client)
        cluster.cluster_api()
        print(f"Unclustered / Total: {len(uncluttered_torch_apis) - i - 1} / {len(uncluttered_torch_apis)}" + "\n")


if __name__ == '__main__':
    # run_randomly()
    run_linearly()
