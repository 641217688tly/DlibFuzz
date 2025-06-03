import json
import random
import traceback
from json import JSONDecodeError
from sqlalchemy import func
from utils import *
from fuzzer.validator import APITestSeedValidator
import numpy as np

# ----------------------------------------------Cluster----------------------------------------------
class EquivalentCluster:
    def __init__(self, api: API, session, rag_client, llm_client):
        self.api = api
        self.session = session
        self.rag_client = rag_client
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
Your answer must be provided strictly in the following JSON format:
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
{'- Parameters: ' + api.parameters if api.parameters else ''}
{'- Attributes:' + api.attributes if api.attributes else ''}
{'- Output:' + api.output if api.output else ''}

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
        print("*" * 80 + "query_llm4cluster()")
        attempt_num = 0
        while attempt_num < max_try:  # 设置最大尝试次数以避免无限循环
            print("-" * 60)
            print(f"query_llm4cluster() Info - messages:\n{messages}")
            self.error_log = []  # 清空错误列表
            try:  # 假如返回的数据不符合JSON格式, 则重新调用OpenAI API, 直到返回的数据符合JSON格式为止
                response = self.rag_client.chat.completions.create(
                    model="gpt-4o-mini",  # gpt-4o-mini  gpt-3.5-turbo
                    response_format={"type": "json_object"},
                    messages=messages,
                    temperature=0.4,
                )
                response = response.choices[0].message['content']
                # 如果response的第一行以"```"开头, 则去掉第一行; 如果response的最后一行以"```"结尾, 则去掉最后一行
                if response.split('\n', 1)[0].startswith("```"):
                    response = response.split('\n', 1)[1]
                if response.split('\n')[-1].endswith("```"):
                    response = '\n'.join(response.split('\n')[:-1])
                messages.append({"role": "assistant", "content": response})
                print(f"query_llm4cluster() Info - Clustered API: {self.api.full_name}\nResponse:\n{response}")
                # 在此处需要检查: 1.响应的数据是否遵循JSON格式; 2.返回的是API的完整函数名(完整函数名 = 模块名.API名)而非函数签名 3.所有的API函数名必须有效(不是虚构的, 也不是被弃用的)
                if self.validate_apis(response):  # 经验证证明返回的数据是有效的
                    print(f"query_llm4cluster() Success - Both json response and apis are valid!")
                    self.error_log = []  # 清空错误列表
                    return json.loads(response)
                else:
                    print(f"query_llm4cluster() Error - self.error_log: {self.error_log}")
                    attempt_num = attempt_num + 1
                    messages.append({"role": "user", "content": f"The JSON response you generated has the following errors: \n{self.error_log} \n Please try again."})
                    print(f"query_llm4cluster() Error - Incorrect JSON format or invalid API. Error Details: {self.error_log} \nRetrying(Current attempt: {attempt_num})...")
            except Exception as e:
                attempt_num = attempt_num + 1
                self.session.rollback()  # 回滚在异常中的任何数据库更改
                print(f"query_llm4cluster() Error - An unexpected error occurred: {e}")
        self.error_log = []  # 清空错误列表
        print("query_llm4cluster() Failed - Max attempts reached. Unable to get valid JSON data.")
        return None

    def construct_twin_test_seed_messages(self, base_api, twin_api_group):  # twin_api_group = [API1, API2, ...]
        # system prompt
        system_prompt = f"""
(1) Role Definition: You are an AI assistant specialized in deep learning framework APIs (e.g., PyTorch, JAX, MindSpore and Jittor). Your primary task is to help users find equivalent APIs (or API groups) across different deep learning libraries.
(2) Output Format: Your response must be pure code. Do not include any explanations, comments, or extra content.
"""

        # twin_api_group 的详情
        if len(twin_api_group) == 1:
            twin_api = twin_api_group[0]
            api_group_info_prompt = f"""
- API Name: {twin_api.full_name}
- API Library: {twin_api.lib} (version{twin_api.version})
- API Signature: {twin_api.signature}
{'- Function Description: ' + twin_api.description if twin_api.description else ''}
{'- Parameters: ' + twin_api.parameters if twin_api.parameters else ''}
{'- Attributes:' + twin_api.attributes if twin_api.attributes else ''}
{'- Output:' + twin_api.output if twin_api.output else ''}
"""
        else:
            api_group_info_prompt = ""
            for count, twin_api in enumerate(twin_api_group):
                api_group_info_prompt = api_group_info_prompt + f"""
Member{count + 1} of API Group:
- API Name: {twin_api.full_name}
- API Signature: {twin_api.signature}
{'- Function Description: ' + twin_api.description if twin_api.description else ''}
{'- Parameters: ' + twin_api.parameters if twin_api.parameters else ''}
{'- Attributes:' + twin_api.attributes if twin_api.attributes else ''}
{'- Output:' + twin_api.output if twin_api.output else ''}
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
- API Library: {base_api.lib} (version{base_api.version})
- API Signature: {base_api.signature}
{'- Function Description: ' + base_api.description if base_api.description else ''}
{'- Parameters: ' + base_api.parameters if base_api.parameters else ''}
{'- Attributes:' + base_api.attributes if base_api.attributes else ''}
{'- Output:' + base_api.output if base_api.output else ''}
"""
        else:
            background_knowledge_prompt = f"""
By combining the APIs in {api_group_brief_info}, it can achieve the similar functionality as the API {base_api.signature} from library {base_api.lib}(v{base_api.version}).
The detail of API ({base_api.signature}) is as follows:
- API Name: {base_api.full_name}
- API Library: {base_api.lib} (version{base_api.version})
- API Signature: {base_api.signature}
{'- Function Description: ' + base_api.description if base_api.description else ''}
{'- Parameters: ' + base_api.parameters if base_api.parameters else ''}
{'- Attributes:' + base_api.attributes if base_api.attributes else ''}
{'- Output:' + base_api.output if base_api.output else ''}
"""

        # 构建最终提示词
        prompt = f"""
Information of the API {'group' if len(twin_api_group) > 1 else ''} to be called:
{api_group_info_prompt}        

Background knowledge:
{background_knowledge_prompt}

Task:
Below is a code snippet calling ({base_api.signature}). Please generate a code snippet that replaces ({base_api.full_name}) with {api_group_brief_info}, ensuring that the input parameters remain unchanged. Additionally, ensure that the code snippet you generate declares the same output variables as the example code. If the example code uses variables like "output1", "output2", "output3", etc., to store multiple return values or affected variables, then your generated code should also declare the same numbered output variables (output1, output2, output3, etc.) to store the corresponding results.
{base_api.example}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        return messages

    def query_llm4TwinTestSeed(self, messages, max_retry=5):
        attempt_num = 0
        while attempt_num < max_retry:  # 设置最大尝试次数以避免无限循环
            try:
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",  # gpt-4o-mini  gpt-3.5-turbo
                    messages=messages,
                    temperature=0.4,
                )
                raw_code = response.choices[0].message.content
                # print(f"\nquery_llm4TwinTestSeed() Info - raw_code:\n{raw_code}")
                code = APITestSeedValidator(llm_client=self.llm_client, raw_code=raw_code).validate4code()
                # print(f"\nquery_llm4TwinTestSeed() Info - code:\n{code}")
                return code
            except Exception as e:
                print(f"Failed to get response due to: \n{e} \nRetrying(Current attempt: {attempt_num + 1})...")
                attempt_num += 1
                self.session.rollback()  # 回滚在异常中的任何数据库更改
        if attempt_num >= 5:  # 设置最大尝试次数以避免无限循环
            print("Max attempts reached. Unable to get valid JSON data.")
            return None

    def construct_extract_base_test_seed_messages(self, base_api: API):
        system_prompt = f"""
(1) Role Definition: You are an AI assistant specialized in deep learning framework APIs (e.g., PyTorch, JAX, MindSpore and Jittor). Your primary task is to help users find equivalent APIs (or API groups) across different deep learning libraries.
(2) Output Format: Your response must be pure code. Do not include any explanations, comments, or extra content.
"""
        # Example 1
        context_query_prompt1 = f"""
API Information:
- API Name: jax.numpy.arccos
- Source Library: JAX
- Version: 0.4.13
- API Signature: jax.numpy.arccos(x,/)
- Function Description: Compute element-wise inverse of trigonometric cosine of input.
- Parameters: x(ArrayLike) – input array or scalar.
- Attributes: Null
- Output: Array
- Examples:
example: >>> x = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])
>>> with jnp.printoptions(precision=3, suppress=True):
...   jnp.arccos(x)
Array([  nan, 3.142, 2.094, 1.571, 1.047, 0.   ,   nan], dtype=float32)

Task:
Extract a usage example from the API's Examples that includes calling the jax.numpy.arccos and return the code. 
When the API has only one return value, you must declare a variable named "output1" to store the result. When the API has multiple return values, you need to declare variables named "output1", "output2", "output3", etc., to store these return values respectively. If the API has no return value but performs a built-in operation on the input, you need to use "output1" to store the input after it has been processed by the API's built-in operation. If the API's built-in operation affects multiple inputs, you need to declare variables named "output1", "output2", "output3", etc., to store these affected inputs respectively.
"""
        context_answer_prompt1 = """
import jax
import jax.numpy as jnp
x = jnp.array([-2, -1, -0.5, 0, 0.5, 1, 2])
output1 = jnp.arccos(x)
"""
        # Example 2 - 多返回值示例
        context_query_prompt2 = f"""
API Information:
- API Name: torch.nn.MultiheadAttention
- Source Library: PyTorch
- Version: 2.4.1
- API Signature: torch.nn.MultiheadAttention(embed_dim,num_heads,dropout=0.0,bias=True,add_bias_kv=False,add_zero_attn=False,kdim=None,vdim=None,batch_first=False,device=None,dtype=None)
- Function Description: Allows the model to jointly attend to information from different representation subspaces.
- Parameters: 
embed_dim – Total dimension of the model.
num_heads – Number of parallel attention heads. Note thatembed_dimwill be split acrossnum_heads(i.e. each head will have dimensionembed_dim//num_heads).
dropout – Dropout probability onattn_output_weights. Default:0.0(no dropout).
bias – If specified, adds bias to input / output projection layers. Default:True.
add_bias_kv – If specified, adds bias to the key and value sequences at dim=0. Default:False.
add_zero_attn – If specified, adds a new batch of zeros to the key and value sequences at dim=1.
kdim – Total number of features for keys. Default:None(useskdim=embed_dim).
vdim – Total number of features for values. Default:None(usesvdim=embed_dim).
batch_first – If True, then the input and output tensors are provided as (batch, seq, feature).
- Examples:
example: >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
>>> attn_output, attn_output_weights = multihead_attn(query, key, value)

Task:
Extract a usage example from the API's Examples that includes calling the torch.nn.MultiheadAttention and return the code. 
When the API has only one return value, you must declare a variable named "output1" to store the result. When the API has multiple return values, you need to declare variables named "output1", "output2", "output3", etc., to store these return values respectively. If the API has no return value but performs a built-in operation on the input, you need to use "output1" to store the input after it has been processed by the API's built-in operation. If the API's built-in operation affects multiple inputs, you need to declare variables named "output1", "output2", "output3", etc., to store these affected inputs respectively.
"""
        context_answer_prompt2 = """
import torch
import numpy as np
embed_dim, num_heads = 128, 8
seq_length, batch_size = 10, 8
query = torch.Tensor(np.random.randn(seq_length, batch_size, embed_dim))
key = torch.Tensor(np.random.randn(seq_length, batch_size, embed_dim))
value = torch.Tensor(np.random.randn(seq_length, batch_size, embed_dim))
multihead_attn = torch.nn.MultiheadAttention(embed_dim, num_heads)
output1, output2 = multihead_attn(query, key, value)
"""
        # query
        query_prompt = f"""
API Information:
- API Name: {base_api.full_name}
- Source Library: {base_api.lib} (version{base_api.version})
- API Signature: {base_api.signature}
{'- Function Description: ' + base_api.description if base_api.description else ''}
{'- Parameters: ' + base_api.parameters if base_api.parameters else ''}
{'- Attributes:' + base_api.attributes if base_api.attributes else ''}
{'- Output:' + base_api.output if base_api.output else ''}
- Examples: \n{base_api.example}

Task:
Extract a usage example from the API's Examples that includes calling the {base_api.full_name} and return the code. 
When the API has only one return value, you must declare a variable named "output1" to store the result. When the API has multiple return values, you need to declare variables named "output1", "output2", "output3", etc., to store these return values respectively. If the API has no return value but performs a built-in operation on the input, you need to use "output1" to store the input after it has been processed by the API's built-in operation. If the API's built-in operation affects multiple inputs, you need to declare variables named "output1", "output2", "output3", etc., to store these affected inputs respectively.
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

    def query_llm4BaseTestSeed(self, messages, max_retry=5):
        attempt_num = 0
        while attempt_num < max_retry:  # 设置最大尝试次数以避免无限循环
            try:
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",  # gpt-4o-mini  gpt-3.5-turbo
                    messages=messages,
                    temperature=0.4,
                )
                raw_code = response.choices[0].message.content
                # print(f"\nquery_llm4BaseTestSeed() Info - raw_code:\n{raw_code}")
                code = APITestSeedValidator(llm_client=self.llm_client, raw_code=raw_code).validate4code()
                # print(f"\nquery_llm4BaseTestSeed() Info - code:\n{code}")
                return code
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
        try:
            module_name, api_name = full_api_name.rsplit('.', 1)
            module_list = module_name.split('.')
            api_lib = self.handle_module_alias(module_list[0])
            if map_module2lib(api_lib) == 'Unknown':
                raise Exception(f"{full_api_name} does not belong to Pytorch, JAX, MindSpore or Jittor.")
            module = importlib.import_module(api_lib)
            if len(module_list) > 1:
                # 将module_name_list进行切片, 只保留除第一个元素以外的部分
                for submodule_name in module_list[1:]:
                    module = getattr(module, submodule_name, None)
                    if module is None:
                        return False
            api = getattr(module, api_name, None)
            if api is None:
                return False
            else:
                return True
        except Exception as e:
            self.error_log.append(f"{full_api_name} is not a valid API. Error: {e}")
            return False

    def validate_apis(self, response):
        """
        验证JSON中API的格式是完整函数名(完整函数名 = 模块名.API名)而非函数签名
        所有的API函数名必须有效(有效的定义为: JSON数据中的API为函数全名(函数全名 = 模块.函数名)而非函数签名, 该API不是被弃用的, 该API必须是函数而非模块或类, 该API可以被导入)
        """
        try:
            is_valid = True
            json_data = json.loads(response)
            print(f"validate_apis() Info - json format is valid")
            for dl_lib, api_groups in json_data.items():  # 逐个访问Pytorch, JAX, MindSpore和Jittor下的API二维数组
                for api_group in api_groups:  # 逐个访问Pytorch, Tensorflow和Jax下的各个API组合
                    for full_api_name in api_group:  # 逐个访问API组合下的各个API
                        if self.validate_api(full_api_name) is False:
                            print(f"validate_apis() Error - {full_api_name} is not a callable API!")
                            self.error_log.append(f"{full_api_name} is not a callable API.")
                            is_valid = False
            return is_valid
        except JSONDecodeError as e:
            self.error_log.append("The response data has an invalid JSON format.")
            return False
        except Exception as e:
            self.error_log.append(str(e))
            return False

    def identify_apis(self, api_groups, whether_supplement_api=False):
        """
        以下列数据为例:
        api_groups = [
            ["tensorflow.keras.losses.CategoricalCrossentropy"],
            ["tensorflow.constant", "tensorflow.nn.softmax_cross_entropy_with_logits"]
        ]
        """
        api_group_objects = []  # [[CategoricalCrossentropy], [constant, softmax_cross_entropy_with_logits]]
        for api_group in api_groups:  # 逐个访问每个API组合
            # 跳过空的API组
            if len(api_group) == 0:
                continue
                
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
            if if_api_group_complete and len(api_list) > 0: 
                api_group_objects.append(tuple(api_list))
        return api_group_objects  # [('CategoricalCrossentropy'), ('constant', 'softmax_cross_entropy_with_logits')]

    def judge_equivalence(self, api_groups, execute_results, threshold=0.95):  # 根据测试用例的运行结果判断API组合的等价关系
        # Example: api_groups = [("jittor.nn.CrossEntropyLoss"), ("jax.nn.log_softmax", "jax.numpy.sum", "jax.numpy.mean"), ...]
        # Example: execute_results = {("jittor.nn.CrossEntropyLoss"):[result1, result2], ("jax.nn.log_softmax", "jax.numpy.sum", "jax.numpy.mean"):[result1, result2], ...}
        target_api_group = [api_group for api_group in api_groups if len(api_group) == 1 and api_group[0] == self.api][0]

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
                target_outputs = target_api_group_results[i]['outputs']
                compare_outputs = results[i]['outputs']
                
                # 比较输出变量的数量
                if len(target_outputs) != len(compare_outputs):
                    is_value_equivalent = False
                    break
                
                # 逐个比较每个输出变量
                for output_key in target_outputs.keys():
                    if output_key not in compare_outputs:
                        is_value_equivalent = False
                        break
                    
                    target_output = target_outputs[output_key]
                    compare_output = compare_outputs[output_key]
                    
                    if isinstance(target_output, str) and isinstance(compare_output, str):  # 输出值为字符串
                        # 字符串完全相同
                        if target_output != compare_output:
                            is_value_equivalent = False
                            break
                    else:
                        # 处理不同类型的输出比较
                        try:
                            # 转换为numpy数组进行比较
                            target_np = convert2numpy(target_output)
                            compare_np = convert2numpy(compare_output)
                            
                            # 检查形状是否一致
                            if target_np.shape != compare_np.shape:
                                is_value_equivalent = False
                                break
                            
                            # 如果是标量（0维数组），直接比较值
                            if target_np.ndim == 0:
                                # 对于标量，检查值是否相等（考虑浮点数精度）
                                if not np.allclose(target_np, compare_np, rtol=1e-5, atol=1e-8):
                                    is_value_equivalent = False
                                    break
                            else:
                                # 对于向量/矩阵，计算余弦相似度
                                similarity = cosine_similarity(target_output, compare_output)
                                if similarity < threshold:
                                    is_value_equivalent = False
                                    break
                        except Exception as e:
                            is_value_equivalent = False
                            break
                if not is_value_equivalent:
                    break
                    
            if is_value_equivalent == True:
                value_equivalent_api_groups.append(list(api_group))
            else:  # 值不等价但状态等价
                state_equivalent_api_groups.append(list(api_group))
        return value_equivalent_api_groups, state_equivalent_api_groups  # [[Taraget_API], [API, API], ...] , [[Taraget_API], [API], [API, API], ...]

    def verify_equivalence(self, json_data):  # 验证API组合的等价关系是否成立(值等价(1)/状态等价(2)/无等价关系(0))
        print("*" * 80 + "verify_equivalence()")
        libs_apis_group_objects = {}  # {"Pytorch" : [(API1), (API2, API3)], "JAX" : [(API1),(API2, API3)], ...}
        for lib, dict_api_groups in json_data.items():
            apis_group_objects = self.identify_apis(api_groups=dict_api_groups, whether_supplement_api=False)
            libs_apis_group_objects[lib] = apis_group_objects
        api_groups = [api_group for api_groups in libs_apis_group_objects.values() for api_group in api_groups] # 例: api_groups = [('CategoricalCrossentropy'), ('constant', 'softmax_cross_entropy_with_logits')]
        if tuple([self.api]) not in api_groups:  # 检查api_group_objects最终是否有包含(self.api,), 如果没有则手动添加
            api_groups.append(tuple([self.api]))

        # 如果没有匹配到任何等价API, 在判定为该API无等价关系
        if len(api_groups) < 2:
            if len(api_groups[0]) == 1 and api_groups[0][0] == self.api:  # 没有匹配到等价API
                print(f"verify_equivalence() Success - Did not find any equivalent API for {self.api.full_name}.")
                self.api.is_clustered = True
                self.session.commit()
                return None, None

        # 使用等价API的文档中API的调用样例作为测试输入
        single_api_groups = [api_group for api_group in api_groups if len(api_group) == 1] # 例: single_api_groups = [('CategoricalCrossentropy'),]
        # 检查是否single_api_groups中的所有APIGroup的api.example都为空
        if all([api_group[0].example is None for api_group in single_api_groups]):
            # 如果没有可用的测试输入作为Oracle, 则直接返回None
            print("verify_equivalence() Error - No test example available for equivalence verification.")
            # self.api.is_clustered = True
            # self.session.commit()
            return None, None

        execute_results = {api_group: [] for api_group in api_groups}
        # 选择有文档usage example的API作为基准API
        base_api_groups = [api_group for api_group in single_api_groups if api_group[0].example is not None and api_group[0].example != "" and len(api_group[0].example) > 5]

        #TODO 为了节约成本, 仅从base_api_groups中随机选择一个API作为基准API
        if len(base_api_groups) > 1:
            # 检查(self.api)是否在base_api_groups中,如果在则优先选择(self.api)作为基准API
            if tuple([self.api]) in base_api_groups:
                base_api_groups = [tuple([self.api])]
            else:
                base_api_groups = random.sample(base_api_groups, 1)

        while base_api_groups:
            base_api_group = base_api_groups[0]
            base_api = base_api_group[0]
            print("-" * 60 + "\n" + f"verify_equivalence() Info - base_api: {base_api.full_name}")

            # 使用LLM从base_api_group[0].example中生提取一个有效的测试用例
            base_messages = self.construct_extract_base_test_seed_messages(base_api)
            base_test_code = self.query_llm4BaseTestSeed(base_messages)
            if base_test_code is None:
                print(f"verify_equivalence() Error - No valid test cases can be obtained from the example of {base_api.full_name}.")
                continue
            print("@" * 40 + "\n" + f"verify_equivalence() Info - {base_api.full_name} Base Test Case:\n{base_test_code}")
            test_cases = {api_group: (base_test_code if api_group == base_api_group else "") for api_group in api_groups}  # 初始化测试用例
            try:
                for twin_api_group in api_groups:
                    if twin_api_group == base_api_group:
                        continue
                    twin_messages = self.construct_twin_test_seed_messages(base_api_group[0], twin_api_group)
                    twin_test_code = self.query_llm4TwinTestSeed(twin_messages)
                    test_cases[twin_api_group] = twin_test_code
                    print("@" * 40 + "\n" + f"verify_equivalence() Info - {twin_api_group[0].full_name} Twin Test Case:\n{twin_test_code}")
            except Exception as e:
                print(f"verify_equivalence() Error - An error occurred when verify equivalence: {e}")
                continue

            # 执行测试用例
            try:
                for api_group, test_case in test_cases.items():
                    if not test_case:
                        execute_results[api_group].append({"status": "Skipped", "outputs": {"error": "Error: No test case provided."}})
                        continue

                    # 执行代码片段
                    exec_namespace = {}  # 使用独立的命名空间来隔离执行环境
                    try:
                        exec(test_case, {}, exec_namespace)
                        # 收集所有output变量（output1, output2, output3等）
                        outputs = {}
                        output_count = 0
                        for i in range(1, 21):  # 最多支持20个输出变量
                            output_var = f'output{i}'
                            if output_var in exec_namespace:
                                outputs[output_var] = exec_namespace[output_var]
                                output_count += 1
                            else:
                                break  # 如果找不到连续的output变量，停止搜索
                        
                        if output_count == 0:
                            # 如果没有找到任何output变量，尝试寻找其他可能的输出变量
                            possible_outputs = []
                            for var_name, var_value in exec_namespace.items():
                                if (not var_name.startswith('_') and 
                                    not callable(var_value) and 
                                    var_name not in ['torch', 'jax', 'mindspore', 'jittor', 'numpy', 'np', 'tensorflow']):
                                    if any(keyword in var_name.lower() for keyword in ['output', 'result', 'attn', 'prediction', 'pred']):
                                        possible_outputs.insert(0, (var_name, var_value))
                                    else:
                                        possible_outputs.append((var_name, var_value))
                            
                            if possible_outputs:
                                outputs['output1'] = possible_outputs[0][1]
                                print(f"verify_equivalence() Info - Using variable '{possible_outputs[0][0]}' as output1 for {api_group}")
                            else:
                                outputs['output1'] = 'No result returned.'
                                print(f"verify_equivalence() Warning - No suitable output variable found for {api_group}")
                        
                        execute_results[api_group].append({"status": "Success", "outputs": outputs})
                    except Exception as exec_e:
                        # 捕获执行中的异常
                        error_trace = traceback.format_exc()
                        execute_results[api_group].append({"status": "Failed", "outputs": {"error": error_trace}})
            except Exception as e:
                print(f"verify_equivalence() Error - An error occurred when execute test cases: {e}")
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
                return False
            if len(api_groups) < 2:
                return False

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
                if cluster_dict:  # 假如已经存在等价簇
                    # Case 2.1 选择已有的等价簇加入
                    equivalent_cluster = max(cluster_dict, key=cluster_dict.get)
                else:  # 假如不存在等价簇
                    # Case 2.2 创建一个新的等价簇并加入
                    equivalent_cluster = Cluster(
                        type=equivalence_type,
                        energy=5,
                    )
                    self.session.add(equivalent_cluster)
                    self.session.flush()
            else:  # 假如不存在由单独的API组成的API组合
                # Case 2.2 创建一个新的等价簇并加入
                equivalent_cluster = Cluster(
                    type=equivalence_type,
                    energy=5,
                )
                self.session.add(equivalent_cluster)
                self.session.flush()

            # 3. 为每个API组合创建对应的APIGroup对象, 之后将它们与新创建的Cluster对象关联
            for api_group in api_groups:  # 逐个访问每个API组合
                group = APIGroup(
                    apis=list(api_group),
                    cluster=equivalent_cluster
                )
                self.session.add(group)
                self.session.flush()

            # 4. 将single_api_groups中的API标记为已经被聚类, 这样能显著提高API匹配的效率
            for single_api_group in single_api_groups:
                single_api_group[0].is_clustered = True
            self.session.flush()
            return True
        except Exception as e:
            self.session.rollback()  # 回滚在异常中的任何数据库更改
            print(f"save_cluster() Error - An error occurred: {e}")

    # ----------------------------------------------run()----------------------------------------------
    def cluster_api(self):
        cluster_query_messages = self.construct_cluster_messages(self.api)
        cluster_json_data = self.query_llm4cluster(cluster_query_messages)
        if cluster_json_data:
            value_equivalent_api_groups, state_equivalent_api_groups = self.verify_equivalence(cluster_json_data)
            # 如果value_equivalent_api_groups和state_equivalent_api_groups不同时为空, 则保存聚类结果
            if value_equivalent_api_groups or state_equivalent_api_groups:
                self.save_cluster('ValueEquivalent', value_equivalent_api_groups)
                self.save_cluster('StateEquivalent', state_equivalent_api_groups)
                self.api.is_clustered = True
                self.session.commit()


def run_randomly():  # 随机挑选未聚类的API进行聚类
    # 创建数据库连接
    session = get_session()
    llm_client = get_llm_client('gpt4o-mini')
    rag_client = get_llm_client('gpt4o-mini-with-rag')

    # 对未聚类的PytorchAPI进行聚类
    unclustered_apis = session.query(API).filter_by(is_clustered=False).all()
    while unclustered_apis:
        # 随机选择一个未聚类的API
        unclustered_api = random.choice(unclustered_apis)
        
        # 打印当前未聚类的API数量
        apis_nums = session.query(API).count()
        unclustered_apis_nums = session.query(API).filter_by(is_clustered=False).count()
        print(f"EquivalentCluster({unclustered_api.full_name})" + "=" * 100 + f"\nUnclustered / Total: {unclustered_apis_nums} / {apis_nums}" + "\n")

        # 对未聚类的API进行聚类
        cluster = EquivalentCluster(unclustered_api, session, rag_client, llm_client)
        cluster.cluster_api()

        # 更新未聚类的API列表
        unclustered_apis = session.query(API).filter_by(is_clustered=False).all()



def run_linearly():  # 线性地对未聚类的API进行聚类
    # 创建数据库连接
    session = get_session()
    llm_client = get_llm_client('gpt4o-mini')
    rag_client = get_llm_client('gpt4o-mini-with-rag')

    # 对未聚类的API进行聚类
    unclustered_apis = session.query(API).filter_by(is_clustered=False).all()
    for unclustered_api in unclustered_apis:
        if unclustered_api.is_clustered:
            continue
        apis_nums = session.query(API).count()
        unclustered_apis_nums = session.query(API).filter_by(is_clustered=False).count()
        print(f"EquivalentCluster({unclustered_api.full_name})" + "=" * 100 + f"\nUnclustered / Total: {unclustered_apis_nums} / {apis_nums}" + "\n")
        
        # 选择一个未聚类的API
        cluster = EquivalentCluster(unclustered_api, session, rag_client, llm_client)
        cluster.cluster_api()
        


if __name__ == '__main__':
    # run_randomly()
    run_linearly()
