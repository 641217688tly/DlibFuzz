import importlib
import json
from json import JSONDecodeError
from sqlalchemy import func
import utils
from fuzzer.validator import APITestSeedValidator
from orm import *
import random
from collections import defaultdict


class Fuzzer:  # 以Cluster为单位生成测试种子
    def __init__(self, session, llm_client, rag_client, error_num_in_context=6, whether_sample_state_equivalent=True, whether_sample_value_equivalent=True):
        self.session = session
        self.llm_client = llm_client
        self.rag_client = rag_client
        self.error_num_in_context = error_num_in_context
        self.whether_sample_state_equivalent = whether_sample_state_equivalent
        self.whether_sample_value_equivalent = whether_sample_value_equivalent
        self.module_alias_mapper = {
            "tf": "tensorflow",
            "ms": "mindspore",
            "np": "numpy",
            "pd": "pandas",
            "jt": "jittor"
        }
        self.error_log = []

    def sample_errors(self, api: API):  # 抽取n个与api相关的错误
        errors = {
            "DirectRelevant": [],
            "StateIndirectRelevant": [],
            "ValueIndirectRelevant": [],
            "Irrelevant": []
        }
        
        # 1.优先寻找与api直接相关的错误
        direct_relevant_errors = api.history_errors
        if direct_relevant_errors:
            errors["DirectRelevant"] = random.sample(direct_relevant_errors, min(self.error_num_in_context, len(direct_relevant_errors)))

        # 2.如果当前错误数量不足n个, 则继续寻找与api间接相关的错误
        if sum(len(v) for v in errors.values()) < self.error_num_in_context and self.whether_sample_state_equivalent:
            single_api_groups = (self.session.query(APIGroup)
                                 .join(APIGroup.apis)
                                 .filter(Cluster.type == 'StateEquivalent')
                                 .group_by(APIGroup.id)
                                 .having(func.count(API.id) == 1,  # 确保每个组合只有一个API
                                         func.min(API.id) == api.id)
                                 .all())
            state_equivalent_apis = []  # 所有与api_obj存在状态等价关系的API
            for single_api_group in single_api_groups:
                state_equivalent_cluster = single_api_group.cluster
                state_equivalent_api_groups = state_equivalent_cluster.api_groups
                filtered_api_list = [
                    api_group.apis[0] for api_group in state_equivalent_api_groups
                    if len(api_group.apis) == 1 and api_group.apis[0] != api
                ]
                state_equivalent_apis.extend(filtered_api_list)
            indirect_relevant_errors = []
            for state_equivalent_api in state_equivalent_apis:
                indirect_relevant_errors.extend(state_equivalent_api.history_errors)
            errors["StateIndirectRelevant"] = random.sample(indirect_relevant_errors,
                                                            min(self.error_num_in_context - sum(
                                                                len(v) for v in errors.values()),
                                                                len(indirect_relevant_errors)))

        # 3.如果当前错误数量不足n个, 则从值等价簇内寻找由单个API组成的APIGroup的直接关联错误
        if sum(len(v) for v in errors.values()) < self.error_num_in_context and self.whether_sample_value_equivalent:
            single_api_groups = (self.session.query(APIGroup)
                                 .join(APIGroup.apis)
                                 .filter(Cluster.type == 'ValueEquivalent')
                                 .group_by(APIGroup.id)
                                 .having(func.count(API.id) == 1,  # 确保每个组合只有一个API
                                         func.min(API.id) == api.id)
                                 .all())
            value_equivalent_apis = []  # 所有与api_obj存在值等价关系的API
            for single_api_group in single_api_groups:
                value_equivalent_cluster = single_api_group.cluster
                value_equivalent_api_groups = value_equivalent_cluster.api_groups
                filtered_api_list = [
                    api_group.apis[0] for api_group in value_equivalent_api_groups
                    if len(api_group.apis) == 1 and api_group.apis[0] != api
                ]
                value_equivalent_apis.extend(filtered_api_list)
            indirect_relevant_errors = []
            for value_equivalent_api in value_equivalent_apis:
                indirect_relevant_errors.extend(value_equivalent_api.history_errors)
            errors["ValueIndirectRelevant"] = random.sample(indirect_relevant_errors,
                                                            min(self.error_num_in_context - sum(
                                                                len(v) for v in errors.values()),
                                                                len(indirect_relevant_errors)))

        # 4.如果当前错误数量不足n个, 则继续寻找与api不相关的错误
        if sum(len(v) for v in errors.values()) < self.error_num_in_context:
            # 此处可以考虑通过比较embedding向量来找到在history_errors中与api相对相关的错误
            relevant_errors = [item for value_list in errors.values() for item in value_list]
            relevant_errors_id_list = [relevant_error.id for relevant_error in relevant_errors]
            # 在APIHistoryError中找到没有出现在relevant_errors中的错误
            irrelevant_errors = (
                self.session.query(APIHistoryError)
                .filter(APIHistoryError.id.notin_(relevant_errors_id_list))
                .all()
            )
            errors["Irrelevant"] = random.sample(irrelevant_errors, min(self.error_num_in_context - sum(
                len(v) for v in errors.values()), len(irrelevant_errors)))
        return errors

    def weighted_sample_base(self, candidates, energy):  # 加权抽取基底API
        if not candidates or energy <= 0:
            return []

        # 用于记录候选项在本次抽取过程中的"已被抽取次数"
        draw_count_map = defaultdict(int)

        # 预先计算所有候选项的 error_count
        error_count_map = {}
        for candidate in candidates:
            api = candidate.apis[0]  # candidate 只包含一个 API
            # 计算 candidate 的各种错误
            errors_dict = self.sample_errors(api)
            # 统计该 candidate 除了 Irrelevant 之外的所有错误数
            total_errors = sum(len(v) for k, v in errors_dict.items() if k != "Irrelevant")
            if total_errors == 0:
                total_errors = 0.1
            error_count_map[candidate] = total_errors

        # 抽取 energy 次
        chosen = []
        for i in range(energy):
            # 计算此次抽取时各个候选基底的权重
            weights = []
            for candidate in candidates:
                error_count = error_count_map[candidate]
                draw_count = draw_count_map[candidate]
                weight = error_count / (draw_count + 1)
                weights.append(weight)

            # 使用 random.choices 进行加权随机抽样(一次抽一个)
            chosen_candidate = random.choices(candidates, weights=weights, k=1)[0]
            chosen.append(chosen_candidate)
            # 更新该 candidate 的 draw_count
            draw_count_map[chosen_candidate] += 1
        return chosen

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
            if utils.map_module2lib(api_lib) == 'Unknown':
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
        try:
            is_valid = True
            json_data = json.loads(response)
            print(f"validate_apis() Info - json format is valid")
            # 获取JSON中键为"APIs"的值
            apis = json_data.get("APIs", [])
            for full_api_name in apis:
                if self.validate_api(full_api_name) is False:
                    print(f"validate_apis() Error - {full_api_name} is not a callable API!")
                    is_valid = False
            return is_valid
        except JSONDecodeError as e:
            self.error_log.append("The response data has an invalid JSON format.")
            return False
        except Exception as e:
            self.error_log.append(str(e))
            return False

    def identify_apis(self, api_name_list, whether_supplement_api=True):
        api_list = []  #  ["tensorflow.constant", "tensorflow.nn.softmax_cross_entropy_with_logits"]
        api_name_list = list(set(api_name_list)) # 去除重复的API名称, 保证每个API只被添加一次
        for full_api_name in api_name_list:  # 获取某个API组合中的每个API
            api = self.session.query(API).filter_by(full_name=full_api_name).first()
            if api is None and whether_supplement_api:
                module_name, api_name = full_api_name.rsplit('.', 1)
                api_info = utils.inspect_api_info(module_name, api_name)
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
            api_list.append(api)
        return api_list

    def construct_messages4base(self, base_api_group: APIGroup):
        base_api = base_api_group.apis[0]

        system_prompt = """
(1) Role Definition
You are an AI assistant specialized in deep learning framework APIs (e.g., PyTorch, JAX, MindSpore and Jittor). Your primary task is to help users find equivalent APIs (or API groups) across different deep learning libraries.

(2) Output Format
Your answer must be provided strictly in the following JSON format:
- Code: A string containing the complete, runnable code snippet. The code should call the specified base_api and implement it according to the user's provided parameters and call combinations.
- APIs: A list containing the full names (including module paths) of all APIs in the code snippet that come from the same deep learning library as the base_api. Each API should be listed only once, in any order.
Example:
{
"Code": "import torch; import torch.nn.functional as F; logits = torch.randn(4, 5, requires_grad=True); target = torch.tensor([1, 4, 3, 0]); loss = F.cross_entropy(input=logits, target=target, weight=torch.tensor([1.0, 2.0, 0.5, 0.8, 1.2]), ignore_index=-1, reduction='mean', label_smoothing=0.1); loss.backward(); print(loss.item())",
"APIs": ["torch.nn.functional.F.cross_entropy", "torch.randn", "torch.tensor"],
}
"""

        # Example
        context_query_prompt1 = f"""
        
"""
        context_answer_prompt1 = f"""
        
"""

        # 基底API的详情
        base_api_info_prompt = f"""
- API Name: {base_api.full_name}
- Source Library: {base_api.lib} (version{base_api.version})
- API Signature: {base_api.signature}
{'- Function Description: ' + base_api.description if base_api.description else ''}
{'- Parameters: ' + base_api.parameters if base_api.parameters else ''}
{'- Attributes:' + base_api.attributes if base_api.attributes else ''}
{'- Output:' + base_api.output if base_api.output else ''} 
"""
        # 触发问题的代码调用样例
        issue_examples = [item for value_list in self.sample_errors(base_api).values() for item in value_list]
        issue_examples_prompt = ""
        for count, issue_example in enumerate(issue_examples):
            issue_examples_prompt = issue_examples_prompt + f"""
History Issue Example{count + 1}:
- Issue Title: {issue_example.title}
- Issue Description: {issue_example.description}
- Issue Trigger API: {issue_example.api.signature}
- Issue Code: 
{issue_example.code}         
"""
        # 构建最终查询提示词
        query_prompt = f"""
Example code snippets that trigger the issue:
{issue_examples_prompt}

Information about the API to be called:
{base_api_info_prompt}

Task Requirements:
Please refer to the input values for API parameters and the API call combinations in the examples above, and generate a code snippet that calls {base_api.full_name}.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            # {"role": "user", "content": context_query_prompt1},
            # {"role": "assistant", "content": context_answer_prompt1},
            {"role": "user", "content": query_prompt},
        ]
        return messages

    def query_llm4BaseSeed(self, messages, max_retry=5, model="gpt-4o-mini"):
        attempt_num = 0
        while attempt_num < max_retry:  # 设置最大尝试次数以避免无限循环
            try:
                response = self.llm_client.chat.completions.create(
                    model=model,  # gpt-4o-mini  gpt-3.5-turbo
                    response_format={"type": "json_object"},
                    messages=messages,
                    temperature=0.4,
                )
                response = response.choices[0].message.content
                # 如果response的第一行以"```"开头, 则去掉第一行; 如果response的最后一行以"```"结尾, 则去掉最后一行
                if response.split('\n', 1)[0].startswith("```"):
                    response = response.split('\n', 1)[1]
                if response.split('\n')[-1].endswith("```"):
                    response = '\n'.join(response.split('\n')[:-1])
                messages.append({"role": "assistant", "content": response})
                # 在此处需要检查: 1.响应的数据是否遵循JSON格式; 2.返回的是API的完整函数名(完整函数名 = 模块名.API名)而非函数签名 3.所有的API函数名必须有效(不是虚构的, 也不是被弃用的)
                if self.validate_apis(response):
                    self.error_log = []
                    return json.loads(response).get("Code", None), json.loads(response).get("APIs", None)
                else:
                    attempt_num = attempt_num + 1
                    messages.append({"role": "user", "content": f"The JSON response you generated has the following errors: \n{self.error_log} \n Please try again."})
            except Exception as e:
                attempt_num += 1
                self.session.rollback()  # 回滚在异常中的任何数据库更改
        self.error_log = []  # 清空错误列表
        return None, None

    def generate_seed4base(self, base_api_group: APIGroup, cluster_seed: ClusterTestSeed):
        print("*" * 40 + "generate_seed4base()" + "*" * 40)
        messages = self.construct_messages4base(base_api_group)
        base_seed_code, api_name_list = self.query_llm4BaseSeed(messages)
        if base_seed_code is None or api_name_list is None:
            raise Exception("Failed to generate base seed for base API.")
        api_combination = self.identify_apis(api_name_list)
        base_seed = APITestSeed(
            cluster_seed_id=cluster_seed.id,
            api_group_id=base_api_group.id,
            raw_code=base_seed_code,
        )
        self.session.add(base_seed)
        self.session.flush()

        # 对基底API进行修复
        valid_code = APITestSeedValidator(self.llm_client, self.session, base_seed).validate4seed()
        if valid_code is None:
            raise Exception("Failed to generate base seed for base API.")
        base_seed.valid_code = valid_code
        self.session.flush()
        print(f"generate_seed4base() Success - Base seed generated and validated for {base_api_group.apis[0].full_name}:\n\n{base_seed.valid_code}")
        return base_seed, api_combination

    def construct_messages4twin(self, twin_api_group, base_api_seed, base_api_invoke_combination):
        system_prompt = """
(1) Role Definition: You are an AI assistant specialized in deep learning framework APIs (e.g., PyTorch, JAX, MindSpore and Jittor).
(2) Output Format: Your response must be pure code. Do not include any explanations, comments, or extra content.  
"""
        # API Group的详情
        if len(twin_api_group.apis) == 1:
            twin_api = twin_api_group.apis[0]
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
            for count, twin_api in enumerate(twin_api_group.apis):
                api_group_info_prompt = api_group_info_prompt + f"""
Member{count + 1} of API Group:
- API Name: {twin_api.full_name}
- API Library: {twin_api.lib} (version{twin_api.version})
- API Signature: {twin_api.signature}
{'- Function Description: ' + twin_api.description if twin_api.description else ''}
{'- Parameters: ' + twin_api.parameters if twin_api.parameters else ''}
{'- Attributes:' + twin_api.attributes if twin_api.attributes else ''}
{'- Output:' + twin_api.output if twin_api.output else ''}
"""
        # twin_api_group brief info
        if len(twin_api_group.apis) == 1:
            api_group_brief_info = f"{twin_api_group.apis[0].signature}"
        else:
            api_group_brief_info = f"[{', '.join([api.signature for api in twin_api_group.apis])}]"

        # 背景知识
        base_api = base_api_seed.api_group.apis[0]
        if len(twin_api_group.apis) == 1:
            twin_api = twin_api_group.apis[0]
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
        query_prompt = f"""
Information of the API {'group' if len(twin_api_group.apis) > 1 else ''} to be called:
{api_group_info_prompt}        

Background Knowledge:
{background_knowledge_prompt}

Task Requirements:
Below is a code snippet calling ({base_api.signature}). Please generate a code snippet that replaces ({base_api.full_name}) with {api_group_brief_info}, ensuring that the input parameters remain unchanged. Additionally, ensure that the code snippet you generate declares the same variables as the example code (for instance, if the example code declares an "output" variable to store the API's result, then your generated code should also declare an "output" variable to store the API's result).
{base_api_seed.valid_code}
"""

        # 为每个api_combination中的API和其对应的equivalent_api生成文档提示词
        if not (len(base_api_invoke_combination) == 1 and base_api_invoke_combination[0] == base_api): # 先检查api_combination内是否有且只有base_api这一个API, 如果不是则处理其他API
            api_mapper = {}
            for api in base_api_invoke_combination:
                api_obj_groups = (self.session.query(APIGroup)
                                  .join(APIGroup.apis)
                                  .group_by(APIGroup.id)
                                  .having(func.count(API.id) == 1,  # 确保当前Group内只包含一个API
                                          func.min(API.id) == api.id)  # 确保当前Group内包含的API是api
                                  .all())
                if api_obj_groups is None:
                    api_mapper[api] = None
                    continue

                equivalent_apis = {
                    "ValueEquivalent": [],
                    "StateEquivalent": []
                }
                for api_obj_group in api_obj_groups:
                    if api_obj_group.cluster.type == 'ValueEquivalent':
                        # 在'ValueEquivalent'的Cluster内寻找twin_api所在库的所有等价API
                        value_equivalent_cluster = api_obj_group.cluster
                        value_equivalent_api_groups = value_equivalent_cluster.api_groups
                        for value_equivalent_api_group in value_equivalent_api_groups:
                            if len(value_equivalent_api_group.apis) == 1 and value_equivalent_api_group.apis[0].lib == twin_api_group.apis[0].lib:
                                equivalent_apis["ValueEquivalent"].append(value_equivalent_api_group.apis[0])
                    else:
                        # 在'StateEquivalent'的Cluster内寻找twin_api所在库的所有等价API
                        state_equivalent_cluster = api_obj_group.cluster
                        state_equivalent_api_groups = state_equivalent_cluster.api_groups
                        for state_equivalent_api_group in state_equivalent_api_groups:
                            if len(state_equivalent_api_group.apis) == 1 and state_equivalent_api_group.apis[0].lib == twin_api_group.apis[0].lib:
                                equivalent_apis["StateEquivalent"].append(state_equivalent_api_group.apis[0])
                if len(equivalent_apis["ValueEquivalent"]) > 0:
                    api_mapper[api] = equivalent_apis["ValueEquivalent"][0]
                elif len(equivalent_apis["StateEquivalent"]) > 0:
                    api_mapper[api] = equivalent_apis["StateEquivalent"][0]
                else:
                    api_mapper[api] = None

            api_invoke_combination_brief_info = f"[{', '.join([api.signature for api in base_api_invoke_combination])}]" # base_api_combination brief info

            invoked_apis_prompt = f"""
Additional Background Knowledge:
The above code snippet invoking the following API combination from {base_api.lib} (version {base_api.version}):
{api_invoke_combination_brief_info}
Below are the detailed information about these APIs and their equivalent APIs in {twin_api_group.apis[0].lib} library (if any):
"""
            for api, equivalent_api in api_mapper.items():
                invoked_apis_prompt += f"""
Source API: {api.full_name}
- Library: {api.lib} (version {api.version})
- Signature: {api.signature}
{'- Description: ' + api.description if api.description else ''}
{'- Parameters: ' + api.parameters if api.parameters else ''}
{'- Output: ' + api.output if api.output else ''}
"""
                if equivalent_api:
                    invoked_apis_prompt += f"""
Corresponding equivalent API in target library: {equivalent_api.full_name}
- Library: {equivalent_api.lib} (version {equivalent_api.version})
- Signature: {equivalent_api.signature}
{'- Description: ' + equivalent_api.description if equivalent_api.description else ''}
{'- Parameters: ' + equivalent_api.parameters if equivalent_api.parameters else ''}
{'- Output: ' + equivalent_api.output if equivalent_api.output else ''}
"""
            query_prompt = query_prompt + invoked_apis_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_prompt},
        ]
        return messages

    def query_llm4TwinSeed(self, messages, max_retry=5, model="gpt-4o-mini"):
        attempt_num = 0
        while attempt_num < max_retry:  # 设置最大尝试次数以避免无限循环
            try:
                response = self.llm_client.chat.completions.create(
                    model=model,  # gpt-4o-mini  gpt-3.5-turbo
                    messages=messages,
                    temperature=0.4,
                )
                response = response.choices[0].message.content
                # 如果response的第一行以"```"开头, 则去掉第一行; 如果response的最后一行以"```"结尾, 则去掉最后一行
                if response.split('\n', 1)[0].startswith("```"):
                    response = response.split('\n', 1)[1]
                if response.split('\n')[-1].endswith("```"):
                    response = '\n'.join(response.split('\n')[:-1])
                return response
            except Exception as e:
                attempt_num += 1
                self.session.rollback()  # 回滚在异常中的任何数据库更改
        return None

    def generate_seed4twin(self, twin_api_group: APIGroup, base_api_seed: APITestSeed, base_api_combination, cluster_seed: ClusterTestSeed):
        messages = self.construct_messages4twin(twin_api_group, base_api_seed, base_api_combination)
        twin_seed_code = self.query_llm4TwinSeed(messages)
        if twin_seed_code is None:
            print(f"generate_seed4twin() Error - Failed to generate seed for equivalent API")
            raise Exception("Failed to generate seed for equivalent API.")
        twin_seed = APITestSeed(
            cluster_seed_id=cluster_seed.id,
            api_group_id=twin_api_group.id,
            raw_code=twin_seed_code
        )
        self.session.add(twin_seed)
        self.session.flush()

        # 对raw_code进行修复
        valid_code = APITestSeedValidator(self.llm_client, self.session, twin_seed).validate4seed()
        if valid_code is None:
            print(f"generate_seed4twin() Error - Failed to validate twin seed")
            raise Exception("Failed to generate seed for equivalent API.")
        twin_seed.valid_code = valid_code
        print(f"generate_seed4twin() Success - Twin seed generated and validated for {twin_api_group.apis[0].full_name}:\n\n{twin_seed.valid_code}")
        self.session.flush()
        return twin_seed

    def fuzz_equivalent_cluster(self, cluster: Cluster):
        print(f"fuzz_equivalent_cluster() Info - Fuzzing cluster ID: {cluster.id}, Type: {cluster.type}")
        if cluster.is_tested:
            print(f"fuzz_equivalent_cluster() Info - Cluster {cluster.id} already tested, skipping")
            return
        elif not cluster.api_groups:  # 如果该等价簇没有API组合, 则直接标记为已测试
            print(f"fuzz_equivalent_cluster() Info - Cluster {cluster.id} has no API groups, marking as tested")
            cluster.is_tested = True
            self.session.commit()
            return

        # 先查询该等价簇已经生成了几个种子
        seeds_num = self.session.query(ClusterTestSeed).filter(ClusterTestSeed.cluster_id == cluster.id).count()
        remaining_energy = cluster.energy - seeds_num
        print(f"fuzz_equivalent_cluster() Info - Cluster {cluster.id} has {seeds_num} existing seeds, remaining energy: {remaining_energy}")
        
        # 从cluster中筛选出仅由一个API组成的APIGroup作为候选基底
        candidate_base_api_groups = [api_group for api_group in cluster.api_groups if len(api_group.apis) == 1]
        print(f"fuzz_equivalent_cluster() Info - Found {len(candidate_base_api_groups)} candidate base API groups")
        
        base_api_groups = self.weighted_sample_base(candidate_base_api_groups, remaining_energy)
        while base_api_groups:  # 生成remaining_energy个ClusterTestSeed
            print("=" * 50 + f"Generating Seed({cluster.energy - len(base_api_groups) + 1})" + "=" * 50)
            # try:  # 开始种子的生成
            base_api_group = base_api_groups[0]
            cluster_seed = ClusterTestSeed(
                cluster_id=cluster.id,
                start_test=datetime.utcnow()
            )
            self.session.add(cluster_seed)
            self.session.flush()

            # 生成基底API的测试用例
            base_api_seed, base_api_combination = self.generate_seed4base(base_api_group, cluster_seed)

            # 生成等价簇中其他API的测试用例
            twin_apis_seeds = []
            for count, twin_api_group in enumerate(cluster.api_groups):
                if twin_api_group == base_api_group:
                    continue
                print("*" * 30 + f"generate_seed4twin() - Twin API Group({count})" + "*" * 30)
                twin_api_seed = self.generate_seed4twin(twin_api_group, base_api_seed, base_api_combination, cluster_seed)
                twin_apis_seeds.append(twin_api_seed)
            cluster_seed.end_test = datetime.utcnow()
            self.session.commit()
            print(f"fuzz_equivalent_cluster() Success - Completed seed generation for base API: {base_api_group.apis[0].full_name}")
            base_api_groups.pop(0)
            # except Exception as e:
            #     print(f"fuzz_equivalent_cluster() Error - Error in generating seed for {cluster.type} Cluster({cluster.id}): {e}")
            #     self.session.rollback()
            #     base_api_groups.pop(0)
            #     continue

        # 检查是否所有的种子都已经生成完毕
        seeds_num = self.session.query(ClusterTestSeed).filter_by(cluster_id=cluster.id).count()
        if seeds_num >= cluster.energy:
            cluster.is_tested = True
            self.session.commit()
            print(f"fuzz_equivalent_cluster() Success - Cluster {cluster.id} completed with {seeds_num} seeds")

    def fuzz_value_equivalent_clusters(self):
        print("=" * 75 +"fuzz_value_equivalent_clusters()" + "=" * 75)
        value_equivalent_clusters = self.session.query(Cluster).filter_by(type='ValueEquivalent').all()
        untested_clusters = self.session.query(Cluster).filter_by(is_tested=False, type='ValueEquivalent').all()
        
        while untested_clusters:
            print("-" * 70 + f"Fuzzing Value Equivalent Clusters: {len(untested_clusters)}/ {len(value_equivalent_clusters)}" + "-" * 70)
            untested_cluster = untested_clusters[0]
            self.fuzz_equivalent_cluster(untested_cluster)
            untested_clusters = self.session.query(Cluster).filter_by(is_tested=False, type='ValueEquivalent').all()
            
        print(f"fuzz_value_equivalent_clusters() Success - All value equivalent clusters fuzzing completed")

    def fuzz_state_equivalent_clusters(self):
        print("=" * 75 + "fuzz_state_equivalent_clusters()" + "=" * 75)
        state_equivalent_clusters = self.session.query(Cluster).filter_by(type='StateEquivalent').all()
        untested_clusters = self.session.query(Cluster).filter_by(is_tested=False, type='StateEquivalent').all()
        
        while untested_clusters:
            print("-" * 70 + f"Fuzzing State Equivalent Clusters: {len(untested_clusters)}/ {len(state_equivalent_clusters)}" + "-" * 70)
            untested_cluster = untested_clusters[0]
            self.fuzz_equivalent_cluster(untested_cluster)
            untested_clusters = self.session.query(Cluster).filter_by(is_tested=False, type='StateEquivalent').all()
            
        print(f"fuzz_state_equivalent_clusters() Success - All state equivalent clusters fuzzing completed")


if __name__ == '__main__':
    session = utils.get_session()
    llm_client = utils.get_llm_client(llm='gpt4o-mini')
    rag_client = utils.get_llm_client(llm='gpt4o-mini-with-rag')
    fuzzer = Fuzzer(session, llm_client, rag_client)
    
    fuzzer.fuzz_value_equivalent_clusters()
    fuzzer.fuzz_state_equivalent_clusters()
    session.close()