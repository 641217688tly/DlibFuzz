import importlib
import json
from json import JSONDecodeError
from sqlalchemy import func
import utils
from fuzzer.validator import APITestSeedValidator
from orm import *
import random
from collections import defaultdict
import threading
import concurrent.futures
import os
from typing import List
import argparse
from datetime import datetime


class Fuzzer:  # 以Cluster为单位生成测试种子
    def __init__(self, cluster: Cluster, session, llm_client, error_num_in_context=6, whether_sample_state_equivalent=True, whether_sample_value_equivalent=True):
        self.cluster = cluster
        self.session = session
        self.llm_client = llm_client
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
            if not isinstance(full_api_name, str) or not full_api_name.strip():
                return False

            parts = full_api_name.split('.')
            # 首段别名映射，与 utils.validate_api_existence 保持一致
            api_lib = utils.map_alias2module(parts[0])
            current_module_obj = importlib.import_module(api_lib)

            # 逐段解析: 优先作为模块导入，失败则回退 getattr
            accumulated = [api_lib]
            for sub in parts[1:]:
                candidate_module = '.'.join(accumulated + [sub])
                try:
                    current_module_obj = importlib.import_module(candidate_module)
                    accumulated.append(sub)
                    continue
                except Exception:
                    attr = getattr(current_module_obj, sub, None)
                    if attr is None:
                        return False
                    current_module_obj = attr
                    accumulated.append(sub)
            return True
        except Exception as e:
            self.error_log.append(f"{full_api_name} is not a valid API belonging to the deep learning libraries. Error: {e}")
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
You are an AI assistant specialized in deep learning framework APIs (e.g., PyTorch, JAX, MindSpore and Jittor).

(2) Output Format
Your answer must be provided strictly in the following JSON format:
- Code: A string containing the complete, runnable code snippet. The code should call the specified base_api and implement it according to the user's provided parameters and call combinations.
- APIs: A list containing the full names (including module paths) of all APIs in the code snippet that come from the same deep learning library as the base_api. Each API should be listed only once, in any order.
"""

        # Example
        context_query_prompt1 = f"""
Code snippets that trigger the issue:
History Issue Example1:
- Issue Title: jax.jit crashes only as a class method (calling jax.scipy.linalg.lu_solve)
- Issue Description: The issue occurs in a Linux environment using Python 3.10.12 and JAX version 0.4.13 with a NVIDIA A100 GPU. The user attempts to solve a large set of LU systems using the `solve_jit` method defined as a class method with a static argument. When calling `solver.solve_jit(rhs)`, the operation results in a crash (kernel dies or segmentation fault), while the non-jit method `solver.solve(rhs)` works correctly.
- Issue Trigger API: jax.jit
- Issue Code: 
class Solver:
  def __init__(self, lu):
    self.lu = lu
  def solve(self, rhs_0):
    return jax.vmap(jax.scipy.linalg.lu_solve)(self.lu, rhs_0)

  @partial(jax.jit, static_argnums=(0,))
  def solve_jit(self, rhs_0):
    return jax.vmap(jax.scipy.linalg.lu_solve)(self.lu, rhs_0)

lu = jax.vmap(jax.scipy.linalg.lu_factor)(lhs)

solver = Solver(lu)
sol = solver.solve(rhs)
sol = solver.solve_jit(rhs)   

History Issue Example2:
- Issue Title: ArgInfo.donated reports donated status for wrong argument
- Issue Description: The issue occurred in an environment using TPU. The user defined a function and applied jax.jit with donate_argnums set to 1. After compiling and inspecting the function's argument information, it incorrectly reported that only one of the input arguments was donated. When the function was called, it resulted in a RuntimeError indicating that an array had been deleted, despite the expectation that the other argument was not donated.
- Issue Trigger API: jax.jit
- Issue Code: 
def fn(x, y):
  return x, y

fn = jax.jit(fn, donate_argnums=1)

x = {{'A': 1.0, 'B': 2.0}}
y = 3.0
x = jax.tree_map(lambda x: jax.device_put(x, jax.local_devices()[0]), x)
y = jax.tree_map(lambda x: jax.device_put(x, jax.local_devices()[0]), y)

fn = fn.lower(x, y)
fn = fn.compile()
print(fn.args_info) # claims only x['B'] is donated

fn(x, y)

print(x) # x wasn't donated at all
print(y) # y was donated (as expected) -> RuntimeError: Array has been deleted.

History Issue Example3:
- Issue Title: JIT donate_argnums slows down execution
- Issue Description: The user is working on a reinforcement learning project on Ubuntu 20.04 with a GPU. They are trying to optimize a replay buffer inside a jitted function using jax.lax.fori_loop. After implementing donate_argnums in their jitted function, they noticed that the training process is significantly slower than expected, despite the assumption that it would improve performance.
- Issue Trigger API: jax.jit
- Issue Code: 
def loop_fn(_, carry):
    loop_state, replay_state = carry
    ...
     # Some modifications to the loop_state and replay_state
    return new_loop_state, new_replay_state

loop_fn = jit(loop_fn)
for i in range(...): 
    loop_state, replay_state = jax.lax.fori_loop(0, FLAGS.log_frequency, loop_fn, (loop_state, replay_state))

def fori_loop_fn(loop_state, replay_state):
    return jax.lax.fori_loop(0, FLAGS.log_frequency, loop_fn, (loop_state, replay_state))

fori_loop_fn = jit(fori_loop_fn, donate_argnums=(1,))
for i in range(...):
    loop_state, replay_state = fori_loop_fn(loop_state, replay_state)

History Issue Example4:
- Issue Title: Jax' transfer guard and XLA-CPU
- Issue Description: The issue occurs in an environment using JAX with a TPU accelerator. The user attempts to transfer numpy arrays to the XLA-CPU device using JAX's transfer guard, expecting it to trigger for host-to-device transfers. However, the transfer guard does not trigger for numpy to XLA-CPU transfers, while it also fails to trigger for XLA-CPU to numpy transfers. This inconsistency in behavior is the main concern.
- Issue Trigger API: jax.jit
- Issue Code: 
jax.jit(jax.random.PRNGKey, backend='cpu')(np.array(0))
np.array(jax.device_put(0, device=jax.devices('cpu')[0]))

History Issue Example5:
- Issue Title: Complex max/min fail on shared gpu device arrays
- Issue Description: The issue occurred in an environment using jax-0.4.13 and jaxlib-0.4.13+cuda12.cudnn89 on NVIDIA A100 GPUs. The user attempted to compute the maximum of a complex array distributed across multiple GPUs using jax.jit and jax.device_put. This operation resulted in an internal error related to the handling of complex arrays, leading to a failure in the computation.
- Issue Trigger API: jax.jit
- Issue Code: 
import jax
import jax.numpy as jnp

x = jnp.ones(128, dtype=jnp.complex64)
sharding = jax.sharding.PositionalSharding(jax.devices())
x = jax.device_put(x, sharding)
jax.debug.visualize_array_sharding(x)
jax.jit(jnp.max)(x)
#jax.jit(jnp.min)(x)

History Issue Example6:
- Issue Title: Crash in Metal plugin if bfloat16 constant is present
- Issue Description: The issue occurred on an Apple M1 Pro with 32.00 GB of system memory and a max cache size of 10.67 GB. The user attempted to execute a JAX function that included a bfloat16 constant. As a result of this operation, an assertion failure occurred in the Metal plugin, indicating that the buffer was not large enough, leading to an abort trap.
- Issue Trigger API: jax.jit
- Issue Code: 
jax.jit(lambda: jnp.exp(jnp.bfloat16(7)))()

Information about the API to be called:
- API Name: jax.jit
- Source Library: JAX (version 0.4.33)
- API Signature: jax.jit(fun,in_shardings=UnspecifiedValue,out_shardings=UnspecifiedValue,static_argnums=None,static_argnames=None,donate_argnums=None,donate_argnames=None,keep_unused=False,device=None,backend=None,inline=False,abstracted_axes=None,compiler_options=None)
- Function Description: Sets up fun for just-in-time compilation with XLA
- Output: pjit.JitWrapped

Task Requirements:
1. Your task is to generate a code snippet that is likely to reveal potential bugs in jax.jit, by mining and learning from the input parameters and API call combinations shown in the above issue examples.
2. Output variable naming rules:
   - You must assign the result or affected variables from jax.jit to a variable named "output".
   - If jax.jit returns multiple values, assign them as a tuple to the "output" variable.
   - If jax.jit performs in-place operations, assign the processed input(s) to the "output" variable.
3. Code complexity requirements:
   - Keep the code simple and straightforward. Avoid defining complex functions or classes unless absolutely necessary.
   - Prefer direct API calls and simple variable assignments over complex control structures.
   - If functions are needed, keep them short and focused on a single purpose.
4. The code should be complete and executable. You are only allowed to use APIs from the JAX (version 0.4.33) library and common utility libraries such as numpy, random, math, and built-in Python functions. Do not use APIs from any other deep learning frameworks or third-party libraries.
"""
        context_answer_prompt1 = f"""
{{
  "Code": "import jax\\nimport jax.numpy as jnp\\nfrom jax import random\\nimport numpy as np\\n\\n# 创建复数数组并进行分片处理\\nx = jnp.ones(128, dtype=jnp.complex64)\\nsharding = jax.sharding.PositionalSharding(jax.devices())\\nx = jax.device_put(x, sharding)\\n\\n# 测试复数数组的最大值计算\\noutput = jax.jit(jnp.max)(x)\\nprint(output)",
  "APIs": [
    "jax.jit",
    "jax.numpy.max",
    "jax.device_put",
    "jax.sharding.PositionalSharding",
    "jax.numpy.ones",
    "jax.devices"
  ]
}}
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
- Issue Trigger API: {issue_example.api.full_name}
- Issue Code: 
{issue_example.code}         
"""
        # 构建最终查询提示词
        query_prompt = f"""
Code snippets that trigger the issue:
{issue_examples_prompt}

Information about the API to be called:
{base_api_info_prompt}

Task Requirements:
1. Your task is to generate a code snippet that is likely to reveal potential bugs in {base_api.full_name}, by mining and learning from the input parameters and API call combinations shown in the above issue examples.
2. Output variable naming rules:
   - You must assign the result or affected variables from {base_api.full_name} to a variable named "output".
   - If {base_api.full_name} returns multiple values, assign them as a tuple to the "output" variable.
   - If {base_api.full_name} performs in-place operations, assign the processed input(s) to the "output" variable.
3. Code complexity requirements:
   - Keep the code simple and straightforward. Avoid defining complex functions or classes unless absolutely necessary.
   - Prefer direct API calls and simple variable assignments over complex control structures.
   - If functions are needed, keep them short and focused on a single purpose.
4. The code should be complete and executable. You are only allowed to use APIs from the {base_api.lib} (version {base_api.version}) library and common utility libraries such as numpy, random, math, and built-in Python functions. Do not use APIs from any other deep learning frameworks or third-party libraries.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context_query_prompt1},
            {"role": "assistant", "content": context_answer_prompt1},
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
                print(f"query_llm4BaseSeed() Error - Attempt {attempt_num} failed with error: {e}")
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
            start_time=datetime.utcnow()
        )
        self.session.add(base_seed)
        self.session.flush()

        # 对基底API进行修复
        valid_code = APITestSeedValidator(self.llm_client, self.session, base_seed).validate4seed()
        if valid_code is None:
            raise Exception("Failed to generate base seed for base API.")
        base_seed.valid_code = valid_code
        base_seed.end_time = datetime.utcnow()
        self.session.flush()
        print(f"generate_seed4base() Success - Base seed generated and validated for {base_api_group.apis[0].full_name}:\n\n{base_seed.valid_code}")
        return base_seed, api_combination

    def construct_messages4twin(self, twin_api_group, base_api_seed, base_api_invoke_combination):
        # System提示词
        system_prompt = """
(1) Role Definition: You are an AI assistant specialized in deep learning framework APIs (e.g., PyTorch, JAX, MindSpore and Jittor).
(2) Output Format: Your response must be pure code. Do not include any explanations, comments, or extra content.  
"""
        # twin_api_group brief info
        if len(twin_api_group.apis) == 1:
            api_group_brief_info = f"{twin_api_group.apis[0].full_name}"
        else:
            api_group_brief_info = f"[{', '.join([api.full_name for api in twin_api_group.apis])}]"

        # Twin API Group的详情
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
Member{count + 1} of API Group {api_group_brief_info}:
- API Name: {twin_api.full_name}
- API Library: {twin_api.lib} (version{twin_api.version})
- API Signature: {twin_api.signature}
{'- Function Description: ' + twin_api.description if twin_api.description else ''}
{'- Parameters: ' + twin_api.parameters if twin_api.parameters else ''}
{'- Attributes:' + twin_api.attributes if twin_api.attributes else ''}
{'- Output:' + twin_api.output if twin_api.output else ''}
"""

        # 背景知识
        base_api = base_api_seed.api_group.apis[0]
        api_invoke_combination_brief_info = f"[{', '.join([api.full_name for api in base_api_invoke_combination])}]"  # base_api_combination brief info
        background_knowledge_prompt = f"""
The above code snippet invoking the following API combination from {base_api.lib} (version {base_api.version}): {api_invoke_combination_brief_info}
Below are the detailed information about these APIs and their equivalent APIs in {twin_api_group.apis[0].lib} library (if any):
"""
        if len(twin_api_group.apis) == 1:
            twin_api = twin_api_group.apis[0]
            background_knowledge_prompt = background_knowledge_prompt + f"""
The detail of API ({base_api.full_name}) is as follows:
- API Name: {base_api.full_name}
- API Library: {base_api.lib} (version{base_api.version})
- API Signature: {base_api.signature}
{'- Function Description: ' + base_api.description if base_api.description else ''}
{'- Parameters: ' + base_api.parameters if base_api.parameters else ''}
{'- Attributes:' + base_api.attributes if base_api.attributes else ''}
{'- Output:' + base_api.output if base_api.output else ''}

The API ({twin_api.full_name}) from library {twin_api.lib}(v{twin_api.version}) has the similar function as the API ({base_api.full_name}) from library {base_api.lib}(v{base_api.version}).
The detail of API ({api_group_brief_info}) is as follows:
{api_group_info_prompt}
"""
        else:
            background_knowledge_prompt = background_knowledge_prompt + f"""
The detail of API ({base_api.full_name}) is as follows:
- API Name: {base_api.full_name}
- API Library: {base_api.lib} (version{base_api.version})
- API Signature: {base_api.signature}
{'- Function Description: ' + base_api.description if base_api.description else ''}
{'- Parameters: ' + base_api.parameters if base_api.parameters else ''}
{'- Attributes:' + base_api.attributes if base_api.attributes else ''}
{'- Output:' + base_api.output if base_api.output else ''}

By combining the APIs in {api_group_brief_info}, it can achieve the similar functionality as the API {base_api.full_name} from library {base_api.lib}(v{base_api.version}).
The detail of API ({api_group_brief_info}) is as follows:
{api_group_info_prompt}
"""

            # 为每个api_combination中的API和其对应的equivalent_api生成文档提示词
            base_api_invoke_combination = [api for api in base_api_invoke_combination if api != base_api]  # 从base_api_invoke_combination中删除base_api
            if (len(base_api_invoke_combination) > 0):  # 先检查api_combination内是否有且只有base_api这一个API, 如果不是则处理其他API
                # 寻找base_api_invoke_combination中每个API在等价簇中的等价API
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
                    elif len(equivalent_apis["StateEquivalent"]) > 0 and self.cluster.type == 'StateEquivalent':
                        api_mapper[api] = equivalent_apis["StateEquivalent"][0]
                    else:
                        api_mapper[api] = None

                # 为每个API和其对应的equivalent_api生成文档提示词
                for api, equivalent_api in api_mapper.items():
                    background_knowledge_prompt += f"""
The detail of API ({api.full_name}) is as follows:
- Library: {api.lib} (version {api.version})
- Signature: {api.signature}
{'- Description: ' + api.description if api.description else ''}
{'- Parameters: ' + api.parameters if api.parameters else ''}
{'- Output: ' + api.output if api.output else ''}
"""
                    if equivalent_api:
                        background_knowledge_prompt += f"""
The API ({equivalent_api.full_name}) from library {equivalent_api.lib}(v{equivalent_api.version}) has the similar function as the API ({api.full_name}) from library {api.lib}(v{api.version}).
The detail of API ({equivalent_api.signature}) is as follows:
- Library: {equivalent_api.lib} (version {equivalent_api.version})
- Signature: {equivalent_api.signature}
{'- Description: ' + equivalent_api.description if equivalent_api.description else ''}
{'- Parameters: ' + equivalent_api.parameters if equivalent_api.parameters else ''}
{'- Output: ' + equivalent_api.output if equivalent_api.output else ''}
"""

        # 构建最终提示词
        query_prompt = f"""
Code Snippet: 
Below is a code snippet that calls the API combination {api_invoke_combination_brief_info} from {base_api.lib} (version {base_api.version}).
{base_api_seed.valid_code}
        
Background Knowledge:
{background_knowledge_prompt}

Task Requirements:
1. Translation Task: Your task is to translate the above code snippet, which calls the API combination {api_invoke_combination_brief_info} from {base_api.lib} (version {base_api.version}), into an equivalent code snippet using the equivalent APIs from {twin_api_group.apis[0].lib} (version {twin_api_group.apis[0].version}).
2. Consistency Requirements: The translated code snippet must maintain consistency with the original code in the following aspects:
   - API call order must remain the same
   - API parameters must remain unchanged
   - API return values handling must be consistent
   - Variable names must be preserved
   - Input parameters must remain unchanged
3. Code complexity requirements:
   - Keep the translated code simple and straightforward, avoiding complex functions or classes unless they exist in the original code.
   - Maintain the same level of complexity as the original code - do not add unnecessary complexity.
   - Prefer direct API calls and simple variable assignments over complex control structures.
4. Output Variable Naming Rules: 
   In the original code, the output variable naming follows this rule:
   - The result or affected variables from {base_api.full_name} are assigned to a variable named "output"
   In the translated code, the equivalent API {'Group' if len(twin_api_group.apis) > 1 else ''} {api_group_brief_info} must follow the same variable naming rule and assign results to a variable named "output".
"""
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
            raw_code=twin_seed_code,
            start_time=datetime.utcnow()
        )
        self.session.add(twin_seed)
        self.session.flush()

        # 对raw_code进行修复
        valid_code = APITestSeedValidator(self.llm_client, self.session, twin_seed).validate4seed()
        if valid_code is None:
            print(f"generate_seed4twin() Error - Failed to validate twin seed")
            raise Exception("Failed to generate seed for equivalent API.")
        twin_seed.valid_code = valid_code
        twin_seed.end_time = datetime.utcnow()
        self.session.flush()
        print(f"generate_seed4twin() Success - Twin seed generated and validated for {twin_api_group.apis[0].full_name}:\n\n{twin_seed.valid_code}")
        return twin_seed

    def fuzz_equivalent_cluster(self):
        print(f"fuzz_equivalent_cluster() Info - Fuzzing cluster ID: {self.cluster.id}, Type: {self.cluster.type}")
        if self.cluster.is_tested:
            print(f"fuzz_equivalent_cluster() Info - Cluster {self.cluster.id} already tested, skipping")
            return
        elif not self.cluster.api_groups:  # 如果该等价簇没有API组合, 则直接标记为已测试
            print(f"fuzz_equivalent_cluster() Info - Cluster {self.cluster.id} has no API groups, marking as tested")
            self.cluster.is_tested = True
            self.session.commit()
            return

        # 先查询该等价簇已经生成了几个种子
        seeds_num = self.session.query(ClusterTestSeed).filter(ClusterTestSeed.cluster_id == self.cluster.id).count()
        remaining_energy = self.cluster.energy - seeds_num
        print(f"fuzz_equivalent_cluster() Info - Cluster {self.cluster.id} has {seeds_num} existing seeds, remaining energy: {remaining_energy}")
        
        # 从cluster中筛选出仅由一个API组成的APIGroup作为候选基底
        candidate_base_api_groups = [api_group for api_group in self.cluster.api_groups if len(api_group.apis) == 1]
        print(f"fuzz_equivalent_cluster() Info - Found {len(candidate_base_api_groups)} candidate base API groups")
        
        base_api_groups = self.weighted_sample_base(candidate_base_api_groups, remaining_energy)
        while base_api_groups:  # 生成remaining_energy个ClusterTestSeed
            print("=" * 50 + f"Generating Seed({self.cluster.energy - len(base_api_groups) + 1})" + "=" * 50)
            cluster_seed = None
            try:  # 开始种子的生成
                base_api_group = base_api_groups[0]
                cluster_seed = ClusterTestSeed(
                    cluster_id=self.cluster.id,
                )
                self.session.add(cluster_seed)
                self.session.flush()
                # 立即提交cluster_seed以确保其在数据库中存在
                self.session.commit()

                # 生成基底API的测试用例
                base_api_seed, base_api_combination = self.generate_seed4base(base_api_group, cluster_seed)

                # 生成等价簇中其他API的测试用例
                twin_apis_seeds = []
                for count, twin_api_group in enumerate(self.cluster.api_groups):
                    if twin_api_group == base_api_group:
                        continue
                    print("*" * 30 + f"generate_seed4twin() - Twin API Group({count})" + "*" * 30)
                    twin_api_seed = self.generate_seed4twin(twin_api_group, base_api_seed, base_api_combination, cluster_seed)
                    twin_apis_seeds.append(twin_api_seed)
                
                # 计算并设置ClusterTestSeed的总耗时
                total_duration = 0.0
                for api_seed in cluster_seed.api_seeds:
                    if api_seed.start_time and api_seed.end_time:
                        total_duration += (api_seed.end_time - api_seed.start_time).total_seconds()
                cluster_seed.duration_time = total_duration
                self.session.commit()
                print(f"fuzz_equivalent_cluster() Success - Completed seed generation for base API: {base_api_group.apis[0].full_name}")
                base_api_groups.pop(0)
            except Exception as e:
                print(f"fuzz_equivalent_cluster() Error - Error in generating seed for {self.cluster.type} Cluster({self.cluster.id}): {e}")
                try:
                    self.session.rollback()
                    # 如果cluster_seed已经创建但发生错误，删除已创建的cluster_seed
                    if cluster_seed and cluster_seed.id:
                        self.session.query(ClusterTestSeed).filter_by(id=cluster_seed.id).delete()
                        self.session.commit()
                except Exception as rollback_error:
                    print(f"fuzz_equivalent_cluster() Error - Error during rollback: {rollback_error}")
                    # 重新创建session以确保数据库连接正常
                    self.session.close()
                    self.session = utils.get_session()
                    # 重新查询cluster对象
                    self.cluster = self.session.query(Cluster).filter_by(id=self.cluster.id).first()
                base_api_groups.pop(0)
                continue

        # 检查是否所有的种子都已经生成完毕
        seeds_num = self.session.query(ClusterTestSeed).filter_by(cluster_id=self.cluster.id).count()
        if seeds_num >= self.cluster.energy:
            self.cluster.is_tested = True
            self.session.commit()
            print(f"fuzz_equivalent_cluster() Success - Cluster {self.cluster.id} completed with {seeds_num} seeds")


def process_clusters_batch(cluster_ids: List[int], thread_id: int, llm_client, progress_lock=None, progress_counter=None, total_clusters=None):
    """处理一批cluster的工作函数，在单独的线程中运行"""
    # 为每个线程创建独立的数据库会话
    thread_session = utils.get_session()
    
    print(f"线程 {thread_id} 开始处理 {len(cluster_ids)} 个 Clusters")
    
    try:
        for i, cluster_id in enumerate(cluster_ids):
            try:
                # 重新查询cluster以确保数据是最新的
                cluster = thread_session.query(Cluster).filter_by(id=cluster_id).first()
                if cluster and not cluster.is_tested:
                    print(f"线程 {thread_id} 正在处理 Cluster {cluster_id} ({i+1}/{len(cluster_ids)})")
                    # 为每个cluster创建独立的Fuzzer实例
                    fuzzer = Fuzzer(cluster, thread_session, llm_client)
                    fuzzer.fuzz_equivalent_cluster()
                    
                    # 更新全局进度
                    if progress_lock and progress_counter is not None and total_clusters:
                        with progress_lock:
                            progress_counter[0] += 1
                            completed = progress_counter[0]
                            print(f"总体进度: {completed}/{total_clusters} ({completed/total_clusters*100:.1f}%)")
                else:
                    print(f"线程 {thread_id} 跳过 Cluster {cluster_id} (已测试或不存在)")
            except Exception as e:
                print(f"线程 {thread_id} 处理 Cluster {cluster_id} 时出错: {e}")
                thread_session.rollback()
                continue
                
    finally:
        thread_session.close()
        print(f"线程 {thread_id} 完成处理")


def split_clusters_for_threads(clusters: List, max_workers: int) -> List[List[int]]:
    """将clusters分配给不同的线程"""
    cluster_ids = [cluster.id for cluster in clusters if not cluster.is_tested]
    
    if not cluster_ids:
        return []
    
    # 计算每个线程应该处理的cluster数量
    clusters_per_thread = len(cluster_ids) // max_workers
    remainder = len(cluster_ids) % max_workers
    
    batches = []
    start_idx = 0
    
    for i in range(max_workers):
        # 为前remainder个线程分配额外的一个cluster
        batch_size = clusters_per_thread + (1 if i < remainder else 0)
        if batch_size > 0:
            end_idx = start_idx + batch_size
            batches.append(cluster_ids[start_idx:end_idx])
            start_idx = end_idx
    
    return batches


def fuzz_value_equivalent_clusters(session, llm_client, max_workers=None):
    """对所有值等价簇进行模糊测试 - 支持多线程"""
    print("=" * 75 +"fuzz_value_equivalent_clusters() - 多线程版本" + "=" * 75)
    
    # 获取系统最大线程数
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)  # 限制最大线程数
    
    # 获取所有未测试的值等价簇
    untested_clusters = session.query(Cluster).filter_by(is_tested=False, type='ValueEquivalent').all()
    total_clusters = session.query(Cluster).filter_by(type='ValueEquivalent').count()
    
    print(f"发现 {len(untested_clusters)}/{total_clusters} 个未测试的值等价簇")
    print(f"使用 {max_workers} 个线程进行并行处理")
    
    if not untested_clusters:
        print("所有值等价簇已完成测试")
        return
    
    # 将clusters分配给不同线程
    cluster_batches = split_clusters_for_threads(untested_clusters, max_workers)
    
    if not cluster_batches:
        print("没有需要处理的clusters")
        return
    
    print(f"将 {len(untested_clusters)} 个clusters分配给 {len(cluster_batches)} 个线程")
    for i, batch in enumerate(cluster_batches):
        print(f"线程 {i} 将处理 {len(batch)} 个clusters")
    
    # 创建进度跟踪
    progress_lock = threading.Lock()
    progress_counter = [0]  # 使用列表来创建可变对象
    
    # 使用线程池执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cluster_batches)) as executor:
        futures = []
        for i, batch in enumerate(cluster_batches):
            future = executor.submit(process_clusters_batch, batch, i, llm_client, 
                                   progress_lock, progress_counter, len(untested_clusters))
            futures.append(future)
        
        # 等待所有线程完成
        concurrent.futures.wait(futures)
        
        # 检查是否有异常
        for i, future in enumerate(futures):
            try:
                future.result()
            except Exception as e:
                print(f"线程 {i} 执行过程中出现异常: {e}")
    
    print("fuzz_value_equivalent_clusters() Success - 所有值等价簇模糊测试完成")


def fuzz_state_equivalent_clusters(session, llm_client, max_workers=None):
    """对所有状态等价簇进行模糊测试 - 支持多线程"""
    print("=" * 75 + "fuzz_state_equivalent_clusters() - 多线程版本" + "=" * 75)
    
    # 获取系统最大线程数
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() or 1) + 4)  # 限制最大线程数
    
    # 获取所有未测试的状态等价簇
    untested_clusters = session.query(Cluster).filter_by(is_tested=False, type='StateEquivalent').all()
    total_clusters = session.query(Cluster).filter_by(type='StateEquivalent').count()
    
    print(f"发现 {len(untested_clusters)}/{total_clusters} 个未测试的状态等价簇")
    print(f"使用 {max_workers} 个线程进行并行处理")
    
    if not untested_clusters:
        print("所有状态等价簇已完成测试")
        return
    
    # 将clusters分配给不同线程
    cluster_batches = split_clusters_for_threads(untested_clusters, max_workers)
    
    if not cluster_batches:
        print("没有需要处理的clusters")
        return
    
    print(f"将 {len(untested_clusters)} 个clusters分配给 {len(cluster_batches)} 个线程")
    for i, batch in enumerate(cluster_batches):
        print(f"线程 {i} 将处理 {len(batch)} 个clusters")
    
    # 创建进度跟踪
    progress_lock = threading.Lock()
    progress_counter = [0]  # 使用列表来创建可变对象
    
    # 使用线程池执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cluster_batches)) as executor:
        futures = []
        for i, batch in enumerate(cluster_batches):
            future = executor.submit(process_clusters_batch, batch, i, llm_client,
                                   progress_lock, progress_counter, len(untested_clusters))
            futures.append(future)
        
        # 等待所有线程完成
        concurrent.futures.wait(futures)
        
        # 检查是否有异常
        for i, future in enumerate(futures):
            try:
                future.result()
            except Exception as e:
                print(f"线程 {i} 执行过程中出现异常: {e}")
    
    print("fuzz_state_equivalent_clusters() Success - 所有状态等价簇模糊测试完成")


# 保留原有的单线程版本作为备用
def fuzz_value_equivalent_clusters_single_thread(session, llm_client):
    """对所有值等价簇进行模糊测试 - 单线程版本"""
    print("=" * 75 +"fuzz_value_equivalent_clusters() - 单线程版本" + "=" * 75)
    value_equivalent_clusters = session.query(Cluster).filter_by(type='ValueEquivalent').all()
    untested_clusters = session.query(Cluster).filter_by(is_tested=False, type='ValueEquivalent').all()
    
    while untested_clusters:
        print("-" * 70 + f"Fuzzing Value Equivalent Clusters: {len(untested_clusters)}/ {len(value_equivalent_clusters)}" + "-" * 70)
        untested_cluster = untested_clusters[0]
        fuzzer = Fuzzer(untested_cluster, session, llm_client)
        fuzzer.fuzz_equivalent_cluster()
        untested_clusters = session.query(Cluster).filter_by(is_tested=False, type='ValueEquivalent').all()
        
    print(f"fuzz_value_equivalent_clusters() Success - All value equivalent clusters fuzzing completed")


def fuzz_state_equivalent_clusters_single_thread(session, llm_client):
    """对所有状态等价簇进行模糊测试 - 单线程版本"""
    print("=" * 75 + "fuzz_state_equivalent_clusters() - 单线程版本" + "=" * 75)
    state_equivalent_clusters = session.query(Cluster).filter_by(type='StateEquivalent').all()
    untested_clusters = session.query(Cluster).filter_by(is_tested=False, type='StateEquivalent').all()

    while untested_clusters:
        print("-" * 70 + f"Fuzzing State Equivalent Clusters: {len(untested_clusters)}/ {len(state_equivalent_clusters)}" + "-" * 70)
        untested_cluster = untested_clusters[0]
        fuzzer = Fuzzer(untested_cluster, session, llm_client)
        fuzzer.fuzz_equivalent_cluster()
        untested_clusters = session.query(Cluster).filter_by(is_tested=False, type='StateEquivalent').all()
        
    print(f"fuzz_state_equivalent_clusters() Success - All state equivalent clusters fuzzing completed")


def clear_all_seeds():
    """
    删除数据库中所有ClusterTestSeed和APITestSeed, 并将所有Cluster的is_tested属性设置为False
    """
    session = utils.get_session()
    try:
        session.query(APITestSeed).delete() # 删除所有APITestSeed
        session.query(ClusterTestSeed).delete() # 删除所有ClusterTestSeed
        session.query(Cluster).update({Cluster.is_tested: False}) # 将所有Cluster的is_tested属性设置为False
        session.commit()
        print("cleare_all_seeds() Success - 所有种子已删除，集群状态已重置")
    except Exception as e:
        # 回滚事务
        session.rollback()
        print(f"cleare_all_seeds() Error - 删除种子时出错: {e}")
        raise e

if __name__ == '__main__':
    session = utils.get_session()
    # clear_all_seeds()
    llm_client = utils.get_llm_client(llm='bianxie')
    #fuzz_value_equivalent_clusters_single_thread(session, llm_client)
    #fuzz_state_equivalent_clusters_single_thread(session, llm_client)
    fuzz_value_equivalent_clusters(session, llm_client, max_workers=16)
    #fuzz_state_equivalent_clusters(session, llm_client, max_workers=16)