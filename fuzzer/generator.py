from sqlalchemy import func
import utils
from fuzzer.validator import APITestSeedValidator
from orm import *
import random
from collections import defaultdict


class Fuzzer:  # 以Cluster为单位生成测试种子
    def __init__(self, session, llm_client, error_num_in_context=6,
                 whether_sample_state_equivalent=True, whether_sample_value_equivalent=True):
        self.session = session
        self.llm_client = llm_client
        self.error_num_in_context = error_num_in_context
        self.whether_sample_state_equivalent = whether_sample_state_equivalent
        self.whether_sample_value_equivalent = whether_sample_value_equivalent

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
            errors["DirectRelevant"] = random.sample(direct_relevant_errors,
                                                     min(self.error_num_in_context, len(direct_relevant_errors)))
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
                    sublist[0] for sublist in state_equivalent_api_groups
                    if len(sublist) == 1 and sublist[0] not in single_api_group
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
                    sublist[0] for sublist in value_equivalent_api_groups
                    if len(sublist) == 1 and sublist[0] not in single_api_group
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
        """
           1. 根据 self.sample_errors(candidate.apis[0]) 得到每个候选项的"错误数"来提高它的抽中概率
           2. 同时，如果候选项在本次抽取过程中被抽过多次，则其再次被抽的概率会递减
           3. 最终抽取 energy 次，返回抽中的 candidate 列表
        """
        if not candidates or energy <= 0:
            return []

        # 用于记录候选项在本次抽取过程中的“已被抽取次数”
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
        for _ in range(energy):
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

    def query_llm(self, prompt, max_retry=5, model="gpt-4o-mini"):
        attempt_num = 0
        while attempt_num < max_retry:  # 设置最大尝试次数以避免无限循环
            try:
                response = self.llm_client.chat.completions.create(
                    model=model,  # gpt-4o-mini  gpt-3.5-turbo
                    messages=[
                        {"role": "system",
                         "content": "You're an AI assistant adept at using multiple deep learning libraries"},
                        {"role": "user", "content": prompt}
                    ],
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

    def fuzz_equivalent_cluster(self, cluster: Cluster):
        if cluster.is_tested:
            return
        elif not cluster.api_groups:  # 如果该等价簇没有API组合, 则直接标记为已测试
            cluster.is_tested = True
            self.session.commit()
            return

        # 先查询该等价簇已经生成了几个种子
        seeds_num = self.session.query(ClusterTestSeed).filter(ClusterTestSeed.cluster_id == cluster.id).count()
        remaining_energy = cluster.energy - seeds_num
        # 从cluster中筛选出仅由一个API组成的APIGroup作为候选基底
        candidate_base_api_groups = [api_group for api_group in cluster.api_groups if len(api_group.apis) == 1]
        base_api_groups = self.weighted_sample_base(candidate_base_api_groups, remaining_energy)

        while base_api_groups:  # 生成remaining_energy个ClusterTestSeed
            try:  # 开始种子的生成
                base_api_group = base_api_groups[0]
                cluster_seed = ClusterTestSeed(
                    cluster_id=cluster.id,
                    start_test=datetime.utcnow()
                )
                self.session.add(cluster_seed)
                self.session.flush()

                # 生成基底API的测试用例
                base_api_seed = self.generate_seed4base(base_api_group, cluster_seed)

                # 生成等价簇中其他API的测试用例
                twin_apis_seeds = []
                for twin_api_group in cluster.api_groups:
                    if twin_api_group == base_api_group:
                        continue
                    twin_api_seed = self.generate_seed4twin(twin_api_group, base_api_seed, cluster_seed)
                    twin_apis_seeds.append(twin_api_seed)
                cluster_seed.end_test = datetime.utcnow()
                self.session.commit()
                base_api_groups.pop(0)
            except Exception as e:
                print(f"Error in generating seed for Value Equivalent Cluster({cluster.id}): {e}")
                self.session.rollback()
                continue

        # 检查是否所有的种子都已经生成完毕
        seeds_num = self.session.query(ClusterTestSeed).filter_by(cluster_id=cluster.id).count()
        if seeds_num >= cluster.energy:
            cluster.is_tested = True
            self.session.commit()

    def generate_seed4base(self, base_api_group: APIGroup, cluster_seed: ClusterTestSeed):
        # 基底API的详情
        base_api = base_api_group.apis[0]
        api_info_prompt = f"""
API Signature: {base_api.signature}
API Description: {base_api.description}
API Library: {base_api.lib} 
API Library Version: {base_api.version}
"""
        # 触发问题的代码调用样例
        issue_examples = [item for value_list in self.sample_errors(base_api).values() for item in value_list]
        issue_examples_prompt = ""
        for count, issue_example in enumerate(issue_examples):
            issue_examples_prompt = issue_examples_prompt + f"""
History Issue Example{count + 1}:
Issue Title: {issue_example.title}
Issue Description: {issue_example.description}
Issue Trigger API: {issue_example.api.signature}
Issue Code: 
{issue_example.code}         

"""
        # 构建最终提示词
        prompt = f"""
触发问题的代码调用样例:
{issue_examples_prompt}

待测试的API信息:
{api_info_prompt}

任务要求:
参考上述样本中的API参数的输入值和API的组合调用, 生成调用{base_api.full_name}的代码片段
"""

        base_seed_code = self.query_llm(prompt)
        if base_seed_code is None:
            raise Exception("Failed to generate base seed for base API.")
        base_seed = APITestSeed(
            cluster_seed_id=cluster_seed.id,
            api_group_id=base_api_group.id,
            raw_code=base_seed_code
        )
        self.session.add(base_seed)
        self.session.flush()

        # 对基底API进行修复
        valid_code = APITestSeedValidator(self.session, self.llm_client, base_seed).validate()
        if valid_code is None:
            raise Exception("Failed to generate base seed for base API.")
        base_seed.valid_code = valid_code
        self.session.flush()
        return base_seed

    def generate_seed4twin(self, twin_api_group: APIGroup, base_api_seed: APITestSeed, cluster_seed: ClusterTestSeed):
        # API Group的详情
        if len(twin_api_group.apis) == 1:
            twin_api = twin_api_group.apis[0]
            api_group_info_prompt = f"""
API Signature: {twin_api.signature}
API Description: {twin_api.description}
API Library: {twin_api.lib} 
API Library Version: {twin_api.version}
"""
        else:
            api_group_info_prompt = ""
            for count, twin_api in enumerate(twin_api_group.apis):
                api_group_info_prompt = api_group_info_prompt + f"""
Member{count + 1} of API Group:
API Signature: {twin_api.signature}
API Description: {twin_api.description}
"""
        # APIGroup brief info
        if len(twin_api_group.apis) == 1:
            api_group_brief_info = f"{twin_api_group.apis[0].signature}"
        else:
            api_group_brief_info = f"({', '.join([api.signature for api in twin_api_group.apis])})"
        # 背景知识
        base_api = base_api_seed.api_group.apis[0]
        if len(twin_api_group.apis) == 1:
            twin_api = twin_api_group.apis[0]
            background_knowledge_prompt = f"""
来自{twin_api.lib}(v{twin_api.version})库的API {twin_api.signature} 与来自{base_api.lib}(v{base_api.version})库的API {base_api.signature} 拥有相同的函数功能.
"""
        else:
            background_knowledge_prompt = f"""
通过组合调用{api_group_brief_info}中的API, 可以与来自{base_api.lib}(v{base_api.version})库的API {base_api.signature}一样的函数功能.
"""

        # 构建最终提示词
        prompt = f"""
待测试的API{'group' if len(twin_api_group.apis) > 1 else ''}的信息:
{api_group_info_prompt}        

背景知识:
{background_knowledge_prompt}

任务要求:
下方的代码片段是对({base_api.signature})的调用. 请你生成使用{api_group_brief_info}替代({base_api.full_name})的代码片段, 要求参数的输入值和最终的输出值保持一致.
{base_api_seed.valid_code} 
"""
        twin_seed_code = self.query_llm(prompt)
        if twin_seed_code is None:
            raise Exception("Failed to generate seed for equivalent API.")
        twin_seed = APITestSeed(
            cluster_seed_id=cluster_seed.id,
            api_group_id=twin_api_group.id,
            raw_code=twin_seed_code
        )
        self.session.add(twin_seed)
        self.session.flush()
        return twin_seed

    def fuzz_value_equivalent_clusters(self):
        value_equivalent_clusters = self.session.query(Cluster).filter_by(type='ValueEquivalent').all()
        untested_clusters = self.session.query(Cluster).filter_by(is_tested=False, type='ValueEquivalent').all()
        while untested_clusters:
            print(f"Fuzzing Value Equivalent Clusters: {len(untested_clusters)}/ {len(value_equivalent_clusters)}")
            untested_cluster = untested_clusters[0]
            self.fuzz_equivalent_cluster(untested_cluster)
            untested_clusters = self.session.query(Cluster).filter_by(is_tested=False, type='ValueEquivalent').all()

    def fuzz_state_equivalent_clusters(self):
        state_equivalent_clusters = self.session.query(Cluster).filter_by(type='StateEquivalent').all()
        untested_clusters = self.session.query(Cluster).filter_by(is_tested=False, type='StateEquivalent').all()
        while untested_clusters:
            print(f"Fuzzing State Equivalent Clusters: {len(untested_clusters)}/ {len(state_equivalent_clusters)}")
            untested_cluster = untested_clusters[0]
            self.fuzz_equivalent_cluster(untested_cluster)
            untested_clusters = self.session.query(Cluster).filter_by(is_tested=False, type='StateEquivalent').all()

if __name__ == '__main__':
    session = utils.get_session()
    llm_client = utils.get_llm_client()
    fuzzer = Fuzzer(session, llm_client)
    fuzzer.fuzz_value_equivalent_clusters()
    fuzzer.fuzz_state_equivalent_clusters()
    session.close()