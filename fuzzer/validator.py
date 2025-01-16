import os
import subprocess
from utils import *


def construct_prompt(code: str, error_details: str):  # 构建提示词
    prompt = f"""
Code Snippet:
{code}

Error Details:
{error_details}

Objective:
Please fix the error in the code snippet based on the error details.

Requirements:
1.Only output the corrected code snippet.
2.Do not include any explanations, comments, or additional text.
    """
    return prompt


class APITestSeedValidator:
    def __init__(self, llm_client, session=None, seed: APITestSeed = None, raw_code=""):
        if seed and session is None:
            raise ValueError("session is required when seed is provided")
        self.session = session
        self.llm_client = llm_client
        self.seed = seed
        self.raw_code = raw_code

    def eliminate_markdown(self, raw_code):  # 去除raw_code中的markdown语法
        code_lines = raw_code.split('\n')  # 将代码按行分割成列表
        # 过滤掉所有以"```"开头的行
        cleaned_lines = [line for line in code_lines if not line.strip().startswith("```")]
        cleaned_code = '\n'.join(cleaned_lines)  # 重新组合代码为单个字符串
        # 返回更新后的代码
        return cleaned_code

    def insert_possible_imports(self, raw_code):  # 向seed.code中插入可能需要的的导入语句
        possible_imports = [
            "import torch",
            "import tensorflow",
            "import jax",
            "import mindspore",
            "import numpy"
        ]
        updated_code = '\n'.join(possible_imports) + '\n' + raw_code
        return updated_code

    def pylint_static_analysis(self, file_path):  # 使用静态分析工具pylint分析Python代码, 如果发现错误, 则返回False和错误信息
        # TODO 该静态分析工具存在误报问题, 暂时放弃使用
        errors2check = [
            'syntax-error',  # 语法错误
            'import-error',  # 导入错误
            'undefined-variable'  # 未定义变量
        ]
        enable_param = ','.join(errors2check)
        result = subprocess.run(
            ['pylint', file_path, '--disable=all', f'--enable={enable_param}', '--score=no'],
            capture_output=True, text=True
        )
        error_details = result.stdout
        if error_details == "":
            return True, error_details
        else:
            errors_lines = error_details.split('\n')
            if errors_lines[0].startswith("*************"):
                errors_cleaned = "\n".join(errors_lines[1:]).strip()
            else:
                errors_cleaned = error_details.strip()
            return False, errors_cleaned

    def flake8_static_analysis(self, file_path):  # 使用静态分析工具flake8分析Python代码, 如果发现错误, 则返回False和错误信息
        result = subprocess.run(
            ['flake8', file_path, '--select=F'],  # TODO flake8中的F和E都包含了一些代码风格建议,这些建议理论上应该被忽略,但需要在配置文件中进一步设置
            capture_output=True, text=True
        )
        error_details = result.stdout
        if error_details == "":
            return True, error_details
        else:
            return False, error_details

    def static_analysis(self, raw_code):  # 静态分析Python代码, 如果发现错误, 则返回False和错误信息
        # 创建一个临时的Python文件:
        file_path = f'../data/tmp/{self.seed.id}.py'
        with open(file_path, 'w') as f:
            f.write(raw_code)
        # 使用静态分析工具对代码文件进行分析
        is_valid, error_details = self.flake8_static_analysis(file_path)
        os.remove(file_path)  # 删除临时文件
        return is_valid, error_details

    def validate4code(self, max_retry=5):
        code_without_markdown = self.eliminate_markdown(self.raw_code)
        code_complemented_import = self.insert_possible_imports(code_without_markdown)
        is_valid, error_details = self.static_analysis(code_complemented_import)

        if is_valid:  # 如果代码没有错误, 则结束修复
            return code_complemented_import  # 返回有效的代码

        print(f"\nError Details:\n {error_details}")

        prompt = construct_prompt(code_complemented_import, error_details)
        messages = [
            {"role": "system", "content": "You're an AI assistant adept at debugging code."},
            {"role": "user", "content": prompt}
        ]
        attempt_num = 0
        while attempt_num < max_retry:
            print(f"Try to fix the code snippet. Current attempt times: {attempt_num + 1}/{max_retry}")
            try:
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",  # gpt-4o-mini  gpt-3.5-turbo
                    messages=messages,
                    temperature=0,
                )
                validated_code = response.choices[0].message.content
                messages.append({"role": "system", "content": validated_code})

                # 检查LLM返回的种子是否有效
                code_without_markdown = self.eliminate_markdown(validated_code)  # 去除code中的markdown语法
                code_complemented_import = self.insert_possible_imports(code_without_markdown)  # 向code中插入可能的导入语句
                validated_code = code_complemented_import
                print(f"Verified Code:\n {validated_code}")

                is_valid, error_details = self.static_analysis(validated_code)
                if is_valid:
                    return validated_code  # 返回修复后的有效代码
                else:
                    print(f"\nError Details:\n {error_details}")
                    prompt = construct_prompt(validated_code, error_details)
                    messages.append({"role": "user", "content": prompt})
                    attempt_num = attempt_num + 1
            except Exception as e:
                attempt_num = attempt_num + 1
                print(f"An unexpected error occurred: {e}")
        print(f"Max attempts reached. Failed to fix the code snippet.")
        return None

    def validate4seed(self, max_retry=5):  # 修复代码中的错误
        code_without_markdown = self.eliminate_markdown(self.seed.raw_code)  # 去除code中的markdown语法
        code_complemented_import = self.insert_possible_imports(code_without_markdown)  # 向code中插入可能的导入语句
        is_valid, error_details = self.static_analysis(code_complemented_import)

        if is_valid:  # 如果代码没有错误, 则结束修复
            self.seed.valid_code = code_complemented_import
            self.session.flush()
            return code_complemented_import  # 返回有效的代码

        print(f"\nError Details:\n {error_details}")

        prompt = construct_prompt(code_complemented_import, error_details)
        messages = [
            {"role": "system", "content": "You're an AI assistant adept at debugging code."},
            {"role": "user", "content": prompt}
        ]
        attempt_num = 0
        while attempt_num < max_retry:
            print(f"Try to fix the code snippet. Current attempt times: {attempt_num + 1}/{max_retry}")
            try:
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",  # gpt-4o-mini  gpt-3.5-turbo
                    messages=messages,
                    temperature=0,
                )
                validated_code = response.choices[0].message.content
                messages.append({"role": "system", "content": validated_code})

                # 检查LLM返回的种子是否有效
                code_without_markdown = self.eliminate_markdown(validated_code)  # 去除code中的markdown语法
                code_complemented_import = self.insert_possible_imports(code_without_markdown)  # 向code中插入可能的导入语句
                validated_code = code_complemented_import
                print(f"Verified Code:\n {validated_code}")

                is_valid, error_details = self.static_analysis(validated_code)
                if is_valid:
                    self.seed.valid_code = validated_code
                    self.session.flush()
                    return validated_code  # 返回修复后的有效代码
                else:
                    print(f"\nError Details:\n {error_details}")
                    prompt = construct_prompt(validated_code, error_details)
                    messages.append({"role": "user", "content": prompt})
                    attempt_num = attempt_num + 1
            except Exception as e:
                attempt_num = attempt_num + 1
                self.session.rollback()  # 回滚在异常中的任何数据库更改
                print(f"An unexpected error occurred: {e}")

        print(f"Max attempts reached. Failed to fix the code snippet.")
        return None


class ClusterTestSeedValidator:
    def __init__(self, session, llm_client, seed: ClusterTestSeed):
        self.session = session
        self.llm_client = llm_client
        self.seed = seed

    def validate(self):
        if self.seed.is_validated:
            return True
        # 寻找当前ClusterSeed中待验证的APISeed
        api_seeds_waiting_validate = self.seed.api_seeds.filter(APITestSeed.is_validated == False).all()
        if_success = True
        for api_seed in api_seeds_waiting_validate:
            api_seed_validator = APITestSeedValidator(session=self.session, llm_client=self.llm_client, seed=api_seed)
            validated_code = api_seed_validator.validate4seed()
            if validated_code is None:
                if_success = False
        if if_success:
            self.seed.is_validated = True
            self.session.commit()
        return if_success


def export_valid_cluster_seed(seed: ClusterTestSeed):  # 导出种子中各个库的测试用例为py文件
    if seed.is_validated:
        cluster_folder_path = 'seeds/validated_seeds/'
        # 首先区分是否利用了历史错误
        # if seed.type == 'WithHistoryError':  # 利用了历史错误
        #    cluster_folder_path = cluster_folder_path + 'WithHistoryError/'
        #    # 然后区分值等价和状态等价
        #    if seed.cluster.type == 'ValueEquivalent':
        #        cluster_folder_path = cluster_folder_path + 'ValueEquivalent/'
        #    else:  # 状态等价
        #        cluster_folder_path = cluster_folder_path + 'StateEquivalent/'
        # else:  # 没有利用历史错误
        #    cluster_folder_path = cluster_folder_path + 'WithoutHistoryError/'
        #    # 然后区分值等价和状态等价
        #    if seed.cluster.type == 'ValueEquivalent':
        #        cluster_folder_path = cluster_folder_path + 'ValueEquivalent/'
        #    else:  # 状态等价
        #        cluster_folder_path = cluster_folder_path + 'StateEquivalent/'
        if seed.cluster.type == 'ValueEquivalent':
            cluster_folder_path = cluster_folder_path + 'ValueEquivalent/'
        else:  # 状态等价
            cluster_folder_path = cluster_folder_path + 'StateEquivalent/'
        cluster_folder_path = cluster_folder_path + f'Cluster_{seed.cluster_id}/'
        if not os.path.exists(cluster_folder_path):
            os.makedirs(cluster_folder_path, exist_ok=True)

        # 查看cluster_folder_path下已经存在了多少个seed文件夹
        cluster_seed_folder_path = cluster_folder_path + f'seed_{len(os.listdir(cluster_folder_path)) + 1}/'
        if not os.path.exists(cluster_seed_folder_path):
            os.makedirs(cluster_seed_folder_path, exist_ok=True)
        for api_seed in seed.api_seeds:
            api_group = api_seed.api_group
            # 将api_group内各个API的full_name用"+"拼接在一起
            api_seed_file_name = '+'.join([api.full_name for api in api_group.apis])
            # 使用api_seed中api_group内各个API的名称作为seed文件夹的名称
            api_seed_file_path = cluster_seed_folder_path + api_seed_file_name + '.py'
            with open(api_seed_file_path, 'w') as f:
                f.write(api_seed.valid_code)


def validate_and_export_all_seeds():
    session = get_session()
    openai_client = get_llm_client()
    # 查询所有未经验证的ClusterSeed
    unvalidated_cluster_seeds = session.query(ClusterTestSeed).filter(ClusterTestSeed.is_validated == False).all()
    while unvalidated_cluster_seeds:
        print("----------------------------------------------------------------------------------")
        cluster_seed = unvalidated_cluster_seeds[0]
        cluster_seed_validator = ClusterTestSeedValidator(session, openai_client, cluster_seed)
        is_success = cluster_seed_validator.validate()
        if is_success:
            export_valid_cluster_seed(cluster_seed)
        # 更新未校验的种子集
        unvalidated_cluster_seeds = session.query(ClusterTestSeed).filter(ClusterTestSeed.is_validated == False).all()
        # 打印未校验的种子数量
        total_seeds_num = session.query(ClusterTestSeed).count()
        print(f"Unvalidated / Total: {len(unvalidated_cluster_seeds)} / {total_seeds_num}")


if __name__ == '__main__':
    validate_and_export_all_seeds()
