import os
import subprocess
from orm import *
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


class SeedValidator:
    def __init__(self, session, openai_client):
        self.session = session
        self.openai_client = openai_client

    def eliminate_markdown(self, seed: ClusterTestSeed):  # 去除markdown语法
        code = seed.code
        code_lines = code.split('\n')  # 将代码按行分割成列表
        if code_lines[0].strip().startswith("```"):  # 检查并去除第一行如果它是"```python"
            code_lines = code_lines[1:]
        if code_lines[-1].strip().startswith("```"):  # 检查并去除最后一行如果它是"```"
            code_lines = code_lines[:-1]
        cleaned_code = '\n'.join(code_lines)  # 重新组合代码为单个字符串
        # 1.更新数据库中的种子代码
        seed.code = cleaned_code
        self.session.commit()
        # 2.更新对应的.py文件中的代码
        if not os.path.exists(seed.unverified_file_path):
            os.makedirs(os.path.dirname(seed.unverified_file_path), exist_ok=True)
        with open(seed.unverified_file_path, 'w') as f:
            f.write(cleaned_code + '\n')
        return cleaned_code

    def static_analysis(self, file_path):  # 使用静态分析工具flake8分析Python代码, 如果发现错误, 则返回False和错误信息
        result = subprocess.run(
            ['flake8', file_path, '--select=F'],
            capture_output=True, text=True
        )
        error_details = result.stdout
        if error_details == "":
            return True, error_details
        else:
            return False, error_details

    def update_seed(self, seed: ClusterTestSeed, seed_path, verified_code: str):  # 更新种子代码
        # 更新数据库中的种子代码
        seed.code = verified_code
        self.session.commit()
        # 更新对应的.py文件中的代码
        if not os.path.exists(seed_path):
            os.makedirs(os.path.dirname(seed_path), exist_ok=True)
        with open(seed_path, 'w') as f:
            f.write(verified_code + '\n')
        self.eliminate_markdown(seed)  # 去除markdown语法

    def validate(self, seed: ClusterTestSeed, max_attempt_limit=5):  # 修复代码中的错误
        self.eliminate_markdown(seed)  # 去除markdown语法
        is_valid, error_details = self.static_analysis(seed.unverified_file_path)

        if is_valid:  # 如果代码没有错误, 则直接标记为已验证然后结束修复
            seed.is_verified = True
            self.update_seed(seed, seed.verified_file_path, seed.code)
            self.session.commit()
            return

        prompt = construct_prompt(seed.code, error_details)
        messages = [
            {"role": "system", "content": "You're an AI assistant adept at debugging code."},
            {"role": "user", "content": prompt}
        ]
        attempt_num = 0
        while attempt_num < max_attempt_limit:
            print(f"Try to fix the code snippet. Current attempt times: {attempt_num + 1}/{max_attempt_limit}")
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",  # gpt-4o-mini  gpt-3.5-turbo
                    messages=messages,
                    temperature=0,
                )
                verified_code = response.choices[0].message.content
                messages.append({"role": "system", "content": verified_code})

                self.update_seed(seed, seed.unverified_file_path, verified_code)  # 更新数据库和py文件中的种子代码
                is_valid, error_details = self.static_analysis(seed.unverified_file_path)
                if is_valid:
                    seed.is_verified = True
                    self.update_seed(seed, seed.verified_file_path, verified_code)
                    self.session.commit()
                    return
                else:
                    prompt = construct_prompt(seed.code, error_details)
                    messages.append({"role": "user", "content": prompt})
                    attempt_num = attempt_num + 1
            except Exception as e:
                attempt_num = attempt_num + 1
                self.session.rollback()  # 回滚在异常中的任何数据库更改
                print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    session = get_session()
    openai_client = get_openai_client()
    validator = SeedValidator(session, openai_client)
    # 查询所有未经验证的种子
    unverified_seeds = session.query(ClusterTestSeed).filter(ClusterTestSeed.is_verified == False).all()
    while unverified_seeds:
        print("----------------------------------------------------------------------------------")
        # 校验第一个未校验的种子
        unverified_seed = unverified_seeds[0]
        validator.validate(unverified_seed)
        # 更新未校验的种子集
        untested_clusters = session.query(Cluster).filter(Cluster.is_tested == False).all()
        # 打印未校验的种子数量
        total_seeds_num = session.query(Cluster).count()
        unverified_seeds_num = len(untested_clusters)
        print(f"Untested / Total: {unverified_seeds_num} / {total_seeds_num}")
