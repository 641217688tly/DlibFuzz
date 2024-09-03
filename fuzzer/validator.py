import os
import subprocess
from orm import *
from utils import *


class SeedValidator:
    def __init__(self, session, openai_client):
        self.session = session
        self.openai_client = openai_client
        self.seed = None

    def eliminate_markdown(self):  # 去除markdown语法
        code = self.seed.code
        code_lines = code.split('\n')  # 将代码按行分割成列表
        if code_lines[0].strip().startswith("```"):  # 检查并去除第一行如果它是"```python"
            code_lines = code_lines[1:]
        if code_lines[-1].strip().startswith("```"):  # 检查并去除最后一行如果它是"```"
            code_lines = code_lines[:-1]
        cleaned_code = '\n'.join(code_lines)  # 重新组合代码为单个字符串
        # 1.更新数据库中的种子代码
        self.seed.code = cleaned_code
        self.session.commit()
        # 2.更新对应的.py文件中的代码
        if not os.path.exists(self.seed.unverified_file_path):
            os.makedirs(os.path.dirname(self.seed.unverified_file_path), exist_ok=True)
        with open(self.seed.unverified_file_path, 'w') as f:
            f.write(cleaned_code + '\n')
        return cleaned_code

    def static_analysis(self):  # 使用静态分析工具flake8分析Python代码, 如果发现错误, 则返回False和错误信息
        result = subprocess.run(
            ['flake8', self.seed.unverified_file_path, '--select=F'],
            capture_output=True, text=True
        )
        errors = result.stdout
        if errors == "":
            return True, errors
        else:
            return False, errors

    def repair(self):
        pass

    def close(self):  # 清除数据
        self.seed = None

    def validate(self, seed: ClusterTestSeed):  # 判断种子是否有效
        self.seed = seed
        self.eliminate_markdown()
        self.repair()
        self.close()


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
