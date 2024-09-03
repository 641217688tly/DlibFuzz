import os
import subprocess
from orm import *
from utils import *


class SeedValidator:
    def __init__(self):
        self.session = get_session()
        self.openai_client = get_openai_client()
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
