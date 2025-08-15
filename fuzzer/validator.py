import os
import subprocess
import time
import threading
from datetime import datetime
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
1. Only output the corrected code snippet.
2. Do not include any explanations, comments, or additional text.
3. Preserve the original indentation, spacing, and formatting as much as possible.
4. Keep all tabs, spaces, and line breaks intact unless they are part of the error.
5. Do not reformat the code style unless it's necessary to fix the error.
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
        # possible_imports = [
        #     "import torch",
        #     "import tensorflow",
        #     "import jax",
        #     "import mindspore",
        #     "import numpy"
        # ]
        # updated_code = '\n'.join(possible_imports) + '\n' + raw_code
        # return updated_code
        return raw_code

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
        # result = subprocess.run(
        #     ['flake8', file_path, '--extend-ignore=F401', '--select=F,E'],  # TODO flake8中的F和E都包含了一些代码风格建议,这些建议理论上应该被忽略,但需要在配置文件中进一步设置
        #     capture_output=True, text=True
        # )
        result = subprocess.run(
            ['flake8', file_path],  # 移除手动指定的参数，让flake8使用配置文件
            capture_output=True, text=True
        )
        error_details = result.stdout
        if error_details == "":
            return True, error_details
        else:
            return False, error_details

    def static_analysis(self, raw_code):  # 静态分析Python代码, 如果发现错误, 则返回False和错误信息
        # 创建一个临时的Python文件:
        timestamp = int(time.time())
        thread_id = threading.get_ident()
        file_path = f'../data/tmp/{timestamp}_{thread_id}.py'
        with open(file_path, 'w') as f:
            f.write(raw_code)
        # 使用静态分析工具对代码文件进行分析
        is_valid, error_details = self.flake8_static_analysis(file_path)
        error_details = "Static Analysis:\n" + error_details if error_details else None
        os.remove(file_path)  # 删除临时文件
        return is_valid, error_details

    def dynamic_import_analysis(self, raw_code):  # 动态分析Python代码, 如果发现错误, 则返回False和错误信息
        result, invalid_imports = validate_code_imports(raw_code)
        if result is True:
            return True, None
        details_lines = [
            "Dynamic Import Analysis:",
            "The following modules cannot be imported in the current environment:",
        ]
        for name in invalid_imports:
            details_lines.append(f" - {name}")
        details = "\n".join(details_lines)
        return False, details
    
    def mixed_analysis(self, raw_code):
        is_syntax_valid, syntax_error_details = self.static_analysis(raw_code)
        is_import_valid, import_error_details = self.dynamic_import_analysis(raw_code)
        
        # 处理错误详情为 None 的情况
        error_parts = []
        if syntax_error_details:
            error_parts.append(syntax_error_details)
        if import_error_details:
            error_parts.append(import_error_details)
        
        combined_error_details = "\n".join(error_parts) if error_parts else ""
        return is_syntax_valid and is_import_valid, combined_error_details

    def validate4code(self, max_retry=5):
        code_without_markdown = self.eliminate_markdown(self.raw_code)
        code_complemented_import = self.insert_possible_imports(code_without_markdown)
        is_valid, error_details = self.mixed_analysis(code_complemented_import)
        
        if is_valid:  # 如果代码没有错误, 则结束修复
            return code_complemented_import  # 返回有效的代码

        print(f"\nError Details:\n {error_details}\n")

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
                    model="gpt-4.1-mini",  # gpt-4o-mini  gpt-3.5-turbo gpt-4.1-mini
                    messages=messages,
                    temperature=0,
                )
                validated_code = response.choices[0].message.content

                # 检查LLM返回的种子是否有效
                code_without_markdown = self.eliminate_markdown(validated_code)  # 去除code中的markdown语法
                code_complemented_import = self.insert_possible_imports(code_without_markdown)  # 向code中插入可能的导入语句
                validated_code = code_complemented_import
                messages.append({"role": "system", "content": validated_code})
                print(f"Verified Code:\n{validated_code}")

                is_valid, error_details = self.mixed_analysis(validated_code)
                if is_valid:
                    return validated_code  # 返回修复后的有效代码
                else:
                    print(f"\nError Details:\n {error_details}\n")
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
        is_valid, error_details = self.mixed_analysis(code_complemented_import)

        if is_valid:  # 如果代码没有错误, 则结束修复
            self.seed.valid_code = code_complemented_import
            self.session.flush()
            return code_complemented_import  # 返回有效的代码

        print(f"\nError Details:\n {error_details}\n")

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
                    model="gpt-4.1-mini",  # gpt-4o-mini  gpt-3.5-turbo gpt-4.1-mini
                    messages=messages,
                    temperature=0,
                )
                validated_code = response.choices[0].message.content

                # 检查LLM返回的种子是否有效
                code_without_markdown = self.eliminate_markdown(validated_code)  # 去除code中的markdown语法
                code_complemented_import = self.insert_possible_imports(code_without_markdown)  # 向code中插入可能的导入语句
                validated_code = code_complemented_import
                messages.append({"role": "system", "content": validated_code})

                print(f"Verified Code:\n{validated_code}")

                is_valid, error_details = self.mixed_analysis(validated_code)
                if is_valid:
                    self.seed.valid_code = validated_code
                    self.session.flush()
                    return validated_code  # 返回修复后的有效代码
                else:
                    print(f"\nError Details:\n {error_details}\n")
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
        api_seeds_waiting_validate = self.session.query(APITestSeed).filter(
            APITestSeed.cluster_seed_id == self.seed.id,
            APITestSeed.is_validated == False
        ).all()
        if_success = True
        for api_seed in api_seeds_waiting_validate:
            api_seed_validator = APITestSeedValidator(llm_client=self.llm_client, session=self.session, seed=api_seed)
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
    llm_client = get_llm_client(llm='bianxie')
    # 查询所有未经验证的ClusterSeed
    unvalidated_cluster_seeds = session.query(ClusterTestSeed).filter(ClusterTestSeed.is_validated == False).all()
    while unvalidated_cluster_seeds:
        print("----------------------------------------------------------------------------------")
        cluster_seed = unvalidated_cluster_seeds[0]
        cluster_seed_validator = ClusterTestSeedValidator(session, llm_client, cluster_seed)
        is_success = cluster_seed_validator.validate()
        if is_success:
            export_valid_cluster_seed(cluster_seed)
        # 更新未校验的种子集
        unvalidated_cluster_seeds = session.query(ClusterTestSeed).filter(ClusterTestSeed.is_validated == False).all()
        # 打印未校验的种子数量
        total_seeds_num = session.query(ClusterTestSeed).count()
        print(f"Unvalidated / Total: {len(unvalidated_cluster_seeds)} / {total_seeds_num}")

def detect_invalid_cluster_seeds(seed_file_path, full_api_name_list):
    """
    检测seed_file_path文件中是否全部调用了full_api_name_list中的API, 如果是, 则返回True, 否则返回False
    """
    try:
        with open(seed_file_path, 'r') as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading file {seed_file_path}: {e}")
        return None
    
    # 如果code为空, 则返回False
    if code == "": 
        return False
    
    # 遍历full_api_name_list, 如果api_name在code中存在, 则返回False
    for full_api_name in full_api_name_list:
        # 取full_api_name的最后一个"."后的字符串作为api_name
        api_name = full_api_name.split('.')[-1]
        # 如果api_name在code中不存在, 则返回False
        if api_name not in code:
            return False
    return True

def label_invalid_cluster_seeds(clusters_folder_path):
    """
    遍历cluster_folder_path下的Cluster文件夹，检测并标记无效的种子文件
    
    Args:
        clusters_folder_path: cluster文件夹的路径
    """
    if not os.path.exists(clusters_folder_path):
        print(f"错误: 路径 {clusters_folder_path} 不存在")
        return
    
    # 获取所有不以valid开头的cluster文件夹
    cluster_folders = [f for f in os.listdir(clusters_folder_path) if os.path.isdir(os.path.join(clusters_folder_path, f)) and f.startswith('Cluster_')]
    
    for cluster_folder in cluster_folders:            
        cluster_path = os.path.join(clusters_folder_path, cluster_folder)
        print(f"处理cluster: {cluster_folder}")
        
        # 获取该cluster下的所有seed文件夹
        seed_folders = [f for f in os.listdir(cluster_path) if os.path.isdir(os.path.join(cluster_path, f))]
        
        all_files_processed = True
        for seed_folder in seed_folders:
            seed_path = os.path.join(cluster_path, seed_folder)
            print(f"  处理seed: {seed_folder}")
            
            # 获取该seed文件夹下的所有.py文件
            py_files = [f for f in os.listdir(seed_path) if f.endswith('.py') and not f.startswith('invalid.')]
            for py_file in py_files:
                py_file_path = os.path.join(seed_path, py_file)
                
                # 从文件名中提取API名称列表, 文件名格式为"api1.py"或者是用"+"连接的多个API"api1+api2+api3.py"
                file_name_without_ext = py_file.replace('.py', '')
                if '+' in file_name_without_ext: # 如果文件名包含"+"，说明是多个API
                    full_api_names = file_name_without_ext.split('+')
                else:
                    full_api_names = [file_name_without_ext]
                is_valid = detect_invalid_cluster_seeds(py_file_path, full_api_names) # 使用detect_invalid_cluster_seeds检测文件
                if is_valid is False:  # 检测返回False，说明文件无效
                    # 重命名文件，在文件名前添加"invalid."
                    invalid_file_name = f"invalid.{py_file}"
                    invalid_file_path = os.path.join(seed_path, invalid_file_name)
                    try:
                        os.rename(py_file_path, invalid_file_path)
                        print(f"    标记无效文件: {py_file} -> {invalid_file_name}")
                    except OSError as e:
                        print(f"    重命名文件失败 {py_file}: {e}")
                        all_files_processed = False
                elif is_valid is None:  # 检测过程中出现错误
                    print(f"    检测文件时出错: {py_file}")
                    all_files_processed = False
        # 如果该cluster下的所有文件都已经检测完成，则给cluster文件夹添加"valid"前缀
        if all_files_processed:
            valid_cluster_name = f"valid_{cluster_folder}"
            valid_cluster_path = os.path.join(clusters_folder_path, valid_cluster_name)
            try:
                os.rename(cluster_path, valid_cluster_path)
                print(f"标记cluster为已处理: {cluster_folder} -> {valid_cluster_name}")
            except OSError as e:
                print(f"重命名cluster文件夹失败 {cluster_folder}: {e}")
        else:
            print(f"cluster {cluster_folder} 在处理期间发生错误")

if __name__ == '__main__':
    validate_and_export_all_seeds()
    #label_invalid_cluster_seeds('seeds/validated_seeds/ValueEquivalent')
    #label_invalid_cluster_seeds('seeds/validated_seeds/StateEquivalent')

