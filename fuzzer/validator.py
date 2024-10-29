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


class SeedValidator:
    def __init__(self, session, openai_client, seed: ClusterTestSeed, raw_code, lib: str):
        self.session = session
        self.openai_client = openai_client
        self.lib = lib
        self.seed = seed
        self.raw_code = raw_code

    def eliminate_markdown(self, raw_code):  # 去除raw_code中的markdown语法
        code_lines = raw_code.split('\n')  # 将代码按行分割成列表
        # 过滤掉所有以"```"开头的行
        cleaned_lines = [line for line in code_lines if not line.strip().startswith("```")]
        cleaned_code = '\n'.join(cleaned_lines)  # 重新组合代码为单个字符串
        # 返回更新后的代码
        return cleaned_code

    def insert_possible_imports(self, raw_code):  # 向seed.code中插入可能的导入语句
        torch_possible_imports = [
            "import torch",
        ]
        tf_possible_imports = [
            "import tensorflow",
        ]
        jax_possible_imports = [
            "import jax",
        ]

        code_lines = raw_code.split('\n')
        # 检查代码中是否已包含了可能的导入语句
        imports_to_add = []
        if self.lib == "Pytorch":
            for import_statement in torch_possible_imports:
                if not import_statement in code_lines:  # 如果代码中没有包含该导入语句, 则将其添加到imports_to_add列表中
                    imports_to_add.append(import_statement)
        elif self.lib == "Tensorflow":
            for import_statement in tf_possible_imports:
                if not import_statement in code_lines:
                    imports_to_add.append(import_statement)
        else:
            for import_statement in jax_possible_imports:
                if not import_statement in code_lines:
                    imports_to_add.append(import_statement)

        # 如果有需要添加的导入语句，将它们插入到代码的开头
        if imports_to_add:
            updated_code = '\n'.join(imports_to_add) + '\n' + raw_code
        else:
            updated_code = raw_code
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

    def validate(self, max_retry_limit=5):  # 修复代码中的错误
        code_without_markdown = self.eliminate_markdown(self.raw_code)  # 去除code中的markdown语法
        code_complemented_import = self.insert_possible_imports(code_without_markdown)  # 向code中插入可能的导入语句
        is_valid, error_details = self.static_analysis(code_complemented_import)

        if is_valid:  # 如果代码没有错误, 则结束修复
            self.save_valid_code(code_complemented_import)
            return code_complemented_import  # 返回有效的代码

        print(f"\nError Details:\n {error_details}")

        prompt = construct_prompt(code_complemented_import, error_details)
        messages = [
            {"role": "system", "content": "You're an AI assistant adept at debugging code."},
            {"role": "user", "content": prompt}
        ]
        attempt_num = 0
        while attempt_num < max_retry_limit:
            print(f"Try to fix the code snippet. Current attempt times: {attempt_num + 1}/{max_retry_limit}")
            try:
                response = self.openai_client.chat.completions.create(
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
                    self.save_valid_code(validated_code)
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

    def save_valid_code(self, valid_code):
        if self.lib == "Pytorch":
            self.seed.valid_pytorch_code = valid_code
        elif self.lib == "Tensorflow":
            self.seed.valid_tensorflow_code = valid_code
        else:
            self.seed.valid_jax_code = valid_code
        self.session.commit()


def export_validated_seed(seed: ClusterTestSeed):  # 导出种子中各个库的测试用例为py文件
    if seed.is_validated:
        # 先构建输出路径
        seed_folder_name = ''
        if seed.pytorch_api_id:
            seed_folder_name = seed_folder_name + f'Pytorch({seed.pytorch_api.name})'
        if seed.tensorflow_api_id:
            seed_folder_name = seed_folder_name + f'Tensorflow({seed.tensorflow_api.name})'
        if seed.jax_api_id:
            seed_folder_name = seed_folder_name + f'JAX({seed.jax_api.name})'
        output_combination_folder_path = f'seeds/validated_seeds/zero-shot/{seed.cluster_id}/' + seed_folder_name
        if not os.path.exists(output_combination_folder_path):  # 创建API组合的文件夹
            os.makedirs(output_combination_folder_path, exist_ok=True)
        # 创建一个新的输出文件夹
        output_folder_path = f"{output_combination_folder_path}/seed_{len(os.listdir(output_combination_folder_path)) + 1}"
        if not os.path.exists(output_folder_path):  # 创建API组合的文件夹
            os.makedirs(output_folder_path, exist_ok=True)
        # 随后在输出路径下导出各个库的测试用例
        # 导出seed.valid_pytorch_code到output_path/torch_seed.py
        if seed.valid_pytorch_code:
            with open(f'{output_folder_path}/torch_seed.py', 'w') as f:
                f.write(seed.valid_pytorch_code)
        # 导出seed.valid_tensorflow_code到output_path/tf.py
        if seed.valid_tensorflow_code:
            with open(f'{output_folder_path}/tf_seed.py', 'w') as f:
                f.write(seed.valid_tensorflow_code)
        # 导出seed.valid_jax_code到output_path/jax.py
        if seed.valid_jax_code:
            with open(f'{output_folder_path}/jax_seed.py', 'w') as f:
                f.write(seed.valid_jax_code)


def validate_all_seeds():
    session = get_session()
    openai_client = get_openai_client()
    # 查询所有未经验证的种子
    unvalidated_seeds = session.query(ClusterTestSeed).filter(ClusterTestSeed.is_validated == False).all()
    while unvalidated_seeds:
        print("----------------------------------------------------------------------------------")
        seed = unvalidated_seeds[0]
        is_valid = True
        if seed.pytorch_api_id and seed.valid_pytorch_code is None:
            torch_validator = SeedValidator(session, openai_client, seed, seed.raw_pytorch_code, "Pytorch")
            valid_code = torch_validator.validate()
            if valid_code is None:
                is_valid = False
                print("Failed to fix the Pytorch code.")
        if seed.tensorflow_api_id and seed.valid_tensorflow_code is None:
            tf_validator = SeedValidator(session, openai_client, seed, seed.raw_tensorflow_code, "Tensorflow")
            valid_code = tf_validator.validate()
            if valid_code is None:
                is_valid = False
                print("Failed to fix the Tensorflow code.")
        if seed.jax_api_id and seed.valid_jax_code is None:
            jax_validator = SeedValidator(session, openai_client, seed, seed.raw_jax_code, "JAX")
            valid_code = jax_validator.validate()
            if valid_code is None:
                is_valid = False
                print("Failed to fix the Jax code.")
        if is_valid:
            seed.is_validated = True
            session.commit()
            print(f"Seed({seed.id}) validated successfully.")

        # 更新未校验的种子集
        unvalidated_seeds = session.query(ClusterTestSeed).filter(ClusterTestSeed.is_validated == False).all()
        # 打印未校验的种子数量
        total_seeds_num = session.query(ClusterTestSeed).count()
        print(f"Unvalidated / Total: {len(unvalidated_seeds)} / {total_seeds_num}")


def export_all_validated_seeds():
    session = get_session()
    # 查询所有已经验证的种子
    validated_seeds = session.query(ClusterTestSeed).filter(ClusterTestSeed.is_validated == True).all()
    for seed in validated_seeds:
        export_validated_seed(seed)


if __name__ == '__main__':
    #validate_all_seeds()
    export_all_validated_seeds()
