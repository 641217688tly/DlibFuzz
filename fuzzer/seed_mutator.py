from utils import *
import json
import os
import shutil
import re

project_root = os.path.dirname(os.path.dirname(__file__))


class SeedMutator:
    def __init__(self, session, openai_client, seeds_dir):
        self.session = session
        self.openai_client = openai_client
        self.seeds_dir = seeds_dir

    def process_analysis_json(self, analysis_json_path):
        with open(analysis_json_path, 'r') as f:
            analysis_data = json.load(f)

        divergent_results = analysis_data.get('divergent_results', [])
        for result in divergent_results:
            self.process_divergent_result(result)

        error_results = analysis_data.get('error_results', [])
        for result in error_results:
            self.process_error_result(result)

    def process_divergent_result(self, result):
        seed_path = os.path.join(self.seeds_dir, result['seed'])
        details = result.get('details', [])

        with open(seed_path, 'r') as f:
            seed_code = f.read()

        prompt = f"""
The following code produces different results across different deep learning frameworks:\n
{seed_code}
The results are:
{details}
Please analyze the cause of the divergence and provide a corrected version of the code that ensures consistent behavior across all frameworks.

Output your response in the following format:
- Provide the corrected code that resolves the divergence without changing the original logic with import.
"""
        response = self.query_openai(prompt)
        if response is None:
            print(f"Failed to get response from OpenAI for seed: {seed_path}")
            return

        corrected_code = self.extract_code_from_response(response)

        if corrected_code:
            self.save_corrected_code(result['seed'], corrected_code, response)
        else:
            print(f"No corrected code returned for {seed_path}")

    def process_error_result(self, result):
        seed_path = os.path.join(self.seeds_dir, result['seed'])
        details = result.get('details', [])

        with open(seed_path, 'r') as f:
            seed_code = f.read()

        prompt = f"""
The following code is composed of different deep learning libraries. The logic is consistent, but some or all of them coused errors during execution:\n
{seed_code}
The errors are:\n
{details}

Here are the definitions of the error types you should use to analyze the results:
- **Run-time Errors**: These are common errors related to code execution, such as shape mismatches, type errors, or invalid input values. Such as "RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x3 and 1x3)", "TypeError: len() of unsized object" and "NameError: name 'input_data' is not defined".
- **Abnormal Errors**: These errors are out of expectation and may indicate a bug in the API. They are not directly related to the execution logic. Such as "NoValidOutput: This might be due to an internal error".


Please determine the error pairs of these deep learning libraries:
1. Have answer: One or more library has **specific values** to output. Please fix the code of other libraries. But keep the original logic.
1. Consistent: All the error messages are **Run-time Errors**. This means that the errors are simple and have the same meaning. Do not fix the code.
2. Inconsistent: All the error messages are **Run-time Errors**. But the error messages are **totally different**. Please fix the code.
2. Abnormal: One or more error messages is **Abnormal Errors**. Do not fix the code. **Be careful with confirming the error type.**

Output your response **strictly** in the following format:
1. If have answer: [Provide the corrected code of all library with import. And simply tell the user why.]
2. If consistent: [Indicate that the error messages are consistent, and no correction is needed. **Do not return any code corrections.**]
3. If inconsistent: [Provide the corrected code with import. And tell the user why.]
4. If Abnormal Errors: [Indicate that these are Abnormal Errors. **Do not return any code corrections.**]
"""
        response_text = self.query_openai(prompt)
        corrected_code = self.extract_code_from_response(response_text)

        if corrected_code:
            self.save_corrected_code(result['seed'], corrected_code, response_text)
        else:
            self.save_abnormal_error(result['seed'], response_text)

    def query_openai(self, prompt, max_retry_limit=5, model="gpt-4o-mini"):
        attempt_num = 0
        while attempt_num < max_retry_limit:
            try:
                response = self.openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system",
                         "content": "You're an AI assistant adept at analyzing deep learning code across multiple frameworks."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=1,
                )
                response_data = response.choices[0].message.content
                print(response_data)
                return response_data
            except Exception as e:
                print(f"Failed to get response due to: \n{e} \nRetrying(Current attempt: {attempt_num + 1})...")
                attempt_num += 1
                self.session.rollback()
        print("Max attempts reached. Unable to get valid response.")
        return None

    def extract_code_from_response(self, response_text):
        code_blocks = re.findall(r'```(?:python)?(.*?)```', response_text, re.DOTALL)
        if code_blocks:
            # 清理提取出的代码，去掉首尾的空白字符
            return code_blocks[0].strip()
        else:
            return None

    def save_corrected_code(self, seed_relative_path, corrected_code, response_text=None):
        # 在 'seeds/run_1/iteration_1/' 路径中创建新的文件夹
        run_folder = os.path.join('seeds', 'run_1', 'iteration_1', os.path.dirname(seed_relative_path))
        os.makedirs(run_folder, exist_ok=True)

        base_name = os.path.basename(seed_relative_path)
        corrected_file_path = os.path.join(run_folder, base_name)

        with open(corrected_file_path, 'w') as f:
            f.write(corrected_code)

        if response_text:
            response_file_path = os.path.join(run_folder, base_name + '_analysis.txt')
            with open(response_file_path, 'w') as f:
                f.write(response_text)

        print(f"Corrected code saved to {corrected_file_path}")

    def save_abnormal_error(self, seed_relative_path, response_text=None):
        abnormal_folder = os.path.join('seeds', 'abnormal_seeds', os.path.dirname(seed_relative_path))
        os.makedirs(abnormal_folder, exist_ok=True)
        base_name = os.path.basename(seed_relative_path)
        destination_path = os.path.join(abnormal_folder, base_name)
        seed_path = os.path.join(self.seeds_dir, seed_relative_path)
        shutil.copy(seed_path, destination_path)
        print(f"Seed file {seed_relative_path} saved to {destination_path} due to Abnormal Error")


def run():
    session = get_session()
    openai_client = get_openai_client()
    # seeds_dir = os.path.join(project_root, 'fuzzer/seeds/initial_seeds')
    seeds_dir = os.path.join(project_root, 'fuzzer/seeds/test_seeds')
    analysis_file = os.path.join(project_root, 'oracle/outputs/run_5/iteration_1/analysis_1.json')
    SeedMutator(session, openai_client, seeds_dir).process_analysis_json(analysis_file)


if __name__ == '__main__':
    import time

    start_time = time.time()
    run()
    print(f"Execution time: {time.time() - start_time} seconds")
