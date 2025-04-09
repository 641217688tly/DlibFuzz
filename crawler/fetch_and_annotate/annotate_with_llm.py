import os
import re

import httpx
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量（例如 API_KEY）
load_dotenv()

# 初始化 OpenAI 客户端，api_key 从环境变量中读取
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", ""),
    http_client= httpx.Client(proxies={
        "http://": "http://127.0.0.1:7890",
        "https://": "http://127.0.0.1:7890"
    })
)


def send_to_openai(prompt_template: str, content_to_be_annotated: str):
    """
    该函数将 prompt_template 和需要标注的文本 content_to_be_annotated
    一起发送给 OpenAI 的对话接口，并返回模型的响应。
    """
    # 构建对话格式的消息，system 角色携带 prompt_template，user 角色携带实际待标注内容
    message = [
        {
            'role': 'system',
            'content': prompt_template
        },
        {
            'role': 'user',
            'content': 'Now, please label the following issue:' + content_to_be_annotated
        }
    ]

    # 创建 ChatCompletion，并指定模型和返回格式为 JSON
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=message,
        temperature=0.0
    )
    # 获取返回内容
    chat_response = response.choices[0].message.content
    return chat_response


def save_to_file(directory: str, prefix: str, title: str, item: str):
    """
    将标注后的结果保存到本地文件系统。
    directory: 根目录
    prefix: 子目录及文件名前缀
    title: 文件名中的标题（通常是 issue ID）
    item: 要保存的文本内容
    """
    # 如果目录不存在，就先创建
    if not os.path.exists(directory):
        os.makedirs(directory)
    # 拼接子目录路径
    file_dir = os.path.join(directory, prefix)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    # 构建文件名和路径
    file_name = f"{prefix}_{title}.json"
    file_path = os.path.join(file_dir, file_name)
    # 以写入的方式打开文件并保存内容
    with open(file_path, 'w') as file:
        file.write(item)


def annotate_and_save_issues(prompt_template: str, issues: dict, directory: str, prefix: str):
    """
    使用指定的 prompt_template 对 issues 字典中的每个 issue 进行标注，
    然后将结果保存到指定目录和前缀对应的文件中。
    """
    annotated = {}
    # 遍历所有 issue
    for key, value in issues.items():
        print(f"Annotating issue {key}...")
        # 调用 send_to_openai 函数进行标注
        annotated[key] = send_to_openai(prompt_template, value)
        # 将标注结果保存到指定路径
        save_to_file(directory, prefix, key, annotated[key])
    return annotated


def extract_number_from_filename(filename):
    # 从文件名中抽取数字并返回，未找到则返回-1
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1


def read_and_annotate_issues(lib: str, prompt_template: str, uid: str):
    target_folder_path = f"results_{uid}/{lib}_issues"
    output_folder_path = f"results_{uid}_annotated/{lib}_issues"

    # 获取文件列表并进行排序
    all_files = os.listdir(target_folder_path)
    # 在这里对文件进行数字排序
    sorted_files = sorted(all_files, key=extract_number_from_filename)

    issues = {}
    for file in sorted_files:
        # 构建输出文件名，根据 save_to_file 中的命名规则进行对应
        file_title = os.path.splitext(file)[0]
        output_file_name = f"{lib}_issues_{file_title}.json"  # prefix_{title}.json
        output_file_path = os.path.join(output_folder_path, output_file_name)

        # 如果已存在输出文件，则跳过
        if os.path.exists(output_file_path):
            print(f"Skipping {file}, as it has already been annotated.")
            continue

        # 否则，读取文件并放入待标注字典
        with open(os.path.join(target_folder_path, file), 'r') as f:
            content = f.read()
            issues[file_title] = content

    if issues:
        print(f"Annotating {lib} issues...")
        annotated = annotate_and_save_issues(
            prompt_template.format(lib.capitalize(), lib.capitalize(), lib.capitalize()),
            issues,
            f"results_{uid}_annotated",
            f"{lib}_issues"
        )
        return annotated
    else:
        print(f"No new {lib} issues to annotate.")
        return {}


if __name__ == "__main__":
    # 从命令行参数获取 index_fetch_results（也可以直接赋值，下面是示例）
    # index_fetch_results = str(sys.argv[1])
    uid = '1229174635'

    # 如果文件夹不存在则退出
    if not os.path.exists(f'results_{uid}'):
        print(f"results_{uid} does not exist.")
        exit(1)

    # 从本地文件 prompt_template.txt 中读取提示模板
    with open('prompt_template.txt', 'r') as f:
        prompt_template = f.read()

    # 使用抽取出的函数来分别标注 PyTorch、JAX、MindSpore 的 issues
    #read_and_annotate_issues("pytorch", prompt_template, uid)
    #read_and_annotate_issues("jax", prompt_template, uid)
    read_and_annotate_issues("jittor", prompt_template, uid)
    print("Done!")
