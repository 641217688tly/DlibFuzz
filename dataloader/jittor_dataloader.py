import logging
import os
import re
from orm import API
from bs4 import BeautifulSoup
from utils import get_session


def parse_jittor_api(html_content, module_name):
    """
    从单个 Jittor 模块的 HTML 文件中解析所有 API 信息。
    一个 HTML 文件即代表一个模块，页面中可能包含多个 <dl class="py function"> 等标签。
    返回一个 API 字典列表。
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    # 找到 API 定义的 dl 标签
    dl_elements = soup.find_all("dl", class_=re.compile(r'py (function|method|class|property|data)'))
    api_list = []
    if not dl_elements:
        logging.warning("No API definitions found in module %s", module_name)
        return api_list
    for dl in dl_elements:
        dt = dl.find("dt")
        dd = dl.find("dd")
        if not dt or not dd:
            continue
        # API Name
        name_span = dt.find("span", class_="descname")
        if name_span:
            api_name = name_span.get_text(strip=True)
        else:
            dt_text = dt.get_text(strip=True)
            m = re.match(r'(.+?)\s*\(', dt_text)
            if m:
                api_name = m.group(1).strip()
            else:
                api_name = dt_text
        # Signature
        signature = dt.get_text(separator=" ", strip=True)
        signature = re.sub(r'\[源代码\]', '', signature).strip()
        # Description
        p_tag = dd.find("p")
        description = p_tag.get_text(" ", strip=True) if p_tag else dd.get_text(" ", strip=True)
        # Parameters & Output
        parameters = ""
        output = ""
        for dt_field in dd.find_all("dt"):
            dt_field_text = dt_field.get_text(strip=True)
            if "参数" in dt_field_text:
                param_dd = dt_field.find_next_sibling("dd")
                if param_dd:
                    parameters = param_dd.get_text(" ", strip=True)
            elif "返回" in dt_field_text:
                output_dd = dt_field.find_next_sibling("dd")
                if output_dd:
                    output = output_dd.get_text(" ", strip=True)
        # Example
        example = ""
        example_div = dd.find("div", class_=re.compile(r'doctest'))
        if example_div:
            example = example_div.get_text("", strip=True).strip()
        full_name = module_name + "." + api_name if module_name else api_name
        api_list.append({
            "name": api_name,
            "lib": "Jittor",
            "version": "1.3.9.2",
            "module": module_name,
            "full_name": full_name,
            "signature": signature,
            "parameters": parameters,
            "attributes": "",
            "output": output,
            "description": description,
            "example": example
        })
    return api_list


def process_jittor_api(root_dir):  # add jittor api from html folder
    """
    遍历指定目录下所有 Jittor HTML 文件（每个文件代表一个模块）
    """
    session = get_session()
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".html"):
                file_path = os.path.join(root, file)
                print(f"Processing Jittor API file: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                # 这里用文件名（不含扩展名）作为模块名称
                module_name = os.path.splitext(file)[0]
                api_list = parse_jittor_api(html_content, module_name)
                if not api_list:
                    print(f"Failed to parse API details from {file_path}")
                    continue
                for api_info in api_list:
                    api_entry = API(
                        name=api_info["name"],
                        lib=api_info["lib"],
                        version=api_info["version"],
                        module=api_info["module"],
                        full_name=api_info["full_name"],
                        signature=api_info["signature"],
                        parameters=api_info["parameters"],
                        attributes=api_info["attributes"],
                        output=api_info["output"],
                        description=api_info["description"],
                        example=api_info["example"],
                    )
                    session.add(api_entry)
                    session.commit()
                    print(f"Inserted Jittor API: {api_entry.full_name}")


def export_plain_text_file(input_folder, output_folder):
    """
    将input_folder下的所有jittor文档(html文件)转换为纯文本格式，
    只保留API文档相关内容，并保存到output_folder下的同名.txt文件中

    Args:
        input_folder (str): 输入文件夹路径，包含HTML文件
        output_folder (str): 输出文件夹路径，将保存转换后的TXT文件
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取输入文件夹中的所有HTML文件
    html_files = [f for f in os.listdir(input_folder) if f.endswith('.html')]

    # 处理每个HTML文件
    for html_file in html_files:
        input_path = os.path.join(input_folder, html_file)
        output_file = os.path.splitext(html_file)[0] + '.txt'
        output_path = os.path.join(output_folder, output_file)

        try:
            # 读取HTML文件
            with open(input_path, 'r', encoding='utf-8') as file:
                html_content = file.read()

            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # 只提取主要内容区域，通常是包含API文档的部分
            main_content = soup.find('div', {'role': 'main'})

            if main_content:
                # 移除可能存在的页脚、翻页按钮等无关元素
                for element in main_content.find_all(['footer', 'nav']):
                    element.decompose()

                # 提取标题和API文档内容
                title = soup.find('title')
                title_text = title.get_text() if title else ""

                # 提取文本内容
                content_text = main_content.get_text(separator='\n', strip=True)

                # 清理文本（移除多余的空行和空格）
                content_text = re.sub(r'\n\s*\n', '\n\n', content_text)

                # 组合最终文本
                final_text = f"{title_text}\n\n{content_text}" if title_text else content_text

                # 写入到输出文件
                with open(output_path, 'w', encoding='utf-8') as file:
                    file.write(final_text)

                print(f"已转换: {html_file} -> {output_file}")
            else:
                # 如果找不到主要内容区域，则提取整个body的内容
                body = soup.find('body')
                if body:
                    # 移除导航栏、页脚等元素
                    for nav in body.find_all(['nav', 'footer', 'div'], class_=['wy-nav-side', 'rst-footer-buttons']):
                        nav.decompose()

                    # 提取文本
                    body_text = body.get_text(separator='\n', strip=True)
                    body_text = re.sub(r'\n\s*\n', '\n\n', body_text)

                    # 写入到输出文件
                    with open(output_path, 'w', encoding='utf-8') as file:
                        file.write(body_text)

                    print(f"已转换(使用body内容): {html_file} -> {output_file}")
                else:
                    print(f"警告: 无法在 {html_file} 中找到有效内容")

        except Exception as e:
            print(f"处理文件 {html_file} 时出错: {str(e)}")

    print(f"完成! 共处理 {len(html_files)} 个HTML文件")


if __name__ == "__main__":
    jittor_docs_folder_path = "./../data/docs/jittor/1.3.9.2/handled"
    rag_docs_folder_path = "./../rag/docs/jittor/1.3.9.2/"
    # process_jittor_api(jittor_docs_folder_path)
    export_plain_text_file(jittor_docs_folder_path, rag_docs_folder_path)
