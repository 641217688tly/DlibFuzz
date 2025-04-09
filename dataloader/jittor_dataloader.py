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

if __name__ == "__main__":
    jittor_docs_folder_path = "./../data/docs/jittor/1.3.9.2/handled"
    rag_docs_folder_path = "./../rag/docs/jittor/1.3.9.2/"
    process_jittor_api(jittor_docs_folder_path)