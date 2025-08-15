import logging
import os
import re
from bs4 import BeautifulSoup

import utils
from utils import get_session
from orm import API


def parse_jax_api(html_content, module_name):
    """
    从单个 JAX API 的 HTML 页面中解析 API 信息 (由 Sphinx 生成的文档)
    """
    soup = BeautifulSoup(html_content, 'html.parser')

    # 找到 API 定义的 dl 标签
    dl_func = soup.find("dl", class_=re.compile(r'py (function|method|class|property|data)'))
    if not dl_func:
        logging.warning("No 'dl' tag found in the HTML content.")
        return {}
    dt = dl_func.find("dt")
    dd = dl_func.find("dd")
    if not dt or not dd:
        logging.warning("No 'dt' or 'dd' tag found in 'dl' tag.")
        return {}

    # API name
    name_span = dt.find("span", class_="descname")
    if name_span:
        api_name = name_span.get_text(strip=True)
    else:
        api_name = dt.get_text(separator=" ", strip=True)

    # 获取 signature(去掉 [source] 等无关部分)
    signature = dt.get_text(separator="", strip=True)
    signature = re.sub(r'\[source\]', '', signature).strip().rstrip('#')

    # 获取 description (第一个 <p> 通常为函数的简短描述)
    description = ""
    p_tags = dd.find_all("p", recursive=False)
    if p_tags:
        description = p_tags[0].get_text(separator="", strip=True)

    # Parameters & Output (查找 dd 中嵌套的 <dl class="field-list simple">)
    parameters = ""
    output = ""
    field_dl = dd.find("dl", class_="field-list simple")
    if field_dl:
        for dt_field in field_dl.find_all("dt"):
            dt_text = dt_field.get_text(strip=True)
            if dt_text.startswith("Parameters"):
                param_dd = dt_field.find_next_sibling("dd")
                if param_dd:
                    parameters = param_dd.get_text(separator="", strip=True)
            elif dt_text.startswith("Return type"):
                return_dd = dt_field.find_next_sibling("dd")
                if return_dd:
                    output = return_dd.get_text(separator=" ", strip=True)

    # Example
    example = ""
    example_section = dd.find("div", class_="doctest highlight-default notranslate")
    if example_section:
        example = example_section.get_text(separator="", strip=False).strip()

    full_name = module_name + "." + api_name if module_name else api_name

    return {
        "name": api_name,
        "lib": "JAX",
        "version": "0.4.13",
        "module": module_name,
        "full_name": full_name,
        "signature": signature,
        "parameters": parameters,
        "attributes": "",
        "output": output,
        "description": description,
        "example": example
    }


def add_jax_apis_from_doc(root_dir, lib_ver):  # add jax api from html folder
    """
    遍历目录下所有 API 详细的 HTML 文件，解析并将信息插入数据库
    """
    session = get_session()
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".html"):
                # 模块首页html与所在目录同名，跳过
                dir_name = os.path.basename(root)
                file_base = file.replace(".html", "")
                if file_base == dir_name:
                    continue
                file_path = os.path.join(root, file)
                print(f"Processing API file: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
                # 通过文件路径获取完整的模块名
                relative_path = os.path.relpath(file_path, root_dir)
                module_from_path = os.path.dirname(relative_path).replace(os.sep, '.')
                api_info = parse_jax_api(html_content, module_from_path)
                if not api_info:
                    print(f"Failed to parse API details from {file_path}")
                    continue
                # 验证API是否已经存在
                api_exists = session.query(API).filter_by(full_name=api_info['full_name'], lib='JAX', version=api_info["version"]).first()
                if api_exists:
                    continue
                # 验证API是否有效
                is_valid = utils.validate_api_existence(f"{api_info['module']}.{api_info['name']}") 
                if not is_valid:
                    continue
                # 如果api_info内的某个键的值为None，则使用utils.inspect_api_info获取的值
                additional_api_info = utils.inspect_api_info(api_info["module"], api_info["name"])
                for key, value in api_info.items():
                    if value is None or value == "":
                        api_info[key] = additional_api_info.get(key, "")
                api_entry = API(
                    name=api_info["name"],
                    lib=api_info["lib"],
                    version=lib_ver,
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
                print(f"Inserted API: {api_entry.full_name}")


if __name__ == "__main__":
    jax_docs_folder_path = "./../data/docs/jax/0.4.13/handled"
    add_jax_apis_from_doc(jax_docs_folder_path, "0.4.13")
