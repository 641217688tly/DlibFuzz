import os
from bs4 import BeautifulSoup
from orm import API
from utils import get_session, inspect_api_info, validate_api_existence
from bs4.element import NavigableString


class MindsporeDocumentationHandler:
    def __init__(self, raw_file_path, core_html_file_path=None, plain_text_file_path="./../rag/docs/Mindspore/2.5.0/"):
        self.api_full_name = raw_file_path.split('/')[-1].replace('.html', '')
        self.raw_file_path = raw_file_path
        if core_html_file_path is None:
            self.core_html_file_path = raw_file_path.replace('raw', 'handled')
        self.plain_text_file_path = os.path.join(plain_text_file_path, f"{self.api_full_name}.txt")

        self.export_core_html_file()
        self.export_plain_text_file()

    def export_core_html_file(self):  # 从HTML中剔除与API无关的内容后导出该HTML文档
        # 读取HTML文件
        with open(self.raw_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')  # 使用BeautifulSoup解析HTML

        # 移除所有的脚本和样式
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        # 移除导航菜单、页脚等无关元素
        for nav in soup.find_all(['nav', 'div'], class_=['wy-nav-side', 'wy-nav-top', 'rst-footer-buttons']):
            nav.decompose()

        # 移除页脚
        footer = soup.find('footer')
        if footer:
            footer.decompose()

        # 提取标题
        title_element = soup.find('title')
        title_text = title_element.get_text() if title_element else ""

        # 提取相关内容 - 在MindSpore文档中，主要内容通常在role="main"的div中
        main_content = soup.find('div', {'role': 'main'})

        # 如果找到主内容区域
        if main_content:
            # 查找API文档的主要部分，通常是section标签
            api_section = main_content.find('section')

            # 查找函数/类定义部分
            api_definition = None
            if api_section:
                # 对于函数，通常在dl标签中，class="py function"
                api_definition = api_section.find('dl',
                                                  {'class': lambda x: x and ('py function' in x or 'py class' in x)})

            # 构建最终的HTML内容
            relevant_content = ""

            # 添加API标题
            if api_section:
                heading = api_section.find('h1')
                if heading:
                    relevant_content += str(heading)

            # 添加API定义和文档内容
            if api_definition:
                relevant_content += str(api_definition)
            else:
                # 如果没有找到特定的API定义，则保留整个section内容
                if api_section:
                    relevant_content += str(api_section)
                else:
                    # 如果没有找到section，则保留整个main_content（去除导航和页脚后）
                    relevant_content += str(main_content)
        else:
            # 如果没有找到主内容区域，尝试直接查找文档内容
            document = soup.find('div', {'class': 'document'})
            if document:
                relevant_content = str(document)
            else:
                # 如果无法找到特定内容区域，则保留整个body内容（已经移除了导航和页脚）
                body = soup.find('body')
                relevant_content = str(body) if body else ""

        # 添加meta标签，明确文件编码为UTF-8
        final_html = f'''<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title_text}</title>
        </head>
        <body>
        {relevant_content}
        </body>
        </html>'''

        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.core_html_file_path), exist_ok=True)

        # 将精简后的内容写入新的HTML文件
        with open(self.core_html_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(final_html)

    def export_plain_text_file(self):  # 从HTML中提取纯文本内容并导出为txt文件
        """
        将HTML文件转换为纯文本格式并保存为txt文件
        - 移除所有HTML标签
        - 保留文本内容的结构和层次
        - 保存到指定的plain_text_file_path目录
        """
        # 读取HTML文件
        with open(self.core_html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # 提取API名称作为标题
        title = self.api_full_name

        # 提取文本内容
        text_content = []

        # 添加标题
        text_content.append(f"# {title}")
        text_content.append("")  # 空行

        # 处理函数/类定义
        api_definition = soup.find('dl', {'class': lambda x: x and ('py function' in x or 'py class' in x)})
        if api_definition:
            # 提取签名信息
            signature = api_definition.find('dt')
            if signature:
                sig_text = signature.get_text().strip()
                # 清理签名文本，移除[source]和多余空格
                sig_text = sig_text.replace('[source]', '').strip()
                text_content.append(f"## Signature:")
                text_content.append(f"```python")
                text_content.append(sig_text)
                text_content.append(f"```")
                text_content.append("")

        # 提取描述信息
        description = None
        if api_definition:
            description = api_definition.find('dd')
            if description and description.find('p'):
                desc_text = description.find('p').get_text().strip()
                text_content.append("## Description:")
                text_content.append(desc_text)
                text_content.append("")

                # 提取描述中的注意事项(Note)
                note_sections = description.find_all('div', {'class': 'admonition note'})
                for note in note_sections:
                    if note.find('p', {'class': 'admonition-title'}):
                        note_title = note.find('p', {'class': 'admonition-title'}).get_text().strip()
                        text_content.append(f"### {note_title}:")

                        # 提取注意事项内容
                        note_content = note.find_all(['p', 'ul'])
                        for content in note_content:
                            if content.get('class') and 'admonition-title' in content.get('class'):
                                continue  # 跳过标题

                            if content.name == 'ul':
                                for li in content.find_all('li'):
                                    li_text = li.get_text().strip()
                                    text_content.append(f"- {li_text}")
                            else:
                                p_text = content.get_text().strip()
                                if p_text and p_text != note_title:  # 避免重复标题
                                    text_content.append(p_text)

                        text_content.append("")  # 空行

        # 提取参数信息
        params_section = None
        if description:
            params_section = description.find('dl', {'class': 'field-list'})

        if params_section:
            # 处理参数、返回值、异常等字段
            for dt in params_section.find_all('dt'):
                field_name = dt.get_text().strip()
                dd = dt.find_next('dd')
                if dd:
                    text_content.append(f"## {field_name}:")
                    # 处理参数列表
                    param_list = dd.find_all('li')
                    if param_list:
                        for li in param_list:
                            param_text = li.get_text().strip().replace('\n', ' ')
                            text_content.append(f"- {param_text}")
                    else:
                        field_text = dd.get_text().strip().replace('\n', ' ')
                        text_content.append(field_text)
                    text_content.append("")

        # 提取支持的平台
        platforms = soup.find('dl', {'class': 'simple'})
        if platforms and "Supported Platforms:" in platforms.get_text():
            platform_text = platforms.find('dd').get_text().strip()
            text_content.append("## Supported Platforms:")
            text_content.append(platform_text)
            text_content.append("")

        # 提取示例代码
        examples_header = soup.find('p', {'class': 'rubric'}, text='Examples')
        if examples_header:
            text_content.append("## Examples:")

            # 检查示例前是否有注意事项
            next_elem = examples_header.next_sibling
            while next_elem and (
                    not next_elem.name or next_elem.name != 'div' or 'doctest' not in next_elem.get('class', '')):
                if next_elem.name == 'div' and 'admonition note' in next_elem.get('class', ''):
                    note = next_elem
                    if note.find('p', {'class': 'admonition-title'}):
                        note_title = note.find('p', {'class': 'admonition-title'}).get_text().strip()
                        text_content.append(f"### {note_title}:")

                        # 提取注意事项内容
                        note_content = note.find_all(['p', 'ul', 'a'])
                        for content in note_content:
                            if content.get('class') and 'admonition-title' in content.get('class'):
                                continue  # 跳过标题

                            if content.name == 'ul':
                                for li in content.find_all('li'):
                                    li_text = li.get_text().strip()
                                    text_content.append(f"- {li_text}")
                            elif content.name == 'a':
                                a_text = content.get_text().strip()
                                a_href = content.get('href', '')
                                text_content.append(f"[{a_text}]({a_href})")
                            else:
                                p_text = content.get_text().strip()
                                if p_text and p_text != note_title:  # 避免重复标题
                                    text_content.append(p_text)

                        text_content.append("")  # 空行
                next_elem = next_elem.next_sibling

            # 提取示例代码
            example_block = soup.find('div', {'class': 'doctest'})
            if example_block:
                code_block = example_block.find('div', {'class': 'highlight'})
                if code_block:
                    example_text = code_block.get_text().strip()
                    text_content.append("```python")
                    text_content.append(example_text)
                    text_content.append("```")
                    text_content.append("")

        # 提取教程链接
        tutorial_section = soup.find('dt', text='Tutorial Examples:')
        if tutorial_section:
            text_content.append("## Tutorial Examples::")
            dd = tutorial_section.find_next('dd')
            if dd:
                links = dd.find_all('a')
                for link in links:
                    text_content.append(f"- {link.get_text().strip()}: {link.get('href')}")
                text_content.append("")

        # 处理类的方法
        methods = soup.find_all('dl', {'class': 'py method'})
        if methods:
            text_content.append("## Methods:")
            for method in methods:
                method_dt = method.find('dt')
                if method_dt:
                    method_name = method_dt.find('span', {'class': 'sig-name'}).get_text() if method_dt.find('span', {
                        'class': 'sig-name'}) else "Unknown Method"
                    text_content.append(f"### {method_name}")

                    # 提取方法签名
                    sig_text = method_dt.get_text().strip().replace('[source]', '').strip()
                    text_content.append(f"```python")
                    text_content.append(sig_text)
                    text_content.append(f"```")

                    # 提取方法描述
                    method_dd = method.find('dd')
                    if method_dd and method_dd.find('p'):
                        method_desc = method_dd.find('p').get_text().strip()
                        text_content.append(method_desc)
                        text_content.append("")

                    # 提取方法中的注意事项
                    if method_dd:
                        note_sections = method_dd.find_all('div', {'class': 'admonition note'})
                        for note in note_sections:
                            if note.find('p', {'class': 'admonition-title'}):
                                note_title = note.find('p', {'class': 'admonition-title'}).get_text().strip()
                                text_content.append(f"#### {note_title}:")

                                # 提取注意事项内容
                                note_content = note.find_all(['p', 'ul'])
                                for content in note_content:
                                    if content.get('class') and 'admonition-title' in content.get('class'):
                                        continue  # 跳过标题

                                    if content.name == 'ul':
                                        for li in content.find_all('li'):
                                            li_text = li.get_text().strip()
                                            text_content.append(f"- {li_text}")
                                    else:
                                        p_text = content.get_text().strip()
                                        if p_text and p_text != note_title:  # 避免重复标题
                                            text_content.append(p_text)

                                text_content.append("")  # 空行

        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.plain_text_file_path), exist_ok=True)

        # 将提取的文本内容写入txt文件
        with open(self.plain_text_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write('\n'.join(text_content))


def process_unhandled_docs(raw_dir='./../data/docs/ms/2.5.0/api mapping docs/2.5.0/raw/',
                           handled_dir='./../data/docs/ms/2.5.0/api mapping docs/2.5.0/handled/',
                           plain_text_dir='./../rag/docs/mindspore/2.5.0'):
    """
    查找raw_dir中未同时在handled_dir和plain_text_dir中处理过的文档，并进行处理
    """
    # 确保输出目录存在
    os.makedirs(handled_dir, exist_ok=True)
    os.makedirs(plain_text_dir, exist_ok=True)

    # 获取已处理的HTML文件列表
    handled_files = set()
    if os.path.exists(handled_dir):
        handled_files = {f for f in os.listdir(handled_dir) if f.endswith('.html')}

    # 获取已处理的纯文本文件列表（注意：需要将.txt转换为.html以便比较）
    plain_text_files = set()
    if os.path.exists(plain_text_dir):
        plain_text_files = {f.replace('.txt', '.html') for f in os.listdir(plain_text_dir) if f.endswith('.txt')}

    # 获取原始文件列表
    raw_files = set()
    if os.path.exists(raw_dir):
        raw_files = {f for f in os.listdir(raw_dir) if f.endswith('.html')}

    # 找出未在两个目录中同时处理的文件
    # 文件必须同时不在handled_dir和plain_text_dir中，或者只在其中一个目录中存在
    unhandled_files = []
    for f in raw_files:
        is_in_handled = f in handled_files
        is_in_plain_text = f in plain_text_files

        # 如果文件不是同时存在于两个目录，则需要处理
        if not (is_in_handled and is_in_plain_text):
            unhandled_files.append(f)

    print(f"发现 {len(unhandled_files)} 个需要处理的文件")
    print(f"总文件数: {len(raw_files)}, 已处理HTML文件: {len(handled_files)}, 已处理文本文件: {len(plain_text_files)}")

    # 处理未完全处理的文件
    for file, i in zip(unhandled_files, range(len(unhandled_files))):
        raw_file_path = os.path.join(raw_dir, file)
        try:
            handler = MindsporeDocumentationHandler(raw_file_path=raw_file_path)
            print(f"process_unhandled_docs(): {i + 1}/{len(unhandled_files)}")
        except Exception as e:
            print(f"process_unhandled_docs({file}) Error: {str(e)}")
            # 打印详细的堆栈跟踪以便调试
            import traceback
            traceback.print_exc()

class MindsporeAPILoader:
    def __init__(self, core_html_file_path, db_session, lib_ver="2.5.0"):
        self.file_path = core_html_file_path
        self.session = db_session
        self.lib_ver = lib_ver
        self.api_full_name = core_html_file_path.split('/')[-1].replace('.html', '')
        self.save2db(self.extract_api_info())

    def extract_api_info(self):
        """
        从HTML中提取API的信息:
        1. name - API名称
        2. module - 所属模块
        3. full_name - 完整名称
        4. lib - 固定为MindSpore
        5. version - API版本
        6. signature - 函数签名
        7. parameters - 参数信息
        8. attributes - 类属性(如果是类)和类方法
        9. output - 返回值信息
        10. description - API描述
        11. example - 使用示例
        """
        with open(self.file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # 基本信息提取
        full_name = self.api_full_name
        name_parts = full_name.split('.')

        api_info = {
            'name': name_parts[-1],
            'lib': 'MindSpore',
            'version': self.lib_ver,
            'module': '.'.join(name_parts[:-1]) if len(name_parts) > 1 else name_parts[0],
            'full_name': full_name,
            'parameters': "",
            'attributes': "",
            'output': "",
            'description': "",
            'example': "",
            'signature': ""
        }

        # 确定API类型(函数或类)
        is_class = bool(soup.find('dl', {'class': 'py class'}))
        is_function = bool(soup.find('dl', {'class': 'py function'}))

        # 提取signature(函数签名)
        sig_element = soup.find('dt', {'class': 'sig sig-object py'})
        if sig_element:
            # 获取原始签名文本
            raw_signature = sig_element.get_text(strip=True)

            # 移除[source]¶和其他可能的无关内容
            if '[source]' in raw_signature:
                raw_signature = raw_signature.split('[source]')[0]
            if '¶' in raw_signature:
                raw_signature = raw_signature.split('¶')[0]
            if raw_signature.startswith('class'):
                raw_signature = raw_signature[5:].strip()  # 移除"class"及其后面的空格

            api_info['signature'] = raw_signature

        # 提取description(描述)
        desc_element = soup.find('dd')
        if desc_element:
            # 提取主要描述文本
            description_text = []

            # 获取所有直接子p元素，这些通常是主要描述
            paragraphs = desc_element.find_all('p', recursive=False)
            for p in paragraphs:
                if not p.find('dl') and not p.find_parent('div', {'class': 'admonition note'}):  # 避免包含参数列表和注释
                    description_text.append(p.get_text(strip=True))

            # 提取数学公式和其他说明文本
            math_sections = desc_element.find_all('div', {'class': 'math'})
            for math in math_sections:
                if not math.parent.name == 'dd' or not math.parent.parent.name == 'dl':  # 避免提取参数或返回值中的公式
                    math_text = math.get_text(strip=True)
                    if math_text:
                        description_text.append(f"Formula: {math_text}")

            # 提取注意事项 - 只提取不在Examples部分的注意事项
            notes = desc_element.find_all('div', {'class': 'admonition note'})
            for note in notes:
                # 检查这个note是否在Examples部分
                is_in_examples = False
                prev_elem = note.previous_sibling
                while prev_elem:
                    if hasattr(prev_elem, 'name') and prev_elem.name == 'p' and 'Examples' in prev_elem.get_text():
                        is_in_examples = True
                        break
                    prev_elem = prev_elem.previous_sibling

                if not is_in_examples:
                    note_content = note.find_all('p')
                    if len(note_content) > 1:  # 跳过只有标题的注释
                        note_text = ' '.join([p.get_text(strip=True) for p in note_content[1:]])
                        description_text.append(f"Note: {note_text}")

            api_info['description'] = '\n'.join(description_text) if description_text else ""

            # 根据API类型提取不同信息
            if is_function:
                # 函数参数提取
                param_section = desc_element.find('dl', {'class': 'field-list simple'})
                if param_section:
                    param_items = []
                    param_terms = param_section.find_all('dt')
                    param_descs = param_section.find_all('dd')

                    for i, term in enumerate(param_terms):
                        if 'Parameters' in term.get_text():
                            if i < len(param_descs):
                                param_list = param_descs[i].find_all('li')
                                for param in param_list:
                                    param_items.append(param.get_text(strip=True))

                    api_info['parameters'] = '\n'.join(param_items) if param_items else ""

                # 返回值提取
                return_section = desc_element.find_all('dt')
                for section in return_section:
                    if 'Returns' in section.get_text():
                        next_dd = section.find_next('dd')
                        if next_dd:
                            api_info['output'] = next_dd.get_text(strip=True)
                    elif 'Return type' in section.get_text():
                        next_dd = section.find_next('dd')
                        if next_dd and not api_info['output']:
                            api_info['output'] = next_dd.get_text(strip=True)

                # 如果有Shape信息，也添加到output
                shape_section = desc_element.find('dl')
                if shape_section and 'Shape:' in shape_section.get_text():
                    shape_text = []
                    shape_items = shape_section.find_all(['dt', 'dd'])
                    for item in shape_items:
                        shape_text.append(item.get_text(strip=True))

                    if api_info['output']:
                        api_info['output'] += '\nShape: ' + ' '.join(shape_text)
                    else:
                        api_info['output'] = 'Shape: ' + ' '.join(shape_text)

            elif is_class:
                # 类参数提取
                param_section = desc_element.find('dl', {'class': 'field-list simple'})
                if param_section:
                    param_items = []
                    param_terms = param_section.find_all('dt')
                    param_descs = param_section.find_all('dd')

                    for i, term in enumerate(param_terms):
                        if 'Parameters' in term.get_text():
                            if i < len(param_descs):
                                param_list = param_descs[i].find_all('li')
                                for param in param_list:
                                    param_items.append(param.get_text(strip=True))

                    api_info['parameters'] = '\n'.join(param_items) if param_items else ""

                # 类属性和方法提取
                attributes = []

                # 1. 提取类属性 (如果有的话)
                attr_sections = desc_element.find_all(['dl', 'div'], {'class': ['field-list', 'attribute']})
                for section in attr_sections:
                    if 'Attributes' in section.get_text():
                        attr_items = section.find_all('li')
                        for attr in attr_items:
                            attributes.append(f"Attribute: {attr.get_text(strip=True)}")

                # 2. 提取类方法 - 查找所有 py method 元素
                method_sections = soup.find_all('dl', {'class': 'py method'})
                for method_section in method_sections:
                    # 获取方法签名
                    method_sig = method_section.find('dt', {'class': 'sig sig-object py'})
                    if method_sig:
                        method_signature = method_sig.get_text(strip=True)

                        # 移除[source]¶和其他可能的无关内容
                        if '[source]' in method_signature:
                            method_signature = method_signature.split('[source]')[0]
                        if '¶' in method_signature:
                            method_signature = method_signature.split('¶')[0]

                        # 获取方法描述
                        method_desc = method_section.find('dd')
                        method_desc_text = ""
                        if method_desc:
                            method_desc_paras = method_desc.find_all('p', recursive=False)
                            if method_desc_paras:
                                method_desc_text = " ".join([p.get_text(strip=True) for p in method_desc_paras])

                        # 将方法信息添加到属性列表
                        method_info = f"Method: {method_signature}"
                        if method_desc_text:
                            method_info += f" - {method_desc_text}"

                        attributes.append(method_info)

                api_info['attributes'] = '\n'.join(attributes) if attributes else ""

                # 提取Shape信息作为output
                shape_section = desc_element.find('dl')
                if shape_section:
                    dt_elements = shape_section.find_all('dt')
                    for dt in dt_elements:
                        if 'Shape:' in dt.get_text():
                            shape_info = []
                            # 找到Shape后的所有dd元素
                            dd = dt.find_next('dd')
                            if dd:
                                # 提取Shape信息中的列表项
                                shape_items = dd.find_all('li')
                                for item in shape_items:
                                    shape_info.append(item.get_text(strip=True))

                                api_info['output'] = '\n'.join(shape_info) if shape_info else ""
                            break

            # 提取示例代码及其注释
            # 首先找到Examples标题
            examples_section = None
            examples_title = None
            for p in desc_element.find_all('p', recursive=True):
                if 'Examples' in p.get_text() and not p.find_parent('div', {'class': 'admonition'}):
                    examples_title = p
                    break

            if examples_title:
                # 收集Examples部分的所有内容
                examples_content = []

                # 添加标题
                examples_content.append(examples_title.get_text(strip=True))
                examples_content.append("")  # 空行

                # 查找Examples部分的注释
                current = examples_title.next_sibling
                while current:
                    if hasattr(current, 'name'):
                        if current.name == 'div' and 'admonition note' in current.get('class', []):
                            # 提取注释内容
                            note_title = current.find('p', {'class': 'admonition-title'})
                            if note_title:
                                examples_content.append(note_title.get_text(strip=True) + ":")

                            # 提取注释正文
                            for p in current.find_all('p'):
                                if not p.has_attr('class') or 'admonition-title' not in p['class']:
                                    examples_content.append(p.get_text(strip=True))

                            examples_content.append("")  # 空行
                        elif current.name == 'div' and 'doctest highlight-default notranslate' in current.get('class',
                                                                                                              []):
                            # 已经到了代码块，跳出循环
                            break
                    current = current.next_sibling

                # 提取代码示例
                code_examples = []
                # 1. 查找所有 doctest 示例代码块
                doctest_sections = desc_element.find_all('div', {'class': 'doctest highlight-default notranslate'})
                for section in doctest_sections:
                    # 获取代码块中的内容
                    code_block = section.find('pre')
                    if code_block:
                        code_examples.append(code_block.get_text(strip=False))

                if not code_examples:
                    # 2. 查找所有普通 highlight 示例代码块
                    highlight_sections = desc_element.find_all('div', {'class': 'highlight-default notranslate'})
                    for section in highlight_sections:
                        # 确保不重复添加已经作为doctest添加的代码块
                        if section not in doctest_sections:
                            code_block = section.find('pre')
                            if code_block:
                                code_examples.append(code_block.get_text(strip=False))

                if not code_examples:
                    # 3. 查找其他可能的代码示例格式
                    other_code_blocks = desc_element.find_all('pre')
                    for block in other_code_blocks:
                        # 确保不重复添加已经处理过的代码块
                        if not block.parent.has_attr('class') or not any(
                                cls in ['doctest highlight-default notranslate', 'highlight-default notranslate']
                                for cls in block.parent.get('class', [])):
                            # 检查是否看起来像代码示例（包含>>>前缀）
                            if '>>>' in block.get_text():
                                code_examples.append(block.get_text(strip=False))

                # 添加代码示例到内容中
                if code_examples:
                    examples_content.extend(code_examples)

                # 合并所有示例内容
                if examples_content:
                    api_info['example'] = '\n'.join(examples_content)

            # 提取Shape段落
            # 方法1: 查找专门的Shape部分 - 通常是一个dl元素，其中dt元素包含"Shape:"文本
            shape_dl = None
            # 首先，直接查找包含"Shape:"的dt元素
            shape_dt = desc_element.find('dt', string=lambda text: text and 'Shape:' in text)
            if shape_dt:
                shape_dl = shape_dt.parent
            # 如果没找到，尝试查找所有dl元素，检查其中是否有包含"Shape:"的dt
            if not shape_dl:
                for dl in desc_element.find_all('dl'):
                    if dl.find('dt', string=lambda text: text and 'Shape:' in text):
                        shape_dl = dl
                        break

            if shape_dl:
                shape_info = []
                # 提取Shape部分的所有信息
                # 首先获取dt元素(通常是"Shape:")
                dt = shape_dl.find('dt', string=lambda text: text and 'Shape:' in text)
                if dt:
                    shape_info.append(dt.get_text(strip=True))
                    # 然后获取对应的dd元素(包含实际的shape描述)
                    dd = dt.find_next('dd')
                    if dd:
                        # 如果dd中有列表项，逐个提取
                        list_items = dd.find_all('li')
                        if list_items:
                            for item in list_items:
                                shape_info.append(f"  - {item.get_text(strip=True)}")
                        else:
                            # 如果没有列表项，提取整个dd的文本
                            shape_info.append(dd.get_text(strip=True))
                # 将shape信息添加到output
                shape_text = '\n'.join(shape_info)
                if api_info['output']:
                    api_info['output'] += f"\n{shape_text}"
                else:
                    api_info['output'] = shape_text

            # 方法2: 查找可能包含Shape信息的其他格式
            # 有时Shape信息可能在一个普通的段落或其他元素中
            shape_p = desc_element.find('p', string=lambda text: text and 'Shape:' in text)
            if shape_p:
                if api_info['output']:
                    api_info['output'] += f"\n{shape_p.get_text(strip=True)}"
                else:
                    api_info['output'] = shape_p.get_text(strip=True)

            # 提取Supported Platforms信息
            platforms_dl = desc_element.find_all('dl', {'class': 'simple'})
            for dl in platforms_dl:
                dt = dl.find('dt')
                if dt and 'Supported Platforms' in dt.get_text():
                    dd = dt.find_next('dd')
                    if dd:
                        platform_text = dd.get_text(strip=True)
                        # 将平台信息添加到description
                        if api_info['description']:
                            api_info['description'] += f"\nSupported Platforms: {platform_text}"
                        else:
                            api_info['description'] = f"Supported Platforms: {platform_text}"

            # 提取Raises信息
            raises_section = desc_element.find_all('dt')
            for section in raises_section:
                if 'Raises' in section.get_text():
                    next_dd = section.find_next('dd')
                    if next_dd:
                        raises_items = next_dd.find_all('li')
                        raises_text = []
                        for item in raises_items:
                            raises_text.append(f"- {item.get_text(strip=True)}")

                        # 将异常信息添加到description
                        if raises_text:
                            if api_info['description']:
                                api_info['description'] += f"\n\nRaises:\n" + '\n'.join(raises_text)
                            else:
                                api_info['description'] = f"Raises:\n" + '\n'.join(raises_text)

            # 对提取到的API数据进行后处理
            # 1. 处理signature - 修复: 不要将返回值添加到签名中
            if is_function:
                if not api_info['signature']:  # 如果signature为空，则默认full_name()为signature
                    api_info['signature'] = f"{api_info['full_name']}()"
            elif is_class:
                if not api_info['signature']:  # 如果signature为空，则默认full_name为signature
                    api_info['signature'] = f"{api_info['full_name']}"

            # 2. 使用Inspect库获取更多参数信息
            additional_api_info = inspect_api_info(api_info['module'], api_info['name'])
            # 逐一检查additional_api_info中的条目, 如果api_info中对应的条目为空，则使用additional_api_info中的内容替换
            for key, value in additional_api_info.items():
                if not api_info[key]:
                    api_info[key] = value

            return api_info

    def save2db(self, api_info):
        try:
            # 检查数据库中是否已存在该API
            api_exists = self.session.query(API).filter_by(full_name=api_info['full_name'], lib='MindSpore',
                                                           version=self.lib_ver).first()
            if api_exists:
                return

            # 创建API实例并添加到session
            new_api = API(
                name=api_info['name'],
                lib=api_info['lib'],
                version=api_info['version'],
                module=api_info['module'],
                full_name=api_info['full_name'],
                signature=api_info['signature'],
                parameters=api_info['parameters'],
                attributes=api_info['attributes'],
                output=api_info['output'],
                description=api_info['description'],
                example=api_info['example']
            )
            self.session.add(new_api)
            self.session.commit()
        except Exception as e:
            print(f"Error: {e}")
            self.session.rollback()


def add_mindspore_apis_from_doc(folder_path):
    # 先获取folder_path下的所有HTML文件
    html_files = [f for f in os.listdir(folder_path) if f.endswith('.html')]
    for file, i in zip(html_files, range(len(html_files))):
        file_path = os.path.join(folder_path, file)
        try:
            full_api_name = file.replace('.html', '')
            module_name = '.'.join(full_api_name.split('.')[:-1])
            api_name = full_api_name.split('.')[-1]
            if validate_api_existence(module_name, api_name):  # 如果API能够被正确导入
                loader = MindsporeAPILoader(file_path, get_session())
            print(f"add_mindspore_apis_from_doc(): {i + 1}/{len(html_files)}")
        except Exception as e:
            print(f"add_mindspore_apis_from_doc({file}) Error: {str(e)}")


if __name__ == '__main__':
    # 处理未处理的文档
    # process_unhandled_docs() # done

    # 向数据库中添加API信息
    core_html_file_folder = './../data/docs/ms/2.5.0/api mapping docs/2.5.0/handled/'
    add_mindspore_apis_from_doc(core_html_file_folder)
    #file_path = './../data/docs/ms/2.5.0/api mapping docs/2.5.0/handled/mindspore.dataset.vision.Normalize.html'
    #file_path = './../data/docs/ms/2.5.0/api mapping docs/2.5.0/handled/mindspore.communication.comm_func.all_gather_into_tensor.html'
    #file_path = './../data/docs/ms/2.5.0/api mapping docs/2.5.0/handled/mindspore.dataset.audio.SlidingWindowCmn.html'
    #loader = MindsporeAPILoader(file_path, get_session())
    #api_info = loader.extract_api_info()
    #for key, value in api_info.items():
    #    print(f"{key}: \n{value}")
    #    print("=" * 50)