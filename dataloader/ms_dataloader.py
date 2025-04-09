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
            print(f"进度: {i + 1}/{len(unhandled_files)}")
        except Exception as e:
            print(f"处理文件 {file} 时出错: {str(e)}")
            # 打印详细的堆栈跟踪以便调试
            import traceback
            traceback.print_exc()
    print("所有需要处理的文件已处理完成")


if __name__ == '__main__':
    # 处理未处理的文档
    process_unhandled_docs()
