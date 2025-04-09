import os
from bs4 import BeautifulSoup
from orm import API
from utils import get_session, inspect_api_info, validate_api_existence


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

        # 移除所有的script和style标签
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        # 提取文本内容，保留结构
        text_content = []

        # 提取API名称和标题
        title = soup.find('title')
        if title:
            text_content.append(title.get_text(strip=True))
            text_content.append('=' * len(title.get_text(strip=True)))
            text_content.append('')

        # 处理标题
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            # 根据标题级别添加适当的格式
            level = int(heading.name[1])
            prefix = '#' * level + ' '
            text_content.append(f"{prefix}{heading.get_text(strip=True)}")
            text_content.append('')  # 添加空行

        # 处理函数/类定义 - 特别处理MindSpore文档中的dl.py.function或dl.py.class
        for dl in soup.find_all('dl', {'class': lambda x: x and ('py function' in x or 'py class' in x)}):
            # 提取函数/类签名
            dt = dl.find('dt')
            if dt:
                signature = dt.get_text(strip=True)
                text_content.append(f"定义: {signature}")
                text_content.append('-' * 80)  # 添加分隔线

            # 提取函数/类描述
            dd = dl.find('dd')
            if dd:
                # 提取主要描述
                description_paras = dd.find_all('p', recursive=False)
                for p in description_paras:
                    text_content.append(p.get_text(strip=True))
                    text_content.append('')

                # 提取参数信息
                param_list = dd.find('dl', {'class': 'field-list'})
                if param_list:
                    # 处理参数、返回值、异常等
                    for dt in param_list.find_all('dt'):
                        param_name = dt.get_text(strip=True)
                        text_content.append(f"{param_name}:")

                        # 查找对应的参数描述
                        dd_elem = dt.find_next('dd')
                        if dd_elem:
                            # 处理参数列表
                            ul = dd_elem.find('ul')
                            if ul:
                                for li in ul.find_all('li'):
                                    text_content.append(f"  - {li.get_text(strip=True)}")
                            else:
                                text_content.append(f"  {dd_elem.get_text(strip=True)}")
                        text_content.append('')

                # 提取平台支持信息
                platforms = dd.find('dl', {'class': 'simple'})
                if platforms:
                    dt_text = platforms.find('dt')
                    dd_text = platforms.find('dd')
                    if dt_text and dd_text:
                        text_content.append(f"{dt_text.get_text(strip=True)}: {dd_text.get_text(strip=True)}")
                        text_content.append('')

                # 提取示例代码 - 修复None类型检查
                # 使用更安全的方式查找含有"示例"或"Examples"的段落
                examples = None
                for p in dd.find_all('p'):
                    p_text = p.get_text(strip=True)
                    if p_text and ('示例' in p_text or 'Examples' in p_text):
                        examples = p
                        break

                if examples:
                    text_content.append("示例:")
                    # 查找示例后的代码块
                    code_block = examples.find_next('div', {'class': 'highlight'})
                    if code_block:
                        text_content.append('```python')
                        text_content.append(code_block.get_text().strip())
                        text_content.append('```')
                        text_content.append('')

        # 处理普通段落
        for paragraph in soup.find_all('p'):
            # 避免重复添加已处理的段落
            if paragraph.parent.name != 'dd':  # 避免重复添加函数描述中的段落
                text = paragraph.get_text(strip=True)
                if text and not any(text in content for content in text_content):
                    text_content.append(text)
                    text_content.append('')  # 添加空行

        # 处理列表
        for ul in soup.find_all(['ul', 'ol']):
            # 避免重复添加已处理的列表
            if ul.parent.name != 'dd' or not ul.parent.parent.name == 'dl':
                for li in ul.find_all('li', recursive=False):
                    text_content.append(f"- {li.get_text(strip=True)}")
                text_content.append('')  # 添加空行

        # 处理代码块
        for pre in soup.find_all('pre'):
            # 避免重复添加已处理的代码块
            if not pre.find_parent('div', {'class': 'highlight'}):
                text_content.append('```')
                text_content.append(pre.get_text())
                text_content.append('```')
                text_content.append('')  # 添加空行

        # 处理表格
        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                row = []
                for cell in tr.find_all(['td', 'th']):
                    row.append(cell.get_text(strip=True))
                rows.append(' | '.join(row))

            if rows:
                text_content.append('\n'.join(rows))
                text_content.append('')  # 添加空行

        # 额外处理：提取所有剩余文本，确保不遗漏内容
        # 获取所有文本节点并过滤掉已处理的内容
        all_text = soup.get_text(separator='\n', strip=True).split('\n')
        all_text = [line.strip() for line in all_text if line.strip()]

        # 合并所有文本内容
        final_text = '\n'.join(text_content)

        # 检查是否有遗漏的重要内容
        for line in all_text:
            if len(line) > 30 and line not in final_text:  # 假设长度超过30的文本是重要内容
                final_text += f"\n{line}\n"

        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.plain_text_file_path), exist_ok=True)

        # 写入文本文件
        with open(self.plain_text_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(final_text)


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
