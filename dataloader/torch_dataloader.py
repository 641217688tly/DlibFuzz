from bs4 import BeautifulSoup
from utils import get_session


class PytorchAPILoader:
    def __init__(self, file_path, db_session, lib_ver="2.3"):
        self.file_path = file_path
        self.output_path = file_path.replace('raw', 'handled')
        self.session = db_session
        self.lib_ver = lib_ver
        self.api_full_name = file_path.split('/')[-1].replace('.html', '')

        self.strip_unnecessary_content()

    def strip_unnecessary_content(self):  # 从HTML中剔除与API无关的内容
        # 读取HTML文件
        with open(self.file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')  # 使用BeautifulSoup解析HTML

        # 移除所有的脚本和样式
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        # 提取相关内容
        content = soup.find('section', {'id': 'pytorch-content-wrap'})

        # 查找所有包含函数定义的部分
        sections = content.find_all('div', {'class': 'section'}) if content else []

        # 使用集合去重，避免重复部分
        unique_sections = set()

        # 筛选和去重API文档内容
        relevant_content = ""
        for section in sections:
            # 根据ID进行去重
            section_id = section.get('id', '')
            if section_id not in unique_sections:
                unique_sections.add(section_id)
                relevant_content += str(section)

        # 添加meta标签，明确文件编码为UTF-8
        final_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyTorch API Documentation</title>
</head>
<body>
{relevant_content}
</body>
</html>'''

        # 将精简后的内容写入新的HTML文件
        with open(self.output_path, 'w', encoding='utf-8') as output_file:
            output_file.write(final_html)

    def extract_api_info(self): # TODO 提取不到Shape
        """
        从HTML中提取API的信息:
        1. name - API名称
        2. module - 所属模块
        3. full_name - 完整名称
        4. lib - 固定为Pytorch
        5. version - API版本
        6. signature - 函数签名
        7. parameters - 参数信息
        8. attributes - 类属性(如果是类)
        9. output - 返回值信息
        10. description - API描述
        11. example - 使用示例
        """
        with open(self.output_path, 'r', encoding='utf-8') as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, 'html.parser')

        # 基本信息提取
        full_name = self.api_full_name
        name_parts = full_name.split('.')

        api_info = {
            'name': name_parts[-1],
            'lib': 'Pytorch',
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

            api_info['signature'] = raw_signature

        # 提取description(描述)
        desc_element = soup.find('dd')
        if desc_element:
            # 提取主要描述文本
            description_text = []

            # 获取所有直接子p元素，这些通常是主要描述
            paragraphs = desc_element.find_all('p', recursive=False)
            for p in paragraphs:
                if not p.find('dl'):  # 避免包含参数列表
                    description_text.append(p.get_text(strip=True))

            # 提取数学公式和其他说明文本
            math_sections = desc_element.find_all('div', {'class': 'math'})
            for math in math_sections:
                if not math.parent.name == 'dd' or not math.parent.parent.name == 'dl':  # 避免提取参数或返回值中的公式
                    math_text = math.get_text(strip=True)
                    if math_text:
                        description_text.append(f"Formula: {math_text}")

            # 提取注意事项
            notes = desc_element.find_all('div', {'class': 'admonition note'})
            for note in notes:
                note_content = note.find_all('p')
                if len(note_content) > 1:  # 跳过只有标题的注释
                    note_text = ' '.join([p.get_text(strip=True) for p in note_content[1:]])
                    description_text.append(f"Note: {note_text}")

            api_info['description'] = '\n'.join(description_text) if description_text else None

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

                api_info['parameters'] = '\n'.join(param_items) if param_items else None

            # 返回值提取
            return_section = desc_element.find_all('dt', {'class': 'field-even'})
            for section in return_section:
                if 'Return type' in section.get_text():
                    next_dd = section.find_next('dd')
                    if next_dd:
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

                api_info['parameters'] = '\n'.join(param_items) if param_items else None

            # 类属性提取 (如果有的话)
            attributes = []
            attr_sections = desc_element.find_all(['dl', 'div'], {'class': ['field-list', 'attribute']})
            for section in attr_sections:
                if 'Attributes' in section.get_text():
                    attr_items = section.find_all('li')
                    for attr in attr_items:
                        attributes.append(attr.get_text(strip=True))

            api_info['attributes'] = '\n'.join(attributes) if attributes else None

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

                            api_info['output'] = '\n'.join(shape_info) if shape_info else None
                        break

        # 提取示例代码
        example_section = soup.find('div', {'class': 'highlight-default notranslate'})
        if example_section:
            api_info['example'] = example_section.get_text(strip=False)  # 保留代码格式

        # 对提取到的API数据进行规范化处理
        # 1. 处理signature
        if is_function:
            if not api_info['signature']:  # 如果signature为空，则默认full_name()为signature
                api_info['signature'] = f"{api_info['full_name']}()"
            api_info['signature'] = f"{api_info['signature']} -> ({api_info['output']})"
        elif is_class:
            if not api_info['signature']:  # 如果signature为空，则默认full_name为signature
                api_info['signature'] = f"{api_info['full_name']}"

        return api_info


# file_path = './../data/docs/torch/docs/2.3/raw/torch._assert.html'
file_path = './../data/docs/torch/docs/2.3/raw/torch.nn.AvgPool1d.html'
# file_path = './../data/docs/torch/docs/2.3/raw/torch.nn.functional.softmax.html'
loader = PytorchAPILoader(file_path, get_session())
api_info = loader.extract_api_info()
for key, value in api_info.items():
    print(f"{key}: \n{value}")
    print("="*50)



