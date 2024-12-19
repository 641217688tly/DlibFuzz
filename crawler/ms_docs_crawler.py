import os
import markdown
from bs4 import BeautifulSoup, Tag
import requests
import re


def sanitize_filename(name):
    """
    清理文件名，移除非法字符。
    """
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def download_file(url, save_path):
    """
    下载指定URL的文件并保存到指定路径。
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"下载成功: {url} -> {save_path}")
    except requests.RequestException as e:
        print(f"下载失败: {url} 错误: {e}")


def extract_and_download_docs(mapper_file_path, docs_dir='../data/docs/ms/docs/'):
    """
    读取Markdown文件，解析API映射关系，并下载相应的HTML资源。

    Args:
        mapper_file_path (str): Markdown文件的路径。
        docs_dir (str): 存储下载资源的根目录。
    """
    # 创建根目录
    os.makedirs(docs_dir, exist_ok=True)

    # 读取Markdown文件内容
    with open(mapper_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 将Markdown内容转换为HTML，启用表格解析
    html = markdown.markdown(md_content, extensions=['tables'])

    # 使用BeautifulSoup解析HTML
    soup = BeautifulSoup(html, 'html.parser')

    # 查找所有二级标题（h2）
    for header in soup.find_all('h2'):
        # 获取h2标题后的所有兄弟元素，直到下一个h2
        for sibling in header.find_next_siblings():
            if isinstance(sibling, Tag):
                if sibling.name == 'h2':
                    break  # 遇到下一个h2，停止处理当前h2下的内容

                if sibling.name == 'table':
                    # 处理表格
                    rows = sibling.find_all('tr')
                    if len(rows) < 2:
                        continue  # 表格没有数据行，跳过

                    # 遍历表格的每一行（跳过表头）
                    for row in rows[1:]:
                        cols = row.find_all('td')
                        if len(cols) < 3:
                            continue  # 列数不足，跳过

                        # 提取PyTorch API的名称和URL
                        pytorch_link_tag = cols[0].find('a')
                        if pytorch_link_tag and 'href' in pytorch_link_tag.attrs:
                            pytorch_api_name = pytorch_link_tag.get_text(strip=True)
                            pytorch_url = pytorch_link_tag['href'].strip()
                        else:
                            pytorch_api_name = cols[0].get_text(strip=True)
                            pytorch_url = ''

                        # 提取MindSpore API的名称和URL
                        ms_link_tag = cols[1].find('a')
                        if ms_link_tag and 'href' in ms_link_tag.attrs:
                            ms_api_name = ms_link_tag.get_text(strip=True)
                            ms_url = ms_link_tag['href'].strip()
                        else:
                            ms_api_name = cols[1].get_text(strip=True)
                            ms_url = ''

                        # 提取说明的文本和URL（如果有）
                        description_tag = cols[2].find('a')
                        description_text = cols[2].get_text(strip=True)
                        if description_tag and 'href' in description_tag.attrs:
                            description_url = description_tag['href'].strip()
                        else:
                            description_url = ''

                        # 创建以MindSpore API名称命名的文件夹
                        ms_api_dir = os.path.join(docs_dir, sanitize_filename(ms_api_name))
                        os.makedirs(ms_api_dir, exist_ok=True)

                        # 下载PyTorch API的HTML
                        if pytorch_url:
                            pytorch_filename = sanitize_filename(pytorch_api_name) + '.html'
                            pytorch_save_path = os.path.join(ms_api_dir, pytorch_filename)
                            download_file(pytorch_url, pytorch_save_path)

                        # 下载MindSpore API的HTML
                        if ms_url:
                            ms_filename = sanitize_filename(ms_api_name) + '.html'
                            ms_save_path = os.path.join(ms_api_dir, ms_filename)
                            download_file(ms_url, ms_save_path)

                        # 条件筛选说明中的超链接下载
                        if description_url and re.search(r'差异对比|不一致|不同', description_text):
                            description_filename = sanitize_filename(description_text) + '.html'
                            description_save_path = os.path.join(ms_api_dir, description_filename)
                            download_file(description_url, description_save_path)


if __name__ == "__main__":
    mapper_file = '../cluster/apis/mindspore/ms_torch_mapping.md'
    extract_and_download_docs(mapper_file)
