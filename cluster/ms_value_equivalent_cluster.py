import markdown
from bs4 import BeautifulSoup, Tag
from utils import *
from sqlalchemy import func


class MSValueEquivalentCluster:
    def __init__(self, db_session, mapper_file_path):
        self.session = db_session
        self.mapper_dic = self.extract_ms_torch_mapper(mapper_file_path)

    def extract_ms_torch_mapper(self, file_path):
        """
        读取Markdown文件中的API映射关系，将说明中包含“一致”或“功能一致”的Pytorch API和MindSpore API存储到字典中
        """
        # 读取Markdown文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # 将Markdown内容转换为HTML，启用表格解析
        html = markdown.markdown(md_content, extensions=['tables'])

        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(html, 'html.parser')

        # 初始化映射字典
        mapper = {}

        # 查找所有二级标题
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

                            # 提取Pytorch API名称并去除超链接
                            pytorch_api = cols[0].get_text(strip=True)

                            # 提取MindSpore API名称并去除超链接
                            ms_api = cols[1].get_text(strip=True)

                            # 提取说明
                            description = cols[2].get_text(strip=True)

                            # 检查说明中是否包含“一致”或“功能一致”
                            if "一致" in description or "功能一致" in description:
                                mapper[pytorch_api] = ms_api
        return mapper

    def value_equivalent_cluster(self, mapper_dic):  # 根据mapper_dic对Mindspore的API进行值等价聚类
        try:
            for torch_api, ms_api in mapper_dic.items():
                torch_module_name, torch_api_name = torch_api.rsplit('.', 1)
                ms_module_name, ms_api_name = ms_api.rsplit('.', 1)
                if not (validate_api_existence(torch_module_name, torch_api_name) and validate_api_existence(
                        ms_module_name, ms_api_name)):
                    continue

                # 查询Pytorch API
                torch_api_obj = self.session.query(API).filter_by(lib='Pytorch', full_name=torch_api).first()
                torch_api_obj_combinations = (session.query(APICombination)
                                              .join(APICombination.apis)
                                              .filter(Cluster.type == 'ValueEquivalent')
                                              .group_by(APICombination.id)
                                              .having(func.count(API.id) == 1,  # 确保每个组合只有一个API
                                                      func.min(API.id) == torch_api_obj.id)
                                              .all())
                if torch_api_obj is None or len(torch_api_obj_combinations) == 0:
                    continue

                # 查询MindSpore API
                ms_api_obj = self.session.query(API).filter_by(lib='MindSpore', full_name=ms_api).first()
                if ms_api_obj is None:
                    # 创建MindSpore API
                    ms_api_info = inspect_api_info(ms_module_name, ms_api_name)
                    ms_api_obj = API(
                        lib='MindSpore',
                        name=ms_api_name,
                        module=ms_module_name,
                        full_name=ms_api,
                        signature=ms_api_info['signature'],
                        description=ms_api_info['description'],
                        is_clustered_by_value=True
                    )
                    self.session.add(ms_api_obj)
                    self.session.commit()

                # 根据torch_api_obj_combinations反向查找值等价簇, 之后以ms_api_obj新建APICombination并加入簇
                for torch_api_obj_combination in torch_api_obj_combinations:
                    cluster = torch_api_obj_combination.cluster
                    ms_api_combination = APICombination(
                        cluster=cluster,
                        apis=[ms_api_obj]
                    )
                    self.session.add(ms_api_combination)
                    self.session.commit()
        except Exception as e:
            print(f"Error: {e}")
            self.session.rollback()
        finally:
            self.session.close()


if __name__ == "__main__":
    mapper_file_path = 'apis/mindspore/ms_torch_mapping.md'
    session = get_session()
    ms_cluster = MSValueEquivalentCluster(session, mapper_file_path)
