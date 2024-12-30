import markdown
from bs4 import BeautifulSoup, Tag
from utils import *
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

class MSStateEquivalentCluster:
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

        # 将Markdown内容转换为HTML,启用表格解析
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
        for torch_api, ms_api in mapper_dic.items():
            try:
                torch_module_name, torch_api_name = torch_api.rsplit('.', 1)
                ms_module_name, ms_api_name = ms_api.rsplit('.', 1)
                if not validate_api_existence(torch_module_name, torch_api_name):
                    continue

                # 查询Pytorch API
                torch_api_obj = self.session.query(API).filter_by(lib='Pytorch', full_name=torch_api).first()
                # 找到APIGroup中group.apis = [torch_api_obj]的APIGroup, 后续将利用这些APIGroup反向查找状态等价簇
                torch_api_obj_groups = (
                    session.query(APIGroup)
                    .join(APIGroup.apis)
                    .filter(Cluster.type == 'StateEquivalent')
                    .group_by(APIGroup.id)
                    .having(func.count(API.id) == 1,  # 确保每个组合只有一个API
                            func.min(API.id) == torch_api_obj.id)
                    .all())
                if torch_api_obj is None or len(torch_api_obj_groups) == 0:
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
                    self.session.flush()

                # 根据torch_api_obj_groups反向查找状态等价簇, 然后从中找到与torch_api_obj状态等价的torch_apis
                state_equivalent_torch_apis = []
                for torch_api_obj_group in torch_api_obj_groups:
                    state_equivalent_cluster = torch_api_obj_group.cluster
                    state_equivalent_torch_api_groups = state_equivalent_cluster.api_groups
                    filtered_api_list = [
                        sublist[0] for sublist in state_equivalent_torch_api_groups
                        if len(sublist) == 1 and sublist[0] not in torch_api_obj_group
                    ]
                    state_equivalent_torch_apis.extend(filtered_api_list)

                # 找到与ms_api_obj状态等价的ms_apis
                state_equivalent_ms_apis = [ms_api_obj] # [api1, api2, api3, ..]
                for state_equivalent_torch_api_obj in state_equivalent_torch_apis:
                    # 根据mapper_dic将state_equivalent_torch_apis中的torch api映射为ms api
                    state_equivalent_ms_api_full_name = mapper_dic.get(state_equivalent_torch_api_obj.full_name, None)
                    # 检查得到的ms_api是否存在
                    if state_equivalent_ms_api_full_name is None:
                        continue
                    state_equivalent_ms_module_name, state_equivalent_ms_api_name = state_equivalent_ms_api_full_name.rsplit('.', 1)
                    if not validate_api_existence(state_equivalent_ms_module_name, state_equivalent_ms_api_name):
                        continue
                    state_equivalent_ms_api_obj = self.session.query(API).filter_by(lib='MindSpore', full_name=state_equivalent_ms_api_full_name).first()
                    if state_equivalent_ms_api_obj is None:
                        # 创建MindSpore API
                        ms_api_info = inspect_api_info(ms_module_name, ms_api_name)
                        state_equivalent_ms_api_obj = API(
                            lib='MindSpore',
                            name=ms_api_name,
                            module=ms_module_name,
                            full_name=state_equivalent_ms_api_full_name,
                            signature=ms_api_info['signature'],
                            description=ms_api_info['description'],
                            version=ms_api_info['version'],
                            is_clustered_by_state=True
                        )
                        self.session.add(state_equivalent_ms_api_obj)
                        self.session.flush()
                    state_equivalent_ms_apis.append(state_equivalent_ms_api_obj)

                # state_equivalent_ms_apis存储了包括ms_api_obj在内的所有与ms_api_obj状态等价的ms_apis
                if len(state_equivalent_ms_apis) >= 2:
                    state_equivalent_cluster_dict = {}
                    for api in state_equivalent_ms_apis:
                        api_obj_groups = (self.session.query(APIGroup)
                                          .join(APIGroup.apis)
                                          .filter(Cluster.type == 'StateEquivalent')
                                          .group_by(APIGroup.id)
                                          .having(func.count(API.id) == 1,  # 确保每个组合只有一个API
                                                  func.min(API.id) == api.id)
                                          .all())
                        for api_obj_group in api_obj_groups:
                            state_equivalent_cluster = api_obj_group.cluster
                            state_equivalent_cluster_dict[state_equivalent_cluster] = state_equivalent_cluster_dict.get(state_equivalent_cluster, 0) + 1
                    if state_equivalent_cluster_dict:
                        state_equivalent_cluster = max(state_equivalent_cluster_dict, key=state_equivalent_cluster_dict.get)
                    else:
                        state_equivalent_cluster = Cluster(
                            type='StateEquivalent',
                            energy=5,
                        )
                        self.session.add(state_equivalent_cluster)
                        self.session.flush()

                    # 为state_equivalent_ms_apis内的每个状态等价API创建对应的APIGroup并关联到state_equivalent_cluster
                    for api in state_equivalent_ms_apis:
                        api_group = APIGroup(cluster=state_equivalent_cluster, apis = [api])
                        self.session.add(api_group)
                        self.session.flush()
                else: # 如果没能找到ms_api的状态等价API, 那么将跳过这个API
                    continue
                    # raise ValueError(f"No state equivalent APIs found for {ms_api}")
                self.session.commit()
            except (Exception, ValueError, SQLAlchemyError) as e:
                self.session.rollback()
                print(f"Error: {e}")
                continue
        self.session.close()


if __name__ == "__main__":
    mapper_file_path = 'apis/mindspore/ms_torch_mapping.md'
    session = get_session()
    ms_cluster = MSStateEquivalentCluster(session, mapper_file_path)
