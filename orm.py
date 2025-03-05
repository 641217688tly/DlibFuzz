import yaml
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Table, Boolean, Enum, DateTime
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

# 读取config.yml文件
with open('config.yml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# 从配置中提取数据库连接信息
db_config = config['db']['mysql']
host = db_config['host']
user = db_config['user']
password = db_config['password']
database = db_config['database']
db_url = f"mysql+pymysql://{user}:{password}@{host}/{database}"  # 创建数据库连接字符串

# 创建数据库连接
Base = declarative_base()

api_group_association = Table('api_group_association', Base.metadata,
                                    Column('api_group_id', Integer,
                                           ForeignKey('api_group.id')),
                                    Column('api_id', Integer, ForeignKey('api.id')))


class API(Base):
    __tablename__ = 'api'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)  # API名
    lib = Column(String(255), nullable=False)  # API所属的库: Pytorch, Tensorflow, JAX, MindSpore
    version = Column(String(255), nullable=True)  # API的版本, 比如2.10
    module = Column(String(255), nullable=True)  # API所在的模块, 比如torch.nn.functional
    full_name = Column(String(255), nullable=True)  # API的完整名字 = 模块名.API名, 比如torch.nn.functional.softmax
    signature = Column(Text, nullable=True)  # API函数签名, 比如torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None) -> Tensor
    parameters = Column(Text, nullable=True)  # API的参数信息, 比如input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None
    attributes = Column(Text, nullable=True) # 如果API的类型是class, 则使用该字段存储类的属性信息
    output = Column(Text, nullable=True)  # API的返回值信息, 比如Tensor, Shape: torch.Size([N, *])
    description = Column(Text, nullable=True)  # 对该API功能的描述
    example = Column(Text, nullable=True)  # 该API的示例调用代码
    is_clustered = Column(Boolean, default=False)  # 该API是否已经被执行匹配
    history_errors = relationship('APIHistoryError', back_populates='api')  # 一个API可能有多个触发bug的代码片段


class APIGroup(Base):
    __tablename__ = 'api_group'
    id = Column(Integer, primary_key=True)
    apis = relationship('API', secondary=api_group_association)
    cluster_id = Column(Integer, ForeignKey('cluster.id'), nullable=True)
    cluster = relationship('Cluster', back_populates='api_groups')
    api_seeds = relationship('APITestSeed', back_populates='api_group')


class APIHistoryError(Base):
    __tablename__ = 'api_history_error'
    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, ForeignKey('api.id'))
    api = relationship('API', back_populates='history_errors')
    issue_url = Column(Text, nullable=True)  # Issue的URL
    title = Column(Text, nullable=False)  # 触发error的Issue标题
    description = Column(Text, nullable=True)  # 描述该代码片段是怎么触发bug的
    code = Column(Text, nullable=False)  # 触发error的代码片段


class Cluster(Base):
    __tablename__ = 'cluster'
    id = Column(Integer, primary_key=True)
    type = Column(Enum('ValueEquivalent', 'StateEquivalent', name='cluster_type_enum'), nullable=False)
    description = Column(Text, nullable=True)
    energy = Column(Integer, default=5)
    api_groups = relationship('APIGroup', back_populates='cluster')  # [APIA], [APIB, APIC], [APID], [APIJ]
    is_tested = Column(Boolean, default=False)  # 该API是否已经生成过了种子
    cluster_seeds = relationship('ClusterTestSeed', back_populates='cluster')


class ClusterTestSeed(Base):
    __tablename__ = 'cluster_seed'
    id = Column(Integer, primary_key=True)
    # type = Column(Enum('WithHistoryError', 'WithoutHistoryError', name='cluster_seed_type_enum'), nullable=False)
    cluster_id = Column(Integer, ForeignKey('cluster.id'), nullable=True)
    cluster = relationship('Cluster', back_populates='cluster_seeds')
    api_seeds = relationship('APITestSeed', back_populates='cluster_seed', cascade="all, delete-orphan")
    is_validated = Column(Boolean, default=False)  # 该种子是否已经修复过了
    start_test = Column(DateTime, default=datetime.utcnow)  # 设置为该种子的创建时间
    end_test = Column(DateTime, nullable=True)  # 设置为该种子结束测试用例生成的时间


class APITestSeed(Base):
    __tablename__ = 'api_seed'
    id = Column(Integer, primary_key=True)
    cluster_seed_id = Column(Integer, ForeignKey('cluster_seed.id'))
    cluster_seed = relationship('ClusterTestSeed', back_populates='api_seeds')
    api_group_id = Column(Integer, ForeignKey('api_group.id'), nullable=True)
    api_group = relationship('APIGroup', back_populates='api_seeds')
    raw_code = Column(Text, nullable=True)
    valid_code = Column(Text, nullable=True)
    is_validated = Column(Boolean, default=False)  # 该种子是否已经修复过了


# 创建表
engine = create_engine(db_url)
Base.metadata.create_all(engine)
