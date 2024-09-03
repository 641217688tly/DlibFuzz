import yaml
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Table, Boolean
from sqlalchemy.orm import relationship, declarative_base

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

pytorch_api_combination_association = Table('pytorch_api_combination_association', Base.metadata,
                                            Column('api_combination_id', Integer,
                                                   ForeignKey('pytorch_api_combination.id')),
                                            Column('api_id', Integer, ForeignKey('pytorch_api.id')))

tensorflow_api_combination_association = Table('tensorflow_api_combination_association', Base.metadata,
                                               Column('api_combination_id', Integer,
                                                      ForeignKey('tensorflow_api_combination.id')),
                                               Column('api_id', Integer, ForeignKey('tensorflow_api.id')))

jax_api_combination_association = Table('jax_api_combination_association', Base.metadata,
                                        Column('api_combination_id', Integer, ForeignKey('jax_api_combination.id')),
                                        Column('api_id', Integer, ForeignKey('jax_api.id')))


# ----------------------------------Pytorch----------------------------------
class PytorchAPI(Base):
    __tablename__ = 'pytorch_api'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)  # API名
    module = Column(String(255), nullable=True)  # API所在的模块
    full_name = Column(String(255), nullable=True)  # API的完整名字 = 模块名.API名
    signature = Column(Text, nullable=True)  # API函数签名
    description = Column(Text, nullable=True)  # 对该API功能的描述
    version = Column(String(255), nullable=True)  # API的版本
    embedding = Column(Text, nullable=True)  # 该API的嵌入向量, 包括函数名和功能描述
    is_clustered = Column(Boolean, default=False)  # 该API是否已经被聚类
    error_triggers = relationship('PytorchErrorTriggerCode', back_populates='api')  # 一个API可能有多个触发bug的代码片段


class PytorchErrorTriggerCode(Base):
    __tablename__ = 'pytorch_error_trigger_code'
    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, ForeignKey('pytorch_api.id'))
    api = relationship('PytorchAPI', back_populates='error_triggers')
    title = Column(Text, nullable=False)  # 触发error的Issue标题
    code = Column(Text, nullable=False)  # 触发error的代码片段
    description = Column(Text, nullable=True)  # 描述该代码片段是怎么触发bug的


class PytorchAPICombination(Base):
    __tablename__ = 'pytorch_api_combination'
    id = Column(Integer, primary_key=True)
    apis = relationship('PytorchAPI', secondary=pytorch_api_combination_association)
    test_seeds = relationship('ClusterTestSeed', back_populates='pytorch_combination')
    cluster_id = Column(Integer, ForeignKey('cluster.id'))
    cluster = relationship('Cluster', back_populates='pytorch_combinations')


# ----------------------------------Tensorflow----------------------------------

class TensorflowAPI(Base):
    __tablename__ = 'tensorflow_api'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)  # API名
    module = Column(String(255), nullable=True)  # API所在的模块
    full_name = Column(String(255), nullable=True)  # API的完整名字 = 模块名.API名
    signature = Column(Text, nullable=True)  # API函数签名
    description = Column(Text, nullable=True)  # 对该API功能的描述
    version = Column(String(255), nullable=True)  # API的版本
    embedding = Column(Text, nullable=True)  # 该API的嵌入向量, 包括函数名和功能描述
    is_clustered = Column(Boolean, default=False)  # 该API是否已经被聚类
    error_triggers = relationship('TensorflowErrorTriggerCode', back_populates='api')  # 一个API可能有多个触发bug的代码片段


class TensorflowErrorTriggerCode(Base):
    __tablename__ = 'tensorflow_error_trigger_code'
    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, ForeignKey('tensorflow_api.id'))
    api = relationship('TensorflowAPI', back_populates='error_triggers')
    title = Column(Text, nullable=False)  # 触发error的Issue标题
    code = Column(Text, nullable=False)  # 触发error的代码片段
    description = Column(Text, nullable=True)  # 描述该代码片段是怎么触发bug的


class TensorflowAPICombination(Base):
    __tablename__ = 'tensorflow_api_combination'
    id = Column(Integer, primary_key=True)
    apis = relationship('TensorflowAPI', secondary=tensorflow_api_combination_association)
    test_seeds = relationship('ClusterTestSeed', back_populates='tensorflow_combination')
    cluster_id = Column(Integer, ForeignKey('cluster.id'))
    cluster = relationship('Cluster', back_populates='tensorflow_combinations')


# ----------------------------------Jax----------------------------------

class JaxAPI(Base):
    __tablename__ = 'jax_api'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)  # API名
    module = Column(String(255), nullable=True)  # API所在的模块
    full_name = Column(String(255), nullable=True)  # API的完整名字 = 模块名.API名
    signature = Column(Text, nullable=True)  # API函数签名
    description = Column(Text, nullable=True)  # 对该API功能的描述
    version = Column(String(255), nullable=True)  # API的版本
    embedding = Column(Text, nullable=True)  # 该API的嵌入向量, 包括函数名和功能描述
    is_clustered = Column(Boolean, default=False)  # 该API是否已经被聚类
    error_triggers = relationship('JaxErrorTriggerCode', back_populates='api')  # 一个API可能有多个触发bug的代码片段


class JaxErrorTriggerCode(Base):
    __tablename__ = 'jax_error_trigger_code'
    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, ForeignKey('jax_api.id'))
    api = relationship('JaxAPI', back_populates='error_triggers')
    title = Column(Text, nullable=False)  # 触发error的Issue标题
    code = Column(Text, nullable=False)  # 触发error的代码片段
    description = Column(Text, nullable=True)  # 描述该代码片段是怎么触发bug的


class JaxAPICombination(Base):
    __tablename__ = 'jax_api_combination'
    id = Column(Integer, primary_key=True)
    apis = relationship('JaxAPI', secondary=jax_api_combination_association)
    test_seeds = relationship('ClusterTestSeed', back_populates='jax_combination')
    cluster_id = Column(Integer, ForeignKey('cluster.id'))
    cluster = relationship('Cluster', back_populates='jax_combinations')


# ----------------------------------Cluster----------------------------------

class Cluster(Base):
    __tablename__ = 'cluster'
    id = Column(Integer, primary_key=True)
    description = Column(Text, nullable=True)
    energy = Column(Integer, default=5)
    is_tested = Column(Boolean, default=False)  # 该API是否已经生成过了种子
    pytorch_combinations = relationship('PytorchAPICombination', back_populates='cluster')
    tensorflow_combinations = relationship('TensorflowAPICombination', back_populates='cluster')
    jax_combinations = relationship('JaxAPICombination', back_populates='cluster')
    test_seeds = relationship('ClusterTestSeed', back_populates='cluster')


class ClusterTestSeed(Base):
    __tablename__ = 'seed'
    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey('cluster.id'))
    cluster = relationship('Cluster', back_populates='test_seeds')
    pytorch_combination_id = Column(Integer, ForeignKey('pytorch_api_combination.id'))
    pytorch_combination = relationship('PytorchAPICombination', back_populates='test_seeds')
    tensorflow_combination_id = Column(Integer, ForeignKey('tensorflow_api_combination.id'))
    tensorflow_combination = relationship('TensorflowAPICombination', back_populates='test_seeds')
    jax_combination_id = Column(Integer, ForeignKey('jax_api_combination.id'))
    jax_combination = relationship('JaxAPICombination', back_populates='test_seeds')
    code = Column(Text, nullable=True)  # pytorch_code + tensorflow_code + jax_code
    pytorch_code = Column(Text, nullable=True)
    tensorflow_code = Column(Text, nullable=True)
    jax_code = Column(Text, nullable=True)
    unverified_file_path = Column(Text, nullable=True)  # 该种子的文件路径
    verified_file_path = Column(Text, nullable=True)  # 该种子的文件路径
    is_verified = Column(Boolean, default=False)  # 该种子是否已经验证过了


# 创建表
engine = create_engine(db_url)
Base.metadata.create_all(engine)