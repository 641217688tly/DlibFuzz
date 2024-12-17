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

mindspore_api_combination_association = Table('mindspore_api_combination_association', Base.metadata,
                                              Column('mindspore_combination_id', Integer,
                                                     ForeignKey('mindspore_api_combination.id')),
                                              Column('api_id', Integer, ForeignKey('mindspore_api.id')))


# ----------------------------------Pytorch----------------------------------
class PytorchAPI(Base):
    __tablename__ = 'pytorch_api'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)  # API名
    module = Column(String(255), nullable=True)  # API所在的模块
    full_name = Column(String(255), nullable=True)  # API的完整名字 = 模块名.API名
    signature = Column(Text, nullable=True)  # API函数签名
    description = Column(Text, nullable=True)  # 对该API功能的描述
    doc_url = Column(String(255), nullable=True)  # 该API的官方文档的URL
    doc_content = Column(Text, nullable=True) # 该API的官方文档的内容
    example = Column(Text, nullable=True) # 该API的示例调用代码
    version = Column(String(255), nullable=True)  # API的版本
    embedding = Column(Text, nullable=True)  # 该API的嵌入向量, 包括函数名和功能描述
    is_clustered = Column(Boolean, default=False)  # 该API是否已经被聚类
    error_triggers = relationship('PytorchErrorTrigger', back_populates='api')  # 一个API可能有多个触发bug的代码片段


class PytorchErrorTrigger(Base):
    __tablename__ = 'pytorch_error_trigger'
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
    cluster_id = Column(Integer, ForeignKey('cluster.id'))
    cluster = relationship('Cluster', back_populates='pytorch_combinations')
    api_seeds = relationship('APITestSeed', back_populates='pytorch_api_combination')


# ----------------------------------Tensorflow----------------------------------

class TensorflowAPI(Base):
    __tablename__ = 'tensorflow_api'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)  # API名
    module = Column(String(255), nullable=True)  # API所在的模块
    full_name = Column(String(255), nullable=True)  # API的完整名字 = 模块名.API名
    signature = Column(Text, nullable=True)  # API函数签名
    description = Column(Text, nullable=True)  # 对该API功能的描述
    doc_url = Column(String(255), nullable=True)  # 该API的官方文档的URL
    doc_content = Column(Text, nullable=True) # 该API的官方文档的内容
    example = Column(Text, nullable=True) # 该API的示例调用代码
    version = Column(String(255), nullable=True)  # API的版本
    embedding = Column(Text, nullable=True)  # 该API的嵌入向量, 包括函数名和功能描述
    is_clustered = Column(Boolean, default=False)  # 该API是否已经被聚类
    error_triggers = relationship('TensorflowErrorTrigger', back_populates='api')  # 一个API可能有多个触发bug的代码片段


class TensorflowErrorTrigger(Base):
    __tablename__ = 'tensorflow_error_trigger'
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
    cluster_id = Column(Integer, ForeignKey('cluster.id'))
    cluster = relationship('Cluster', back_populates='tensorflow_combinations')
    api_seeds = relationship('APITestSeed', back_populates='tensorflow_api_combination')


# ----------------------------------JAX----------------------------------

class JAXAPI(Base):
    __tablename__ = 'jax_api'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)  # API名
    module = Column(String(255), nullable=True)  # API所在的模块
    full_name = Column(String(255), nullable=True)  # API的完整名字 = 模块名.API名
    signature = Column(Text, nullable=True)  # API函数签名
    description = Column(Text, nullable=True)  # 对该API功能的描述
    doc_url = Column(String(255), nullable=True)  # 该API的官方文档的URL
    doc_content = Column(Text, nullable=True) # 该API的官方文档的内容
    example = Column(Text, nullable=True) # 该API的示例调用代码
    version = Column(String(255), nullable=True)  # API的版本
    embedding = Column(Text, nullable=True)  # 该API的嵌入向量, 包括函数名和功能描述
    is_clustered = Column(Boolean, default=False)  # 该API是否已经被聚类
    error_triggers = relationship('JAXErrorTrigger', back_populates='api')  # 一个API可能有多个触发bug的代码片段


class JAXErrorTrigger(Base):
    __tablename__ = 'jax_error_trigger'
    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, ForeignKey('jax_api.id'))
    api = relationship('JAXAPI', back_populates='error_triggers')
    title = Column(Text, nullable=False)  # 触发error的Issue标题
    code = Column(Text, nullable=False)  # 触发error的代码片段
    description = Column(Text, nullable=True)  # 描述该代码片段是怎么触发bug的


class JAXAPICombination(Base):
    __tablename__ = 'jax_api_combination'
    id = Column(Integer, primary_key=True)
    apis = relationship('JAXAPI', secondary=jax_api_combination_association)
    cluster_id = Column(Integer, ForeignKey('cluster.id'))
    cluster = relationship('Cluster', back_populates='jax_combinations')
    api_seeds = relationship('APITestSeed', back_populates='jax_api_combination')


# ----------------------------------MindSpore----------------------------------
class MindSporeAPI(Base):
    __tablename__ = 'mindspore_api'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)  # API名
    module = Column(String(255), nullable=True)  # API所在的模块
    full_name = Column(String(255), nullable=True)  # API的完整名字 = 模块名.API名
    signature = Column(Text, nullable=True)  # API函数签名
    description = Column(Text, nullable=True)  # 对该API功能的描述
    doc_url = Column(String(255), nullable=True)  # 该API的官方文档的URL
    doc_content = Column(Text, nullable=True) # 该API的官方文档的内容
    example = Column(Text, nullable=True) # 该API的示例调用代码
    version = Column(String(255), nullable=True)  # API的版本
    embedding = Column(Text, nullable=True)  # 该API的嵌入向量, 包括函数名和功能描述
    is_clustered = Column(Boolean, default=False)  # 该API是否已经被聚类
    error_triggers = relationship('MindSporeErrorTrigger', back_populates='api')  # 一个API可能有多个触发bug的代码片段


class MindSporeErrorTrigger(Base):
    __tablename__ = 'mindspore_error_trigger'
    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, ForeignKey('mindspore_api.id'))
    api = relationship('MindSporeAPI', back_populates='error_triggers')
    title = Column(Text, nullable=False)  # 触发error的Issue标题
    code = Column(Text, nullable=False)  # 触发error的代码片段
    description = Column(Text, nullable=True)  # 描述该代码片段是怎么触发bug的


class MindSporeAPICombination(Base):
    __tablename__ = 'mindspore_api_combination'
    id = Column(Integer, primary_key=True)
    apis = relationship('MindSporeAPI', secondary=mindspore_api_combination_association)
    cluster_id = Column(Integer, ForeignKey('cluster.id'))
    cluster = relationship('Cluster', back_populates='mindspore_combinations')
    api_seeds = relationship('APITestSeed', back_populates='mindspore_api_combination')


# ----------------------------------Cluster----------------------------------

class Cluster(Base):
    __tablename__ = 'cluster'
    id = Column(Integer, primary_key=True)
    description = Column(Text, nullable=True)
    energy = Column(Integer, default=5)
    pytorch_combinations = relationship('PytorchAPICombination', back_populates='cluster')
    tensorflow_combinations = relationship('TensorflowAPICombination', back_populates='cluster')
    jax_combinations = relationship('JAXAPICombination', back_populates='cluster')
    mindspore_combinations = relationship('MindSporeAPICombination', back_populates='cluster')
    is_tested = Column(Boolean, default=False)  # 该API是否已经生成过了种子
    cluster_seeds = relationship('ClusterTestSeed', back_populates='cluster')


class ClusterTestSeed(Base):
    __tablename__ = 'seed'
    id = Column(Integer, primary_key=True)
    cluster_id = Column(Integer, ForeignKey('cluster.id'))
    cluster = relationship('Cluster', back_populates='cluster_seeds')
    api_seeds = relationship('APITestSeed', back_populates='cluster_seed', cascade="all, delete-orphan")
    is_validated = Column(Boolean, default=False)  # 该种子是否已经修复过了
    start_test = Column(DateTime, default=datetime.utcnow)  # 设置为该种子的创建时间
    end_test = Column(DateTime, nullable=True)  # 设置为该种子结束测试用例生成的时间


class APITestSeed(Base):
    __tablename__ = 'api_seed'
    id = Column(Integer, primary_key=True)

    api_type = Column(Enum('Pytorch', 'Tensorflow', 'JAX', name='api_type_enum'), nullable=False)

    cluster_seed_id = Column(Integer, ForeignKey('seed.id'))
    cluster_seed = relationship('ClusterTestSeed', back_populates='api_seeds')

    pytorch_api_combination_id = Column(Integer, ForeignKey('pytorch_api_combination.id'), nullable=True)
    pytorch_api_combination = relationship('PytorchAPICombination', back_populates='api_seeds')

    tensorflow_api_combination_id = Column(Integer, ForeignKey('tensorflow_api_combination.id'), nullable=True)
    tensorflow_api_combination = relationship('TensorflowAPICombination', back_populates='api_seeds')

    jax_api_combination_id = Column(Integer, ForeignKey('jax_api_combination.id'), nullable=True)
    jax_api_combination = relationship('JAXAPICombination', back_populates='api_seeds')

    mindspore_api_combination_id = Column(Integer, ForeignKey('mindspore_api_combination.id'), nullable=True)
    mindspore_api_combination = relationship('MindSporeAPICombination', back_populates='api_seeds')

    raw_code = Column(Text, nullable=True)
    valid_code = Column(Text, nullable=True)
    is_validated = Column(Boolean, default=False)  # 该种子是否已经修复过了


# 创建表
engine = create_engine(db_url)
Base.metadata.create_all(engine)
