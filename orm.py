import yaml
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Table, Boolean, Enum
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

pytorch_api_cluster_association = Table('pytorch_api_cluster_association', Base.metadata,
                                        Column('cluster_id', Integer, ForeignKey('cluster.id')),
                                        Column('pytorch_api_id', Integer, ForeignKey('pytorch_api.id'))
                                        )

tensorflow_api_cluster_association = Table('tensorflow_api_cluster_association', Base.metadata,
                                           Column('cluster_id', Integer, ForeignKey('cluster.id')),
                                           Column('tensorflow_api_id', Integer, ForeignKey('tensorflow_api.id'))
                                           )

jax_api_cluster_association = Table('jax_api_cluster_association', Base.metadata,
                                    Column('cluster_id', Integer, ForeignKey('cluster.id')),
                                    Column('jax_api_id', Integer, ForeignKey('jax_api.id'))
                                    )


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
    clusters = relationship('Cluster', secondary=pytorch_api_cluster_association, back_populates='pytorch_apis')
    error_triggers = relationship('PytorchErrorTrigger', back_populates='api')  # 一个API可能有多个触发bug的代码片段
    seeds = relationship('ClusterTestSeed', back_populates='pytorch_api')


class PytorchErrorTrigger(Base):
    __tablename__ = 'pytorch_error_trigger'
    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, ForeignKey('pytorch_api.id'))
    api = relationship('PytorchAPI', back_populates='error_triggers')
    title = Column(Text, nullable=False)  # 触发error的Issue标题
    code = Column(Text, nullable=False)  # 触发error的代码片段
    description = Column(Text, nullable=True)  # 描述该代码片段是怎么触发bug的


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
    clusters = relationship('Cluster', secondary=tensorflow_api_cluster_association, back_populates='tensorflow_apis')
    error_triggers = relationship('TensorflowErrorTrigger', back_populates='api')  # 一个API可能有多个触发bug的代码片段
    seeds = relationship('ClusterTestSeed', back_populates='tensorflow_api')


class TensorflowErrorTrigger(Base):
    __tablename__ = 'tensorflow_error_trigger'
    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, ForeignKey('tensorflow_api.id'))
    api = relationship('TensorflowAPI', back_populates='error_triggers')
    title = Column(Text, nullable=False)  # 触发error的Issue标题
    code = Column(Text, nullable=False)  # 触发error的代码片段
    description = Column(Text, nullable=True)  # 描述该代码片段是怎么触发bug的


# ----------------------------------JAX----------------------------------

class JAXAPI(Base):
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
    clusters = relationship('Cluster', secondary=jax_api_cluster_association, back_populates='jax_apis')
    error_triggers = relationship('JAXErrorTrigger', back_populates='api')  # 一个API可能有多个触发bug的代码片段
    seeds = relationship('ClusterTestSeed', back_populates='jax_api')


class JAXErrorTrigger(Base):
    __tablename__ = 'jax_error_trigger'
    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, ForeignKey('jax_api.id'))
    api = relationship('JAXAPI', back_populates='error_triggers')
    title = Column(Text, nullable=False)  # 触发error的Issue标题
    code = Column(Text, nullable=False)  # 触发error的代码片段
    description = Column(Text, nullable=True)  # 描述该代码片段是怎么触发bug的


# ----------------------------------Cluster----------------------------------

class Cluster(Base):
    __tablename__ = 'cluster'
    id = Column(Integer, primary_key=True)
    description = Column(Text, nullable=True)
    energy = Column(Integer, default=5)
    base = Column(Enum('Pytorch', 'JAX', 'Tensorflow', name='base_type'), nullable=False)  # 添加枚举列
    pytorch_apis = relationship('PytorchAPI', secondary=pytorch_api_cluster_association, back_populates='clusters')
    tensorflow_apis = relationship('TensorflowAPI', secondary=tensorflow_api_cluster_association,
                                   back_populates='clusters')
    jax_apis = relationship('JAXAPI', secondary=jax_api_cluster_association, back_populates='clusters')
    is_tested = Column(Boolean, default=False)  # 该API是否已经生成过了种子
    seeds = relationship('ClusterTestSeed', back_populates='cluster')


class ClusterTestSeed(Base):
    __tablename__ = 'seed'
    id = Column(Integer, primary_key=True)

    cluster_id = Column(Integer, ForeignKey('cluster.id'))
    cluster = relationship('Cluster', back_populates='seeds')

    pytorch_api_id = Column(Integer, ForeignKey('pytorch_api.id'), nullable=True)
    pytorch_api = relationship('PytorchAPI', back_populates='seeds')

    tensorflow_api_id = Column(Integer, ForeignKey('tensorflow_api.id'), nullable=True)
    tensorflow_api = relationship('TensorflowAPI', back_populates='seeds')

    jax_api_id = Column(Integer, ForeignKey('jax_api.id'), nullable=True)
    jax_api = relationship('JAXAPI', back_populates='seeds')

    raw_pytorch_code = Column(Text, nullable=True)
    raw_tensorflow_code = Column(Text, nullable=True)
    raw_jax_code = Column(Text, nullable=True)

    valid_pytorch_code = Column(Text, nullable=True)
    valid_tensorflow_code = Column(Text, nullable=True)
    valid_jax_code = Column(Text, nullable=True)

    is_validated = Column(Boolean, default=False)  # 该种子是否已经修复过了


# 创建表
engine = create_engine(db_url)
Base.metadata.create_all(engine)
