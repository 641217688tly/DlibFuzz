import yaml
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Enum, Table
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.orm import sessionmaker

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
engine = create_engine(db_url)
Base = declarative_base()

output_equivalence_association_pytorch = Table('output_equivalence_association_pytorch', Base.metadata,
                                               Column('pytorch_id', Integer, ForeignKey('pytorch.id'),
                                                      primary_key=True),
                                               Column('cluster_id', Integer,
                                                      ForeignKey('output_equivalence_cluster.id'), primary_key=True)
                                               )

output_equivalence_association_tensorflow = Table('output_equivalence_association_tensorflow', Base.metadata,
                                                  Column('tensorflow_id', Integer, ForeignKey('tensorflow.id'),
                                                         primary_key=True),
                                                  Column('cluster_id', Integer,
                                                         ForeignKey('output_equivalence_cluster.id'), primary_key=True)
                                                  )

output_equivalence_association_jax = Table('output_equivalence_association_jax', Base.metadata,
                                           Column('jax_id', Integer, ForeignKey('jax.id'), primary_key=True),
                                           Column('cluster_id', Integer, ForeignKey('output_equivalence_cluster.id'),
                                                  primary_key=True)
                                           )

function_equivalence_association_pytorch = Table('function_equivalence_association_pytorch', Base.metadata,
                                                 Column('pytorch_id', Integer, ForeignKey('pytorch.id'),
                                                        primary_key=True),
                                                 Column('cluster_id', Integer,
                                                        ForeignKey('function_equivalence_cluster.id'), primary_key=True)
                                                 )

function_equivalence_association_tensorflow = Table('function_equivalence_association_tensorflow', Base.metadata,
                                                    Column('tensorflow_id', Integer, ForeignKey('tensorflow.id'),
                                                           primary_key=True),
                                                    Column('cluster_id', Integer,
                                                           ForeignKey('function_equivalence_cluster.id'),
                                                           primary_key=True)
                                                    )

function_equivalence_association_jax = Table('function_equivalence_association_jax', Base.metadata,
                                             Column('jax_id', Integer, ForeignKey('jax.id'), primary_key=True),
                                             Column('cluster_id', Integer,
                                                    ForeignKey('function_equivalence_cluster.id'), primary_key=True)
                                             )


class Pytorch(Base):
    __tablename__ = 'pytorch'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    signature = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    energy = Column(Integer, default=10)
    code_snippets = relationship('CodeSnippet', back_populates='pytorch')
    output_clusters = relationship('OutputEquivalenceCluster', secondary=output_equivalence_association_pytorch,
                                   back_populates='pytorches')
    function_clusters = relationship('FunctionEquivalenceCluster', secondary=function_equivalence_association_pytorch,
                                     back_populates='pytorches')


class Tensorflow(Base):
    __tablename__ = 'tensorflow'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    signature = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    energy = Column(Integer, default=10)
    code_snippets = relationship('CodeSnippet', back_populates='tensorflow')
    output_clusters = relationship('OutputEquivalenceCluster', secondary=output_equivalence_association_tensorflow,
                                   back_populates='tensorflows')
    function_clusters = relationship('FunctionEquivalenceCluster',
                                     secondary=function_equivalence_association_tensorflow,
                                     back_populates='tensorflows')


class JAX(Base):
    __tablename__ = 'jax'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    signature = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    energy = Column(Integer, default=10)
    code_snippets = relationship('CodeSnippet', back_populates='jax')
    output_clusters = relationship('OutputEquivalenceCluster', secondary=output_equivalence_association_jax,
                                   back_populates='jaxes')
    function_clusters = relationship('FunctionEquivalenceCluster', secondary=function_equivalence_association_jax,
                                     back_populates='jaxes')


class CodeSnippet(Base):
    __tablename__ = 'code_snippet'
    id = Column(Integer, primary_key=True)
    code = Column(Text, nullable=False)
    pytorch_id = Column(Integer, ForeignKey('pytorch.id'))
    tensorflow_id = Column(Integer, ForeignKey('tensorflow.id'))
    jax_id = Column(Integer, ForeignKey('jax.id'))
    pytorch = relationship('Pytorch', back_populates='code_snippets')
    tensorflow = relationship('Tensorflow', back_populates='code_snippets')
    jax = relationship('JAX', back_populates='code_snippets')


class OutputEquivalenceCluster(Base):
    __tablename__ = 'output_equivalence_cluster'
    id = Column(Integer, primary_key=True)
    pytorches = relationship('Pytorch', secondary=output_equivalence_association_pytorch,
                             back_populates='output_clusters')
    tensorflows = relationship('Tensorflow', secondary=output_equivalence_association_tensorflow,
                               back_populates='output_clusters')
    jaxes = relationship('JAX', secondary=output_equivalence_association_jax, back_populates='output_clusters')


class FunctionEquivalenceCluster(Base):
    __tablename__ = 'function_equivalence_cluster'
    id = Column(Integer, primary_key=True)
    pytorches = relationship('Pytorch', secondary=function_equivalence_association_pytorch,
                             back_populates='function_clusters')
    tensorflows = relationship('Tensorflow', secondary=function_equivalence_association_tensorflow,
                               back_populates='function_clusters')
    jaxes = relationship('JAX', secondary=function_equivalence_association_jax, back_populates='function_clusters')


# 创建表
Base.metadata.create_all(engine)


def add_data():
    Session = sessionmaker(bind=engine)
    session = Session()
    # 从文件读取Pytorch APIs并添加到数据库
    try:
        with open('cluster/api_signatures/pytorch/torch_valid_apis.txt', 'r', encoding='utf-8') as file:
            for line in file:
                # 移除空格和换行符
                api_name = line.strip()
                # 创建Pytorch实例并添加到session
                if api_name:  # 确保不是空行
                    pytorch_api = Pytorch(name=api_name)
                    session.add(pytorch_api)
    except Exception as e:
        print(f"Error reading Pytorch APIs file: {e}")

    # 从文件读取Tensorflow APIs并添加到数据库
    try:
        with open('cluster/api_signatures/tensorflow/tf_valid_apis.txt', 'r', encoding='utf-8') as file:
            for line in file:
                # 移除空格和换行符
                api_name = line.strip()
                # 创建Tensorflow实例并添加到session
                if api_name:  # 确保不是空行
                    tensorflow_api = Tensorflow(name=api_name)
                    session.add(tensorflow_api)
    except Exception as e:
        print(f"Error reading Tensorflow APIs file: {e}")

    # 提交到数据库
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error committing to database: {e}")
    finally:
        session.close()
        print("Data loaded successfully!")

# add_data()
