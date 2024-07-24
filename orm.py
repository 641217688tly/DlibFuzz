import yaml
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Enum, Table, Boolean
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

cluster_pytorch_combination_association = Table(
    'cluster_pytorch_combination_association', Base.metadata,
    Column('cluster_id', Integer, ForeignKey('cluster.id'), primary_key=True),
    Column('pytorch_combination_id', Integer, ForeignKey('pytorch_api_combination.id'), primary_key=True)
)

cluster_tensorflow_combination_association = Table(
    'cluster_tensorflow_combination_association', Base.metadata,
    Column('cluster_id', Integer, ForeignKey('cluster.id'), primary_key=True),
    Column('tensorflow_combination_id', Integer, ForeignKey('tensorflow_api_combination.id'), primary_key=True)
)

cluster_jax_combination_association = Table(
    'cluster_jax_combination_association', Base.metadata,
    Column('cluster_id', Integer, ForeignKey('cluster.id'), primary_key=True),
    Column('jax_combination_id', Integer, ForeignKey('jax_api_combination.id'), primary_key=True)
)


# ----------------------------------Pytorch----------------------------------
class PytorchAPI(Base):
    __tablename__ = 'pytorch_api'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    signature = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    is_clustered = Column(Boolean, default=False)
    error_triggers = relationship('PytorchErrorTriggerCode', back_populates='api')


class PytorchErrorTriggerCode(Base):
    __tablename__ = 'pytorch_error_trigger_code'
    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, ForeignKey('pytorch_api.id'))
    api = relationship('PytorchAPI', back_populates='error_triggers')
    code = Column(Text, nullable=False)
    description = Column(Text, nullable=True)  # 描述该代码片段是怎么触发bug的


class PytorchAPICombination(Base):
    __tablename__ = 'pytorch_api_combination'
    id = Column(Integer, primary_key=True)
    apis = relationship('PytorchAPI', secondary=pytorch_api_combination_association)
    test_seeds = relationship('ClusterTestSeed', back_populates='pytorch_combination')


# ----------------------------------Tensorflow----------------------------------

class TensorflowAPI(Base):
    __tablename__ = 'tensorflow_api'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    signature = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    is_clustered = Column(Boolean, default=False)
    error_triggers = relationship('TensorflowErrorTriggerCode', back_populates='api')


class TensorflowErrorTriggerCode(Base):
    __tablename__ = 'tensorflow_error_trigger_code'
    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, ForeignKey('tensorflow_api.id'))
    api = relationship('TensorflowAPI', back_populates='error_triggers')
    code = Column(Text, nullable=False)
    description = Column(Text, nullable=True)


class TensorflowAPICombination(Base):
    __tablename__ = 'tensorflow_api_combination'
    id = Column(Integer, primary_key=True)
    apis = relationship('TensorflowAPI', secondary=tensorflow_api_combination_association)
    test_seeds = relationship('ClusterTestSeed', back_populates='tensorflow_combination')


# ----------------------------------Jax----------------------------------

class JaxAPI(Base):
    __tablename__ = 'jax_api'
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    signature = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    is_clustered = Column(Boolean, default=False)
    error_triggers = relationship('JaxErrorTriggerCode', back_populates='api')


class JaxErrorTriggerCode(Base):
    __tablename__ = 'jax_error_trigger_code'
    id = Column(Integer, primary_key=True)
    api_id = Column(Integer, ForeignKey('jax_api.id'))
    api = relationship('JaxAPI', back_populates='error_triggers')
    code = Column(Text, nullable=False)
    description = Column(Text, nullable=True)


class JaxAPICombination(Base):
    __tablename__ = 'jax_api_combination'
    id = Column(Integer, primary_key=True)
    apis = relationship('JaxAPI', secondary=jax_api_combination_association)
    test_seeds = relationship('ClusterTestSeed', back_populates='jax_combination')


# ----------------------------------Cluster----------------------------------

class Cluster(Base):
    __tablename__ = 'cluster'
    id = Column(Integer, primary_key=True)
    description = Column(Text, nullable=True)
    energy = Column(Integer, default=10)
    pytorch_combinations = relationship('PytorchAPICombination', secondary=cluster_pytorch_combination_association)
    tensorflow_combinations = relationship('TensorflowAPICombination',
                                           secondary=cluster_tensorflow_combination_association)
    jax_combinations = relationship('JaxAPICombination', secondary=cluster_jax_combination_association)
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
    pytorch_code = Column(Text, nullable=False)
    tensorflow_code = Column(Text, nullable=False)
    jax_code = Column(Text, nullable=False)


# 创建表
Base.metadata.create_all(engine)


def add_data():
    Session = sessionmaker(bind=engine)
    session = Session()
    if not session.query(PytorchAPI).first() and not session.query(TensorflowAPI).first() and not session.query(
            JaxAPI).first():
        # 从文件读取Pytorch APIs并添加到数据库
        try:
            with open('cluster/api_signatures/pytorch/torch_valid_apis.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    # 移除空格和换行符
                    api_name = line.strip()
                    # 创建Pytorch实例并添加到session
                    if api_name:  # 确保不是空行
                        pytorch_api = PytorchAPI(name=api_name)
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
                        tensorflow_api = TensorflowAPI(name=api_name)
                        session.add(tensorflow_api)
        except Exception as e:
            print(f"Error reading Tensorflow APIs file: {e}")

        # JAX APIs并添加到数据库
        try:
            with open('cluster/api_signatures/jax/jax_valid_apis.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    # 移除空格和换行符
                    api_name = line.strip()
                    # 创建Tensorflow实例并添加到session
                    if api_name:  # 确保不是空行
                        jax_api = JaxAPI(name=api_name)
                        session.add(jax_api)
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


def attach_code_snippet():
    print("Attach code snippets")
    # TODO


if __name__ == '__main__':
    # 如果JAX/Tensorflow/Pytorch数据库为空，添加数据
    add_data()
