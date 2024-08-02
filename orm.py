import json
import yaml
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey, Text, Table, Boolean
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.orm import sessionmaker
import utils

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
    code = Column(Text, nullable=True) # pytorch_code + tensorflow_code + jax_code
    pytorch_code = Column(Text, nullable=True)
    tensorflow_code = Column(Text, nullable=True)
    jax_code = Column(Text, nullable=True)


# 创建表
Base.metadata.create_all(engine)


def add_data_from_txt():
    Session = sessionmaker(bind=engine)
    session = Session()

    if not session.query(PytorchAPI).first():  # 从文件读取Pytorch APIs并添加到数据库
        try:
            with open('cluster/apis/pytorch/torch_valid_apis.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    full_api_name = line.strip()
                    if full_api_name:  # 确保不是空行
                        module_name, api_name = full_api_name.rsplit('.', 1)
                        is_valid = utils.validate_api_existence(module_name, api_name)
                        if is_valid:
                            pytorch_api = PytorchAPI(
                                name=api_name,
                                module=module_name,
                                full_name=full_api_name,
                                version="1.12"
                            )
                            session.add(pytorch_api)
                session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error processing Pytorch APIs file: {e}")
        finally:
            session.close()
            print("Pytorch API data loaded successfully!")

    if not session.query(TensorflowAPI).first():  # 从文件读取Tensorflow APIs并添加到数据库
        try:
            with open('cluster/apis/tensorflow/tf_valid_apis.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    full_api_name = line.strip()
                    if full_api_name:  # 确保不是空行
                        module_name, api_name = full_api_name.rsplit('.', 1)
                        is_valid = utils.validate_api_existence(module_name, api_name)
                        if is_valid:
                            tensorflow_api = TensorflowAPI(
                                name=api_name,
                                module=module_name,
                                full_name=full_api_name,
                                version="2.10"
                            )
                            session.add(tensorflow_api)
                session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error processing Tensorflow APIs file: {e}")
        finally:
            session.close()
            print("Tensorflow API data loaded successfully!")

    # if not session.query(JaxAPI).first():  # JAX APIs并添加到数据库
    #     try:
    #         with open('cluster/apis/jax/jax_valid_apis.txt', 'r', encoding='utf-8') as file:
    #             for line in file:
    #                 full_api_name = line.strip()
    #                 if full_api_name:  # 确保不是空行
    #                     module_name, api_name = full_api_name.rsplit('.', 1)
    #                     is_valid = utils.validate_api(module_name, api_name)
    #                     if is_valid:
    #                         jax_api = JaxAPI(
    #                             name=api_name,
    #                             module=module_name,
    #                             full_name=full_api_name,
    #                             version="0.4.13"
    #                         )
    #                         session.add(jax_api)
    #             session.commit()
    #     except Exception as e:
    #         session.rollback()
    #         print(f"Error processing Jax APIs file: {e}")
    #     finally:
    #         session.close()
    #         print("Jax API data loaded successfully!")


def add_data_from_json():
    Session = sessionmaker(bind=engine)
    session = Session()

    if not session.query(JaxAPI).first():  # 从JSON文件读取Jax APIs并添加到数据库
        try:
            with open('cluster/apis/jax/jax_apis.json', 'r', encoding='utf-8') as file:
                torch_apis = json.load(file)
                for api_id, api_info in torch_apis.items():
                    # 检查数据库中是否已存在该API
                    is_valid = utils.validate_api_existence(api_info['module'], api_info['name'])
                    api_exists = session.query(JaxAPI).filter_by(name=api_info['name']).first()
                    if not (api_exists and is_valid):
                        # 创建JaxAPI实例并添加到session
                        new_api = JaxAPI(
                            name=api_info['name'],
                            module=api_info['module'],
                            full_name=api_info['fullName'],
                            signature=api_info['signature'],
                            description=api_info['description'],
                            version="0.4.13"
                        )
                        session.add(new_api)
                session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error processing JAX APIs file: {e}")
        finally:
            session.close()
            print("JAX API data loaded successfully!")

    if not session.query(PytorchAPI).first():  # 从JSON文件读取Pytorch APIs并添加到数据库
        try:
            with open('cluster/apis/pytorch/torch_apis.json', 'r', encoding='utf-8') as file:
                torch_apis = json.load(file)
                for api_id, api_info in torch_apis.items():
                    # 检查数据库中是否已存在该API
                    is_valid = utils.validate_api_existence(api_info['module'], api_info['name'])
                    api_exists = session.query(PytorchAPI).filter_by(name=api_info['name']).first()
                    if not (api_exists and is_valid):
                        # 创建PytorchAPI实例并添加到session
                        new_api = PytorchAPI(
                            name=api_info['name'],
                            module=api_info['module'],
                            full_name=api_info['fullName'],
                            signature=api_info['signature'],
                            description=api_info['description'],
                            version="1.12"
                        )
                        session.add(new_api)
                session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error processing Pytorch APIs file: {e}")
        finally:
            session.close()
            print("Pytorch API data loaded successfully!")

    if not session.query(TensorflowAPI).first():  # 从JSON文件读取Tensorflow APIs并添加到数据库
        try:
            with open('cluster/apis/tensorflow/tf_apis.json', 'r', encoding='utf-8') as file:
                torch_apis = json.load(file)
                for api_id, api_info in torch_apis.items():
                    # 检查数据库中是否已存在该API
                    is_valid = utils.validate_api_existence(api_info['module'], api_info['name'])
                    api_exists = session.query(TensorflowAPI).filter_by(name=api_info['name']).first()
                    if not (api_exists and is_valid):
                        # 创建TensorflowAPI实例并添加到session
                        new_api = TensorflowAPI(
                            name=api_info['name'],
                            module=api_info['module'],
                            full_name=api_info['fullName'],
                            signature=api_info['signature'],
                            description=api_info['description'],
                            version="2.10"
                        )
                        session.add(new_api)
                session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error processing Tensorflow APIs file: {e}")
        finally:
            session.close()
            print("Tensorflow API data loaded successfully!")


def attach_error_trigger_code():
    print("Attach code snippets")
    # TODO


if __name__ == '__main__':
    # 如果JAX/Tensorflow/Pytorch数据库为空，添加数据
    add_data_from_txt()
    add_data_from_json()
