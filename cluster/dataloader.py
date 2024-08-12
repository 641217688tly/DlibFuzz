import json
import os

from sqlalchemy.exc import NoResultFound

from orm import *
import utils


def add_apis_from_txt(session):
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


def add_apis_from_json(session):
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


def attach_error_trigger_code(api_class, dir_path, session):
    try:
        # 读取目录下所有json文件
        for filename in os.listdir(dir_path):
            if filename.endswith(".json"):
                file_path = os.path.join(dir_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    apis = data.get("API", [])
                    title = data.get("Title", "")
                    code = data.get("Code", "")
                    description = data.get("Description", "")
                    if not code or not apis:  # 如果code为""或apis为空列表，则跳过
                        continue

                    for full_api_name in apis:
                        module_name, api_name = full_api_name.rsplit('.', 1)
                        if utils.validate_api_existence(module_name, api_name):  # 验证API是否存在
                            api = session.query(api_class).filter_by(full_name=full_api_name).first()
                            if not api:
                                api = api_class(name=api_name, module=module_name, full_name=full_api_name)
                                session.add(api)
                                session.flush()  # 确保api对象有id

                            # 检查api.error_triggers中是否已经存在相同的错误触发代码
                            existing_trigger = session.query(api.error_triggers.class_).filter_by(
                                api_id=api.id,
                                title=title,
                                code=code
                            ).first()

                            if not existing_trigger:
                                # 创建新的错误触发代码实例并添加到数据库
                                new_trigger = api.error_triggers.class_(
                                    api_id=api.id,
                                    title=title,
                                    code=code,
                                    description=description
                                )
                                session.add(new_trigger)
                        session.commit()  # 提交所有更改
    except Exception as e:
        session.rollback()  # 出现异常时回滚
        print(f"An error occurred: {e}")

def attach_error_trigger_code_backup(api_class, dir_path, session):
    # 从dir_path下读取所有json文件: dir_path/jax_issue_0.json, dir_path/jax_issue_1.json, ...

    # 逐个读取每个json文件中的信息
    # 从json["API"]这获得API List, 然后逐个遍历API
    # if(当前API经过utils.validate_api_existence(api_module, api_name)验证为True)
    # if(数据库中不存在该API):
    # 将它添加到数据库
    # 新建一个ErrorTrigger实例并添加到新建的API的error_triggers属性中
    # else:
    # 获取该API对象(PytorchAPI/TensorflowAPI/JaxAPI)
    # if(该API对象的error_triggers属性中已经存在该ErrorTrigger实例)
    # 跳过
    # else:
    # 将ErrorTrigger实例添加到该API对象的error_triggers属性中
    print("Attach code snippets")


if __name__ == '__main__':
    session = utils.get_session()
    # 如果JAX/Tensorflow/Pytorch数据库中为空，添加数据
    # add_apis_from_txt(session)
    # add_apis_from_json(session)
