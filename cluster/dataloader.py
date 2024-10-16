import json
import os
from orm import *
import utils


def add_apis_from_txt(session, torch_version="1.12", tf_version="2.10", jax_version="0.4.13"):
    if not session.query(PytorchAPI).first():  # 从文件读取Pytorch APIs并添加到数据库
        try:
            with open('apis/pytorch/torch_valid_apis.txt', 'r', encoding='utf-8') as file:
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
                                version=torch_version
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
            with open('apis/tensorflow/tf_valid_apis.txt', 'r', encoding='utf-8') as file:
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
                                version=tf_version
                            )
                            session.add(tensorflow_api)
                session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error processing Tensorflow APIs file: {e}")
        finally:
            session.close()
            print("Tensorflow API data loaded successfully!")

    # if not session.query(JAXAPI).first():  # JAX APIs并添加到数据库
    #     try:
    #         with open('apis/jax/jax_valid_apis.txt', 'r', encoding='utf-8') as file:
    #             for line in file:
    #                 full_api_name = line.strip()
    #                 if full_api_name:  # 确保不是空行
    #                     module_name, api_name = full_api_name.rsplit('.', 1)
    #                     is_valid = utils.validate_api_existence(module_name, api_name)
    #                     if is_valid:
    #                         jax_api = JAXAPI(
    #                             name=api_name,
    #                             module=module_name,
    #                             full_name=full_api_name,
    #                             version=jax_version
    #                         )
    #                         session.add(jax_api)
    #             session.commit()
    #     except Exception as e:
    #         session.rollback()
    #         print(f"Error processing JAX APIs file: {e}")
    #     finally:
    #         session.close()
    #         print("JAX API data loaded successfully!")


def add_apis_from_json(session, torch_version="1.12", tf_version="2.10", jax_version="0.4.13"):
    if not session.query(JAXAPI).first():  # 从JSON文件读取JAX APIs并添加到数据库
        try:
            with open('apis/jax/jax_apis.json', 'r', encoding='utf-8') as file:
                torch_apis = json.load(file)
                for api_id, api_info in torch_apis.items():
                    # 检查数据库中是否已存在该API
                    is_valid = utils.validate_api_existence(api_info['module'], api_info['name'])
                    api_exists = session.query(JAXAPI).filter_by(name=api_info['name']).first()
                    if not (api_exists and is_valid):
                        # 创建JAXAPI实例并添加到session
                        new_api = JAXAPI(
                            name=api_info['name'],
                            module=api_info['module'],
                            full_name=api_info['fullName'],
                            signature=api_info['signature'],
                            description=api_info['description'],
                            version=jax_version
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
            with open('apis/pytorch/torch_apis.json', 'r', encoding='utf-8') as file:
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
                            version=torch_version
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
            with open('apis/tensorflow/tf_apis.json', 'r', encoding='utf-8') as file:
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
                            version=tf_version
                        )
                        session.add(new_api)
                session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error processing Tensorflow APIs file: {e}")
        finally:
            session.close()
            print("Tensorflow API data loaded successfully!")


def attach_error_trigger_code(api_class, error_trigger_class, dir_path, session):
    # 获取所有.json文件的列表
    json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
    files_num = len(json_files)  # 总文件数

    # 读取目录下所有json文件
    for count, filename in enumerate(json_files, start=1):  # start=1表示从1开始计数
        file_path = os.path.join(dir_path, filename)
        print(
            f"----------------------------------------------------------Loading {api_class.__name__}'s error trigger: {count}----------------------------------------------------------")
        with open(file_path, 'r', encoding='utf-8') as file:
            print(f"Current JSON File: {file_path}\n")
            data = json.load(file)
            apis = data.get("API", [])
            title = data.get("Title", "")
            code = data.get("Code", "")
            description = data.get("Description", "")
            if not code or not apis:  # 如果code为""或apis为空列表，则跳过
                print(f"Skipping {file_path} due to missing code or APIs")
                continue

            for full_api_name in apis:
                try:
                    print(f"Processing {full_api_name}...")
                    module_name, api_name = full_api_name.rsplit('.', 1)
                    if utils.validate_api_existence(module_name, api_name):  # 验证API是否存在
                        api = session.query(api_class).filter_by(full_name=full_api_name).first()
                        if not api:
                            api = api_class(name=api_name, module=module_name, full_name=full_api_name)
                            session.add(api)
                            session.flush()  # 确保api对象有id

                        # 检查api.error_triggers中是否已经存在相同的错误触发代码
                        existing_trigger = session.query(error_trigger_class).filter_by(
                            api_id=api.id,
                            title=title,
                            code=code
                        ).first()

                        if not existing_trigger:
                            # 创建新的错误触发代码实例并添加到数据库
                            new_trigger = error_trigger_class(
                                api_id=api.id,
                                title=title,
                                code=code,
                                description=description
                            )
                            session.add(new_trigger)
                        print(f"Successfully processed {full_api_name}\n")
                    else:
                        print(
                            f"WARNING: The {full_api_name} does not exist or is deprecated in the current version of the library!\n")
                    session.commit()  # 提交所有更改
                except Exception as e:
                    session.rollback()  # 出现异常时回滚
                    print(f"An error occurred: {e}")
        print(f"Processed {count}/{files_num} files")


if __name__ == '__main__':
    session = utils.get_session()

    # 如果JAX/Tensorflow/Pytorch数据库中为空则添加数据
    add_apis_from_txt(session, torch_version="1.12", tf_version="2.10", jax_version="0.4.13")
    add_apis_from_json(session, torch_version="1.12", tf_version="2.10", jax_version="0.4.13")

    # 将错误触发代码附加到Pytorch/Tensorflow/JAX API下
    torch_dir = '../data/error_triggers/pytorch_issue'
    tf_dir = '../data/error_triggers/tensorflow_issue'
    jax_dir = '../data/error_triggers/jax_issue'
    # attach_error_trigger_code(PytorchAPI, PytorchErrorTrigger, torch_dir, session)
    # attach_error_trigger_code(TensorflowAPI, TensorflowErrorTrigger, tf_dir, session)
    # attach_error_trigger_code(JAXAPI, JAXErrorTrigger, jax_dir, session)
