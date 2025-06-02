import json
import os
from orm import *
import utils
from torch_dataloader import add_pytorch_apis_from_doc
from ms_dataloader import add_mindspore_apis_from_doc
from jax_dataloader import add_jax_apis_from_doc
from jittor_dataloader import add_jittor_apis_from_doc

def process_signature(full_api_name, raw_signature):
    # 将raw_signature按照最后一个'->'分割为输入参数和输出参数
    parts = raw_signature.strip().rsplit('->', 1)
    if len(parts) == 2:
        input_params, output_params = parts
        input_params = input_params.strip()
        output_params = output_params.strip()
        # 如果input_params没有被"()"包围，则为其添加括号
        if not input_params.startswith('(') and not input_params.endswith(')'):
            input_params = f"({input_params})"
        # 如果output_params没有被"()"包围，则为其添加括号
        if not output_params.startswith('(') and not output_params.endswith(')'):
            output_params = f"({output_params})"
        signature = f"{full_api_name}{input_params} -> {output_params}"
        return signature
    else:  # 如果函数没有输出值
        input_params = raw_signature.strip()
        # output_params = "()"
        # 如果signature没有被"()"包围，则添加括号
        if not input_params.startswith('(') and not input_params.endswith(')'):
            input_params = f"({input_params})"
        # signature = f"{full_api_name}{input_params} -> {output_params}"
        signature = f"{full_api_name}{input_params}"
        return signature


def add_apis_from_json(db_session, file_path, lib, version):
    # 首先检查数据库中是否已存在该库的API
    lib_exists = db_session.query(API).filter_by(lib=lib, version=version).first()
    if lib_exists is not None:
        print(f"{lib} API data already exists in the database!")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            apis = json.load(file)
            for api_id, api_info in apis.items():
                module = api_info['module']
                full_api_name = api_info['fullName']

                # 检查数据库中是否已存在该API
                is_valid = utils.validate_api_existence(module, api_info['name'])
                api_exists = db_session.query(API).filter_by(full_name=full_api_name, lib=lib, version=version).first()
                if api_exists is None and is_valid:  # 如果API不存在且API是有效的:
                    # 创建TensorflowAPI实例并添加到session
                    new_api = API(
                        lib=lib,
                        name=api_info['name'],
                        module=module,
                        full_name=full_api_name,
                        signature=process_signature(full_api_name, api_info['signature']),
                        description=api_info['description'],
                        version=version
                    )
                    db_session.add(new_api)
            db_session.commit()
    except Exception as e:
        db_session.rollback()
        print(f"Error processing Tensorflow APIs file: {e}")
    finally:
        db_session.close()
        print(f"{lib} API data loaded successfully!")


def attach_history_errors(db_session, dir_path, whether_supplement_api=True):
    # 获取所有.json文件的列表
    json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
    files_num = len(json_files)  # 总文件数
    added_errors_num = 0
    # 读取目录下所有json文件
    for count, filename in enumerate(json_files, start=1):  # start=1表示从1开始计数
        file_path = os.path.join(dir_path, filename)
        print(
            f"----------------------------------------------------------Loading History Errors: {count}----------------------------------------------------------")
        with open(file_path, 'r', encoding='utf-8') as file:
            print(f"Current JSON File: {file_path}\n")
            # 尝试解析JSON文件, 如果解析失败则跳过
            try:
                data = json.load(file)
            except Exception:
                continue
            apis = data.get("API", [])
            title = data.get("Title", "")
            code = data.get("Code", "")
            url = data.get("URL", "")
            description = data.get("Description", "")
            if not isinstance(description, str):  # 如果解析得到的description不是字符串类型而是字典类型或数组类型, 则将其转换为字符串
                description = str(description)

            if not code or not apis:  # 如果code为""或apis为空列表，则跳过
                print(f"Skipping {file_path} due to missing code or APIs")
                continue

            for full_api_name in apis:
                try:
                    print(f"Processing {full_api_name}...")
                    module_name, api_name = full_api_name.rsplit('.', 1)
                    lib = utils.map_module2lib(full_api_name.split('.')[0])
                    if utils.validate_api_existence(module_name, api_name):  # 验证API在当前Python环境中的当前版本的DL库内是否存在
                        api = db_session.query(API).filter_by(lib=lib, full_name=full_api_name).first()
                        if not api and whether_supplement_api:  # 如果API不存在且需要补充API
                            api_info = utils.inspect_api_info(module_name, api_name)
                            api = API(
                                name=api_name,
                                module=module_name,
                                full_name=full_api_name,
                                signature=api_info['signature'],
                                description=api_info['description'],
                                lib=lib,
                                version=api_info['version'],
                            )
                            db_session.add(api)
                            db_session.flush()  # 确保api对象有id
                        elif not api and not whether_supplement_api:  # 如果API不存在且不允许补充API
                            continue
                        # 检查api.history_errors中是否已经存在相同的错误触发代码
                        existing_trigger = db_session.query(APIHistoryError).filter_by(
                            api_id=api.id,
                            title=title,
                            code=code
                        ).first()

                        if not existing_trigger:
                            # 创建新的错误触发代码实例并添加到数据库
                            new_errors = APIHistoryError(
                                api_id=api.id,
                                title=title,
                                code=code,
                                issue_url=url,
                                description=description
                            )
                            db_session.add(new_errors)
                        print(f"Successfully processed {full_api_name}\n")
                    else:
                        print(f"WARNING: The {full_api_name} does not exist or is deprecated in the current version of the library!\n")
                    db_session.commit()  # 提交所有更改
                except Exception as e:
                    db_session.rollback()  # 出现异常时回滚
                    print(f"An error occurred: {e}")
            # 检索APIHistoryError表, 查找当前History Error是否已经被添加
            existing_error = db_session.query(APIHistoryError).filter_by(title=title, code=code,
                                                                         description=description).first()
            if existing_error:
                added_errors_num = added_errors_num + 1
        print(f"Processed {count}/{files_num} files")
    return added_errors_num


if __name__ == '__main__':
    session = utils.get_session()
    torch_docs_folder_path = './../data/docs/torch/2.3.0/handled/'
    add_pytorch_apis_from_doc(torch_docs_folder_path, "2.4.1")
    ms_docs_folder_path = './../data/docs/ms/2.5.0/api mapping docs/2.5.0/handled/'
    add_mindspore_apis_from_doc(ms_docs_folder_path, "2.5.0")
    jittor_docs_folder_path = "./../data/docs/jittor/1.3.9.2/handled"
    add_jittor_apis_from_doc(jittor_docs_folder_path, "1.3.9.14")
    jax_docs_folder_path = "./../data/docs/jax/0.4.13/handled"
    add_jax_apis_from_doc(jax_docs_folder_path, "0.4.33")

    # 如果JAX/Tensorflow/Pytorch数据库中为空则添加数据
    # add_apis_from_json(session, '../data/apis/jax/jax_apis.json', 'JAX', "0.4.33")
    # add_apis_from_json(session, '../data/apis/jittor/jt_apis.json', 'Jittor', "1.3.9.14")

    # 将错误触发代码附加到Pytorch/JAX/MindSpore/Jittor APIs下
    torch_dir = '../data/history_errors/pytorch_issues'
    jax_dir = '../data/history_errors/jax_issues'
    ms_dir = '../data/history_errors/mindspore_issues'
    jt_dir = '../data/history_errors/jittor_issues'
    added_torch_errors_num = attach_history_errors(session, torch_dir)
    added_jax_errors_num = attach_history_errors(session, jax_dir)
    added_ms_errors_num = attach_history_errors(session, ms_dir)
    added_jittor_errors_num = attach_history_errors(session, jt_dir)
    print(f"Total number of added Pytorch history issues: {added_torch_errors_num}")  # 1791
    print(f"Total number of added JAX history issues: {added_jax_errors_num}")  # 1534
    print(f"Total number of added MindSpore history issues: {added_ms_errors_num}")  # 481
    print(f"Total number of added Jittor error triggers: {added_jittor_errors_num}")  # 107
