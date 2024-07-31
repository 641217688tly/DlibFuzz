from json import JSONDecodeError
import httpx
from openai import OpenAI
from orm import *
from utils import *

TORCH_VERSION = "1.12"
TF_VERSION = "2.10"
JAX_VERSION = "0.4.13"


# ----------------------------------------------clusterer----------------------------------------------
class Clusterer:
    def __init__(self, api, session, openai_client):
        self.api = api
        self.session = session
        self.openai_client = openai_client
        self.model = "gpt-4o-mini",  # gpt-4o-mini gpt-3.5-turbo
        self.torch_ver = TORCH_VERSION
        self.tf_ver = TF_VERSION
        self.jax_ver = JAX_VERSION
        self.messages = self.initialize_message()
        self.responses = []
        self.errors = []

    def initialize_message(self):  # 构建clusterer的初始提词并返回对话消息
        # TODO 后续可能会从将Prompt中的Example存入JSON以避免硬编码
        clusterer_example_1 = """json
        {   
            "Pytorch" : {
                "1" : ["torch.tensor", "torch.nn.CrossEntropyLoss"],
            },
            "Tensorflow" : {
                "1" : ["tensorflow.keras.losses.CategoricalCrossentropy"], // tensorflow.keras.losses.CategoricalCrossentropy internal will automatically array into Tensorflow tensor, so there is no need to be used with tensorflow.constant
                "2" : ["tensorflow.constant", "tensorflow.nn.softmax_cross_entropy_with_logits"] // Before using tensorflow.nn.softmax_cross_entropy_with_logits, it needs to use tensorflow.constant to convert the input value into a tensor
            },
            "JAX" : {
                "1" : ["jax.numpy.array", "jax.nn.log_softmax", "jax.numpy.sum"] // Before using jax.nn.softmax_cross_entropy, it needs to use jax.numpy.array to convert the input value into a tensor. After using jax.nn.softmax_cross_entropy, it needs to use jax.numpy.sum to calculate the sum of the cross entropy loss
            }
        }
        """

        clusterer_example_2 = """json
        {   
            // When there are no TensorFlow APIs whose combined output with Pytorch's API has the same value, output an empty json
            "Pytorch" : {}, 
            "Tensorflow" : {}, 
            "JAX" : {}
        }
        
        """
        clusterer_prompt = f"""
        Which apis or combinations of api calls in TensorFlow (v{self.tf_ver}) and JAX (v{self.jax_ver}) have the exact same functionality as {self.api.name} in PyTorch (v{self.torch_ver})?
        Note: "The same functionality" means that these APIs are responsible for performing exactly the same tasks. When these APIs have no return value, using these APIs to perform the same operations on inputs with the same structure or element values (such as tensors) should result in consistent changes to the original input. For example, PyTorch's torch.scatter_, TensorFlow's tf.scatter_update, and JAX's jax.ops.index_update all have the functionality to update tensors, and when the tensors being updated and the update strategies are the same, the updated tensors should be consistent. When these APIs have return values, PyTorch's torch.nn.ReLU, TensorFlow's tf.nn.relu or tf.keras.layers.ReLU, and JAX's jax.nn.relu all produce the same output values when given the same input values.
        Please output the function names or combinations of function names in PyTorch, TensorFlow, and JAX that meet the above conditions in JSON format, with an example shown below:
        Example 1: 
        {clusterer_example_1}
        Example 2: 
        {clusterer_example_2}
        """
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": clusterer_prompt}
        ]
        return messages

    def dynamic_api_call(self, module_name, function_name):
        try:
            # 动态导入模块
            module = importlib.import_module(module_name)
            # 尝试获取函数引用
            func = getattr(module, function_name, None)
            if func is None:
                return False, f"Function {function_name} not found in {module_name}"
            return True, "Function exists and is callable"
        except ImportError:
            return False, f"Module {module_name} not found"
        except Exception as e:
            return False, str(e)

    def validate_api_function_names(self, api_data):
        results = {}
        for tech, apis in api_data.items():
            for api_group_id, api_names in apis.items():
                for api_name in api_names:
                    module_name, func_name = api_name.rsplit('.', 1)
                    is_valid, message = self.dynamic_api_call(module_name, func_name)
                    if not is_valid:
                        print(f"Validation failed for {api_name}: {message}")
                    results[api_name] = is_valid
        return results

    def generate_cluster(self):
        attempt_num = 0
        while attempt_num < 5:  # 设置最大尝试次数以避免无限循环
            try:  # 假如返回的数据不符合JSON格式, 则重新调用OpenAI API, 直到返回的数据符合JSON格式为止
                # 调用 OpenAI API
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    messages=self.messages,
                    temperature=0.0,
                )
                response = response.choices[0].message.content
                self.responses.append(response)
                self.messages.append({"role": "assistant", "content": response})
                print(f"Clustered Pytorch API: {self.api.name}\nResponse:\n{response}")

                # 在此处需要检查: 1.响应的数据是否遵循JSON格式; 2.返回的是API的完整函数名(完整函数名 = 模块名.API名)而非函数签名 3.所有的API函数名必须有效(不是虚构的, 也不是被弃用的)
                json_data = json.loads(response)
                validation_results = self.validate_api_function_names(json_data)
                if not all(validation_results.values()):  # 如果有任何 API 名称验证失败
                    raise ValueError("One or more API names are invalid or unknown.")
                break  # 成功解析 JSON,跳出循环
            except (JSONDecodeError, ValueError) as e:
                print(f"Failed to decode JSON due to: {e}.\nCurrent attempt: {attempt_num + 1}. Retrying...")
                attempt_num += 1
            except Exception as e:
                self.session.rollback()  # 回滚在异常中的任何数据库更改
                print(f"An unexpected error occurred: {e}")
                break

        if attempt_num == 5:  # 设置最大尝试次数以避免无限循环
            print("Max attempts reached. Unable to get valid JSON data.")
            return

        return json_data

    # --------------------------------------store API Combinations and Cluster into database--------------------------------------
    def supplement_apis(self, api_combinations, api_class):  # 将Jax/Pytorch/Tensorflow的API组合内不在数据库中的API添加到数据库中
        """
        以下列数据为例:
        api_combinations = "Tensorflow" : {
            "1" : ["tensorflow.keras.losses.CategoricalCrossentropy"], // tensorflow.keras.losses.CategoricalCrossentropy internal will automatically array into Tensorflow tensor, so there is no need to be used with tensorflow.constant
            "2" : ["tensorflow.constant", "tensorflow.nn.softmax_cross_entropy_with_logits"] // Before using tensorflow.nn.softmax_cross_entropy_with_logits, it needs to use tensorflow.constant to convert the input value into a tensor
        }
        api_class = Tensorflow
        """
        api_combination_objects = {}
        for api_id, api_combination in api_combinations.items():  # 逐个访问每个API组合
            api_combination_objects[api_id] = [] # api_combination_objects = { "1" : [], "2" : [] }
            for full_api_name in api_combination:  # 获取某个API组合中的每个API
                api = self.session.query(api_class).filter_by(full_name=full_api_name).first()
                if not api:
                    module_name, api_name = full_api_name.rsplit('.', 1)
                    api = api_class(
                        name=api_name,
                        module=module_name,
                        full_name=full_api_name,# 根据api_class来设置version
                        version= self.torch_ver if api_class == PytorchAPI else self.tf_ver if api_class == TensorflowAPI else self.jax_ver
                    )
                    self.session.add(api)
                    self.session.commit()
                api_combination_objects[api_id].append(api) # { "1" : [CategoricalCrossentropy], "2" : [constant, softmax_cross_entropy_with_logits] }
        return api_combination_objects

    def associate_api_combinations_to_cluster(self, cluster, api_combination_objects, combination_class):
        for api_id, api_combination in api_combination_objects.items(): # 逐个访问每个API组合
            combination = combination_class(apis=api_combination, cluster=cluster)
            self.session.add(combination)
            self.session.commit()

    def save_cluster(self, json_data):
        """
            接收并处理clusterer的响应结果, 创建cluster聚类和关联的API组合
        """
        try:
            self.api.is_clustered = True
            # 1. 解析返回的JSON数据并检查Pytorch,Tensorflow和Jax中的所有API名,如果PytorchAPI表或TensorflowAPI表或JAX表中没有对应的API,则先在对应表中创建对应的数据
            torch_apis_combination_objects = self.supplement_apis(json_data['Pytorch'], PytorchAPI)
            tf_apis_combination_objects = self.supplement_apis(json_data['Tensorflow'], TensorflowAPI)
            jax_apis_combination_objects = self.supplement_apis(json_data['JAX'], JaxAPI)

            # 2. 创建Cluster对象
            new_cluster = Cluster()
            self.session.add(new_cluster)
            self.session.commit()

            # 3. 为Pytorch, Tensorflow和Jax的每个API组合创建对应的PytorchAPICombination, TensorflowAPICombination和JaxAPICombination对象, 之后将它们与新创建的Cluster对象关联
            self.associate_api_combinations_to_cluster(new_cluster, torch_apis_combination_objects, PytorchAPICombination)
            self.associate_api_combinations_to_cluster(new_cluster, tf_apis_combination_objects, TensorflowAPICombination)
            self.associate_api_combinations_to_cluster(new_cluster, jax_apis_combination_objects, JaxAPICombination)
            self.session.commit()
        except Exception as e:
            self.session.rollback()  # 回滚在异常中的任何数据库更改
            print(f"An error occurred: {e}")

    # ----------------------------------------------run()----------------------------------------------
    def cluster_api(self):
        # 利用clusterer进行聚类
        clusterer_json_data = self.generate_cluster()
        self.save_cluster(clusterer_json_data)


def run():
    # 创建数据库连接
    session = get_session()
    openai_client = get_openai_client()

    # 对未聚类的PytorchAPI进行聚类
    uncluttered_torch_apis = session.query(PytorchAPI).filter_by(is_clustered=False).all()
    while uncluttered_torch_apis:
        print("----------------------------------------------------------------------------------")
        cluster = Clusterer(uncluttered_torch_apis[0], session, openai_client)
        cluster.cluster_api()
        uncluttered_torch_apis = session.query(PytorchAPI).filter_by(is_clustered=False).all()
        total_apis_num = session.query(PytorchAPI).count()
        unclustered_torch_apis_num = len(uncluttered_torch_apis)
        print(f"Unclustered / Total: {unclustered_torch_apis_num} / {total_apis_num}")


if __name__ == '__main__':
    run()
