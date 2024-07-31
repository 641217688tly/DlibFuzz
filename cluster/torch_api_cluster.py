import inspect
from json import JSONDecodeError
from orm import *
from utils import *

TORCH_VERSION = "1.12"
TF_VERSION = "2.10"
JAX_VERSION = "0.4.13"


# ----------------------------------------------Clusterer----------------------------------------------
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

    def validate_api(self, full_api_name):
        module_name = ""
        api_name = ""
        try:
            module_name, api_name = full_api_name.rsplit('.', 1)
            module = importlib.import_module(module_name)
            func = getattr(module, api_name, None)

            if func is None or not callable(func):
                self.errors.append(f"{full_api_name} is not callable or does not exist.")
                return False

            if inspect.ismodule(func):
                self.errors.append(f"{full_api_name} is a module, not a function.")
                return False

            if inspect.isclass(func):
                self.errors.append(f"{full_api_name} is a class, not a function.")
                return False

            # if validate_api_availability(func):
            #     self.errors.append(f"{full_api_name} is deprecated.")
            #     return False
            return True
        except ImportError as e:
            self.errors.append(f"Module {module_name} not found: {str(e)}")
            return False
        except AttributeError:
            self.errors.append(f"{api_name} does not exist in {module_name}.")
            return False
        except Exception as e:
            self.errors.append(str(e))
            return False

    def validate_apis(self, response):
        """
        验证JSON中API的格式是完整函数名(完整函数名 = 模块名.API名)而非函数签名
        所有的API函数名必须有效(有效的定义为: JSON数据中的API为函数全名(函数全名 = 模块.函数名)而非函数签名, 该API不是被弃用的, 该API必须是函数而非模块或类, 该API可以被导入)
        """
        try:
            is_valid = True
            json_data = json.loads(response)
            for dl_lib, api_combinations in json_data.items(): # 逐个访问Pytorch, Tensorflow和Jax
                for api_combination_id, api_combination in api_combinations.items(): # 逐个访问Pytorch, Tensorflow和Jax下的各个API组合
                    for full_api_name in api_combination: # 逐个访问API组合下的各个API
                        if not self.validate_api(full_api_name):
                            is_valid = False
            return is_valid
        except JSONDecodeError:
            self.errors.append("The response data has an invalid JSON format.")
            return False
        except Exception as e:
            self.errors.append(str(e))
            return False

    def conduct_cluster(self):
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
                if self.validate_apis(response):
                    self.errors = []  # 清空错误列表
                    return json.loads(response)
                else:
                    attempt_num += 1
                    self.messages.append({"role": "user", "content": f"The JSON data you generated has the following errors: \n{self.errors} \n Please try again."})
                    self.errors = []  # 清空错误列表
                    print(f"Incorrect JSON format or invalid API. Current attempt: {attempt_num + 1}. Retrying...")
                    break
            except Exception as e:
                self.session.rollback()  # 回滚在异常中的任何数据库更改
                print(f"An unexpected error occurred: {e}")
                break
        self.errors = []  # 清空错误列表
        print("Max attempts reached. Unable to get valid JSON data.")
        return None

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
            api_combination_objects[api_id] = []  # api_combination_objects = { "1" : [], "2" : [] }
            for full_api_name in api_combination:  # 获取某个API组合中的每个API
                api = self.session.query(api_class).filter_by(full_name=full_api_name).first()
                if not api:
                    module_name, api_name = full_api_name.rsplit('.', 1)
                    api = api_class(
                        name=api_name,
                        module=module_name,
                        full_name=full_api_name,  # 根据api_class来设置version
                        version=self.torch_ver if api_class == PytorchAPI else self.tf_ver if api_class == TensorflowAPI else self.jax_ver
                    )
                    self.session.add(api)
                    self.session.commit()
                api_combination_objects[api_id].append(
                    api)  # { "1" : [CategoricalCrossentropy], "2" : [constant, softmax_cross_entropy_with_logits] }
        return api_combination_objects

    def associate_api_combinations_to_cluster(self, cluster, api_combination_objects, combination_class):
        for api_id, api_combination in api_combination_objects.items():  # 逐个访问每个API组合
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
            self.associate_api_combinations_to_cluster(new_cluster, torch_apis_combination_objects,
                                                       PytorchAPICombination)
            self.associate_api_combinations_to_cluster(new_cluster, tf_apis_combination_objects,
                                                       TensorflowAPICombination)
            self.associate_api_combinations_to_cluster(new_cluster, jax_apis_combination_objects, JaxAPICombination)
            self.session.commit()
        except Exception as e:
            self.session.rollback()  # 回滚在异常中的任何数据库更改
            print(f"An error occurred: {e}")

    # ----------------------------------------------run()----------------------------------------------
    def cluster_api(self):
        json_data = self.conduct_cluster()
        if json_data:
            self.save_cluster(json_data)


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
