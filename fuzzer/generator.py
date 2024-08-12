from utils import *
from tqdm.contrib import itertools
import os
from orm import *
import random

ERROR_TRIGGER_TARGET = 8

OUTPUT_EXAMPLE = """
logits = [[4.0, 1.0, 0.2]]
# Labels (one-hot encoded)
labels = [[1.0, 0.0, 0.0]]

# PyTorch
logits_pt = torch.tensor(logits, requires_grad=True)
labels_pt = torch.tensor(labels)
loss_fn_pt = torch.nn.CrossEntropyLoss()
output_pt = loss_fn_pt(logits_pt, torch.argmax(labels_pt, dim=1))
print("PyTorch Loss:", output_pt.item())

# TensorFlow: tf.nn.softmax_cross_entropy_with_logits
logits_tf = tf.constant(logits)
labels_tf = tf.constant(labels)
output_tf = tf.nn.softmax_cross_entropy_with_logits(labels=labels_tf, logits=logits_tf)
print("TensorFlow NN Loss:", output_tf.numpy()[0])

# JAX
logits_jax = jnp.array(logits)
labels_jax = jnp.array(labels)
log_softmax = jax.nn.log_softmax(logits_jax)
output_jax = -jnp.sum(labels_jax * log_softmax)
print("JAX Loss:", output_jax)
"""


def get_apis_error_triggers(apis, error_trigger_class, session):
    error_triggers = []
    for api in apis:
        if api.error_triggers:
            error_triggers.extend(api.error_triggers)
    if not error_triggers:  # 如果error_triggers为空, 则从数据库中随机选择6个error_trigger
        error_triggers = session.query(error_trigger_class).all()
        error_triggers = random.sample(error_triggers, 6)
    # error_triggers_dict将记录每个error_trigger被使用的次数, 其key为int, 代表error_trigger被使用的次数, 其value为list, 代表被使用次数为key的error_trigger的列表
    error_triggers_dict = {0: error_triggers}
    return error_triggers_dict


def weighted_sampling(error_triggers_dict, target):
    # TODO 函数功能有待检查
    # 创建一个列表来存储所有API及其权重
    weighted_apis = []
    for count, apis in error_triggers_dict.items():
        # 权重是与使用次数的倒数成正比
        weight = 1 / (count + 1)  # 避免除以零错误
        for api in apis:
            weighted_apis.append((api, weight))

    # 从weighted_apis中根据权重抽样
    selected_apis = random.choices(
        [api for api, weight in weighted_apis],  # 提取API列表
        weights=[weight for api, weight in weighted_apis],  # 提取权重列表
        k=target
    )

    # 更新字典以反映API的新使用次数
    for api in selected_apis:
        # 找出API当前的使用次数
        for count in list(error_triggers_dict):
            if api in error_triggers_dict[count]:
                # 从当前计数列表中移除API
                error_triggers_dict[count].remove(api)
                # 如果当前计数列表为空，删除该键
                if not error_triggers_dict[count]:
                    del error_triggers_dict[count]

                # 将API添加到下一个使用次数的列表中
                next_count = count + 1
                if next_count in error_triggers_dict:
                    error_triggers_dict[next_count].append(api)
                else:
                    error_triggers_dict[next_count] = [api]
                break

    return selected_apis


def construct_error_trigger_examples_prompt(lib_name: str, error_trigger_examples: list):
    prompt = ""
    for error_trigger in error_trigger_examples:
        prompt += f"""
        {lib_name}'s Error Trigger Example:
        Title: {error_trigger.title}
        Description: {error_trigger.description}
        Code:{error_trigger.code}\n
        """
    return prompt


def generate_seeds(session, openai_client, cluster, seeds_num=5):
    folder_path = f'seeds/unverified_seeds/{cluster.id}'
    # 在folder_path下创建一个新的文件夹, 文件夹名为cluster.id
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 先从cluster中获取所有的Pytorch对象, Tensorflow对象和JAX对象的组合. 比如当前有一个cluster, 其中包含了Pytorch对象A1和A2, Tensorflow对象B1和JAX对象C1, 则有以下组合: (A1,B1,C1), (A2,B1,C1); 再比如当前有一个cluster, 其中包含了Pytorch对象A1, Tensorflow对象为null, JAX对象C1和C2, 则有以下组合: (A1,C1), (A1,C2)
    all_combinations = list(itertools.product(
        cluster.pytorch_combinations if cluster.pytorch_combinations else [None],
        cluster.tensorflow_combinations if cluster.tensorflow_combinations else [None],
        cluster.jax_combinations if cluster.jax_combinations else [None]
    ))

    for multi_lib_combinations in all_combinations:  # 对于每种组合都生成5个seed
        # 在folder_path下创建一个新的文件夹, 文件夹名为combination中Pytorch, Tensorflow和JAX对象的id组合. 比如当前的combination为(A1,B1,C1), 则创建的文件夹名为(A1.id)_(B1.id)_(C1.id)
        seed_folder_name = "_".join(
            [str(single_lib_combination.id) for single_lib_combination in multi_lib_combinations if
             single_lib_combination])
        if not os.path.exists(f'{folder_path}/{seed_folder_name}'):
            os.makedirs(f'{folder_path}/{seed_folder_name}')

        # 分别获取PytorchAPICombination, TensorFlowAPICombination和JaxAPICombination内所有的API
        torch_apis = multi_lib_combinations[0].apis if multi_lib_combinations[0] else []
        torch_error_triggers_dict = get_apis_error_triggers(torch_apis,
                                                            PytorchErrorTriggerCode,
                                                            session)
        tf_apis = multi_lib_combinations[1].apis if multi_lib_combinations[1] else []
        tf_error_triggers_dict = get_apis_error_triggers(tf_apis,
                                                         TensorflowErrorTriggerCode,
                                                         session)
        jax_apis = multi_lib_combinations[2].apis if multi_lib_combinations[2] else []
        jax_error_triggers_dict = get_apis_error_triggers(jax_apis, JaxErrorTriggerCode,
                                                          session)
        target = [round(ERROR_TRIGGER_TARGET * len(torch_apis) / (len(torch_apis) + len(tf_apis) + len(jax_apis))),
                  round(ERROR_TRIGGER_TARGET * len(tf_apis) / (len(torch_apis) + len(tf_apis) + len(jax_apis))),
                  round(ERROR_TRIGGER_TARGET * len(jax_apis) / (len(torch_apis) + len(tf_apis) + len(jax_apis)))]

        for i in range(seeds_num):  # 生成n个seed
            # 构建prompt
            torch_error_trigger_examples = weighted_sampling(torch_error_triggers_dict, target[0])
            tf_error_trigger_examples = weighted_sampling(tf_error_triggers_dict, target[1])
            jax_error_trigger_examples = weighted_sampling(jax_error_triggers_dict, target[2])
            error_trigger_examples_prompt = construct_error_trigger_examples_prompt("Pytorch", torch_error_trigger_examples) + \
                                                construct_error_trigger_examples_prompt("Tensorflow", tf_error_trigger_examples) + \
                                                construct_error_trigger_examples_prompt("JAX", jax_error_trigger_examples)
            # TODO Prompt有待优化
            prompt = f"""
            {error_trigger_examples_prompt}
            
            Given the API combinations{torch_apis} of the Pytorch library, the {tf_apis} of the TensorFlow library and the {jax_apis} of the Jax library all have the same functionality.
            Please Generate test code snippets for the apis of the different libraries mentioned above.
            Requirements:
            1. The output values of test code snippets generated for different apis need to remain the same.
            2. Import the required modules or apis in your code.
            3. Output only code, nothing else.

            Following is a example:
            API combinations ["torch.tensor", "torch.nn.CrossEntropyLoss"] of the Pytorch library, the ["tensorflow.constant", "tensorflow.nn.softmax_cross_entropy_with_logits"] of the TensorFlow library and the ["jax.numpy.array", "jax.nn.log_softmax", "jax.numpy.sum"] of the Jax library all have the same functionality.
            The code snippet to test these API combinations is as follows:
            {OUTPUT_EXAMPLE}
            """

            response_data = None
            attempt_num = 0
            while attempt_num < 5:  # 设置最大尝试次数以避免无限循环
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",  # gpt-4o-mini  gpt-3.5-turbo
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=1,
                    )
                    response_data = response.choices[0].message.content
                    print(response_data)
                    break
                except Exception as e:
                    print(f"Failed to get response due to: \n{e} \nRetrying(Current attempt: {attempt_num + 1})...")
                    attempt_num += 1
                    session.rollback()  # 回滚在异常中的任何数据库更改
                    break
            if attempt_num == 5:  # 设置最大尝试次数以避免无限循环
                print("Max attempts reached. Unable to get valid JSON data.")
                return
            # 创建一个新的ClusterTestSeed对象
            new_seed = ClusterTestSeed(
                cluster_id=cluster.id,
                pytorch_combination_id=multi_lib_combinations[0].id if multi_lib_combinations[0] else None,
                tensorflow_combination_id=multi_lib_combinations[1].id if multi_lib_combinations[1] else None,
                jax_combination_id=multi_lib_combinations[2].id if multi_lib_combinations[2] else None,
                code=response_data)
            session.add(new_seed)
            session.commit()

            # 在seed_folder_name文件夹下创建一个新的json文件, 文件名为seed_{i}.py
            with open(f'{folder_path}/{seed_folder_name}/seed_{i}.py', 'w') as file:
                # 读取json_data中的code字段, 并将其写入到文件中
                file.write(response_data)
    # 在所有的seed生成完毕后, 将cluster的is_tested字段设置为True
    cluster.is_tested = True
    session.commit()


def run():
    session = get_session()
    openai_client = get_openai_client()

    # 获得所有的cluster未测试的cluster
    untested_clusters = session.query(Cluster).filter(Cluster.is_tested == False).all()
    while untested_clusters:
        print("----------------------------------------------------------------------------------")
        generate_seeds(session, openai_client, untested_clusters[0])
        untested_clusters = session.query(Cluster).filter(Cluster.is_tested == False).all()
        total_clusters_num = session.query(Cluster).count()
        untested_clusters_num = len(untested_clusters)
        print(f"Untested / Total: {untested_clusters_num} / {total_clusters_num}")


if __name__ == '__main__':
    run()
