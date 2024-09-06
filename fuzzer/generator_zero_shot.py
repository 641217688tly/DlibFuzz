from utils import *
from tqdm.contrib import itertools
import os
from orm import *
import threading

ERROR_TRIGGER_TARGET = 6

TORCH_VERSION = "1.12"
TF_VERSION = "2.10"
JAX_VERSION = "0.4.13"

STEPS_PROMPT = """
1.Consider: For given APIs, think about which edge cases and combinations of API calls may expose potential vulnerabilities in the API.
2.Generate Differential Testing Code: Create code snippets for differential testing using the identified API combinations from PyTorch, TensorFlow, and JAX.
Step 1: Define common variable values that will be used across all three libraries.
Step 2: Write code for PyTorch using the provided API combination ({torch_apis}).
Step 3: Write code for TensorFlow using the provided API combination ({tf_apis}).
Step 4: Write code for JAX using the provided API combination ({jax_apis}).
"""

REQUIREMENTS_PROMPT = """
1.Imports: Ensure that all necessary modules or APIs are imported.
2.Consistency in Input: Use the same input values for API combinations across different libraries.
3.Consistency in Output: The output values from the code snippets should be identical when using the same inputs.
4.Clear Separation: Use comments # PyTorch, # TensorFlow, and # JAX to clearly separate the code snippets for each library.
5.Code-Only Format: Only output code and comments in the required format, avoiding any additional text or Markdown syntax.
6.Simplicity: Avoid creating custom functions or classes for code that is not reused multiple times.
7.Correctness: Ensure the generated code does not contain syntax errors (e.g., SyntaxError, NameError) or invalid input errors (e.g., ValueError, InvalidArgumentError).
"""

OUTPUT_EXAMPLE_PROMPT = """
Known API combinations ["torch.tensor", "torch.nn.CrossEntropyLoss"] from the PyTorch library, ["tensorflow.constant", "tensorflow.nn.softmax_cross_entropy_with_logits"] from the TensorFlow library, and ["jax.numpy.array", "jax.nn.log_softmax", "jax.numpy.sum"] from the JAX library all have the same functionality. The code snippet for differential testing of these API combinations is as follows:
```python
logits = [[4.0, 1.0, 0.2]]
# Labels (one-hot encoded)
labels = [[1.0, 0.0, 0.0]]

# PyTorch
logits_pt = torch.tensor(logits, requires_grad=True)
labels_pt = torch.tensor(labels)
loss_fn_pt = torch.nn.CrossEntropyLoss()
output_pt = loss_fn_pt(logits_pt, torch.argmax(labels_pt, dim=1))
print("PyTorch Loss:", output_pt.item())

# TensorFlow
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
```
"""


def generate_seeds4cluster(session, openai_client, cluster, seeds_num=5):
    folder_path = f'seeds/unverified_seeds/zero-shot/{cluster.id}'
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
        tf_apis = multi_lib_combinations[1].apis if multi_lib_combinations[1] else []
        jax_apis = multi_lib_combinations[2].apis if multi_lib_combinations[2] else []
        for i in range(seeds_num):  # 生成n个seed
            # 构建prompt
            prompt = f"""
Objective:
Generate code snippets that can be used to differentially test API combinations from PyTorch (v{TORCH_VERSION}), TensorFlow (v{TF_VERSION}), and JAX (v{JAX_VERSION}), which have identical functionalities. The goal is to identify potential crashes or inconsistencies across these libraries.

Background:
API combinations from the PyTorch {torch_apis}, TensorFlow {tf_apis}, and JAX {jax_apis} all have identical functionalities. The definition of "Identical Functionality" is as follows:
1.Consistency in Input Transformation: When these APIs have no return value, applying them to inputs with the same structure or element values (such as tensors) should result in consistent transformations or changes to the original input.
2.Consistency in Output: When these APIs have return values, they should produce the same output values when given the same input values.

Steps:
{STEPS_PROMPT}

Requirements:
{REQUIREMENTS_PROMPT}

Output Format Example:
{OUTPUT_EXAMPLE_PROMPT}
"""
            response_data = None
            attempt_num = 0
            while attempt_num < 5:  # 设置最大尝试次数以避免无限循环
                try:
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",  # gpt-4o-mini  gpt-3.5-turbo
                        messages=[
                            {"role": "system",
                             "content": "You're an AI assistant adept at using multiple deep learning libraries"},
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
                code=response_data,
                unverified_file_path=f'{folder_path}/{seed_folder_name}/seed_{i}.py',
                verified_file_path=f'{folder_path}/{seed_folder_name}/seed_{i}.py'.replace("unverified_seeds",
                                                                                           "verified_seeds")
            )
            session.add(new_seed)
            session.commit()

            # 在seed_folder_name文件夹下创建一个新的json文件, 文件名为seed_{i}.py
            with open(f'{folder_path}/{seed_folder_name}/seed_{i}.py', 'w') as file:
                # 读取json_data中的code字段, 并将其写入到文件中
                file.write(response_data)
    # 在所有的seed生成完毕后, 将cluster的is_tested字段设置为True
    cluster.is_tested = True
    session.commit()


def generate_seeds4clusters(session, openai_client, clusters, seeds_num=5):
    for cluster in clusters:
        print(f"Processing cluster ID: {cluster.id}")
        generate_seeds4cluster(session, openai_client, cluster, seeds_num)


def multithreaded_run(thread_num=3):
    sessions = []
    openai_clients = []
    for i in range(thread_num):
        sessions.append(get_session())
        openai_clients.append(get_openai_client())

    untested_clusters = get_session().query(Cluster).filter(Cluster.is_tested == False).all()
    print(f"Total clusters to process: {len(untested_clusters)}")

    # 划分任务
    split_clusters = [untested_clusters[i::thread_num] for i in range(
        thread_num)]  # split_clusters = [[cluster1,cluster2,..],[cluster2000,cluster2001,...],[cluster4000,cluster40001,...]]

    threads = []
    for i, clusters in enumerate(split_clusters):
        thread = threading.Thread(target=generate_seeds4clusters, args=(sessions[i], openai_clients[i], clusters))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    for session in sessions:
        session.close()


def run():
    session = get_session()
    openai_client = get_openai_client()

    # 获得所有的cluster未测试的cluster
    untested_clusters = session.query(Cluster).filter(Cluster.is_tested == False).all()
    while untested_clusters:
        print("----------------------------------------------------------------------------------")
        generate_seeds4cluster(session, openai_client, untested_clusters[0])
        untested_clusters = session.query(Cluster).filter(Cluster.is_tested == False).all()
        total_clusters_num = session.query(Cluster).count()
        untested_clusters_num = len(untested_clusters)
        print(f"Untested / Total: {untested_clusters_num} / {total_clusters_num}")


if __name__ == '__main__':
    run()
    #multithreaded_run()
