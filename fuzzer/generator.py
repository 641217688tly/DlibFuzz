from utils import *
from tqdm.contrib import itertools
import os
from orm import *


def generate_seeds(session, openai_client, cluster, seeds_num=3):
    folder_path = f'seeds/unverified_seeds/{cluster.__class__.__name__}/{cluster.id}'
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
        seed_folder_name = "_".join([str(single_lib_combination.id) for single_lib_combination in multi_lib_combinations if single_lib_combination])
        if not os.path.exists(f'{folder_path}/{seed_folder_name}'):
            os.makedirs(f'{folder_path}/{seed_folder_name}')

        # 分别获取PytorchAPICombination, TensorFlowAPICombination和JaxAPICombination内所有的API
        torch_apis = multi_lib_combinations[0].apis if multi_lib_combinations[0] else []
        tf_apis = multi_lib_combinations[1].apis if multi_lib_combinations[1] else []
        jax_apis = multi_lib_combinations[2].apis if multi_lib_combinations[2] else []

        # 构建prompt
        example1 = """json
        {
            "code": "\n# Logits\nlogits = [[4.0, 1.0, 0.2]]\n# Labels (one-hot encoded)\nlabels = [[1.0, 0.0, 0.0]]\n\n# PyTorch\nlogits_pt = torch.tensor(logits, requires_grad=True)\nlabels_pt = torch.tensor(labels)\nloss_fn_pt = torch.nn.CrossEntropyLoss()\noutput_pt = loss_fn_pt(logits_pt, torch.argmax(labels_pt, dim=1))\nprint(\"PyTorch Loss:\", output_pt.item())\n\n# TensorFlow: tf.nn.softmax_cross_entropy_with_logits\nlogits_tf = tf.constant(logits)\nlabels_tf = tf.constant(labels)\noutput_tf = tf.nn.softmax_cross_entropy_with_logits(labels=labels_tf, logits=logits_tf)\nprint(\"TensorFlow NN Loss:\", output_tf.numpy()[0])\n\n# Jax\nlogits_jax = jnp.array(logits)\nlabels_jax = jnp.array(labels)\nlog_softmax = jax.nn.log_softmax(logits_jax)\noutput_jax = -jnp.sum(labels_jax * log_softmax)\nprint(\"JAX Loss:\", output_jax)\n"
        }
        """
        prompt = f"""
        Given the API combinations{torch_apis} of the Pytorch library, the {tf_apis} of the TensorFlow library and the {jax_apis} of the Jax library all have the same functionality.
        Please Generate test code snippets for the apis of the different libraries mentioned above.
        Requirements:
        1. The output values of test code snippets generated for different apis need to remain the same.
        2. Your answers should be in JSON format.
        
        Following is a example:
        API combinations ["torch.tensor", "torch.nn.CrossEntropyLoss"] of the Pytorch library, the ["tensorflow.constant", "tensorflow.nn.softmax_cross_entropy_with_logits"] of the TensorFlow library and the ["jax.numpy.array", "jax.nn.log_softmax", "jax.numpy.sum"] of the Jax library all have the same functionality.
        The code snippet to test these API combinations is as follows:
        {example1}
        """
        for i in range(seeds_num):  # 生成n个seed
            json_data = None
            attempt_num = 0
            while attempt_num < 5: # 设置最大尝试次数以避免无限循环
                try:  # 假如返回的数据不符合JSON格式, 则重新调用OpenAI API, 直到返回的数据符合JSON格式为止
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",  # gpt-4o-mini  gpt-3.5-turbo
                        response_format={"type": "json_object"},
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=1,
                    )
                    response_data = response.choices[0].message.content
                    print(response_data)
                    json_data = json.loads(response_data)
                    break  # 成功解析 JSON,跳出循环
                except json.JSONDecodeError as e:
                    print(
                        f"Failed to decode JSON due to.\nError Details: \n{e} \nRetrying(Current attempt: {attempt_num + 1})...")
                    attempt_num += 1
                except Exception as e:
                    session.rollback()  # 回滚在异常中的任何数据库更改
                    print(f"An unexpected error occurred: {e}")
                    break
            if attempt_num == 5: # 设置最大尝试次数以避免无限循环
                print("Max attempts reached. Unable to get valid JSON data.")
                return

            # 创建一个新的ClusterTestSeed对象
            new_seed = ClusterTestSeed(
                cluster_id=cluster.id,
                pytorch_combination_id= multi_lib_combinations[0].id if multi_lib_combinations[0] else None,
                tensorflow_combination_id=multi_lib_combinations[1].id if multi_lib_combinations[1] else None,
                jax_combination_id=multi_lib_combinations[2].id if multi_lib_combinations[2] else None,
                code=json_data['code'])
            session.add(new_seed)
            session.commit()

            # 在seed_folder_name文件夹下创建一个新的json文件, 文件名为seed_{i}.py
            with open(f'{folder_path}/{seed_folder_name}/seed_{i}.py', 'w') as file:
                # 读取json_data中的code字段, 并将其写入到文件中
                file.write(json_data['code'])
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
