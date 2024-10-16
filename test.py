# # 设置代理
# import httpx
# from openai import OpenAI
#
# proxy = httpx.Client(proxies={
#     "http://": "http://127.0.0.1:7890",
#     "https://": "http://127.0.0.1:7890"
# })
# openai_client = OpenAI(api_key="", http_client=proxy)
#
# response = openai_client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     response_format={"type": "json_object"},
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
#         {"role": "user", "content": "Hello!"}
#     ],
#     temperature=0.0,
# )
# data = response.choices[0].message.content
# print(data)
from orm import ClusterTestSeed
from utils import get_session


# import torch
# import tensorflow as tf
# import jax
# import jax.numpy as jnp
#
# # Logits
# logits = [[4.0, 1.0, 0.2]]
# # Labels (one-hot encoded)
# labels = [[1.0, 0.0, 0.0]]
#
# # PyTorch
# logits_pt = torch.tensor(logits, requires_grad=True)
# labels_pt = torch.tensor(labels)
# loss_fn_pt = torch.nn.CrossEntropyLoss()
# output_pt = loss_fn_pt(logits_pt, torch.argmax(labels_pt, dim=1))
# print("PyTorch Loss:", output_pt.item())
#
# # TensorFlow: tf.keras.losses.CategoricalCrossentropy
# loss_fn_tf_keras = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# output_tf_keras = loss_fn_tf_keras(labels, logits).numpy()
# print("TensorFlow Keras Loss:", output_tf_keras)
#
# # TensorFlow: tf.nn.softmax_cross_entropy_with_logits
# logits_tf = tf.constant(logits)
# labels_tf = tf.constant(labels)
# output_tf = tf.nn.softmax_cross_entropy_with_logits(labels=labels_tf, logits=logits_tf)
# print("TensorFlow NN Loss:", output_tf.numpy()[0])
#
# # JAX
# logits_jax = jnp.array(logits)
# labels_jax = jnp.array(labels)
# log_softmax = jax.nn.log_softmax(logits_jax)
# output_jax = -jnp.sum(labels_jax * log_softmax)
# print("JAX Loss:", output_jax)

# import subprocess
#
# file_path = 'fuzzer/seeds/unverified_seeds/1/1_1_1/seed_0.py'
#
#
# def static_analysis(file_path):  # 静态分析Python代码, 如果发现错误, 则返回False和错误信息
#    errors2check = [
#        'syntax-error',  # 语法错误
#        'import-error',  # 导入错误
#        'undefined-variable'  # 未定义变量
#    ]
#    enable_param = ','.join(errors2check)
#    result = subprocess.run(
#        ['pylint', file_path, '--disable=all', f'--enable={enable_param}', '--score=no'],
#        capture_output=True, text=True
#    )
#    errors = result.stdout
#    if errors == "":
#        return True, errors
#    else:
#        errors_lines = errors.split('\n')
#        if errors_lines[0].startswith("*************"):
#            errors_cleaned = "\n".join(errors_lines[1:]).strip()
#        else:
#            errors_cleaned = errors.strip()
#        return False, errors_cleaned
# _, errors = static_analysis(file_path)
# print(errors)
#
# import subprocess
#
# def run_flake8(file_path):
#     result = subprocess.run(
#         ['flake8', file_path, '--select=F'],
#         capture_output=True, text=True
#     )
#     errors = result.stdout
#     if errors == "":
#         return True, errors
#     else:
#         return False, errors
#
# file_path = 'fuzzer/seeds/unverified_seeds/1/1_1_1/seed_0.py'
# _, errors = run_flake8(file_path)
# print(errors)
#
# import tensorflow as tf
#
# # 使用dtype为float32的张量与int32的张量相乘
# a = tf.constant([1, 2, 3], dtype=tf.float32)
# b = tf.constant([4, 5, 6], dtype=tf.int32)
# tf.add(a, b)


#def insert_possible_imports(code):  # 向seed.code中插入可能的导入语句
#    print(code)
#    possible_imports = [
#        "import torch",
#        "import tensorflow",
#        "import jax"
#    ]
#    code_lines = code.split('\n')
#
#    # 检查代码中是否已包含了可能的导入语句
#    imports_to_add = []
#    for import_statement in possible_imports:
#        if not import_statement in code_lines:  # 如果代码中没有包含该导入语句, 则将其添加到imports_to_add列表中
#            imports_to_add.append(import_statement)
#
#    # 如果有需要添加的导入语句，将它们插入到代码的开头
#    if imports_to_add:
#        updated_code = '\n'.join(imports_to_add) + '\n' + code
#        print("=" * 10)
#        print(updated_code)
#
## 读取fuzzer/seeds/unverified_seeds/zero-shot/1/1_1_1/seed_0.py中的代码
#file_path = 'fuzzer/seeds/unverified_seeds/zero-shot/1/1_1_1/seed_0.py'
#with open(file_path, 'r') as file:
#    code = file.read()
#    insert_possible_imports(code)


import tensorflow
import jax
import torch
import torch.nn.functional as F
import tensorflow as tf
import jax.numpy as jnp
from jax import random
# Common variable values
input_tensor = [[1.0, 2.0, 3.0]]
kernel = [[0.5, 0.5], [0.5, 0.5]]

# PyTorch
# input_pt = torch.tensor(input_tensor)
# kernel_pt = torch.tensor(kernel)
# output_pt = F.conv2d(input_pt.unsqueeze(0).unsqueeze(0), kernel_pt.unsqueeze(0).unsqueeze(0), stride=1)
# print("PyTorch Output:", output_pt)

# TensorFlow
input_tf = tf.constant(input_tensor, dtype=tf.float32)
kernel_tf = tf.constant(kernel, dtype=tf.float32)
output_tf = tf.nn.conv2d(tf.expand_dims(input_tf, axis=0), tf.expand_dims(kernel_tf, axis=0), strides=[1, 1, 1, 1], padding='VALID')
print("TensorFlow Output:", output_tf.numpy())

input_tf = tf.constant(input_tensor, dtype=tf.float32)
kernel_tf = tf.constant(kernel, dtype=tf.float32)
output_tf = tf.nn.conv2d(input_tf, kernel_tf, strides=[1, 1, 1, 1], padding='VALID')
print("TensorFlow Output:", output_tf.numpy())


# JAX
input_jax = jnp.array(input_tensor)
kernel_jax = jnp.array(kernel)
output_jax = jax.lax.conv_general_dilated(jnp.expand_dims(input_jax, axis=0).reshape(1, 1, 1, 3),
                                          jnp.expand_dims(kernel_jax, axis=0).reshape(1, 1, 2, 2),
                                          (1, 1),
                                          'VALID')
print("JAX Output:", output_jax)