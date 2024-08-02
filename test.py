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


import torch
import tensorflow as tf
import jax
import jax.numpy as jnp

# Logits
logits = [[4.0, 1.0, 0.2]]
# Labels (one-hot encoded)
labels = [[1.0, 0.0, 0.0]]

# PyTorch
logits_pt = torch.tensor(logits, requires_grad=True)
labels_pt = torch.tensor(labels)
loss_fn_pt = torch.nn.CrossEntropyLoss()
output_pt = loss_fn_pt(logits_pt, torch.argmax(labels_pt, dim=1))
print("PyTorch Loss:", output_pt.item())

# TensorFlow: tf.keras.losses.CategoricalCrossentropy
loss_fn_tf_keras = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
output_tf_keras = loss_fn_tf_keras(labels, logits).numpy()
print("TensorFlow Keras Loss:", output_tf_keras)

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


