#!/bin/bash

# 设置环境变量以确保仅使用 CPU 并抑制大多数日志输出
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=3  # 仅输出 ERROR 级别的日志(0: INFO, 1: WARN, 2: ERROR, 3: FATAL)

# 打印执行的命令，便于调试
echo "Running driver.py with arguments: $@"

# 使用 Python 解释器运行 driver.py 脚本
python /mnt/e/SoftwareCourses/DeepLearning/DlibFuzz/oracle/tf_driver.py "$@"

# 如果需要查看执行结果目录中的内容，可以在此列出目录
# ls /mnt/e/SoftwareCourses/DeepLearning/DlibFuzz/fuzzer/seeds/tf.add
