import tensorflow as tf

input = "text"
input_encoding = "utf-8"
errors = "replace"
replacement_char = 65533
replace_control_characters = False
Tsplits = 3.0
result = tf.raw_ops.UnicodeDecodeWithOffsets(
    input=input,
    input_encoding=input_encoding,
    errors=errors,
    replacement_char=replacement_char,
    replace_control_characters=replace_control_characters,
    Tsplits=Tsplits,
)
"""
CPU Env:
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-08-06 17:21:52.452648: F tensorflow/core/framework/tensor.cc:725] Check failed: dtype() == expected_dtype (9 vs. 3) int32 expected, got int64
/tmp/tmplxnwsmt6: line 3: 12851 Aborted                 python /mnt/e/1University/Research/DlibFuzz/Code/DlibFuzz/data/crash_examples/tf_crash_example.py
ERROR conda.cli.main_run:execute(124): `conda run python /mnt/e/1University/Research/DlibFuzz/Code/DlibFuzz/data/crash_examples/tf_crash_example.py` failed. (See above for error)
进程已结束，退出代码为 134

GPU Env:
2024-08-06 17:29:03.185225: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2024-08-06 17:29:03.192289: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
2024-08-06 17:29:03.252938: F tensorflow/core/framework/tensor.cc:725] Check failed: dtype() == expected_dtype (9 vs. 3) int32 expected, got int64
进程已结束，退出代码为 -1073740791 (0xC0000409)
"""