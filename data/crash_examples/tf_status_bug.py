import tensorflow as tf

data = ["a", "b", "c", "d", "e"]
partitions = [3, -2, 2, -1, 2]
num_partitions = 5
# t1 = tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions)  # Succeed
# print(t1)
"""
2024-08-06 17:19:48.937092: W tensorflow/core/framework/op_kernel.cc:1780] OP_REQUIRES failed at bincount_op.cc:228 : INVALID_ARGUMENT: Input arr must be non-negative!
Traceback (most recent call last):
  File "/mnt/e/1University/Research/DlibFuzz/Code/DlibFuzz/data/crash_examples/tf_status_bug.py", line 6, in <module>
    t1 = tf.ragged.stack_dynamic_partitions(data, partitions, num_partitions)  # Succeed
  File "/home/tly/Applications/Programming/Python/Anaconda3/envs/DlibFuzz/lib/python3.9/site-packages/tensorflow/python/util/traceback_utils.py", line 153, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/home/tly/Applications/Programming/Python/Anaconda3/envs/DlibFuzz/lib/python3.9/site-packages/tensorflow/python/framework/ops.py", line 7209, in raise_from_not_ok_status
    raise core._status_to_exception(e) from None  # pylint: disable=protected-access
tensorflow.python.framework.errors_impl.InvalidArgumentError: {{function_node __wrapped__Bincount_device_/job:localhost/replica:0/task:0/device:CPU:0}} Input arr must be non-negative! [Op:Bincount]
ERROR conda.cli.main_run:execute(124): `conda run python /mnt/e/1University/Research/DlibFuzz/Code/DlibFuzz/data/crash_examples/tf_status_bug.py` failed. (See above for error)
进程已结束，退出代码为 1
"""

t2 = tf.ragged.stack(tf.dynamic_partition(data, partitions, num_partitions)) # Raise InvalidArgumentError
print(t2)
"""
Traceback (most recent call last):
  File "/mnt/e/1University/Research/DlibFuzz/Code/DlibFuzz/data/crash_examples/tf_status_bug.py", line 22, in <module>
    t2 = tf.ragged.stack(tf.dynamic_partition(data, partitions, num_partitions)) # Raise InvalidArgumentError
  File "/home/tly/Applications/Programming/Python/Anaconda3/envs/DlibFuzz/lib/python3.9/site-packages/tensorflow/python/ops/gen_data_flow_ops.py", line 653, in dynamic_partition
    return dynamic_partition_eager_fallback(
  File "/home/tly/Applications/Programming/Python/Anaconda3/envs/DlibFuzz/lib/python3.9/site-packages/tensorflow/python/ops/gen_data_flow_ops.py", line 706, in dynamic_partition_eager_fallback
    _result = _execute.execute(b"DynamicPartition", num_partitions,
  File "/home/tly/Applications/Programming/Python/Anaconda3/envs/DlibFuzz/lib/python3.9/site-packages/tensorflow/python/eager/execute.py", line 54, in quick_execute
    tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
tensorflow.python.framework.errors_impl.InvalidArgumentError: {{function_node __wrapped__DynamicPartition_num_partitions_5_device_/job:localhost/replica:0/task:0/device:CPU:0}} partitions[1] = -2 is not in [0, 5) [Op:DynamicPartition]
ERROR conda.cli.main_run:execute(124): `conda run python /mnt/e/1University/Research/DlibFuzz/Code/DlibFuzz/data/crash_examples/tf_status_bug.py` failed. (See above for error)

进程已结束，退出代码为 1
"""