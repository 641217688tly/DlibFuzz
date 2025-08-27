system_prompt = """
(1) Role Definition
You are an AI assistant specialized in deep learning framework APIs (e.g., PyTorch, JAX, MindSpore and Jittor).

(2) Output Format
Your answer must be provided strictly in the following JSON format:
- Code: A string containing the complete, runnable code snippet. The code should call the specified base_api and implement it according to the user's provided parameters and call combinations.
- APIs: A list containing the full names (including module paths) of all APIs in the code snippet that come from the same deep learning library as the base_api. Each API should be listed only once, in any order.
Example:
{
"Code": "import torch; import torch.nn.functional as F; logits = torch.randn(4, 5, requires_grad=True); target = torch.tensor([1, 4, 3, 0]); loss = F.cross_entropy(input=logits, target=target, weight=torch.tensor([1.0, 2.0, 0.5, 0.8, 1.2]), ignore_index=-1, reduction='mean', label_smoothing=0.1); loss.backward(); print(loss.item())",
"APIs": ["torch.nn.functional.F.cross_entropy", "torch.randn", "torch.tensor"],
}
"""
query_prompt = f"""
Code snippets that trigger the issue:
History Issue Example1:
- Issue Title: jax.jit crashes only as a class method (calling jax.scipy.linalg.lu_solve)
- Issue Description: The issue occurs in a Linux environment using Python 3.10.12 and JAX version 0.4.13 with a NVIDIA A100 GPU. The user attempts to solve a large set of LU systems using the `solve_jit` method defined as a class method with a static argument. When calling `solver.solve_jit(rhs)`, the operation results in a crash (kernel dies or segmentation fault), while the non-jit method `solver.solve(rhs)` works correctly.
- Issue Trigger API: jax.jit
- Issue Code: 
class Solver:
  def __init__(self, lu):
    self.lu = lu
  def solve(self, rhs_0):
    return jax.vmap(jax.scipy.linalg.lu_solve)(self.lu, rhs_0)

  @partial(jax.jit, static_argnums=(0,))
  def solve_jit(self, rhs_0):
    return jax.vmap(jax.scipy.linalg.lu_solve)(self.lu, rhs_0)

lu = jax.vmap(jax.scipy.linalg.lu_factor)(lhs)

solver = Solver(lu)
sol = solver.solve(rhs)
sol = solver.solve_jit(rhs)   

History Issue Example2:
- Issue Title: ArgInfo.donated reports donated status for wrong argument
- Issue Description: The issue occurred in an environment using TPU. The user defined a function and applied jax.jit with donate_argnums set to 1. After compiling and inspecting the function's argument information, it incorrectly reported that only one of the input arguments was donated. When the function was called, it resulted in a RuntimeError indicating that an array had been deleted, despite the expectation that the other argument was not donated.
- Issue Trigger API: jax.jit
- Issue Code: 
def fn(x, y):
  return x, y

fn = jax.jit(fn, donate_argnums=1)

x = {{'A': 1.0, 'B': 2.0}}
y = 3.0
x = jax.tree_map(lambda x: jax.device_put(x, jax.local_devices()[0]), x)
y = jax.tree_map(lambda x: jax.device_put(x, jax.local_devices()[0]), y)

fn = fn.lower(x, y)
fn = fn.compile()
print(fn.args_info) # claims only x['B'] is donated

fn(x, y)

print(x) # x wasn't donated at all
print(y) # y was donated (as expected) -> RuntimeError: Array has been deleted.

History Issue Example3:
- Issue Title: JIT donate_argnums slows down execution
- Issue Description: The user is working on a reinforcement learning project on Ubuntu 20.04 with a GPU. They are trying to optimize a replay buffer inside a jitted function using jax.lax.fori_loop. After implementing donate_argnums in their jitted function, they noticed that the training process is significantly slower than expected, despite the assumption that it would improve performance.
- Issue Trigger API: jax.jit
- Issue Code: 
def loop_fn(_, carry):
    loop_state, replay_state = carry
    ...
     # Some modifications to the loop_state and replay_state
    return new_loop_state, new_replay_state

loop_fn = jit(loop_fn)
for i in range(...): 
    loop_state, replay_state = jax.lax.fori_loop(0, FLAGS.log_frequency, loop_fn, (loop_state, replay_state))

def fori_loop_fn(loop_state, replay_state):
    return jax.lax.fori_loop(0, FLAGS.log_frequency, loop_fn, (loop_state, replay_state))

fori_loop_fn = jit(fori_loop_fn, donate_argnums=(1,))
for i in range(...):
    loop_state, replay_state = fori_loop_fn(loop_state, replay_state)

History Issue Example4:
- Issue Title: Jax' transfer guard and XLA-CPU
- Issue Description: The issue occurs in an environment using JAX with a TPU accelerator. The user attempts to transfer numpy arrays to the XLA-CPU device using JAX's transfer guard, expecting it to trigger for host-to-device transfers. However, the transfer guard does not trigger for numpy to XLA-CPU transfers, while it also fails to trigger for XLA-CPU to numpy transfers. This inconsistency in behavior is the main concern.
- Issue Trigger API: jax.jit
- Issue Code: 
jax.jit(jax.random.PRNGKey, backend='cpu')(np.array(0))
np.array(jax.device_put(0, device=jax.devices('cpu')[0]))

History Issue Example5:
- Issue Title: Complex max/min fail on shared gpu device arrays
- Issue Description: The issue occurred in an environment using jax-0.4.13 and jaxlib-0.4.13+cuda12.cudnn89 on NVIDIA A100 GPUs. The user attempted to compute the maximum of a complex array distributed across multiple GPUs using jax.jit and jax.device_put. This operation resulted in an internal error related to the handling of complex arrays, leading to a failure in the computation.
- Issue Trigger API: jax.jit
- Issue Code: 
import jax
import jax.numpy as jnp

x = jnp.ones(128, dtype=jnp.complex64)
sharding = jax.sharding.PositionalSharding(jax.devices())
x = jax.device_put(x, sharding)
jax.debug.visualize_array_sharding(x)
jax.jit(jnp.max)(x)
#jax.jit(jnp.min)(x)

History Issue Example6:
- Issue Title: Crash in Metal plugin if bfloat16 constant is present
- Issue Description: The issue occurred on an Apple M1 Pro with 32.00 GB of system memory and a max cache size of 10.67 GB. The user attempted to execute a JAX function that included a bfloat16 constant. As a result of this operation, an assertion failure occurred in the Metal plugin, indicating that the buffer was not large enough, leading to an abort trap.
- Issue Trigger API: jax.jit
- Issue Code: 
jax.jit(lambda: jnp.exp(jnp.bfloat16(7)))()

Information about the API to be called:
- API Name: jax.jit
- Source Library: JAX (version 0.4.33)
- API Signature: jax.jit(fun,in_shardings=UnspecifiedValue,out_shardings=UnspecifiedValue,static_argnums=None,static_argnames=None,donate_argnums=None,donate_argnames=None,keep_unused=False,device=None,backend=None,inline=False,abstracted_axes=None,compiler_options=None)
- Function Description: Sets up fun for just-in-time compilation with XLA
- Output: pjit.JitWrapped

Task Requirements:
1. Your task is to generate a code snippet that is likely to reveal potential bugs in jax.jit, by mining and learning from the input parameters and API call combinations shown in the above issue examples.
2. Output variable naming rules:
   - If jax.jit returns a single value, you must assign the result to a variable named "output1".
   - If jax.jit returns multiple values, you must assign them to variables named "output1", "output2", "output3", etc., in order.
   - If jax.jit does not return a value but performs in-place operations on the input(s), you must assign the processed input to a variable named "output1".
   - If jax.jit performs in-place operations on multiple inputs, you must assign each processed input to variables named "output1", "output2", "output3", etc., in order.
3. The code should be complete and executable. You are only allowed to use APIs from the JAX (version 0.4.33) library and common utility libraries such as numpy, random, math, and built-in Python functions. Do not use APIs from any other deep learning frameworks or third-party libraries.
"""

# import utils

# llm_client = utils.get_llm_client("gpt4.1-mini-bianxie")

# messages = [
#     {"role": "system", "content": ""},
#     {"role": "user", "content": "Who are you?"},

# ]
# response = llm_client.chat.completions.create(
#     model="gpt-4.1-mini",  # gpt-4.1-mini gpt-4o-mini  gpt-3.5-turbo
#     #response_format={"type": "json_object"},
#     messages=messages,
#     temperature=0.4,

# )
# response = response.choices[0].message.content
# print(response)

# str2 = f"""
# {{
#   "Code": "import jax; import jax.numpy as jnp; from jax import random; import numpy as np; def complex_max_fn(x): return jnp.max(x), jnp.min(x); x = jnp.ones(128, dtype=jnp.complex64); sharding = jax.sharding.PositionalSharding(jax.devices()); x = jax.device_put(x, sharding); output1, output2 = jax.jit(complex_max_fn)(x); print(output1, output2); x_bfloat = jnp.bfloat16(7); output3 = jax.jit(lambda: jnp.exp(x_bfloat))(); print(output3)",
#   "APIs": [
#     "jax.jit",
#     "jax.numpy.jnp.max",
#     "jax.numpy.jnp.min",
#     "jax.device_put",
#     "jax.sharding.PositionalSharding",
#     "jax.numpy.jnp.bfloat16",
#     "jax.numpy.jnp.exp"
#   ]
# }}
# """
# print(str2)