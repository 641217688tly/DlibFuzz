# DlibFuzz

## Introduction

该仓库存储了北京工业大学的“星火基金”重点项目(编号**XH-2024-13-07**)的实验代码. 该代码的最终版权属于北京工业大学.

该项目致力于使用基于大语言模型驱动的模糊测试策略来检测目前主流深度学习库的潜在漏洞和错误.

我们的学术论文和实验代码预计将在2024年完成.

## Setup

### Environment

|     **依赖项**     |                         **具体细节**                         |
| :----------------: | :----------------------------------------------------------: |
|    **操作系统**    | 需要Linux系统, 因为jaxlib没有提供Windows版本, 推荐在本地的wsl2上运行该项目 |
|  **Python解释器**  |                         版本v3.9.19                          |
|     **依赖库**     |               见项目根目录下的requirements.txt               |
|      **GPU**       |                  实验的当前阶段尚不需要GPU                   |
|     **数据库**     |      需要本地或远程mysql中有一个名为“dlibfuzz”的数据库       |
| **OpenAI API Key** |     你需要一个Openai的API Key以驱动我们的聚类器和模糊器      |
|    **网络代理**    | 如果你所在的网络环境无法直接与Openai的服务器建立连接, 你需要在运行该项目时始终保持网络代理为开启状态; 该项目默认用户使用的网络代理为Clash(Port=7890), 如果你希望自定义网络代理的信息, 请前往***util.py***对***get_openai_client()***进行修改 |



### **Test Target**

我们将TitanFuzz和FuzzGPT作为我们实验的基线以验证我们测试策略的先进性, 因此在实验初期我们将选用以下版本的深度学习库作为测试对象:

| **深度学习库** |              **版本**              |
| :------------: | :--------------------------------: |
|  **Pytorch**   | v1.12(与TitanFuzz&FuzzGPT保持一致) |
| **TensorFlow** | v2.10(与TitanFuzz&FuzzGPT保持一致) |
|    **JAX**     |              v0.4.13               |

## Running

**可以通过以下步骤使用我们的模糊器:**

1. 在Linux系统中搭建运行所需的各种环境
2. 分别为项目根目录, cluster模块和fuzzer模块下的config.yml配置文件填充必要的信息(包括mysql数据库的用户名和密码, Openai的秘钥)
3. 运行***orm.py***以在mysql数据库中初始化表
4. 运行***cluster/dataloader***以添加Pytorch, Tensorflow和Jax的API信息以及触发崩溃的历史代码调用片段到数据库
5. 分别运行***cluster/torch_api_cluster.py***, ***cluster/tf_api_cluster.py***和***cluster/jax_api_cluster.py***来分别为先前添加进数据库中的添加Pytorch, Tensorflow和Jax的API进行聚类**(这一步需要开启网络代理)**
6. 在聚类完成后, 运行***fuzzer/generator.py***来对所有聚类逐个生成测试种子**(这一步需要开启网络代理)**
7. 当模糊器完成初始种子生成后, 运行***fuzzer/validator.py***来修复存在错误的种子



## Clarification

当前代码不是我们研究的最终版本, 当前的实验效果也不代表我们最终的实验结果.

在我们的论文发表前, 请您不要传播我们的实验代码.