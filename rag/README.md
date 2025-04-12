# RAG Module

RAG 模块目前主要有 5 个主要的文件，分别是：

1. `rag_llm.py`
2. `embeddings.py`
3. `llm.py`
4. `gen_api.py`
5. `rag_client.py`

## 对各个文件的说明

### rag_llm.py

这是 RAG 模块的核心代码，定义了整个 RAG 管线，包括创建向量嵌入、检索和生成等等。

在首次使用时，应该调用 `build_embeddings` 函数来创建向量数据库。该函数接受 1 个参数：`documents_dir`。其中，`documents_dir` 是知识库的文件的路径。不同的深度学习库的文档应该放在以其名字命名的子路径下面。例如 PyTorch 的路径就应该是 `/docs/pytorch/`。

如果想测试 RAG 模块，可以直接在命令行中运行该文件。

```python
python3 rag_llm.py
```

每次问答的记录将会以 *generated_code_<当前时间>.txt* 的格式被自动保存，以便调查。

### embeddings.py and llm.py

这两个文件是对嵌入模型和生成模型的调用的处理。

### gen_api.py

这个文件使用了 `FastAPI` 来把 RAG 模块以 REST API 的形式暴露了出来。目前一共包含三个 API：

- `/generate` 能够直接接受用户的 Query 并且调用 `rag_generate` 来回答问题。
- `/generate_without_rag` 则直接将 Query 发给大语言模型来生成，不使用文档检索和提示词模版。
- `retrieve_documents` 不使用大语言模型来生成，仅使用向量数据库来检索相关文档。

这里的 `qa_chain` 和 `vector_store` 对象则会在一开始被创建。使用者可以通过修改参数来指定大语言模型和 API Key。

可以使用 `uvicorn gen_api:app --reload` 来运行服务器程序。

### rag_client.py

这个文件在 `gen_api.py` 中的 REST API 的基础上再次将其封装成了一个类似于 OpenAIClient 的类，以方便调用。

## 运行流程

在使用 rag_client 来调用 RAG 模块时，需要首先运行 FastAPI 服务器和 Ollama 服务器。在初次运行时，还需要使用 `ollama pull` 来下载 `bge-m3` 模型，以及通过 `pip` 等 Python 包管理器来安装 `faiss_cpu`。

