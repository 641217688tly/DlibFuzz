from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from transformers.utils import is_torch_cuda_available, is_torch_mps_available
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import ChatPromptTemplate

def get_model():
    codeqwen_llm = Llama.from_pretrained(
        repo_id="Qwen/CodeQwen1.5-7B-Chat-GGUF",
        filename="codeqwen-1_5-7b-chat-q2_k.gguf",
    )

    return codeqwen_llm


def get_docs(directory):
    loader = BSHTMLLoader(directory)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    docs = text_splitter.split_documents(documents)

    return docs


def preprocess(docs):
    model_name = "moka-ai/m3e-base"
    local_model_dir = "/mnt/workspace/m3e-base"

    # 词嵌入模型
    EMBEDDING_DEVICE = "cuda" if is_torch_cuda_available() else "mps" if is_torch_mps_available() else "cpu"
    embeddings_model = HuggingFaceEmbeddings(
         model_name='model\m3e-base', 
         model_kwargs={'device': EMBEDDING_DEVICE}
    )

    # vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings_model)

    # 指定保存的路径
    save_path = "./faiss_index"

    # 加载向量数据库
    vectorstore = FAISS.load_local(save_path, embeddings_model, allow_dangerous_deserialization=True)
    return vectorstore


def get_metadata(vectorstore):
    return vectorstore.docstore._dict.values()


def retrieve_docs(query, vectorstore):
    #向量检索
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    
    return retriever, docs


def get_llm_response(question):
    unprocessed_docs = get_docs("demo_docs/demo.html") # TODO: Change to a more general form
    vectorstore = preprocess(unprocessed_docs) # TODO: 预加载向量数据库
    llm = get_model()


    retriever, docs = retrieve_docs(question, vectorstore)

    system_template = "You are an experienced programmer with profound knowledge in deep learning."
    human_template = "Please answer my question based on the following documents: \n\n{docs}\n\nQuestion: {question}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human" , human_template)
    ])

    output_parser = StrOutputParser()

    docs = retriever.get_relevant_documents(question)
    formatted_prompt = prompt.format(docs=docs, question=question)

    llm_response = llm(formatted_prompt)
    parsed_response = output_parser.parse(llm_response)

    return parsed_response, docs


if __name__ == "__main__":
    question = ""
    response, docs = get_llm_response(question)
    print(response)
    print(docs)