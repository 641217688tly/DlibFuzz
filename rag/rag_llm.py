import datetime
import os
import time
import pickle
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from llm import CodeQwenLLM, OpenAILLM
from transformers_llm import TransformersLLM
from embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA


def load_files(directory: str, kind: str):
    documents = []
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(('.html', '.htm', '.md')):
                filepath = os.path.join(root, filename)
                try:
                    if filename.endswith(('.html', '.htm')):
                        with open(filepath, 'r', encoding='utf-8') as file:
                            soup = BeautifulSoup(file, 'html.parser')
                            if kind == 'pytorch' or 'jax':
                                sections = soup.find_all('div', class_='section')
                                if sections:
                                    text = "\n".join(section.get_text(separator='') for section in sections)
                                else:
                                    # If no sections found, get all text
                                    text = soup.get_text(separator='\n')
                            elif kind == 'mindspore' or kind == 'jittor':
                                sections = soup.find_all('div', class_='section')
                                if sections:
                                    text = "\n".join(section.get_text(separator=' ') for section in sections)
                                else:
                                    # If no sections found, get all text
                                    text = soup.get_text(separator='')
                            else:
                                text = soup.get_text(separator='\n')
                            documents.append(text)
                    else:  # .md files
                        with open(filepath, 'r', encoding='utf-8') as file:
                            text = file.read()
                            documents.append(text)
                except Exception as e:
                    print(f"Error processing file {filepath}: {str(e)}")
                    continue
        
    return documents

def create_vector_store_batched(documents, embeddings, batch_size=100):
    """Batch process documents to create vector store"""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    vector_store = None
    
    try:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            split_docs = text_splitter.split_documents(
                [Document(page_content=doc) for doc in batch]
            )
            
            if vector_store is None:
                vector_store = FAISS.from_documents(split_docs, embeddings)
            else:
                vector_store.add_documents(split_docs)
                
            print(f'Processed batch {i//batch_size + 1}/{len(documents)//batch_size + 1}')
        
        return vector_store
    except Exception as e:
        print(f"Error in create_vector_store_batched: {str(e)}")
        return None




def initialize_rag_system(documents_dir: list, 
                          is_local: bool,
                          openai_model: str = "gpt-4o-mini", 
                          openai_api_key: str = None
                          ):
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="llama3.1")
    print('Embeddings initialized.')

    if os.path.exists('vector_store.faiss'):
        vector_store = FAISS.load_local('vector_store.faiss', embeddings=embeddings)
        print('Vector store loaded.')
    else:
        print('Vector store not found. Creating new vector store...')
        # Load documents
        print('Loading documents...')
        docs = []
        for directory in documents_dir:
            # docs += load_files(directory, kind=directory.strip('docs/'))
            docs += load_files(directory, kind=directory.strip('docs/'))
        print('Documents loaded.')

        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        # split_docs = text_splitter.split_documents([Document(page_content=doc) for doc in docs])
        # print('Documents split.')
    
    
        # Create a FAISS vector store from the documents and their embeddings
        # vector_store = FAISS.from_documents(split_docs, embeddings)
        vector_store = create_vector_store_batched(docs, embeddings)
        print('Vector store created.')

        vector_store.save_local('vector_store.faiss')
        print('Vector store saved.')

    
    
    # Step 4: Initialize LLM
    # llm = CodeQwenLLM()
    if is_local:
        llm = TransformersLLM(
            model_id="Qwen/Qwen2.5-Coder-14B-Instruct",
            device="auto",          # 自动选择设备
            load_in_4bit=False,      # 4-bit量化
            # load_in_8bit=True,      # 8-bit量化
            torch_dtype="bfloat16"  # 使用 bfloat16 精度
        )
    else:
        llm = OpenAILLM(openai_model, openai_api_key)
    print('LLM initialized.')
    
    # Step 5: Establish RAG pipeline
    prompt_template = """
    Instructions:
    You are an AI assistant specialized in processing deep learning code based on user requirements.
    Answer the User Query using the following retrieved documents. If the documents are not relevant, rely on your training data.

    Retrieved Documents:
    {context}
    
    User Query:
    {question}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template) # PROMPT?

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vector_store.as_retriever(), 
        chain_type_kwargs={"prompt": prompt}
    )

    
    # prompt = ChatPromptTemplate.from_template(prompt_template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # # Create a stuff chain with the custom prompt
    # combine_documents_chain = create_stuff_documents_chain(
    #     llm=llm,
    #     prompt=prompt,
    # )
    
    # qa_chain = (
    #     {
    #         "context": vector_store.as_retriever() | format_docs,
    #         "question": RunnablePassthrough(),
    #     } 
    #     | prompt 
    #     | llm 
    #     # | StrOutputParser()
    # )
    
    return qa_chain, vector_store


def rag_generate(query: str, qa_chain):
    answer = qa_chain.run(query)
    return answer

def bare_llm_generate(query: str, llm):
    answer = llm.run(query)
    return answer


def retrieve_documents(query: str, vector_store):
    retrieved_docs = vector_store.as_retriever().invoke(query)
    return retrieved_docs


def retrieve_documents_only(query: str, vector_store):
    retrieved_docs = vector_store.as_retriever().invoke(query)
    return retrieved_docs


if __name__ == "__main__":
    print("Welcome to the RAG System!")
    print("Type 'exit' or 'quit' to terminate the program.\n")

    directories = ['docs/pytorch', 'docs/jax', 'docs/mindspore', 'docs/jittor']
    # directories = ['demo_docs']

    qa_chain, vector_store = initialize_rag_system(directories, is_local=True)

    while True:
        query = input("Enter your code-related query: ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        try:
            start_time = time.time()
            retrieved_docs = vector_store.as_retriever().invoke(query)

            answer = qa_chain.invoke(query)

            end_time = time.time()

            total_time = end_time - start_time

            print("\nGenerated Code:\n")
            print(answer['result'])
            print("\n" + "=" * 50 + "\n")

            print(f"total time: {total_time}")

            current_time = datetime.datetime.now().strftime('%m%d%H%M%S')
            with open(f'generated_code_{current_time}.txt', 'w', encoding='utf-8') as file:
                file.write(f"User Query: {query}\n")

                file.write("\nRetrieved Documents:\n")

                for idx, doc in enumerate(retrieved_docs, 1):
                    file.write(f"\nDocument {idx}:\n")
                    file.write(doc.page_content)
                    file.write("\n" + "-" * 40 + "\n")

                file.write("\nGenerated Code:\n")
                file.write(answer['result'])
                file.write("\n" + "=" * 50 + "\n")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("\n" + "=" * 50 + "\n")

    # directories = ['docs/pytorch', 'docs/jax', 'docs/mindspore', 'docs/jittor']
    # with open('documents3.txt', 'w') as f:
    #     docs = []
    #     for directory in directories:
    #         docs += load_files(directory, kind=directory.strip('docs/'))
    #     for doc in docs:
    #         f.write(doc)
    #         f.write('\n\n')