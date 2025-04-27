import datetime
import os
import time
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from embeddings import OllamaEmbeddings, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from typing import List, Dict, Union


def load_files(directory: str, kind: str):
    documents = []
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith(('.html', '.htm', '.md', '.txt')):
                continue
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
                elif filename.endswith(('.md', '.txt')):
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


def build_embeddings(documents_dir: list,
                     use_third_party_hosted: bool = False,
                     openai_api_key: str = ''):
    if use_third_party_hosted:
        try:
            print('Using model from OpenAI for text embedding.')
            embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        except Exception as e:
            print(f"Error happend: {e}")
    else:
        print('Using self-hosted model for text embedding')
        embeddings = OllamaEmbeddings(model='nomic-embed-text')

    if os.path.exists('vector_store.faiss'):
        print('Vector store already existed.')
        return

    print('--------------Creating vector store--------------')

    try:
        docs = []
        for directory in documents_dir:
            print(f"Loading from {directory}")
            kind = directory.split('/')[-1] if '/' in directory else directory
            loaded_docs = load_files(directory, kind=kind)
            print(f"Loaded {len(loaded_docs)} from {directory}...")
            docs += loaded_docs

        print(f'{len(docs)} documents loaded.')

        # 创建FAISS向量存储
        vector_store = create_vector_store_batched(docs, embeddings)

        # 添加错误检查
        if vector_store is None:
            print("Error: Cannot build vectortore.")
            return None

        print('Vectorstore built as expected.')

        # 保存向量存储
        try:
            vector_store.save_local('vector_store.faiss')
            print('Vectorstore has been saved locally.')
        except Exception as e:
            print(f"Error when saving vectorstore: {str(e)}")
            return None

        return vector_store
    except Exception as e:
        print(f"Error when building embeddings: {str(e)}")
        return None


def initialize_rag_system(is_local: bool,
                          openai_model: str = 'gpt-4o-mini', 
                          openai_api_key: str = '',
                          instructions_template: str = None
                          ):
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    print('Embeddings initialized.')

    if os.path.exists('vector_store.faiss'):
        vector_store = FAISS.load_local('vector_store.faiss', 
                                        embeddings=embeddings, 
                                        allow_dangerous_deserialization=True)
        print('Vector store loaded.')
    else:
        raise Exception('Vector store not found. Please invoke build_embeddings to build the vector store.')
    
    # Initialize LLM
    if is_local:
        from llm import OllamaLLM
        llm = OllamaLLM(model_name='llama3.1')
    else:
        from llm import OpenAILLM
        llm = OpenAILLM(model_name=openai_model, api_key=openai_api_key)
    
    # Establish RAG pipeline
    if instructions_template is None:
        prompt_template = """
        Instructions:
        You are an AI assistant specialized in processing deep learning code based on user requirements.
        Answer the User Query using the following retrieved documents. If the documents are not relevant, rely on your training data.

        Retrieved Documents:
        {context}
        
        User Query:
        {question}

        """
    else:
        prompt_template = "Instructions:\n" + instructions_template + """

        Retrieved Documents:
        {context}
        
        User Query:
        {question}

        """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vector_store.as_retriever(), 
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain, vector_store


def extract_query_and_context_from_messages(messages: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Extract the query and relevant context from a list of messages.

    Args:
        messages: A list of message dictionaries with 'role' and 'content'

    Returns:
        A dictionary with 'query' and 'context' keys
    """
    if not messages:
        return {"query": "", "context": ""}

    # Extract system message if present
    system_content = ""
    system_messages = [msg["content"] for msg in messages if msg["role"] == "system"]
    if system_messages:
        system_content = system_messages[0]

    # Find all user messages
    user_message_indices = [i for i, msg in enumerate(messages) if msg["role"] == "user"]
    if not user_message_indices:
        return {"query": "", "context": system_content}

    # Get the last user message as the query
    last_user_idx = user_message_indices[-1]
    query = messages[last_user_idx]["content"]

    # Build context from conversation history
    context_parts = []
    if system_content:
        context_parts.append(f"System: {system_content}")

    # Get conversation history (limit to a few turns before the query)
    if last_user_idx > 0:
        # Get up to 3 conversation turns before the latest query
        start_idx = max(0, last_user_idx - 6)  # Get up to 3 turns (6 messages)
        for i in range(start_idx, last_user_idx):
            msg = messages[i]
            prefix = "User: " if msg["role"] == "user" else "Assistant: "
            context_parts.append(f"{prefix}{msg['content']}")

    context = "\n\n".join(context_parts) if context_parts else ""

    return {"query": query, "context": context}


def rag_generate(query_or_messages: Union[str, List[Dict[str, str]]], qa_chain):
    """
    Generate a response using the RAG system.

    Args:
        query_or_messages: Either a query string or a list of message dictionaries
        qa_chain: The retrieval QA chain to use

    Returns:
        The generated response
    """
    # Check if input is messages or a direct query
    if isinstance(query_or_messages, list):
        # Extract query and context from messages
        extracted = extract_query_and_context_from_messages(query_or_messages)
        query = extracted["query"]

        # If there's significant context, we could modify the query to include it
        # For now, we'll keep it simple and just use the extracted query
    else:
        query = query_or_messages

    # Generate the response using the retrieval QA chain
    answer = qa_chain.run(query)
    return answer


def bare_llm_generate(query_or_messages: Union[str, List[Dict[str, str]]], llm):
    """
    Generate a response without using retrieval.

    Args:
        query_or_messages: Either a query string or a list of message dictionaries
        llm: The language model to use

    Returns:
        The generated response
    """
    # Check if input is messages or a direct query
    if isinstance(query_or_messages, list):
        # Process messages format
        answer = llm._call(query_or_messages)
    else:
        # Process direct query
        answer = llm.run(query_or_messages)

    return answer


def retrieve_documents(query: str, vector_store):
    """
    Retrieve relevant documents for a query.

    Args:
        query: The query to retrieve documents for
        vector_store: The vector store to retrieve from

    Returns:
        A list of retrieved documents
    """
    retrieved_docs = vector_store.as_retriever().invoke(query)
    return retrieved_docs


if __name__ == "__main__":
    print("RAG Module Activated.\n")
    print("Type 'exit' or 'quit' to terminate the program.\n")

    directories = ['docs/pytorch', 'docs/jax', 'docs/mindspore', 'docs/jittor']

    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    print(f'The API Key is {OPENAI_API_KEY}')
    build_embeddings(documents_dir=directories, use_third_party_hosted=False, openai_api_key=OPENAI_API_KEY)
    qa_chain, vector_store = initialize_rag_system(openai_api_key=OPENAI_API_KEY, is_local=False)

    while True:
        query = input("Enter your code-related query (or type 'exit'/'quit' to end): ")
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

