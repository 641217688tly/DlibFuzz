import datetime
import os
from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from llm import CodeQwenLLM, CodeGemmaLLM
from embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA


def load_html_files(directory: str):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.html') or filename.endswith('.htm'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                text = soup.get_text(separator='\n')
                documents.append(text)
    return documents


def initialize_rag_system(documents_dir: str):
    # Step 1: Load documents
    docs = load_html_files(documents_dir)
    
    # Step 2: Preprocess documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents([Document(page_content=doc) for doc in docs])
    
    # Step 3: Initialize embeddings
    embeddings = OllamaEmbeddings(model="llama3.1")
    
    # Create a FAISS vector store from the documents and their embeddings
    vector_store = FAISS.from_documents(split_docs, embeddings)
    
    # Step 4: Initialize LLM
    # llm = CodeQwenLLM()
    llm = CodeGemmaLLM()
    
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


def retrieve_documents(query: str, vector_store):
    retrieved_docs = vector_store.as_retriever().invoke(query)
    return retrieved_docs


if __name__ == "__main__":
    print("Welcome to the RAG System!")
    print("Type 'exit' or 'quit' to terminate the program.\n")

    qa_chain, vector_store = initialize_rag_system("demo_docs")

    while True:
        query = input("Enter your code-related query: ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        try:
            retrieved_docs = vector_store.as_retriever().invoke(query)

            answer = qa_chain.run(query) #TODO 十分奇怪，CodeGemma在使用qa_chain.run时不会有问题，但在使用qa_chain.invoke时就会报错

            print("\nGenerated Code:\n")
            print(answer)
            print("\n" + "=" * 50 + "\n")

            current_time = datetime.datetime.now().strftime('%m%d%H%M%S')
            with open(f'generated_code_{current_time}.txt', 'w', encoding='utf-8') as file:
                file.write(f"User Query: {query}\n")

                file.write("\nRetrieved Documents:\n")

                for idx, doc in enumerate(retrieved_docs, 1):
                    file.write(f"\nDocument {idx}:\n")
                    file.write(doc.page_content)
                    file.write("\n" + "-" * 40 + "\n")

                file.write("\nGenerated Code:\n")
                file.write(answer)
                file.write("\n" + "=" * 50 + "\n")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("\n" + "=" * 50 + "\n")
