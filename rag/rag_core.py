import os
import requests
from bs4 import BeautifulSoup
from typing import List
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from llama_cpp import Llama
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from embeddings import OllamaEmbeddings
from llm import CodeQwenLLM

# Step 1: Load HTML files
def load_html_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.html') or filename.endswith('.htm'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                text = soup.get_text(separator='\n')
                documents.append(text)
    return documents

# Step 2: Process documents
docs = load_html_files('demo_docs')
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents([Document(page_content=doc) for doc in docs])

# Step 3: Initialize the custom Ollama embeddings
embeddings = OllamaEmbeddings(model="llama3")

# Create a FAISS vector store from the documents and their embeddings
vector_store = FAISS.from_documents(split_docs, embeddings)

# Step 4: Initialize the custom LLM
llm = CodeQwenLLM()

# Step 5: Build the Retrieval-Augmented Generation pipeline
prompt_template = """
You are an AI assistant specialized in generating code based on user requirements.

Use the following retrieved documents to inform your code generation. If the documents are not relevant, rely on your training data.

Retrieved Documents:
{context}

User Query:
{question}

Generate the appropriate code in response to the user's query.
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    prompt=prompt
)

# Step 6: Interactive interface
def main():
    print("Welcome to the Code Generation RAG System!")
    print("Type 'exit' or 'quit' to terminate the program.\n")

    while True:
        query = input("Enter your code-related query: ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        try:
            answer = qa_chain.run(query)
            print("\nGenerated Code:\n")
            print(answer)
            print("\n" + "-"*50 + "\n")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
