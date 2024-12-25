import datetime
import os
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from bs4 import BeautifulSoup
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA

@dataclass
class RAGResponse:
    """Response object for RAG queries"""
    result: str
    retrieved_docs: List[Document]
    total_time: float
    timestamp: str

class RAGClient:
    """A client for RAG-based code generation using OpenAI models"""
    
    def __init__(
        self,
        documents_dir: str,
        llm_model: str,
        openai_model: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        embeddings_model: str = "llama3.1",
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        """
        Initialize the RAG OpenAI Client
        
        Args:
            documents_dir: Directory containing the documents for RAG
            llm_model: LLM model to use ("openai" or "codellama")
            embeddings_model: Embeddings model to use
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
        """
        self.documents_dir = documents_dir
        self.llm_model = llm_model
        self.embeddings_model = embeddings_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize the RAG system
        self.qa_chain, self.vector_store = self._initialize_rag_system()
        
    def _load_files(self, kind: str = "pytorch") -> List[str]:
        """Load documents from the specified directory"""
        documents = []
        for filename in os.listdir(self.documents_dir):
            if filename.endswith(('.html', '.htm')):
                filepath = os.path.join(self.documents_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'html.parser')
                    if kind == 'pytorch':
                        sections = soup.find_all('div', class_='section')
                        text = "\n".join(section.get_text(separator='') for section in sections)
                        documents.append(text)
            elif filename.endswith('.md'):
                filepath = os.path.join(self.documents_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
                    documents.append(text)
        return documents
    
    def _initialize_rag_system(self):
        """Initialize the RAG system components"""
        # Load and preprocess documents
        docs = self._load_files()
        text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        split_docs = text_splitter.split_documents(
            [Document(page_content=doc) for doc in docs]
        )
        
        # Initialize embeddings
        from embeddings import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model=self.embeddings_model)
        
        # Create vector store
        vector_store = FAISS.from_documents(split_docs, embeddings)
        
        # Initialize LLM
        if self.llm_model == "openai":
            from llm import OpenAILLM
            llm = OpenAILLM(self.openai_model, self.openai_api_key)
        else:
            from llm import CodeQwenLLM
            llm = CodeQwenLLM()
            
        # Setup RAG prompt
        prompt_template = """
        Instructions:
        You are an AI assistant specialized in processing deep learning code based on user requirements.
        Answer the User Query using the following retrieved documents. If the documents are not relevant, rely on your training data.

        Retrieved Documents:
        {context}
        
        User Query:
        {question}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt}
        )
        
        return qa_chain, vector_store
    
    def generate(self, query: str, save_output: bool = True) -> RAGResponse:
        """
        Generate code based on the input query
        
        Args:
            query: User's code-related query
            save_output: Whether to save the output to a file
            
        Returns:
            RAGResponse object containing the result and metadata
        """
        try:
            start_time = time.time()
            
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.as_retriever().invoke(query)
            
            # Generate answer
            answer = self.qa_chain.invoke(query)
            
            # Calculate timing and create timestamp
            total_time = time.time() - start_time
            timestamp = datetime.datetime.now().strftime('%m%d%H%M%S')
            
            # Create response object
            response = RAGResponse(
                result=answer['result'],
                retrieved_docs=retrieved_docs,
                total_time=total_time,
                timestamp=timestamp
            )
            
            # Save output if requested
            if save_output:
                self._save_output(query, response)
            
            return response
            
        except Exception as e:
            raise Exception(f"Generation failed: {str(e)}")
    
    def _save_output(self, query: str, response: RAGResponse):
        """Save the generation output to a file"""
        filename = f'generated_code_{response.timestamp}.txt'
        
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"User Query: {query}\n")
            
            file.write("\nRetrieved Documents:\n")
            for idx, doc in enumerate(response.retrieved_docs, 1):
                file.write(f"\nDocument {idx}:\n")
                file.write(doc.page_content)
                file.write("\n" + "-" * 40 + "\n")
            
            file.write("\nGenerated Code:\n")
            file.write(response.result)
            file.write("\n" + "=" * 50 + "\n")
    
    def retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query without generating code"""
        return self.vector_store.as_retriever().invoke(query)