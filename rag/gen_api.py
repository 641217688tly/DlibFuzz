'''
This is the API for the RAG system. It uses FastAPI to create a REST API.
Start the server by running `uvicorn gen_api:app --reload` in the terminal.
'''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_llm import initialize_rag_system, rag_generate, retrieve_documents, bare_llm_generate, build_embeddings
import os
from dotenv import load_dotenv


app = FastAPI()

load_dotenv()
# Initialize RAG system at startup
directories = ['docs/pytorch', 'docs/jax', 'docs/mindspore', 'docs/jittor']
build_embeddings(documents_dir=directories)
qa_chain, vector_store = initialize_rag_system(is_local=False,
                                               openai_model='gpt4o-mini', 
                                               openai_api_key=os.getenv('OPENAI_API_KEY', '')
                                               )


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


class RetrieveDocumentsResponse(BaseModel):
    documents: list


@app.post("/generate", response_model=QueryResponse)
def generate_code(request: QueryRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        retrieved_docs = retrieve_documents(query, vector_store)
        answer = rag_generate(query, qa_chain)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_without_rag", response_model=QueryResponse)
def generate_code(request: QueryRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        retrieved_docs = retrieve_documents(query, vector_store)
        answer = bare_llm_generate(query, qa_chain)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve_documents", response_model=RetrieveDocumentsResponse)
def retrieve_documents_api(request: QueryRequest):
    query = request.query
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        retrieved_docs = retrieve_documents(query, vector_store)
        return RetrieveDocumentsResponse(documents=[doc.page_content for doc in retrieved_docs])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
