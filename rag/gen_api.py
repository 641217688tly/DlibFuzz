'''
This is the API for the RAG system. It uses FastAPI to create a REST API.
Start the server by running `uvicorn rag.gen_api:app --reload` in the terminal.
'''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_llm import initialize_rag_system, rag_generate, retrieve_documents


app = FastAPI()

# Initialize RAG system at startup
qa_chain, vector_store = initialize_rag_system('demo_docs')


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


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
