'''
This is the API for the RAG system. It uses FastAPI to create a REST API.
Start the server by running `uvicorn gen_api:app --reload` in the terminal.
'''

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from rag_llm import initialize_rag_system, rag_generate, retrieve_documents, bare_llm_generate, build_embeddings
import os
from dotenv import load_dotenv


app = FastAPI()

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
# Initialize RAG system at startup
directories = ['docs/pytorch', 'docs/jax', 'docs/mindspore', 'docs/jittor']
build_embeddings(documents_dir=directories)
qa_chain, vector_store = initialize_rag_system(is_local=False,
                                               openai_model='gpt-4o-mini',
                                               openai_api_key=OPENAI_API_KEY
                                               )


class Message(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    messages: Optional[List[Dict[str, str]]] = None


class QueryResponse(BaseModel):
    answer: str


class RetrieveDocumentsResponse(BaseModel):
    documents: list


def extract_query_from_messages(messages: List[Dict[str, str]]) -> str:
    """Extract the most relevant query from a list of messages."""
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided.")

    # Extract the last user message as the primary query
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user messages found.")

    return user_messages[-1]["content"]


@app.post("/generate", response_model=QueryResponse)
def generate_code(request: QueryRequest):
    # Extract query either directly or from messages
    query = request.query
    messages = request.messages

    if not query and messages:
        query = extract_query_from_messages(messages)

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Process the context from messages if available
        if messages:
            # You might want to enhance your system prompt based on conversation context
            system_messages = [msg["content"] for msg in messages if msg["role"] == "system"]
            system_context = system_messages[0] if system_messages else ""

            # Get conversation history
            conv_context = ""
            user_messages = [i for i, msg in enumerate(messages) if msg["role"] == "user"]
            if len(user_messages) > 1:
                # There's conversation history to consider
                last_few_msgs = messages[max(0, user_messages[-2]):user_messages[-1]]
                conv_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in last_few_msgs])

        retrieved_docs = retrieve_documents(query, vector_store)
        answer = rag_generate(query, qa_chain)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_without_rag", response_model=QueryResponse)
def generate_without_rag(request: QueryRequest):
    # Extract query either directly or from messages
    query = request.query
    messages = request.messages

    if not query and messages:
        query = extract_query_from_messages(messages)

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer = bare_llm_generate(query, qa_chain)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve_documents", response_model=RetrieveDocumentsResponse)
def retrieve_documents_api(request: QueryRequest):
    # Extract query either directly or from messages
    query = request.query
    messages = request.messages

    if not query and messages:
        query = extract_query_from_messages(messages)

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        retrieved_docs = retrieve_documents(query, vector_store)
        return RetrieveDocumentsResponse(documents=[doc.page_content for doc in retrieved_docs])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

