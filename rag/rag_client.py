import requests
from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Choice:
    """Represents a single choice in the chat completion response"""
    index: int
    message: Dict[str, str]
    finish_reason: str


class ChatCompletion:
    """Helper class to match OpenAI's ChatCompletion response structure"""
    def __init__(self, response_data: str):
        self.id = f"rag-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.object = "chat.completion"
        self.created = int(datetime.now().timestamp())
        self.model = "rag-model"
        self.usage = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None
        }
        
        # Create a Choice object with the proper structure
        choice = Choice(
            index=0,
            message={"role": "assistant", "content": response_data},
            finish_reason="stop"
        )
        
        # Store choices as a list of Choice objects
        self.choices = [choice]

    def to_dict(self) -> Dict:
        """Convert the response to a dictionary format"""
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "usage": self.usage,
            "choices": [{
                "index": choice.index,
                "message": choice.message,
                "finish_reason": choice.finish_reason
            } for choice in self.choices]
        }

class RagClient:
    """
    A client that mimics the OpenAI client interface but uses the RAG API backend.
    This allows for drop-in replacement in code that uses OpenAI's client.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        """
        Initialize the RAG client.
        
        Args:
            base_url: The base URL of the RAG API
            api_key: Optional API key (included for OpenAI compatibility)
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.chat = self.Chat(self)  # Mirror OpenAI's structure
        
    class Chat:
        def __init__(self, client):
            self.client = client
            self.completions = self
        
        def create(self, 
                  messages: List[Dict[str, str]], 
                  model: Optional[str] = None,
                  temperature: Optional[float] = None,
                  max_tokens: Optional[int] = None,
                  **kwargs) -> ChatCompletion:
            """
            Create a chat completion using the RAG system.
            
            Args:
                messages: List of message dictionaries with 'role' and 'content'
                model: Model identifier (ignored, included for compatibility)
                temperature: Temperature for generation (ignored, included for compatibility)
                max_tokens: Maximum tokens to generate (ignored, included for compatibility)
                **kwargs: Additional arguments (ignored, included for compatibility)
            
            Returns:
                ChatCompletion: A response object matching OpenAI's structure
            """
            # Extract the last user message as the query
            user_messages = [msg for msg in messages if msg["role"] == "user"]
            if not user_messages:
                raise ValueError("No user messages found in the conversation")
            
            query = user_messages[-1]["content"]
            
            # Prepare the request
            headers = {"Content-Type": "application/json"}
            data = {"query": query}
            
            # Make the request to the RAG API
            try:
                response = requests.post(
                    f"{self.client.base_url}/generate",
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                
                # Parse the response
                result = response.json()
                return ChatCompletion(result["answer"])
                
            except requests.exceptions.RequestException as e:
                raise Exception(f"Error communicating with RAG API: {str(e)}")
    
    def get_documents(self, query: str) -> List[str]:
        """
        Retrieve relevant documents for a query (additional method not in OpenAI's interface).
        
        Args:
            query: The query to retrieve documents for
            
        Returns:
            List[str]: List of retrieved documents
        """
        headers = {"Content-Type": "application/json"}
        data = {"query": query}
        
        try:
            response = requests.post(
                f"{self.base_url}/retrieve_documents",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            return result["documents"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error retrieving documents: {str(e)}")


if __name__ == "__main__":
    # 初始化 RagClient
    # 模型和文档路径需要在 gen_api.py 中设置
    client = RagClient(base_url="http://localhost:8000")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I use PyTorch's DataLoader?"}
    ]
    
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4" # 这个参数没有用，仅仅是为了兼容性
    )
    
    print(response.choices[0].message["content"])
    
    # 仅仅检索文档而不生成内容
    documents = client.get_documents("How do I use PyTorch's DataLoader?")
    print("\nRelevant documents:", documents)