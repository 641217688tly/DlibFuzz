import requests
from typing import List, Dict, Any, Optional
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
                  response_format: Optional[Dict] = None,
                  **kwargs) -> ChatCompletion:
            """
            Create a chat completion using the RAG system.

            Args:
                messages: List of message dictionaries with 'role' and 'content'
                model: Model identifier (ignored, included for compatibility)
                temperature: Temperature for generation (ignored, included for compatibility)
                max_tokens: Maximum tokens to generate (ignored, included for compatibility)
                response_format: Response format options (ignored, included for compatibility)
                **kwargs: Additional arguments (ignored, included for compatibility)

            Returns:
                ChatCompletion: A response object matching OpenAI's structure
            """
            # Process the full conversation history
            # Extract system message (if present) for context
            system_messages = [msg["content"] for msg in messages if msg["role"] == "system"]
            system_context = system_messages[0] if system_messages else ""

            # Extract the conversation history focusing on the latest query context
            # First, identify the latest user message
            user_messages_indices = [i for i, msg in enumerate(messages) if msg["role"] == "user"]
            if not user_messages_indices:
                raise ValueError("No user messages found in the conversation")

            latest_user_msg_idx = user_messages_indices[-1]

            # Extract query and relevant context
            query = messages[latest_user_msg_idx]["content"]

            # Look at recent conversation context
            # If the latest user message is not the first message, include some context
            context = ""
            if latest_user_msg_idx > 0:
                # Get the last few turns of conversation before the latest user message
                context_window = messages[max(0, latest_user_msg_idx-4):latest_user_msg_idx]
                context_parts = []
                for msg in context_window:
                    prefix = "User: " if msg["role"] == "user" else "Assistant: "
                    context_parts.append(f"{prefix}{msg['content']}")

                if context_parts:
                    context = "Previous conversation:\n" + "\n".join(context_parts)

            # Build a comprehensive query that includes the system prompt and context
            enhanced_query = query
            if system_context or context:
                # Only add necessary context to avoid diluting the query
                if len(query.split()) < 100:  # If the query is short, add more context
                    additional_context = []
                    if system_context:
                        additional_context.append(f"Context: {system_context}")
                    if context:
                        additional_context.append(context)

                    enhanced_query = "\n\n".join([*additional_context, f"Query: {query}"])

            # Prepare the request
            headers = {"Content-Type": "application/json"}
            data = {
                "query": enhanced_query,
                "messages": messages  # Also send the full messages array for future compatibility
            }

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

    def get_documents(self, query: str, messages: Optional[List[Dict[str, str]]] = None) -> List[str]:
        """
        Retrieve relevant documents for a query (additional method not in OpenAI's interface).

        Args:
            query: The query to retrieve documents for
            messages: Optional full message history for context

        Returns:
            List[str]: List of retrieved documents
        """
        headers = {"Content-Type": "application/json"}
        data = {"query": query}

        if messages:
            data["messages"] = messages

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
    # Initialize RagClient
    client = RagClient(base_url="http://localhost:8000")

    # Test with a simple message structure
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How do I use PyTorch's DataLoader?"}
    ]

    response = client.chat.completions.create(
        messages=messages,
        model="gpt-4"  # This parameter is unused, just for compatibility
    )

    print(response.choices[0].message["content"])

    # Test retrieving documents only
    documents = client.get_documents("How do I use PyTorch's DataLoader?")
    print("\nRelevant documents:", documents)

