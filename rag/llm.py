import os
import logging
from typing import Any, Dict, List, Optional, Union

from langchain.llms.base import LLM
from openai import OpenAI
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAILLM(LLM):
    """
    Custom LLM class to interface with OpenAI models.
    """
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    _client: OpenAI = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = kwargs.get('api_key', os.getenv('OPENAI_API_KEY'))
        self.model_name = kwargs.get('model_name', self.model_name)
        self.temperature = kwargs.get('temperature', self.temperature)
        self.max_tokens = kwargs.get('max_tokens', self.max_tokens)

        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided either through kwargs or OPENAI_API_KEY environment variable")

        try:
            self._client = OpenAI(
                api_key=self.api_key,
            )
            logger.info(f"OpenAILLM initialized successfully with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAILLM: {e}")
            raise

        logger.info(f"LLM initialized as expected. Model: {self.model_name}")

    @property
    def _llm_type(self) -> str:
        return "openai"

    def _convert_messages_to_chat(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain message types to OpenAI chat format."""
        converted_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, HumanMessage):
                role = "user"
            else:
                role = "user"  # Default to user for other message types

            converted_messages.append({
                "role": role,
                "content": message.content
            })

        return converted_messages

    def _call(self, prompt: Union[str, List[Dict[str, str]], List[BaseMessage]], 
              stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Call the OpenAI API with the given prompt.
        
        Args:
            prompt: The input text or message list to send to the model
            stop: Optional list of stop sequences
            **kwargs: Additional keyword arguments
            
        Returns:
            Generated text from the model
        """
        try:
            # Handle different prompt types
            messages = []

            # Case 1: prompt is a string
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]

            # Case 2: prompt is a list of dictionary messages (like from RagClient)
            elif isinstance(prompt, list) and prompt and isinstance(prompt[0], dict) and "role" in prompt[0]:
                messages = prompt

            # Case 3: prompt is a list of LangChain BaseMessage objects
            elif isinstance(prompt, list) and prompt and isinstance(prompt[0], BaseMessage):
                messages = self._convert_messages_to_chat(prompt)

            # Default case: convert to string and pass as user message
            else:
                messages = [{"role": "user", "content": str(prompt)}]

            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
                **kwargs
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error during OpenAI API call: {e}")
            raise

    def get_num_tokens(self, text: str) -> int:
        """
        Get the number of tokens in the text. This is a simple approximation.
        For more accurate token counting, consider using tiktoken.
        """
        # Simple approximation: ~4 characters per token
        return len(text) // 4


class OllamaLLM(LLM):
    """
    Custom LLM class to interact with Ollama server locally.
    """
    model_name: str = "llama3"
    api_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 1024

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs.get('model_name', self.model_name)
        self.api_url = kwargs.get('api_url', self.api_url)
        self.temperature = kwargs.get('temperature', self.temperature)
        self.max_tokens = kwargs.get('max_tokens', self.max_tokens)

        logger.info(f"LLM initialized as expected. Model: {self.model_name}")


    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: Union[str, List[Dict[str, str]], List[BaseMessage]],
              stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Call the Ollama local server with the given prompt.
        """
        import requests

        if isinstance(prompt, list):
            if isinstance(prompt[0], BaseMessage):
                prompt = "\n".join([msg.content for msg in prompt])
            elif isinstance(prompt[0], dict) and "content" in prompt[0]:
                prompt = "\n".join([msg["content"] for msg in prompt])
            else:
                prompt = str(prompt)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "options": {
                "num_predict": self.max_tokens
            }
        }

        response = requests.post(f"{self.api_url}/v1/chat/completions", json=payload)
        if response.status_code != 200:
            raise ValueError(f"Error communicating with Ollama: {response.text}")
        data = response.json()
        return data["message"]["content"]

    def get_num_tokens(self, text: str) -> int:
        return len(text) // 4  # rough estimate

