import os
import logging
from typing import Any, Dict, List, Optional, Union
import requests
import json

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
    Custom LLM class to interface with Ollama models running locally.
    """
    model_name: str = "qwen3:14b"
    api_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 1024

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs.get('model_name', self.model_name)
        self.api_url = kwargs.get('api_url', self.api_url).rstrip('/')
        self.temperature = kwargs.get('temperature', self.temperature)
        self.max_tokens = kwargs.get('max_tokens', self.max_tokens)

        # Test connection to Ollama
        try:
            self._test_connection()
            logger.info(f"OllamaLLM initialized successfully with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise

    def _test_connection(self):
        """Test if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            response.raise_for_status()

            # Check if the model is available
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]

            if self.model_name not in available_models:
                logger.warning(f"Model {self.model_name} not found in available models: {available_models}")
                logger.info(f"Attempting to pull model {self.model_name}...")
                self._pull_model()

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Cannot connect to Ollama at {self.api_url}. Make sure Ollama is running. Error: {e}")

    def _pull_model(self):
        """Pull the model if it's not available locally."""
        try:
            response = requests.post(
                f"{self.api_url}/api/pull",
                json={"name": self.model_name},
                stream=True,
                timeout=300  # 5 minutes timeout for model pulling
            )
            response.raise_for_status()

            # Process streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if 'status' in data:
                            logger.info(f"Model pull status: {data['status']}")
                        if data.get('status') == 'success':
                            logger.info(f"Model {self.model_name} pulled successfully")
                            break
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to pull model {self.model_name}: {e}")

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _convert_messages_to_ollama_format(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain message types to Ollama chat format."""
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
        Call the Ollama API with the given prompt.

        Args:
            prompt: The input text or message list to send to the model
            stop: Optional list of stop sequences
            **kwargs: Additional keyword arguments

        Returns:
            Generated text from the model
        """
        try:
            # Handle different prompt types
            if isinstance(prompt, str):
                # Simple string prompt - use generate endpoint
                return self._generate_from_string(prompt, stop, **kwargs)

            elif isinstance(prompt, list) and prompt:
                if isinstance(prompt[0], dict) and "role" in prompt[0]:
                    # List of message dictionaries - use chat endpoint
                    return self._chat_from_messages(prompt, stop, **kwargs)
                elif isinstance(prompt[0], BaseMessage):
                    # List of LangChain BaseMessage objects - convert and use chat endpoint
                    messages = self._convert_messages_to_ollama_format(prompt)
                    return self._chat_from_messages(messages, stop, **kwargs)

            # Default case: convert to string and use generate endpoint
            return self._generate_from_string(str(prompt), stop, **kwargs)

        except Exception as e:
            logger.error(f"Error during Ollama API call: {e}")
            raise

    def _generate_from_string(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Generate response from a string prompt using Ollama's generate endpoint."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        if stop:
            payload["options"]["stop"] = stop

        response = requests.post(
            f"{self.api_url}/api/generate",
            json=payload,
            timeout=120  # 2 minutes timeout
        )
        response.raise_for_status()

        result = response.json()
        return result.get("response", "")

    def _chat_from_messages(self, messages: List[Dict[str, str]], stop: Optional[List[str]] = None, **kwargs) -> str:
        """Generate response from messages using Ollama's chat endpoint."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }

        if stop:
            payload["options"]["stop"] = stop

        response = requests.post(
            f"{self.api_url}/api/chat",
            json=payload,
            timeout=120  # 2 minutes timeout
        )
        response.raise_for_status()

        result = response.json()
        return result.get("message", {}).get("content", "")

    def get_num_tokens(self, text: str) -> int:
        """
        Get the number of tokens in the text. This is a simple approximation.
        For more accurate token counting, you would need to use the model's tokenizer.
        """
        # Simple approximation: ~4 characters per token for most models
        return len(text) // 4

