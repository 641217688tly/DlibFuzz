import os
import logging
from typing import Any, ClassVar, Dict, List, Optional

import httpx
from pydantic import PrivateAttr
from langchain.llms.base import LLM
# from llama_cpp import Llama
from openai import OpenAI
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage, ChatMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# class CodeQwenLLM(LLM):
#     model_path: ClassVar[str] = "models/qwen2.5-coder-14b-instruct-q4_k_m.gguf"
#     _llm: Llama = PrivateAttr()  # Define as a private attribute
#
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)  # Initialize the superclass
#         self._llm = Llama(
#             model_path=self.model_path,
#             n_ctx=2048,
#             n_parts=-1,
#             seed=0,
#             n_gpu_layers=15000,
#             n_batch=512,
#             f16_kv=False,
#             logits_all=False,
#             vocab_only=False,
#             use_mlock=False,  # Set to False to avoid memory locking issues
#             embedding=False,
#             **kwargs
#         )
#
#     @property
#     def _llm_type(self):
#         return "llama_cpp"
#
#     def _call(self, prompt, stop=None):
#         response = self._llm.create_chat_completion(
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             stop=stop
#         )
#         return response['choices'][0]['message']['content']


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

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """
        Call the OpenAI API with the given prompt.
        
        Args:
            prompt: The input text to send to the model
            stop: Optional list of stop sequences
            **kwargs: Additional keyword arguments
            
        Returns:
            Generated text from the model
        """
        try:
            # Handle both string prompts and message lists
            messages = []
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
            elif isinstance(prompt, list):
                messages = self._convert_messages_to_chat(prompt)
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

# if __name__ == "__main__":
#     llm = CodeQwenLLM()
#     prompt = "Generate a python program to draw a line"
#     response = llm.invoke(prompt)
#
#     print(f"response: {response}")
