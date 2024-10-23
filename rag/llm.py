import os
import logging
from typing import ClassVar
from pydantic import PrivateAttr
from langchain.llms.base import LLM
from llama_cpp import Llama
from langchain.schema import BaseMessage, AIMessage, HumanMessage, SystemMessage, ChatMessage



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeQwenLLM(LLM):
    model_path: ClassVar[str] = "models/codeqwen-1_5-7b-chat-q8_0.gguf"
    _llm: Llama = PrivateAttr()  # Define as a private attribute

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize the superclass
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_parts=-1,
            seed=0,
            n_gpu_layers=15000,
            n_batch=512,
            f16_kv=False,
            logits_all=False,
            vocab_only=False,
            use_mlock=False,  # Set to False to avoid memory locking issues
            embedding=False,
            **kwargs
        )

    @property
    def _llm_type(self):
        return "llama_cpp"

    def _call(self, prompt, stop=None):
        response = self._llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stop=stop
        )
        return response['choices'][0]['message']['content']

class CodeGemmaLLM(LLM):
    """
    Custom LLM class to interface with CodeGemma for code generation.
    """
    model_path: ClassVar[str] = os.getenv("CODEGEMMA_MODEL_PATH", "models/codegemma-7b-it-f16.gguf")
    _llm: Llama = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Verify model file exists and is readable
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not os.access(self.model_path, os.R_OK):
            logger.error(f"No read permission for model file at {self.model_path}")
            raise PermissionError(f"No read permission for model file at {self.model_path}")

        try:
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_gpu_layers=20,
                use_fp16=True,
                **kwargs
            )
            logger.info("CodeGemmaLLM initialized successfully with GPU support.")
        except Exception as e:
            logger.error(f"Failed to initialize CodeGemmaLLM: {e}")
            raise

    @property
    def _llm_type(self):
        return "llama_cpp"

    def _call(self, prompt, stop=None):
        if isinstance(prompt, ChatMessage):
            # Convert ChatPromptValue to a string prompt
            prompt = prompt.to_string()
        elif isinstance(prompt, list):
            # Convert list of messages to a string
            prompt = "\n".join([message.content for message in prompt])
        try:
            response = self._llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=1024,
                temperature=0.7,
                stop=["[/INST]"] if stop else None,
                repeat_penalty=1.0
            )

            logger.debug(f"Raw Response: {response}")

            # Extract the generated text
            generated_text = response['choices'][0]['message']['content']

            # Remove any trailing [INST] tags if present
            if generated_text.endswith("[/INST]"):
                generated_text = generated_text[:-len("[/INST]")].strip()

            return generated_text
        except Exception as e:
            logger.error(f"Error during LLM call: {e}")
            raise
        

if __name__ == "__main__":
    llm = CodeGemmaLLM()
    prompt = "Generate a python program to draw a line"
    response = llm.invoke(prompt)

    print(f"response: {response}")