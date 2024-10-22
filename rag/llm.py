from typing import ClassVar
from pydantic import PrivateAttr
from langchain.llms.base import LLM
from llama_cpp import Llama

class CodeQwenLLM(LLM):
    model_path: ClassVar[str] = "models/codeqwen-1_5-7b-chat-q8_0.gguf"  # Annotated as ClassVar
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
