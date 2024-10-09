from langchain.llms.base import LLM
from llama_cpp import Llama

# Define the custom LLM
class CodeQwenLLM(LLM):
    model_path = "path/to/codeqwen-1_5-7b-chat-q2_k.gguf"  # Update this path

    def __init__(self, **kwargs):
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_parts=-1,
            seed=0,
            f16_kv=False,
            logits_all=False,
            vocab_only=False,
            use_mlock=True,
            embedding=False,
            **kwargs
        )

    @property
    def _llm_type(self):
        return "llama_cpp"

    def _call(self, prompt, stop=None):
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stop=stop
        )
        return response['choices'][0]['message']['content']
