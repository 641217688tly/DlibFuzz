from typing import Any, List, Optional, Dict, Sequence, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    BitsAndBytesConfig
)
import logging
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from langchain_core.runnables import RunnableConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformersLLM(LLM):
    model_id: str = None
    model_path: str = None
    device: str = "auto"
    torch_dtype: str = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    
    _model: Optional[PreTrainedModel] = None
    _tokenizer: Optional[PreTrainedTokenizer] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._initialize_model()

    def _initialize_model(self):
        try:
            model_identifier = self.model_path or self.model_id
            if not model_identifier:
                raise ValueError("Either model_id or model_path must be provided")

            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Handle dtype
            if isinstance(self.torch_dtype, str):
                if self.torch_dtype == "auto":
                    self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
                elif self.torch_dtype == "float16":
                    self.torch_dtype = torch.float16
                elif self.torch_dtype == "bfloat16":
                    self.torch_dtype = torch.bfloat16
                elif self.torch_dtype == "float32":
                    self.torch_dtype = torch.float32

            # Configure quantization
            quantization_config = None
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=self.torch_dtype
                )

            # Initialize tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_identifier,
                trust_remote_code=True
            )
            
            # Initialize model with proper configuration
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": self.torch_dtype,
                "device_map": "auto" if self.device == "cuda" else None,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config

            self._model = AutoModelForCausalLM.from_pretrained(
                model_identifier,
                **model_kwargs
            )

            # Move model to device if not using device_map="auto"
            if self.device != "cuda" and not quantization_config:
                self._model.to(self.device)

            logger.info(f"Model initialized successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            if not self._model or not self._tokenizer:
                raise RuntimeError("Model or tokenizer not initialized")

            inputs = self._tokenizer(prompt, return_tensors="pt")
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_tokens', self.max_tokens),
                    temperature=kwargs.get('temperature', self.temperature),
                    top_p=kwargs.get('top_p', self.top_p),
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # Get the full decoded output
            full_output = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the beginning
            input_tokens = self._tokenizer(prompt, return_tensors="pt")["input_ids"]
            prompt_length = len(self._tokenizer.decode(input_tokens[0], skip_special_tokens=True))
            response = full_output[prompt_length:].strip()
            
            return response

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    def generate(
        self,
        prompts: Union[str, List[str]],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate text from prompt(s)."""
        if isinstance(prompts, str):
            prompts = [prompts]
        
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, callbacks=callbacks, **kwargs)
            generations.append([Generation(text=text)])
        
        return LLMResult(generations=generations)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in a text string."""
        if not self._tokenizer:
            raise RuntimeError("Tokenizer not initialized")
        return len(self._tokenizer.encode(text))

    @property
    def _llm_type(self) -> str:
        return "transformers"

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the model."""
        return {
            "model_id": self.model_id or self.model_path,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "is_quantized": self.load_in_4bit or self.load_in_8bit
        }