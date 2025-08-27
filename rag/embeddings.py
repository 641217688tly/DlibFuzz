from langchain.embeddings.base import Embeddings
from typing import List
import requests
import os
from typing import Optional


# Create embeddings with Ollama and vector store
class OllamaEmbeddings(Embeddings):
    """
    Custom Embeddings class to interact with Ollama's API.
    """

    def __init__(self, model: str = "embedding-model-name", api_url: str = "http://localhost:11434"):
        """
        Initialize with the model name and Ollama API URL.
        """
        self.model = model
        self.api_url = api_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents by sending them to Ollama's API.
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query/document.
        """
        payload = {
            "model": self.model,
            "input": text
        }
        response = requests.post(f"{self.api_url}/v1/embeddings", json=payload)
        if response.status_code != 200:
            raise ValueError(f"Error fetching embedding from Ollama: {response.text}")
        data = response.json()
        return data["data"][0]["embedding"]

    @property
    def embedding_dimension(self) -> int:
        """
        Return the dimension of the embeddings.
        """
        return 768


class OpenAIEmbeddings(Embeddings):
    """
    使用OpenAI API的嵌入类
    """

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        """
        初始化OpenAI嵌入模型
        
        Args:
            model: OpenAI嵌入模型名称
            api_key: OpenAI API密钥
        """
        self.model = model
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API Key must be provided via parameters or the environment variable OPENAI_API_KEY.")
        
        self.client = None
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "OpenAI Python package not installed.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        嵌入多个文本
        """
        # 处理空文本的情况
        if not texts:
            return []
            
        try:
            # 调用OpenAI API获取嵌入
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            # 提取嵌入向量
            embeddings = [item.embedding for item in response.data]
            return embeddings
            
        except Exception as e:
            print(f"Error when obtaining OpenAI embeddings: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding

    @property
    def embedding_dimension(self) -> int:
        """
        返回嵌入维度
        """
        # text-embedding-3-small: 1536维
        # text-embedding-3-large: 3072维
        # text-embedding-ada-002: 1536维
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        return dimensions.get(self.model, 1536)  # 默认返回1536

