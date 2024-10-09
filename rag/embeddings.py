from langchain.embeddings.base import Embeddings
from typing import List
import requests

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
        Adjust this based on the model you are using.
        """
        return 768  # Example dimension; replace with actual if different
