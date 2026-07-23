import httpx
from typing import List


class EmbeddingClient:
    def __init__(self, endpoint: str):
        self.url = endpoint.rstrip("/") + "/v1/embeddings"
        self._client = httpx.Client(timeout=60)

    def embed(self, text: str) -> List[float]:
        resp = self._client.post(self.url, json={
            "input": text,
            "model": "default",
        })
        resp.raise_for_status()
        data = resp.json()
        vec = data["data"][0]["embedding"]
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def close(self):
        self._client.close()
