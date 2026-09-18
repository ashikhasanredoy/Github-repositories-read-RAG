import json
import logging
from typing import List, Optional
import httpx
from src.code_rag.config import settings

logger = logging.getLogger(__name__)

class OllamaService:
    def __init__(
        self,
        base_url: str = settings.OLLAMA_BASE_URL,
        llm_model: str = settings.LLM_MODEL,
        embedding_model: str = settings.EMBEDDING_MODEL,
        temperature: float = settings.TEMPERATURE
    ):
        self.base_url = base_url.rstrip("/")
        self.preferred_llm = llm_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self._active_llm: Optional[str] = None

    def list_installed_models(self) -> List[str]:
        try:
            with httpx.Client(timeout=4.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    models = [m["name"] for m in res.json().get("models", [])]
                    return [m for m in models if "embed" not in m]
        except Exception:
            pass
        return []

    def get_active_model(self) -> str:
        if self._active_llm:
            return self._active_llm

        installed = self.list_installed_models()
        if not installed:
            return self.preferred_llm

        for m in installed:
            if self.preferred_llm.split(":")[0] in m:
                self._active_llm = m
                return self._active_llm

        self._active_llm = installed[0]
        return self._active_llm

    def set_model(self, model_name: str) -> None:
        if model_name:
            self._active_llm = model_name

    def embed_query(self, text: str) -> List[float]:
        results = self.embed_documents([text])
        return results[0] if results else []

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        bounded_texts = [t[:1200] for t in texts]

        with httpx.Client(timeout=180.0) as client:
            for i in range(0, len(bounded_texts), batch_size):
                batch = bounded_texts[i:i + batch_size]
                try:
                    resp = client.post(
                        f"{self.base_url}/api/embed",
                        json={"model": self.embedding_model, "input": batch}
                    )
                    if resp.status_code == 200:
                        embs = resp.json().get("embeddings", [])
                        if len(embs) == len(batch):
                            all_embeddings.extend(embs)
                            continue
                except Exception as err:
                    logger.warning("Batch embedding request failed: %s", err)

                # Fallback mini-batches
                for text in batch:
                    try:
                        resp = client.post(
                            f"{self.base_url}/api/embeddings",
                            json={"model": self.embedding_model, "prompt": text[:800]}
                        )
                        if resp.status_code == 200:
                            all_embeddings.append(resp.json().get("embedding", []))
                        else:
                            all_embeddings.append([0.0] * 768)
                    except Exception:
                        all_embeddings.append([0.0] * 768)

        return all_embeddings

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        tokens = list(self.generate_stream(prompt=prompt, system_prompt=system_prompt))
        return "".join(tokens).strip()

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        model = self.get_active_model()
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "num_predict": 320,
                "num_ctx": 2048
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                if response.status_code == 404:
                    installed = self.list_installed_models()
                    if installed and installed[0] != model:
                        payload["model"] = installed[0]
                        self._active_llm = installed[0]
                        with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as retry_resp:
                            for line in retry_resp.iter_lines():
                                if line:
                                    try:
                                        chunk = json.loads(line)
                                        yield chunk.get("response", "")
                                    except Exception:
                                        pass
                        return

                if response.status_code != 200:
                    raise RuntimeError(f"Ollama generation failed ({response.status_code})")

                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            yield chunk.get("response", "")
                        except Exception:
                            pass
