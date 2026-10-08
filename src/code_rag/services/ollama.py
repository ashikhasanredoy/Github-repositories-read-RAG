import os
import json
import logging
import concurrent.futures
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
        self._active_embed: Optional[str] = None

    def list_installed_models(self) -> List[str]:
        try:
            with httpx.Client(timeout=4.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    raw_models = res.json().get("models", [])
                    models = []
                    for m in raw_models:
                        caps = m.get("details", {}).get("families", []) or m.get("capabilities", [])
                        name = m.get("name", "")
                        if "capabilities" in m and "completion" not in m["capabilities"]:
                            continue
                        if any(k in name.lower() for k in ["embed", "minilm", "bge", "bert"]):
                            continue
                        models.append(name)
                    return models if models else [m.get("name", "") for m in raw_models]
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

    def get_active_embed_model(self) -> str:
        if self._active_embed:
            return self._active_embed

        installed = self.list_installed_models()
        # Look for standard high-accuracy embedding models first
        for preferred in ["nomic-embed-text", "all-minilm", "bge", "embed"]:
            for m in installed:
                if preferred in m.lower():
                    self._active_embed = m
                    return self._active_embed

        self._active_embed = self.embedding_model
        return self._active_embed

    def embed_query(self, text: str, model_override: Optional[str] = None) -> List[float]:
        results = self.embed_documents([text], model_override=model_override)
        return results[0] if results else []

    def embed_documents(self, texts: List[str], batch_size: int = 48, model_override: Optional[str] = None) -> List[List[float]]:
        if not texts:
            return []

        embed_model = model_override or self.get_active_embed_model()
        embed_limit = getattr(settings, "EMBED_MAX_CHARS", 1500)
        bounded_texts = [t[:embed_limit] for t in texts]
        batches = [bounded_texts[i:i + batch_size] for i in range(0, len(bounded_texts), batch_size)]
        
        batch_results: List[Optional[List[List[float]]]] = [None] * len(batches)

        def _process_batch(idx: int, batch: List[str]) -> None:
            try:
                with httpx.Client(timeout=120.0, limits=httpx.Limits(max_keepalive_connections=10)) as client:
                    resp = client.post(
                        f"{self.base_url}/api/embed",
                        json={"model": embed_model, "input": batch}
                    )
                    if resp.status_code == 200:
                        embs = resp.json().get("embeddings", [])
                        if len(embs) == len(batch):
                            batch_results[idx] = embs
                            return
            except Exception as err:
                logger.warning("Batch %d embedding failed: %s, falling back to sequential", idx, err)

            fallback_embs = []
            with httpx.Client(timeout=60.0) as client:
                for text in batch:
                    try:
                        resp = client.post(
                            f"{self.base_url}/api/embeddings",
                            json={"model": embed_model, "prompt": text[:embed_limit]}
                        )
                        if resp.status_code == 200:
                            fallback_embs.append(resp.json().get("embedding", []))
                        else:
                            fallback_embs.append([0.0] * 768)
                    except Exception:
                        fallback_embs.append([0.0] * 768)
            batch_results[idx] = fallback_embs

        max_workers = min(6, len(batches))
        if max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_process_batch, i, b) for i, b in enumerate(batches)]
                concurrent.futures.wait(futures)
        elif len(batches) == 1:
            _process_batch(0, batches[0])

        all_embeddings: List[List[float]] = []
        for res in batch_results:
            if res:
                all_embeddings.extend(res)
            else:
                all_embeddings.extend([[0.0] * 768] * batch_size)

        return all_embeddings[:len(texts)]

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        tokens = list(self.generate_stream(prompt=prompt, system_prompt=system_prompt))
        return "".join(tokens).strip()

    def _build_payload(self, model: str, prompt: str, system_prompt: Optional[str] = None) -> dict:
        prompt_len = len(prompt) + (len(system_prompt) if system_prompt else 0)
        est_prompt_tokens = prompt_len // 3
        # Dynamically scale num_ctx: default 4096 for rapid prompt eval, scales up to settings.NUM_CTX for large files
        target_ctx = min(settings.NUM_CTX, max(4096, est_prompt_tokens + 2048))
        max_predict = min(settings.MAX_TOKENS, max(2048, target_ctx - est_prompt_tokens))

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": "15m",
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": max_predict,
                "num_ctx": target_ctx
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
        return payload

    def generate_stream(self, prompt: str, system_prompt: Optional[str] = None):
        model = self.get_active_model()
        payload = self._build_payload(model, prompt, system_prompt)

        with httpx.Client(timeout=300.0, limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=30.0)) as client:
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
                                        if chunk.get("done", False):
                                            break
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
                            if chunk.get("done", False):
                                break
                        except Exception:
                            pass

    async def generate_stream_async(self, prompt: str, system_prompt: Optional[str] = None):
        model = self.get_active_model()
        payload = self._build_payload(model, prompt, system_prompt)

        async with httpx.AsyncClient(timeout=300.0, limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=30.0)) as client:
            async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                if response.status_code == 404:
                    installed = self.list_installed_models()
                    if installed and installed[0] != model:
                        payload["model"] = installed[0]
                        self._active_llm = installed[0]
                        async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as retry_resp:
                            async for line in retry_resp.aiter_lines():
                                if line:
                                    try:
                                        chunk = json.loads(line)
                                        yield chunk.get("response", "")
                                        if chunk.get("done", False):
                                            break
                                    except Exception:
                                        pass
                        return

                if response.status_code != 200:
                    raise RuntimeError(f"Ollama generation failed ({response.status_code})")

                async for line in response.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            yield chunk.get("response", "")
                            if chunk.get("done", False):
                                break
                        except Exception:
                            pass
