import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from src.code_rag.config import settings
from src.code_rag.core.models import CodeChunk
from src.code_rag.services.ollama import OllamaService

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        ollama_service: Optional[OllamaService] = None
    ):
        self.persist_dir = str(persist_dir or settings.CHROMA_DIR)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.ollama = ollama_service or OllamaService()

    @staticmethod
    def _sanitize_collection_name(repo_id: str) -> str:
        clean = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in f"repo_{repo_id}".lower())
        clean = clean.strip("._-")
        if not clean or not clean[0].isalnum():
            clean = f"repo_{clean}".strip("._-")
        return clean[:63]

    def index(self, repo_id: str, chunks: List[CodeChunk]) -> int:
        if not chunks:
            return 0

        coll_name = self._sanitize_collection_name(repo_id)
        try:
            self.client.delete_collection(name=coll_name)
        except Exception:
            pass

        collection = self.client.create_collection(
            name=coll_name,
            metadata={"hnsw:space": "cosine", "repo_id": repo_id}
        )

        texts = [c.formatted_content for c in chunks]
        metas = [c.to_metadata() for c in chunks]
        ids = [c.chunk_id for c in chunks]
        embeddings = self.ollama.embed_documents(texts)

        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            collection.add(
                ids=ids[i:i + batch_size],
                embeddings=embeddings[i:i + batch_size],
                documents=texts[i:i + batch_size],
                metadatas=metas[i:i + batch_size]
            )

        return len(chunks)

    def search(self, repo_id: str, query: str, top_k: int = settings.VECTOR_TOP_K) -> List[Dict[str, Any]]:
        coll_name = self._sanitize_collection_name(repo_id)
        try:
            collection = self.client.get_collection(name=coll_name)
        except Exception:
            return []

        query_vector = self.ollama.embed_query(query)
        if not query_vector:
            return []

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, max(1, collection.count())),
            include=["documents", "metadatas", "distances"]
        )

        matched_docs: List[Dict[str, Any]] = []
        if results and results.get("ids") and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                dist = results["distances"][0][i] if "distances" in results else 1.0
                matched_docs.append({
                    "chunk_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": round(1.0 - dist, 4),
                    "source": "vector"
                })

        return matched_docs

    def list_repos(self) -> List[str]:
        return [
            c.name[5:] for c in self.client.list_collections()
            if c.name.startswith("repo_")
        ]
