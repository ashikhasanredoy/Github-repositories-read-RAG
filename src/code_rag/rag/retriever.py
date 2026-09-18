import logging
from typing import List, Dict, Any, Optional
from src.code_rag.config import settings
from src.code_rag.storage.vector_store import VectorStore
from src.code_rag.storage.bm25_store import BM25Store

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_store: Optional[BM25Store] = None,
        rrf_k: int = settings.RRF_K
    ):
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()
        self.rrf_k = rrf_k

    def retrieve(self, repo_id: str, query: str, top_k: int = settings.FINAL_TOP_K) -> List[Dict[str, Any]]:
        vector_results = self.vector_store.search(repo_id, query, top_k=settings.VECTOR_TOP_K)
        bm25_results = self.bm25_store.search(repo_id, query, top_k=settings.BM25_TOP_K)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(vector_results):
            cid = item["chunk_id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        for rank, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        sorted_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_k]

        final_chunks: List[Dict[str, Any]] = []
        for cid in sorted_ids:
            doc = chunk_map[cid]
            doc["rrf_score"] = round(rrf_scores[cid], 5)
            final_chunks.append(doc)

        return final_chunks

    @staticmethod
    def build_context(chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No relevant code chunks found in repository."

        blocks = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            file_path = meta.get("rel_path", "unknown")
            lang = meta.get("language", "python")
            start = meta.get("start_line", 1)
            end = meta.get("end_line", 1)
            symbol = meta.get("symbol_name", "snippet")
            sym_type = meta.get("symbol_type", "code")

            blocks.append(
                f"### [{i}] {file_path} (Lines {start}–{end})\n"
                f"**Symbol**: `{symbol}` ({sym_type})\n"
                f"```{lang}\n"
                f"{chunk.get('content', '')}\n"
                f"```"
            )

        return "\n\n".join(blocks)
