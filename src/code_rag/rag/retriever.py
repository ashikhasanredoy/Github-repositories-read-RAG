import re
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

        # Target file extraction & strict filtering
        target_files = [t for t in re.findall(r'[\w\-]+\.[\w]+', query) if not t.endswith('.com') and not t.endswith('.org')]
        exact_file_chunks = []
        if target_files:
            for tf in target_files:
                tf_lower = tf.lower()
                for cid, item in chunk_map.items():
                    rel = item.get("metadata", {}).get("rel_path", "").lower()
                    if tf_lower in rel or rel.endswith(tf_lower):
                        exact_file_chunks.append(cid)
                        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0

        if exact_file_chunks:
            sorted_ids = sorted(list(set(exact_file_chunks)), key=lambda k: rrf_scores[k], reverse=True)[:top_k]
        else:
            sorted_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_k]

        final_chunks: List[Dict[str, Any]] = []
        for cid in sorted_ids:
            doc = chunk_map[cid]
            doc["rrf_score"] = round(rrf_scores[cid], 5)
            final_chunks.append(doc)

        return final_chunks

    @staticmethod
    def build_context(chunks: List[Dict[str, Any]], query: str = "") -> str:
        if not chunks:
            return "No relevant code chunks found in repository."

        blocks = []

        # Check if user asked about a specific file that is absent
        target_files = [t for t in re.findall(r'[\w\-]+\.[\w]+', query) if not t.endswith('.com') and not t.endswith('.org')]
        if target_files:
            found_any = False
            for chunk in chunks:
                rel = chunk.get("metadata", {}).get("rel_path", "").lower()
                if any(tf.lower() in rel for tf in target_files):
                    found_any = True
                    break
            if not found_any:
                missing = ", ".join(target_files)
                blocks.append(
                    f"⚠️ [SYSTEM NOTICE: The user explicitly asked about '{missing}', but '{missing}' DOES NOT EXIST in this repository.\n"
                    f"DO NOT attribute code from other files to '{missing}'. State clearly that '{missing}' is not present in the repository.]\n"
                )

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
