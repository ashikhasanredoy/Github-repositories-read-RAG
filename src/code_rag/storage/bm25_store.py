import re
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Plus
from src.code_rag.config import settings
from src.code_rag.core.models import CodeChunk

logger = logging.getLogger(__name__)

class BM25Store:
    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or settings.BM25_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        split_camel = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
        words = re.findall(r'[a-zA-Z0-9]+', split_camel.lower())
        compounds = [token.lower() for token in re.findall(r'[a-zA-Z0-9_]+', text)]
        combined = words + compounds
        return [t for t in combined if len(t) > 1]

    def index(self, repo_id: str, chunks: List[CodeChunk]) -> bool:
        if not chunks:
            return False

        corpus_tokens = [self.tokenize(c.formatted_content) for c in chunks]
        bm25 = BM25Plus(corpus_tokens)

        payload = {
            "bm25": bm25,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "content": c.formatted_content,
                    "metadata": c.to_metadata()
                }
                for c in chunks
            ]
        }

        save_path = self.storage_dir / f"{repo_id}.pkl"
        save_path.write_bytes(pickle.dumps(payload))
        return True

    def search(self, repo_id: str, query: str, top_k: int = settings.BM25_TOP_K) -> List[Dict[str, Any]]:
        index_file = self.storage_dir / f"{repo_id}.pkl"
        if not index_file.exists():
            return []

        try:
            data = pickle.loads(index_file.read_bytes())
        except Exception as err:
            logger.error("Failed to load BM25 index for repo '%s': %s", repo_id, err)
            return []

        tokens = self.tokenize(query)
        if not tokens:
            return []

        bm25: BM25Plus = data["bm25"]
        stored_chunks: List[Dict[str, Any]] = data["chunks"]

        scores = bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        matched = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.0:
                matched.append({
                    "chunk_id": stored_chunks[idx]["chunk_id"],
                    "content": stored_chunks[idx]["content"],
                    "metadata": stored_chunks[idx]["metadata"],
                    "score": round(score, 4),
                    "source": "bm25"
                })

        return matched
