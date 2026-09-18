import tempfile
from pathlib import Path
from src.code_rag.core.models import CodeChunk
from src.code_rag.storage.bm25_store import BM25Store

def test_bm25_indexer():
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = Path(tmp_dir)
        bm25_store = BM25Store(storage_dir=storage)

        chunks = [
            CodeChunk(
                chunk_id="chunk_1",
                repo_id="test_repo",
                rel_path="api/predict.py",
                language="python",
                symbol_name="predict_endpoint",
                symbol_type="function",
                start_line=10,
                end_line=25,
                code_content="@app.post('/predict')\ndef predict_endpoint(): pass",
                formatted_content="File: api/predict.py\nSymbol: predict_endpoint\n@app.post('/predict')\ndef predict_endpoint(): pass"
            ),
            CodeChunk(
                chunk_id="chunk_2",
                repo_id="test_repo",
                rel_path="auth/jwt.py",
                language="python",
                symbol_name="verify_token",
                symbol_type="function",
                start_line=1,
                end_line=15,
                code_content="def verify_token(token: str): pass",
                formatted_content="File: auth/jwt.py\nSymbol: verify_token\ndef verify_token(token: str): pass"
            )
        ]

        saved = bm25_store.index("test_repo", chunks)
        assert saved is True

        results = bm25_store.search("test_repo", "Where is the predict endpoint?", top_k=2)
        assert len(results) > 0
        assert results[0]["chunk_id"] == "chunk_1"

        results_auth = bm25_store.search("test_repo", "verify token authentication", top_k=2)
        assert len(results_auth) > 0
        assert results_auth[0]["chunk_id"] == "chunk_2"
