from pathlib import Path
from src.code_rag.core.parser import CodeParser
from src.code_rag.core.chunker import CodeChunker
from src.code_rag.services.ollama import OllamaService
from src.code_rag.storage.vector_store import VectorStore
from src.code_rag.storage.bm25_store import BM25Store
from src.code_rag.rag.retriever import HybridRetriever
from src.code_rag.rag.graph import CodeRAGGraph

def test_full_pipeline_sample_repo():
    repo_path = str(Path(__file__).parent / "sample_repo")
    repo_id = "sample_churn_repo"

    parser = CodeParser()
    chunker = CodeChunker(parser=parser)
    chunks = chunker.chunk_repository(repo_id=repo_id, repo_path=repo_path)
    assert len(chunks) >= 4

    ollama = OllamaService()
    v_store = VectorStore(ollama_service=ollama)
    b_store = BM25Store()

    v_store.index(repo_id=repo_id, chunks=chunks)
    b_store.index(repo_id=repo_id, chunks=chunks)

    retriever = HybridRetriever(vector_store=v_store, bm25_store=b_store)
    results = retriever.retrieve(repo_id=repo_id, query="Where is the prediction API endpoint implemented?", top_k=3)
    assert len(results) > 0
    top_file = results[0]["metadata"]["rel_path"]
    assert "model_api.py" in top_file

    graph = CodeRAGGraph(retriever=retriever, ollama_service=ollama).build()
    state = {
        "question": "Where is the prediction API implemented?",
        "original_question": "Where is the prediction API implemented?",
        "repo_id": repo_id,
        "query_intent": "code_explanation",
        "extracted_entities": [],
        "retrieved_docs": [],
        "is_relevant": True,
        "retry_count": 0,
        "context_str": "",
        "answer": "",
        "sources": [],
        "trace_steps": []
    }

    final = graph.invoke(state)
    assert len(final["sources"]) > 0
    assert len(final["answer"]) > 0
