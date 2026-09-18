import json
import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from src.code_rag.config import settings
from src.code_rag.api.schemas import (
    IndexRepoRequest, IndexRepoResponse,
    QueryRequest, QueryResponse, SourceCitation,
    RepoListResponse
)
from src.code_rag.core.cloner import RepoCloner
from src.code_rag.core.parser import CodeParser
from src.code_rag.core.chunker import CodeChunker
from src.code_rag.services.ollama import OllamaService
from src.code_rag.storage.vector_store import VectorStore
from src.code_rag.storage.bm25_store import BM25Store
from src.code_rag.rag.retriever import HybridRetriever
from src.code_rag.rag.graph import CodeRAGGraph

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("code_rag.api")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="REST API for indexing GitHub repositories and querying with OLMo."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ollama_service = OllamaService()
cloner = RepoCloner()
parser = CodeParser()
chunker = CodeChunker(parser=parser)
vector_store = VectorStore(ollama_service=ollama_service)
bm25_store = BM25Store()
retriever = HybridRetriever(vector_store=vector_store, bm25_store=bm25_store)
rag_graph = CodeRAGGraph(retriever=retriever, ollama_service=ollama_service)
rag_pipeline = rag_graph.build()

@app.get("/", include_in_schema=False)
async def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "active_llm": ollama_service.get_active_model()
    }

@app.get("/api/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "active_llm": ollama_service.get_active_model()
    }

@app.get("/api/models", tags=["Health"])
async def list_models():
    models = await asyncio.to_thread(ollama_service.list_installed_models)
    return {
        "models": models,
        "active_model": ollama_service.get_active_model()
    }

@app.post("/api/index", response_model=IndexRepoResponse, tags=["Indexing"])
async def index_repository(req: IndexRepoRequest):
    try:
        def _index_job():
            repo = cloner.clone_or_load(req.repo_source, force_reclone=req.force_reindex)
            chunks = chunker.chunk_repository(repo["repo_id"], repo["local_path"])
            if not chunks:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No supported code files found in repository."
                )
            vector_store.index(repo["repo_id"], chunks)
            bm25_store.index(repo["repo_id"], chunks)
            unique_file_count = len({c.rel_path for c in chunks})
            return repo, chunks, unique_file_count

        repo, chunks, unique_file_count = await asyncio.to_thread(_index_job)

        return IndexRepoResponse(
            repo_id=repo["repo_id"],
            repo_name=repo["repo_name"],
            source=req.repo_source,
            total_files=unique_file_count,
            total_chunks=len(chunks),
            status="indexed",
            message=f"Indexed {len(chunks)} code chunks across {unique_file_count} files."
        )
    except HTTPException:
        raise
    except Exception as err:
        logger.error("Error during repository indexing: %s", err)
        raise HTTPException(status_code=500, detail=str(err))

@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def query_repository(req: QueryRequest):
    try:
        if req.model_name:
            ollama_service.set_model(req.model_name)

        initial_state = {
            "question": req.question,
            "original_question": req.question,
            "repo_id": req.repo_id,
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

        result = rag_pipeline.invoke(initial_state)

        citations = [
            SourceCitation(
                file_path=s["file_path"],
                language=s["language"],
                symbol_name=s["symbol_name"],
                symbol_type=s["symbol_type"],
                start_line=s["start_line"],
                end_line=s["end_line"],
                score=s.get("score", 0.0),
                snippet=s.get("snippet", "")
            )
            for s in result.get("sources", [])
        ]

        return QueryResponse(
            repo_id=req.repo_id,
            question=req.question,
            query_intent=result.get("query_intent", "code_explanation"),
            answer=result.get("answer", ""),
            sources=citations,
            trace_steps=result.get("trace_steps", [])
        )
    except Exception as err:
        logger.error("Query execution error: %s", err)
        raise HTTPException(status_code=500, detail=str(err))

@app.post("/api/query/stream", tags=["Query"])
def query_repository_stream(req: QueryRequest):
    if req.model_name:
        ollama_service.set_model(req.model_name)

    initial_state = {
        "question": req.question,
        "original_question": req.question,
        "repo_id": req.repo_id,
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

    def event_stream():
        try:
            for item in rag_graph.stream_rag(initial_state):
                yield f"data: {json.dumps(item)}\n\n"
        except Exception as err:
            logger.error("Streaming error: %s", err)
            yield f"data: {json.dumps({'type': 'error', 'error': str(err)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/api/repos", response_model=RepoListResponse, tags=["Repositories"])
async def list_repositories():
    repos = await asyncio.to_thread(vector_store.list_repos)
    return RepoListResponse(repos=repos, count=len(repos))
