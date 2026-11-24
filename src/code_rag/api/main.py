import time
import json
import asyncio
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from src.code_rag.config import settings
from src.code_rag.api.schemas import (
    IndexRepoRequest, IndexRepoResponse,
    QueryRequest, QueryResponse, SourceCitation,
    RepoListResponse, DeleteRepoResponse,
    ConversationSummary, ConversationListResponse,
    ChatMessage, ConversationDetailResponse,
    ConversationMessagesResponse, RenameConversationRequest,
    DeleteTurnResponse, DeleteConversationResponse
)
from src.code_rag.core.cloner import RepoCloner
from src.code_rag.core.parser import CodeParser
from src.code_rag.core.chunker import CodeChunker
from src.code_rag.services.ollama import OllamaService
from src.code_rag.storage.vector_store import VectorStore
from src.code_rag.storage.bm25_store import BM25Store
from src.code_rag.storage.sqlite_store import SQLiteStore
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
sqlite_store = SQLiteStore()
retriever = HybridRetriever(vector_store=vector_store, bm25_store=bm25_store, cloner=cloner)
rag_graph = CodeRAGGraph(retriever=retriever, ollama_service=ollama_service)
rag_pipeline = rag_graph.build()

active_stream_cancellations: dict = {}

@app.get("/", include_in_schema=False)
async def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(
            index_file,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
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
    except ValueError as val_err:
        logger.warning("Validation error during repository indexing: %s", val_err)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as err:
        logger.error("Error during repository indexing: %s", err)
        raise HTTPException(status_code=500, detail=str(err))

# -------------------------------------------------------------
# Conversation Management APIs
# -------------------------------------------------------------

@app.get("/api/conversations", response_model=ConversationListResponse, tags=["Conversations"])
async def list_conversations():
    convs = await asyncio.to_thread(sqlite_store.list_conversations)
    return ConversationListResponse(
        conversations=[
            ConversationSummary(
                conversation_id=c["conversation_id"],
                title=c["title"],
                repo_id=c.get("repo_id", ""),
                created_at=c["created_at"],
                updated_at=c["updated_at"],
                turn_count=c.get("turn_count", 0)
            )
            for c in convs
        ],
        count=len(convs)
    )

@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetailResponse, tags=["Conversations"])
async def get_conversation(
    conversation_id: str,
    limit: int = Query(None, description="Optional maximum messages to retrieve"),
    offset: int = Query(0, description="Offset for pagination")
):
    conv = await asyncio.to_thread(sqlite_store.get_conversation, conversation_id, limit=limit, offset=offset)
    if not conv:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Conversation '{conversation_id}' not found")

    messages = [
        ChatMessage(
            message_id=m["message_id"],
            conversation_id=m["conversation_id"],
            turn_id=m["turn_id"],
            role=m["role"],
            content=m["content"],
            model_name=m.get("model_name"),
            query_intent=m.get("query_intent"),
            sources=m.get("sources"),
            status=m["status"],
            created_at=m["created_at"]
        )
        for m in conv.get("messages", [])
    ]
    return ConversationDetailResponse(
        conversation_id=conv["conversation_id"],
        title=conv["title"],
        repo_id=conv.get("repo_id", ""),
        created_at=conv["created_at"],
        updated_at=conv["updated_at"],
        messages=messages
    )

@app.get("/api/conversations/{conversation_id}/messages", response_model=ConversationMessagesResponse, tags=["Conversations"])
async def get_conversation_messages(conversation_id: str):
    raw_messages = await asyncio.to_thread(sqlite_store.get_messages, conversation_id)
    messages = [
        ChatMessage(
            message_id=m["message_id"],
            conversation_id=m["conversation_id"],
            turn_id=m["turn_id"],
            role=m["role"],
            content=m["content"],
            model_name=m.get("model_name"),
            query_intent=m.get("query_intent"),
            sources=m.get("sources"),
            status=m["status"],
            created_at=m["created_at"]
        )
        for m in raw_messages
    ]
    return ConversationMessagesResponse(
        conversation_id=conversation_id,
        messages=messages,
        count=len(messages)
    )

@app.delete("/api/conversations/{conversation_id}/turns/{turn_id}", response_model=DeleteTurnResponse, tags=["Conversations"])
async def delete_turn(conversation_id: str, turn_id: str):
    success, deleted_count = await asyncio.to_thread(sqlite_store.delete_turn, conversation_id, turn_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Turn '{turn_id}' not found in conversation '{conversation_id}'."
        )
    return DeleteTurnResponse(
        success=True,
        conversation_id=conversation_id,
        turn_id=turn_id,
        deleted_messages=deleted_count,
        message="Question and corresponding AI answer deleted successfully."
    )

@app.delete("/api/conversations/{conversation_id}", response_model=DeleteConversationResponse, tags=["Conversations"])
async def delete_conversation(conversation_id: str):
    cancel_event = active_stream_cancellations.get(conversation_id)
    if cancel_event:
        cancel_event.set()

    success = await asyncio.to_thread(sqlite_store.delete_conversation, conversation_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found."
        )
    return DeleteConversationResponse(
        success=True,
        conversation_id=conversation_id,
        message=f"Conversation '{conversation_id}' and all its turns were deleted successfully."
    )

@app.patch("/api/conversations/{conversation_id}", response_model=ConversationSummary, tags=["Conversations"])
async def rename_conversation(conversation_id: str, req: RenameConversationRequest):
    success = await asyncio.to_thread(sqlite_store.rename_conversation, conversation_id, req.title)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation '{conversation_id}' not found."
        )
    conv = await asyncio.to_thread(sqlite_store.get_conversation, conversation_id)
    return ConversationSummary(
        conversation_id=conv["conversation_id"],
        title=conv["title"],
        repo_id=conv.get("repo_id", ""),
        created_at=conv["created_at"],
        updated_at=conv["updated_at"]
    )

# -------------------------------------------------------------
# Query Execution & Real-Time SSE Streaming
# -------------------------------------------------------------

@app.post("/api/query", response_model=QueryResponse, tags=["Query"])
async def query_repository(req: QueryRequest):
    try:
        if req.model_name:
            ollama_service.set_model(req.model_name)

        conv_id = req.conversation_id or f"conv_{int(time.time() * 1000)}"
        turn_id = req.turn_id or f"turn_{int(time.time() * 1000)}"

        # 1-7. Save question & create turn in SQLite3
        await asyncio.to_thread(sqlite_store.create_or_get_conversation, conv_id, repo_id=req.repo_id)
        await asyncio.to_thread(sqlite_store.create_turn, conv_id, turn_id)
        await asyncio.to_thread(sqlite_store.add_user_message, conv_id, turn_id, req.question)
        await asyncio.to_thread(
            sqlite_store.create_pending_assistant_message,
            conv_id, turn_id, model_name=req.model_name or ollama_service.get_active_model()
        )

        initial_state = {
            "question": req.question,
            "original_question": req.question,
            "repo_id": req.repo_id,
            "conversation_id": conv_id,
            "turn_id": turn_id,
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

        citations_dicts = [c.model_dump() for c in citations]
        # Update assistant message in SQLite3 (only if turn still exists)
        await asyncio.to_thread(
            sqlite_store.update_assistant_message,
            conv_id,
            turn_id,
            content=result.get("answer", ""),
            sources=citations_dicts,
            status="completed",
            query_intent=result.get("query_intent", "code_explanation"),
            model_name=req.model_name or ollama_service.get_active_model()
        )

        return QueryResponse(
            repo_id=req.repo_id,
            question=req.question,
            query_intent=result.get("query_intent", "code_explanation"),
            answer=result.get("answer", ""),
            sources=citations,
            trace_steps=result.get("trace_steps", []),
            conversation_id=conv_id,
            turn_id=turn_id
        )
    except Exception as err:
        logger.error("Query execution error: %s", err)
        raise HTTPException(status_code=500, detail=str(err))

@app.post("/api/query/stream", tags=["Query"])
async def query_repository_stream(req: QueryRequest):
    if req.model_name:
        ollama_service.set_model(req.model_name)

    conv_id = req.conversation_id or f"conv_{int(time.time() * 1000)}"
    turn_id = req.turn_id or f"turn_{int(time.time() * 1000)}"

    # 1. Save question & pending assistant message before streaming
    await asyncio.to_thread(sqlite_store.create_or_get_conversation, conv_id, repo_id=req.repo_id)
    await asyncio.to_thread(sqlite_store.create_turn, conv_id, turn_id)
    await asyncio.to_thread(sqlite_store.add_user_message, conv_id, turn_id, req.question)
    await asyncio.to_thread(
        sqlite_store.create_pending_assistant_message,
        conv_id, turn_id, model_name=req.model_name or ollama_service.get_active_model()
    )

    initial_state = {
        "question": req.question,
        "original_question": req.question,
        "repo_id": req.repo_id,
        "conversation_id": conv_id,
        "turn_id": turn_id,
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

    cancel_event = asyncio.Event()
    active_stream_cancellations[conv_id] = cancel_event

    async def event_stream():
        accumulated_answer = []
        sources = []
        query_intent = "code_explanation"
        stream_completed_normally = False

        try:
            # Send meta event so client immediately receives conversation_id and turn_id
            yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conv_id, 'turn_id': turn_id})}\n\n"

            async for item in rag_graph.stream_rag_async(initial_state):
                if cancel_event.is_set():
                    logger.info("Stream cancelled via cancellation event for conv %s.", conv_id)
                    break
                if item.get("type") == "token":
                    accumulated_answer.append(item.get("content", ""))
                elif item.get("type") == "done":
                    sources = item.get("sources", [])
                    query_intent = item.get("query_intent", query_intent)
                    item["conversation_id"] = conv_id
                    item["turn_id"] = turn_id
                    stream_completed_normally = True

                yield f"data: {json.dumps(item)}\n\n"

            if stream_completed_normally and not cancel_event.is_set():
                full_answer = "".join(accumulated_answer)
                # Save completed response to existing assistant message (safe against deletion)
                await asyncio.to_thread(
                    sqlite_store.update_assistant_message,
                    conv_id,
                    turn_id,
                    content=full_answer,
                    sources=sources,
                    status="completed",
                    query_intent=query_intent,
                    model_name=req.model_name or ollama_service.get_active_model()
                )
        except (asyncio.CancelledError, GeneratorExit):
            logger.info("Stream cancelled/aborted for turn %s in conv %s.", turn_id, conv_id)
            partial_answer = "".join(accumulated_answer)
            await asyncio.to_thread(
                sqlite_store.update_assistant_message,
                conv_id,
                turn_id,
                content=partial_answer,
                sources=sources,
                status="interrupted",
                query_intent=query_intent,
                model_name=req.model_name or ollama_service.get_active_model()
            )
            return
        except Exception as err:
            logger.error("Streaming error for turn %s: %s", turn_id, err)
            await asyncio.to_thread(
                sqlite_store.update_assistant_message,
                conv_id,
                turn_id,
                content="".join(accumulated_answer),
                sources=[],
                status="failed",
                model_name=req.model_name or ollama_service.get_active_model()
            )
            yield f"data: {json.dumps({'type': 'error', 'error': str(err)})}\n\n"
        finally:
            active_stream_cancellations.pop(conv_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/api/repos", response_model=RepoListResponse, tags=["Repositories"])
async def list_repositories():
    repos = await asyncio.to_thread(vector_store.list_repos)
    return RepoListResponse(repos=repos, count=len(repos))

@app.delete("/api/repos/{repo_id}", response_model=DeleteRepoResponse, tags=["Repositories"])
async def delete_repository(repo_id: str):
    try:
        def _delete_job():
            v_del = vector_store.delete_repo(repo_id)
            b_del = bm25_store.delete_repo(repo_id)
            c_del = cloner.delete_repo(repo_id)
            return v_del or b_del or c_del

        await asyncio.to_thread(_delete_job)
        return DeleteRepoResponse(
            repo_id=repo_id,
            status="deleted",
            message=f"Repository '{repo_id}' removed from vector database, BM25 index, and disk storage."
        )
    except Exception as err:
        logger.error("Error deleting repository %s: %s", repo_id, err)
        raise HTTPException(status_code=500, detail=str(err))
