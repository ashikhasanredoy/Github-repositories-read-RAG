from typing import List, Optional, Any
from pydantic import BaseModel, Field

class IndexRepoRequest(BaseModel):
    repo_source: str = Field(..., description="GitHub URL or local folder path")
    force_reindex: bool = Field(False, description="Reclone and re-index from scratch")

class IndexRepoResponse(BaseModel):
    repo_id: str
    repo_name: str
    source: str
    total_files: int
    total_chunks: int
    status: str
    message: str

class QueryRequest(BaseModel):
    repo_id: str = Field(..., description="Target repository ID")
    question: str = Field(..., description="Codebase question")
    model_name: Optional[str] = Field(None, description="Ollama model override")
    conversation_id: Optional[str] = Field(None, description="Existing or new conversation ID")
    turn_id: Optional[str] = Field(None, description="Unique turn ID for this question/answer pair")

class SourceCitation(BaseModel):
    file_path: str
    language: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    score: float
    snippet: str

class QueryResponse(BaseModel):
    repo_id: str
    question: str
    query_intent: str
    answer: str
    sources: List[SourceCitation]
    trace_steps: List[str]
    conversation_id: Optional[str] = None
    turn_id: Optional[str] = None

class RepoListResponse(BaseModel):
    repos: List[str]
    count: int

class DeleteRepoResponse(BaseModel):
    repo_id: str
    status: str
    message: str

# -------------------------------------------------------------
# Conversation & Turn History Schemas
# -------------------------------------------------------------

class ConversationSummary(BaseModel):
    conversation_id: str
    title: str
    repo_id: Optional[str] = ""
    created_at: str
    updated_at: str
    turn_count: Optional[int] = 0

class ConversationListResponse(BaseModel):
    conversations: List[ConversationSummary]
    count: int

class ChatMessage(BaseModel):
    message_id: str
    conversation_id: str
    turn_id: str
    role: str
    content: str
    model_name: Optional[str] = None
    query_intent: Optional[str] = None
    sources: Optional[List[Any]] = None
    status: str
    created_at: str

class ConversationDetailResponse(BaseModel):
    conversation_id: str
    title: str
    repo_id: Optional[str] = ""
    created_at: str
    updated_at: str
    messages: List[ChatMessage]

class ConversationMessagesResponse(BaseModel):
    conversation_id: str
    messages: List[ChatMessage]
    count: int

class RenameConversationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120, description="New conversation title")

class DeleteTurnResponse(BaseModel):
    success: bool
    conversation_id: str
    turn_id: str
    deleted_messages: int
    message: str

class DeleteConversationResponse(BaseModel):
    success: bool
    conversation_id: str
    message: str
