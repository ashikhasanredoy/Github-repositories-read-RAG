from typing import List, Optional
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

class RepoListResponse(BaseModel):
    repos: List[str]
    count: int
