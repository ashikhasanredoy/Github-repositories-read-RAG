from typing import TypedDict, List, Dict, Any

class CodeRAGState(TypedDict):
    question: str
    original_question: str
    repo_id: str
    query_intent: str
    extracted_entities: List[str]
    retrieved_docs: List[Dict[str, Any]]
    is_relevant: bool
    retry_count: int
    context_str: str
    answer: str
    sources: List[Dict[str, Any]]
    trace_steps: List[str]
