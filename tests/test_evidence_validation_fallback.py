import pytest
from unittest.mock import MagicMock, patch
from src.code_rag.rag.graph import (
    CodeRAGGraph,
    PROFESSIONAL_FALLBACK_RESPONSE,
    is_general_programming_query,
)
from src.code_rag.rag.state import CodeRAGState


def test_professional_fallback_response_text():
    """Verify that the exact required professional fallback response string is used."""
    expected_text = (
        "I couldn't find sufficient information in the indexed repository to provide an accurate answer to your question. "
        "The available source code does not contain enough relevant evidence to verify the requested information.\n\n"
        "To ensure accuracy, I won't make assumptions or generate unsupported information. Please make sure the latest "
        "version of your repository has been indexed, or provide the relevant source file so I can investigate further."
    )
    assert PROFESSIONAL_FALLBACK_RESPONSE == expected_text


def test_is_general_programming_query_detection():
    """Verify classification of general programming queries vs repository inquiries."""
    # General queries
    assert is_general_programming_query("What is a binary search tree in Python?") is True
    assert is_general_programming_query("How does list comprehension work in Python?") is True
    assert is_general_programming_query("Explain what recursion means") is True
    assert is_general_programming_query("Difference between REST and GraphQL") is True

    # Repo-specific queries
    assert is_general_programming_query("Where is process_payment in this repo?", "my_repo") is False
    assert is_general_programming_query("Show me main.py", "my_repo") is False
    assert is_general_programming_query("Explain how retriever.py works", "my_repo") is False
    assert is_general_programming_query("How does this project handle database migrations?", "my_repo") is False


def test_verify_evidence_nonexistent_function():
    """Verify that asking for a nonexistent function marks evidence_status as 'none'."""
    graph = CodeRAGGraph()
    state: CodeRAGState = {
        "question": "Explain how crypto_payout() works in the billing module",
        "original_question": "Explain how crypto_payout() works in the billing module",
        "repo_id": "test_repo",
        "query_intent": "general_qa",
        "query_type": "repository_specific",
        "extracted_entities": ["crypto_payout"],
        "target_files": [],
        "retrieved_docs": [],
    }

    # Mock retriever disk search to return empty
    with patch.object(graph.retriever, "search_disk_for_symbols", return_value=[]), \
         patch.object(graph.retriever, "search_disk_for_keywords", return_value=[]):
        result = graph.verify_evidence(state)
        assert result["evidence_status"] == "none"
        assert result.get("is_relevant") is not True


def test_verify_evidence_nonexistent_file():
    """Verify that asking for a nonexistent file marks evidence_status as 'none'."""
    graph = CodeRAGGraph()
    state: CodeRAGState = {
        "question": "Show me the contents of payment_processor_stripe.py",
        "original_question": "Show me the contents of payment_processor_stripe.py",
        "repo_id": "test_repo",
        "query_intent": "full_file_code",
        "query_type": "repository_specific",
        "extracted_entities": [],
        "target_files": ["payment_processor_stripe.py"],
        "retrieved_docs": [],
    }

    with patch.object(graph.retriever, "_find_full_files_on_disk", return_value=[]):
        result = graph.verify_evidence(state)
        assert result["evidence_status"] == "none"
        assert result.get("is_relevant") is not True


def test_verify_evidence_sufficient():
    """Verify that valid retrieved docs matching the query mark evidence_status as 'sufficient'."""
    graph = CodeRAGGraph()
    state: CodeRAGState = {
        "question": "How does the hybrid retriever work?",
        "original_question": "How does the hybrid retriever work?",
        "repo_id": "test_repo",
        "query_intent": "general_qa",
        "query_type": "repository_specific",
        "extracted_entities": ["hybrid", "retriever"],
        "target_files": [],
        "retrieved_docs": [
            {
                "chunk_id": "test_1",
                "content": "class HybridRetriever:\n    def retrieve(self, query):\n        pass",
                "metadata": {"rel_path": "src/retriever.py", "start_line": 1, "end_line": 10},
                "rrf_score": 0.85
            }
        ],
    }

    result = graph.verify_evidence(state)
    assert result["evidence_status"] == "sufficient"


def test_generate_answer_fallback_on_none_evidence():
    """When evidence_status is 'none', generate_answer returns exact fallback response without calling LLM."""
    graph = CodeRAGGraph()
    state: CodeRAGState = {
        "question": "Where is the bitcoin transaction logic?",
        "original_question": "Where is the bitcoin transaction logic?",
        "repo_id": "test_repo",
        "query_intent": "general_qa",
        "query_type": "repository_specific",
        "evidence_status": "none",
        "retrieved_docs": [],
    }

    result = graph.generate_answer(state)
    assert result["answer"] == PROFESSIONAL_FALLBACK_RESPONSE
    # No sources should be cited
    assert result.get("sources", []) == []


def test_stream_rag_async_fallback_on_none_evidence():
    """Verify stream_rag_async emits verification trace and yields exact fallback chunks when evidence is 'none'."""
    import asyncio
    graph = CodeRAGGraph()

    async def run_stream():
        events = []
        with patch.object(graph.retriever, "retrieve", return_value=[]), \
             patch.object(graph.retriever, "search_disk_for_symbols", return_value=[]), \
             patch.object(graph.retriever, "search_disk_for_keywords", return_value=[]):
            async for event in graph.stream_rag_async({
                "question": "How is the imaginary blockchain integrated?",
                "repo_id": "test_repo"
            }):
                events.append(event)
        return events

    events = asyncio.run(run_stream())
    event_types = [e.get("type") for e in events]
    assert "trace" in event_types
    assert "token" in event_types
    assert "done" in event_types

    # Collect streamed tokens
    tokens = "".join([e["content"] for e in events if e.get("type") == "token"])
    assert tokens == PROFESSIONAL_FALLBACK_RESPONSE

    # Find done event payload
    done_payload = [e for e in events if e.get("type") == "done"][0]
    assert done_payload["sources"] == []
    assert done_payload["query_intent"] == "no_evidence_fallback"
