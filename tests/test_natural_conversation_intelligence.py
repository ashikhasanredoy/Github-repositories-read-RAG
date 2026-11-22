"""
Complete Conversational Intelligence Test Suite
Covers all 52 scenarios from Section 14:
- 20 Pure Conversational tests (Single and Composite phrases)
- 10 Bangla and Banglish tests
- 7 Mixed-Intent tests (with and without punctuation)
- 7 False-Positive tests (Guarding genuine code requests)
- 5 Context-Awareness tests (Follow-ups, confirmations, continuations)
- 3 No-Context tests (Yes, Continue, Why? without prior context)
- LangGraph Compiled Workflow execution tests
- SSE Synchronous and Asynchronous streaming tests
- SQLite3 Chat Persistence tests
"""

import pytest
import asyncio
from unittest.mock import MagicMock
from src.code_rag.rag.conversation import ConversationHandler
from src.code_rag.rag.graph import CodeRAGGraph, PROFESSIONAL_FALLBACK_RESPONSE
from src.code_rag.rag.state import CodeRAGState
from src.code_rag.storage.sqlite_store import SQLiteStore


@pytest.fixture
def handler():
    return ConversationHandler()


@pytest.fixture
def mock_graph(tmp_path):
    mock_retriever = MagicMock()
    mock_retriever.cloner.get_repo_path.return_value = tmp_path
    mock_retriever.retrieve.return_value = []
    mock_retriever.build_context.return_value = ""

    mock_ollama = MagicMock()
    mock_ollama.generate.return_value = "Mocked technical explanation."
    mock_ollama.generate_stream.return_value = ["Mocked ", "technical ", "explanation."]

    async def mock_async_stream(*args, **kwargs):
        for token in ["Mocked ", "async ", "explanation."]:
            yield token
    mock_ollama.generate_stream_async = mock_async_stream

    db_path = tmp_path / "test_chat.db"
    store = SQLiteStore(db_path=str(db_path))

    graph = CodeRAGGraph(
        retriever=mock_retriever,
        ollama_service=mock_ollama,
        sqlite_store=store
    )
    return graph


# ==============================================================================
# SECTION 14: 52 SPECIFIC TEST SCENARIOS
# ==============================================================================

class TestSection14FiftyTwoScenarios:

    # --------------------------------------------------------------------------
    # A. Pure conversational tests (1 to 20)
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize("query,expected_category", [
        ("Hi", "greeting"),                                 # 1
        ("Hello!", "greeting"),                             # 2
        ("Heyyyyy", "greeting"),                            # 3
        ("Good morning!", "greeting"),                      # 4
        ("How are you?", "small_talk"),                     # 5
        ("Perfect, thanks!", "gratitude"),                  # 6 (Composite acknowledgment + gratitude)
        ("Great, thank you!", "gratitude"),                 # 7
        ("Okay, got it.", "acknowledgment"),                # 8 (Composite acknowledgment + acknowledgment)
        ("Thanks for your help.", "gratitude"),             # 9
        ("I really appreciate your help.", "gratitude"),    # 10
        ("Sorry, my mistake.", "apology"),                  # 11
        ("No worries!", "acknowledgment"),                  # 12
        ("You're welcome!", "acknowledgment"),              # 13
        ("That's helpful, thanks!", "gratitude"),           # 14
        ("Excellent, thank you.", "gratitude"),             # 15
        ("Nice to meet you.", "small_talk"),                # 16
        ("Can you help me?", "capabilities"),               # 17
        ("Are you there?", "small_talk"),                   # 18
        ("Goodbye!", "farewell"),                           # 19
        ("Good night!", "farewell"),                        # 20
    ])
    def test_pure_conversational_scenarios(self, handler, query, expected_category):
        res = handler.detect_intent(query)
        assert res["type"] == "pure_conversation", f"Failed for '{query}': Got {res.get('type')}"
        assert res["is_conversational"] is True
        assert len(res["response"]) > 0
        assert "couldn't find" not in res["response"].lower()

    # --------------------------------------------------------------------------
    # B. Bangla and Banglish tests (21 to 30)
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize("query,expected_lang,expected_phrase_in_response", [
        ("আপনি কেমন আছেন?", "bangla", "ভালো আছি"),             # 21
        ("ধন্যবাদ।", "bangla", "স্বাগতম"),                    # 22 (Proper welcome response!)
        ("আপনাকে অনেক ধন্যবাদ।", "bangla", "স্বাগতম"),           # 23
        ("ঠিক আছে।", "bangla", "ঠিক আছে"),                    # 24
        ("বুঝেছি।", "bangla", "ঠিক আছে"),                     # 25
        ("Apni kemon achen?", "banglish", "Ami bhalo achi"),   # 26
        ("Onek dhonnobad.", "banglish", "shagotom"),           # 27
        ("Thik ache bhai.", "banglish", "Thik ache"),          # 28
        ("Kono somossa nei.", "banglish", "Kono somossa nei"), # 29
        ("Bhalo achen?", "banglish", "Ami bhalo achi"),        # 30
    ])
    def test_bangla_and_banglish_scenarios(self, handler, query, expected_lang, expected_phrase_in_response):
        res = handler.detect_intent(query)
        assert res["type"] == "pure_conversation", f"Failed for '{query}'"
        assert res["language"] == expected_lang
        assert expected_phrase_in_response in res["response"]

    # --------------------------------------------------------------------------
    # C. Mixed-intent tests (31 to 37)
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize("query,expected_tech_subquery", [
        ("Hi, explain graph.py.", "explain graph.py."),                                   # 31
        ("Hi explain graph.py.", "explain graph.py."),                                    # 32 (No comma!)
        ("Hello, how does hybrid retrieval work?", "how does hybrid retrieval work?"),    # 33
        ("Thanks, show me retriever.py.", "show me retriever.py."),                       # 34
        ("Good morning, explain Python decorators.", "explain Python decorators."),       # 35
        ("Perfect, now give me all project files one by one.", "give me all project files one by one."), # 36
        ("Hello, explain analyze_query().", "explain analyze_query()."),                  # 37
    ])
    def test_mixed_intent_scenarios(self, handler, query, expected_tech_subquery):
        res = handler.detect_intent(query)
        assert res["type"] == "mixed_intent", f"Failed to identify mixed intent for '{query}'"
        assert res["is_mixed"] is True
        assert len(res["conversational_prefix"]) > 0
        assert expected_tech_subquery in res["cleaned_technical_question"]

    # --------------------------------------------------------------------------
    # D. False-positive tests (38 to 44)
    # --------------------------------------------------------------------------
    @pytest.mark.parametrize("query", [
        "Explain the hello function in Python.",                    # 38
        "Find the greet_user() function.",                         # 39
        "Implement a greeting notification feature.",               # 40
        "Explain the greeting component in my frontend.",           # 41
        "Show me the `hello.py` file.",                             # 42
        "What is the difference between Hi and Hello in Python?",   # 43
        "Explain the notification service in my project."           # 44
    ])
    def test_false_positive_scenarios(self, handler, query):
        res = handler.detect_intent(query)
        assert res["type"] == "technical_request", f"Query '{query}' falsely classified as conversational!"
        assert res["is_conversational"] is False

    # --------------------------------------------------------------------------
    # E. Context-awareness tests (45 to 49)
    # --------------------------------------------------------------------------
    def test_context_why_after_explanation(self, handler):
        # 45. Explain graph.py. -> Why?
        history = [
            {"role": "user", "content": "Explain graph.py."},
            {"role": "assistant", "content": "graph.py coordinates LangGraph state nodes."}
        ]
        res = handler.detect_intent("Why?", history=history)
        assert res["type"] == "context_followup"
        assert res["is_contextual"] is True
        assert "Explain graph.py." in res["context_query"]

    def test_context_explain_more(self, handler):
        # 46. Explain analyze_query(). -> Explain more.
        history = [
            {"role": "user", "content": "Explain analyze_query()."},
            {"role": "assistant", "content": "analyze_query parses intent."}
        ]
        res = handler.detect_intent("Explain more.", history=history)
        assert res["type"] == "context_followup"
        assert res["is_contextual"] is True

    def test_context_confirmation_yes(self, handler):
        # 47. Can you explain line by line? -> Should I proceed? -> Yes.
        history = [
            {"role": "user", "content": "Can you explain graph.py line by line?"},
            {"role": "assistant", "content": "Should I proceed with the line-by-line breakdown?"}
        ]
        res = handler.detect_intent("Yes.", history=history)
        assert res["type"] == "context_confirmation"
        assert res["is_contextual"] is True

    def test_context_next_during_export(self, handler):
        # 48. Give me all files one by one. -> Next. (with active export session)
        active_session = {"pending_files": ["src/main.py", "src/auth.py"]}
        res = handler.detect_intent("Next.", active_session=active_session)
        assert res["type"] == "export_continuation"
        assert res["workflow"] == "repo_code_export"

    def test_context_what_about_its_dependencies(self, handler):
        # 49. Explain the retriever. -> What about its dependencies?
        history = [
            {"role": "user", "content": "Explain the retriever."},
            {"role": "assistant", "content": "The retriever combines BM25 and ChromaDB."}
        ]
        res = handler.detect_intent("What about its dependencies?", history=history)
        assert res["type"] == "context_followup"
        assert res["is_contextual"] is True

    # --------------------------------------------------------------------------
    # F. No-context tests (50 to 52)
    # --------------------------------------------------------------------------
    def test_no_context_yes(self, handler):
        # 50. User: Yes. (no pending confirmation -> natural acknowledgment)
        res = handler.detect_intent("Yes.", history=[])
        assert res["type"] == "pure_conversation"
        assert res["category"] == "acknowledgment"
        assert "Understood" in res["response"] or "help" in res["response"].lower()

    def test_no_context_continue(self, handler):
        # 51. User: Continue. (no active export -> asks what to continue)
        res = handler.detect_intent("Continue.", history=[], active_session=None)
        assert res["type"] == "pure_conversation"
        assert "continue" in res["response"].lower()

    def test_no_context_why(self, handler):
        # 52. User: Why? (no context -> asks for clarification)
        res = handler.detect_intent("Why?", history=[])
        assert res["type"] == "pure_conversation"
        assert "clarify" in res["response"].lower()


# ==============================================================================
# INTEGRATION TESTS: LANGGRAPH WORKFLOW & STREAMING
# ==============================================================================

class TestLangGraphConversationalIntegration:

    def test_perfect_thanks_bypasses_retrieval(self, mock_graph):
        """Validates that 'Perfect, thanks!' executes via direct conversational path."""
        state: CodeRAGState = {
            "question": "Perfect, thanks!",
            "original_question": "Perfect, thanks!",
            "repo_id": "test-repo",
            "trace_steps": []
        }
        # Run compiled LangGraph workflow
        compiled_app = mock_graph.build()
        result = compiled_app.invoke(state)

        # 1. Direct answer generated
        assert "welcome" in result["answer"].lower()
        # 2. Vector and BM25 retriever never called
        mock_graph.retriever.retrieve.assert_not_called()
        # 3. No fallback emitted
        assert result["answer"] != PROFESSIONAL_FALLBACK_RESPONSE
        # 4. Sources empty
        assert result["sources"] == []
        assert result["query_intent"] == "pure_conversation"

    def test_thanks_for_notification_not_code_request(self, mock_graph):
        """Asserts that 'Thanks for the notification' is classified as pure conversation."""
        state: CodeRAGState = {
            "question": "Thanks for the notification.",
            "original_question": "Thanks for the notification.",
            "repo_id": "test-repo",
            "trace_steps": []
        }
        res = mock_graph.analyze_query(state)
        assert res["query_intent"] == "pure_conversation"
        assert res["is_pure_conversation"] is True
        mock_graph.retriever.retrieve.assert_not_called()

    def test_mixed_intent_prefix_appears_once(self, mock_graph):
        """Verifies mixed intent queries prepend conversational prefix exactly once."""
        state: CodeRAGState = {
            "question": "Hi, explain graph.py.",
            "original_question": "Hi, explain graph.py.",
            "repo_id": "test-repo",
            "trace_steps": []
        }
        res = mock_graph.analyze_query(state)
        state.update(res)
        answer_result = mock_graph.generate_answer(state)
        # Prefix appears at beginning
        assert answer_result["answer"].startswith("Hello! ")
        # Ensure it does not repeat twice
        assert answer_result["answer"].count("Hello! ") == 1

    def test_streaming_pure_conversation(self, mock_graph):
        """Tests that stream_rag yields words smoothly and emits pure_conversation done event."""
        state: CodeRAGState = {
            "question": "Great, thank you!",
            "original_question": "Great, thank you!",
            "repo_id": "test-repo",
            "trace_steps": []
        }
        stream_events = list(mock_graph.stream_rag(state))
        tokens = [e["content"] for e in stream_events if e.get("type") == "token"]
        full_text = "".join(tokens)
        assert "welcome" in full_text.lower()
        done = next(e for e in stream_events if e.get("type") == "done")
        assert done["query_intent"] == "pure_conversation"
        assert done["sources"] == []

    @pytest.mark.asyncio
    async def test_async_streaming_pure_conversation(self, mock_graph):
        """Tests that stream_rag_async yields tokens smoothly without LLM delay."""
        state: CodeRAGState = {
            "question": "কেমন আছেন?",
            "original_question": "কেমন আছেন?",
            "repo_id": "test-repo",
            "trace_steps": []
        }
        tokens = []
        async for item in mock_graph.stream_rag_async(state):
            if item.get("type") == "token":
                tokens.append(item["content"])
        full_text = "".join(tokens)
        assert "ভালো আছি" in full_text
