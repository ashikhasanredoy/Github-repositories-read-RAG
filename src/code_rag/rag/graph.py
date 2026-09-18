import re
import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from src.code_rag.rag.state import CodeRAGState
from src.code_rag.rag.retriever import HybridRetriever
from src.code_rag.services.ollama import OllamaService

logger = logging.getLogger(__name__)

class CodeRAGGraph:
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        ollama_service: Optional[OllamaService] = None
    ):
        self.retriever = retriever or HybridRetriever()
        self.ollama = ollama_service or OllamaService()

    def analyze_query(self, state: CodeRAGState) -> Dict[str, Any]:
        question = state["question"]
        trace = list(state.get("trace_steps", []))
        trace.append(f"Analyzing query: '{question}'")

        entities = re.findall(r'/[a-zA-Z0-9_\-]+|[a-zA-Z0-9_]+\(\)|[a-zA-Z0-9_]{3,}', question)
        stopwords = {
            "where", "what", "which", "how", "when", "does", "implemented",
            "handle", "works", "work", "find", "show", "tell", "explain",
            "the", "and", "for", "with", "this"
        }
        cleaned_entities = [e for e in entities if e.lower() not in stopwords]

        intent = "code_explanation"
        q_lower = question.lower()
        if any(w in q_lower for w in ["where", "which file", "location", "find"]):
            intent = "symbol_lookup"
        elif any(w in q_lower for w in ["api", "endpoint", "route", "url"]):
            intent = "api_endpoint"

        trace.append(f"Intent: '{intent}' (found {len(cleaned_entities)} potential symbols)")
        return {
            "query_intent": intent,
            "extracted_entities": cleaned_entities,
            "trace_steps": trace
        }

    def retrieve_code(self, state: CodeRAGState) -> Dict[str, Any]:
        trace = list(state.get("trace_steps", []))
        trace.append(f"Hybrid retrieval on repo '{state['repo_id']}'")

        docs = self.retriever.retrieve(
            repo_id=state["repo_id"],
            query=state["question"],
            top_k=3
        )
        context = self.retriever.build_context(docs)
        trace.append(f"Retrieved {len(docs)} chunks")

        return {
            "retrieved_docs": docs,
            "context_str": context,
            "trace_steps": trace
        }

    def grade_relevance(self, state: CodeRAGState) -> Dict[str, Any]:
        docs = state.get("retrieved_docs", [])
        trace = list(state.get("trace_steps", []))
        entities = state.get("extracted_entities", [])
        retry = state.get("retry_count", 0)

        all_text = " ".join([d.get("content", "").lower() for d in docs])
        has_symbol_matches = any(e.lower().strip("()") in all_text for e in entities) if entities else True

        is_relevant = len(docs) > 0 and (has_symbol_matches or retry >= 1)
        trace.append(f"Relevance grade: {'Passed' if is_relevant else 'Retrying with query rewrite'}")

        return {
            "is_relevant": is_relevant,
            "trace_steps": trace
        }

    def rewrite_query(self, state: CodeRAGState) -> Dict[str, Any]:
        trace = list(state.get("trace_steps", []))
        entities = state.get("extracted_entities", [])
        retry = state.get("retry_count", 0) + 1

        rewritten = f"{' '.join(entities)} {state['question']} definition function class"
        trace.append(f"Rewriting query: '{rewritten}'")

        return {
            "question": rewritten,
            "retry_count": retry,
            "trace_steps": trace
        }

    def generate_answer(self, state: CodeRAGState) -> Dict[str, Any]:
        trace = list(state.get("trace_steps", []))
        trace.append("Generating response with OLMo...")

        system_prompt = (
            "You are a principal software engineer and codebase expert.\n"
            "Answer the user's question using ONLY the provided repository context.\n"
            "Do not invent functions or file paths that are not in the context.\n"
            "Always cite the exact file path and line numbers where logic is implemented."
        )

        user_prompt = f"""Repository Context:
----------------------------------------
{state.get('context_str', '')}
----------------------------------------

User Question: {state.get('original_question', state['question'])}

Provide a clear and accurate explanation of the implementation.
Conclude with a bulleted list of source files and line ranges.
"""

        try:
            answer = self.ollama.generate(prompt=user_prompt, system_prompt=system_prompt)
        except Exception as err:
            logger.error("Error generating answer with OLMo: %s", err)
            answer = f"Error generating answer with OLMo: {err}\n\nRetrieved context:\n{state.get('context_str', '')}"

        sources = []
        for d in state.get("retrieved_docs", []):
            m = d.get("metadata", {})
            sources.append({
                "file_path": m.get("rel_path", ""),
                "language": m.get("language", ""),
                "symbol_name": m.get("symbol_name", ""),
                "symbol_type": m.get("symbol_type", ""),
                "start_line": m.get("start_line", 1),
                "end_line": m.get("end_line", 1),
                "score": d.get("rrf_score", 0.0),
                "snippet": d.get("content", "")
            })

        trace.append(f"Answer generated with {len(sources)} source citations.")
        return {
            "answer": answer,
            "sources": sources,
            "trace_steps": trace
        }

    def build(self) -> Any:
        workflow = StateGraph(CodeRAGState)

        workflow.add_node("analyze_query", self.analyze_query)
        workflow.add_node("retrieve_code", self.retrieve_code)
        workflow.add_node("grade_relevance", self.grade_relevance)
        workflow.add_node("rewrite_query", self.rewrite_query)
        workflow.add_node("generate_answer", self.generate_answer)

        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "retrieve_code")
        workflow.add_edge("retrieve_code", "grade_relevance")

        def route_after_grading(s: CodeRAGState) -> str:
            if s.get("is_relevant", True) or s.get("retry_count", 0) >= 1:
                return "generate_answer"
            return "rewrite_query"

        workflow.add_conditional_edges("grade_relevance", route_after_grading, {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query"
        })
        workflow.add_edge("rewrite_query", "retrieve_code")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def stream_rag(self, initial_state: CodeRAGState):
        state = dict(initial_state)
        # 1. Analyze
        state.update(self.analyze_query(state))
        yield {"type": "trace", "step": state["trace_steps"][-1]}

        # 2. Retrieve
        state.update(self.retrieve_code(state))
        yield {"type": "trace", "step": state["trace_steps"][-1]}

        # 3. Grade
        state.update(self.grade_relevance(state))
        yield {"type": "trace", "step": state["trace_steps"][-1]}

        if not state.get("is_relevant", True) and state.get("retry_count", 0) < 1:
            state.update(self.rewrite_query(state))
            yield {"type": "trace", "step": state["trace_steps"][-1]}
            state.update(self.retrieve_code(state))
            yield {"type": "trace", "step": state["trace_steps"][-1]}

        # 4. Stream tokens
        yield {"type": "trace", "step": "Generating response..."}
        system_prompt = (
            "You are a principal software engineer and codebase expert.\n"
            "Answer the user's question concisely using ONLY the provided repository context.\n"
            "Do not invent functions or file paths that are not in the context.\n"
            "Always cite the exact file path and line numbers where logic is implemented."
        )
        user_prompt = f"""Repository Context:
----------------------------------------
{state.get('context_str', '')}
----------------------------------------

User Question: {state.get('original_question', state['question'])}

Provide a clear and concise explanation of the implementation. Conclude with source files and line ranges."""

        sources = []
        for d in state.get("retrieved_docs", []):
            m = d.get("metadata", {})
            sources.append({
                "file_path": m.get("rel_path", ""),
                "language": m.get("language", ""),
                "symbol_name": m.get("symbol_name", ""),
                "symbol_type": m.get("symbol_type", ""),
                "start_line": m.get("start_line", 1),
                "end_line": m.get("end_line", 1),
                "score": d.get("rrf_score", 0.0),
                "snippet": d.get("content", "")
            })

        for token in self.ollama.generate_stream(prompt=user_prompt, system_prompt=system_prompt):
            yield {"type": "token", "content": token}

        yield {
            "type": "done",
            "sources": sources,
            "trace_steps": state["trace_steps"]
        }
