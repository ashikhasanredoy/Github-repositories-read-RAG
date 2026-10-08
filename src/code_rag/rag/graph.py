import re
import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from src.code_rag.config import settings
from src.code_rag.rag.state import CodeRAGState
from src.code_rag.rag.retriever import HybridRetriever
from src.code_rag.services.ollama import OllamaService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert AI code assistant and educator analyzing source code.

CRITICAL RULES:
1. LINE-BY-LINE CODE EXPLANATION: When the user provides a code snippet or asks for a line-by-line breakdown, systematically explain every line (or concise logical group of lines) with line numbers, syntax details, variable mutations, logic flow, and execution traces.
2. SPECIFIC FILE/CODE REQUESTS: When asked to show/explain a repository file, output the 100% COMPLETE, EXACT, UNTRUNCATED CODE inside a fenced code block with appropriate language tag, followed by an in-depth breakdown.
3. FILE STRUCTURE REQUESTS: Output the ASCII tree inside a ```markdown ... ``` block followed by bulleted descriptions.
4. Base repository answers strictly and accurately on the provided repository context."""

def extract_user_code_from_query(question: str) -> Optional[Dict[str, Any]]:
    """Extracts explicit code blocks or multi-line pasted code snippets from user queries."""
    if not question:
        return None

    # 1. Check for fenced code blocks ```lang\ncode\n```
    fenced_match = re.search(r'```([a-zA-Z0-9_+\-]*)\s*\n([\s\S]+?)```', question)
    if fenced_match:
        lang = fenced_match.group(1).lower() or "python"
        code = fenced_match.group(2).strip()
        if len(code) > 5:
            # Strip code block to get surrounding instruction
            text_around = question.replace(fenced_match.group(0), "").strip()
            return {
                "code": code,
                "language": lang,
                "user_instruction": text_around or "Explain this code line by line."
            }

    # 2. Check for multi-line pasted code without markdown fences
    lines = question.splitlines()
    code_keywords = (
        "def ", "class ", "import ", "from ", "function ", "const ", "let ", "var ",
        "return ", "for ", "while ", "if ", "elif ", "else:", "print(", "console.",
        "public ", "private ", "protected ", "static ", "void ", "int ", "float ",
        "double ", "bool ", "String ", "#include", "using namespace", "package ",
        "SELECT ", "INSERT ", "UPDATE ", "DELETE ", "CREATE TABLE", "WHERE ",
        "<html>", "<div", "<template", "export default", "async ", "await ", "fn ",
        "struct ", "impl ", "trait ", "enum ", "type ", "interface ", "lambda ",
        "SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY"
    )

    code_lines = []
    text_lines = []
    code_score = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if code_lines:
                code_lines.append(line)
            continue
        is_code = any(stripped.startswith(kw) for kw in code_keywords) or (
            any(sym in stripped for sym in ["=>", "{", "}", ";", "==", "!=", "+=", "-="]) and len(stripped) > 3
        ) or (
            line.startswith("    ") or line.startswith("\t")
        )
        if is_code:
            code_score += 1
            code_lines.append(line)
        else:
            text_lines.append(line)

    if code_score >= 2 and len(code_lines) >= 2:
        return {
            "code": "\n".join(code_lines).strip(),
            "language": "python",
            "user_instruction": "\n".join(text_lines).strip() or "Explain this code line by line."
        }

    return None

def build_line_by_line_prompt(user_code: str, user_instruction: str = "", language: str = "python") -> str:
    lines = user_code.strip().splitlines()
    numbered_lines = []
    for idx, l in enumerate(lines, 1):
        numbered_lines.append(f"Line {idx:2d} | {l}")
    numbered_str = "\n".join(numbered_lines)

    return f"""You are an elite software engineer and educator. 
The user provided the following source code and requested a comprehensive, crystal-clear LINE-BY-LINE breakdown:

Provided Code with Line Numbers:
----------------------------------------
{numbered_str}
----------------------------------------

User Request: {user_instruction or 'Explain this code line by line in detail.'}

Provide your explanation formatted in clean, professional markdown according to this structure:

### 🎯 Overview & Purpose
A concise high-level summary of what this code does, its main purpose, expected inputs, and output.

### 🔍 Detailed Line-by-Line Breakdown
Walk through every line (or concise logical group of lines) in order from Line 1 to Line {len(lines)}:
- **`Line X`**: `exact line code`
  - **Syntax & Mechanism**: Explain the language constructs, keywords, operators, and functions used.
  - **Role & State**: Explain how variables change, what conditions are evaluated, or what is returned.

### 💡 Step-by-Step Logic Flow & Example Trace
Provide a concrete walkthrough with sample input values (e.g. tracing step-by-step state changes).

### ⚡ Complexity & Key Takeaways
- **Time Complexity**: $O(...)$ with rationale.
- **Space Complexity**: $O(...)$ with rationale.
- **Best Practices / Key Insights**: Any edge cases, potential pitfalls, or idiomatic improvements.
"""

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

        # 1. Check for user-provided code
        code_info = extract_user_code_from_query(question)
        if code_info:
            line_count = len(code_info["code"].splitlines())
            trace.append(f"Detected user-provided code snippet ({line_count} lines). Initializing line-by-line explainer.")
            return {
                "query_intent": "user_code_line_by_line",
                "extracted_entities": [],
                "user_code_snippet": code_info,
                "trace_steps": trace
            }

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
        elif any(w in q_lower for w in ["line by line", "line-by-line", "break down line", "explain line"]):
            intent = "line_by_line_explanation"

        trace.append(f"Intent: '{intent}' (found {len(cleaned_entities)} potential symbols)")
        return {
            "query_intent": intent,
            "extracted_entities": cleaned_entities,
            "trace_steps": trace
        }

    def retrieve_code(self, state: CodeRAGState) -> Dict[str, Any]:
        trace = list(state.get("trace_steps", []))
        
        # If user provided their own code snippet, skip repository search
        if state.get("user_code_snippet"):
            trace.append("User provided direct code snippet. Skipping repository search.")
            return {
                "retrieved_docs": [],
                "context_str": "",
                "trace_steps": trace
            }

        trace.append(f"Hybrid retrieval on repo '{state['repo_id']}'")

        docs = self.retriever.retrieve(
            repo_id=state["repo_id"],
            query=state["question"],
            top_k=settings.FINAL_TOP_K
        )
        context = self.retriever.build_context(docs, query=state.get("original_question", state["question"]))
        trace.append(f"Retrieved {len(docs)} chunks")

        return {
            "retrieved_docs": docs,
            "context_str": context,
            "trace_steps": trace
        }

    def grade_relevance(self, state: CodeRAGState) -> Dict[str, Any]:
        trace = list(state.get("trace_steps", []))
        
        if state.get("user_code_snippet"):
            trace.append("Relevance grade: Passed (Direct user code).")
            return {
                "is_relevant": True,
                "trace_steps": trace
            }

        docs = state.get("retrieved_docs", [])
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

    @staticmethod
    def _build_user_prompt(state: CodeRAGState) -> str:
        docs = state.get("retrieved_docs", [])
        has_full_file = any(d.get("metadata", {}).get("is_full_file", False) for d in docs)
        question = state.get("original_question", state["question"])
        intent = state.get("query_intent", "code_explanation")

        instructions = "Provide a direct, precise, and helpful answer to the user's question based strictly on the repository context above."
        if intent == "line_by_line_explanation":
            instructions = (
                "CRITICAL INSTRUCTION: The user requested a LINE-BY-LINE explanation of the code.\n"
                "1. Provide a concise Overview & Purpose of the code.\n"
                "2. Walk through EVERY line (or concise logical group) with line numbers, syntax mechanisms, variable mutations, and logic flow.\n"
                "3. Provide a Step-by-Step Execution Trace with sample inputs.\n"
                "4. State Time & Space Complexity ($O(...)$) and best practice insights."
            )
        elif has_full_file:
            instructions = (
                "CRITICAL INSTRUCTION: The user asked for a file or code, and the complete source file is provided in the repository context above.\n"
                "1. Output the COMPLETE, EXACT, 100% UNTRUNCATED CODE from line 1 to the end in a fenced code block with appropriate language tag.\n"
                "2. DO NOT skip or omit any lines (no `...` or placeholders). Output the entire code same to same.\n"
                "3. Immediately after the code block, provide a full in-depth explanation and description of that code (purpose, imports, functions, classes, logic flow)."
            )

        return f"""Repository Context:
----------------------------------------
{state.get('context_str', '')}
----------------------------------------

User Question: {question}

{instructions}"""

    def generate_answer(self, state: CodeRAGState) -> Dict[str, Any]:
        trace = list(state.get("trace_steps", []))
        trace.append("Generating response...")

        # 1. User-provided code line-by-line explanation
        if state.get("user_code_snippet"):
            code_info = state["user_code_snippet"]
            prompt = build_line_by_line_prompt(
                user_code=code_info["code"],
                user_instruction=code_info.get("user_instruction", ""),
                language=code_info.get("language", "python")
            )
            try:
                answer = self.ollama.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
            except Exception as err:
                logger.error("Error generating line-by-line explanation: %s", err)
                answer = f"Error generating line-by-line explanation: {err}"

            trace.append("Line-by-line code explanation generated.")
            return {
                "answer": answer,
                "sources": [],
                "trace_steps": trace
            }

        docs = state.get("retrieved_docs", [])
        full_file_doc = next((d for d in docs if d.get("metadata", {}).get("is_full_file", False)), None)
        question = state.get("original_question", state["question"])

        if full_file_doc:
            rel_p = full_file_doc.get("metadata", {}).get("rel_path", "")
            lang = full_file_doc.get("metadata", {}).get("language", "python")
            raw_content = full_file_doc.get("content", "")
            
            # Instant exact code block (ONLY ONCE)
            answer_parts = [f"Here is the complete, untruncated source code for `{rel_p}`:\n\n```{lang}\n{raw_content}\n```\n\n### 📝 Explanation & Breakdown:\n\n"]
            
            # Fast explanation via LLM
            imports_and_structure = "\n".join([
                line for line in raw_content.splitlines()[:60]
                if any(line.strip().startswith(kw) for kw in ["import ", "from ", "export ", "def ", "class ", "function ", "const "])
            ])
            summary_prompt = (
                f"File: `{rel_p}`\n"
                f"Symbols/Imports Overview:\n{imports_and_structure or raw_content[:300]}\n\n"
                f"User Question: {question}\n\n"
                f"CRITICAL INSTRUCTION: The complete source code has ALREADY been displayed to the user in the block above.\n"
                f"DO NOT repeat, re-output, or reprint any code blocks or file contents.\n"
                f"Directly start your answer with the explanation breakdown:\n"
                f"- **Summary & Purpose**: High-level overview of what this file does.\n"
                f"- **Key Components & Functions**: What each main part does.\n"
                f"- **Logic Flow**: How data and control flow through this file."
            )
            try:
                expl = self.ollama.generate(prompt=summary_prompt, system_prompt=SYSTEM_PROMPT)
                answer_parts.append(expl)
            except Exception:
                answer_parts.append(f"This file `{rel_p}` contains the complete implementation shown above.")
            
            answer = "".join(answer_parts)
        else:
            system_prompt = SYSTEM_PROMPT
            user_prompt = self._build_user_prompt(state)
            try:
                answer = self.ollama.generate(prompt=user_prompt, system_prompt=system_prompt)
            except Exception as err:
                logger.error("Error generating answer: %s", err)
                answer = f"Error generating answer: {err}\n\nRetrieved context:\n{state.get('context_str', '')}"

        sources = []
        for d in docs:
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

        trace.append("Answer generated.")
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

        # 2. Check for direct user code
        if state.get("user_code_snippet"):
            yield {"type": "trace", "step": "Generating line-by-line breakdown of provided code..."}
            code_info = state["user_code_snippet"]
            prompt = build_line_by_line_prompt(
                user_code=code_info["code"],
                user_instruction=code_info.get("user_instruction", ""),
                language=code_info.get("language", "python")
            )
            for token in self.ollama.generate_stream(prompt=prompt, system_prompt=SYSTEM_PROMPT):
                yield {"type": "token", "content": token}
            yield {
                "type": "done",
                "sources": [],
                "trace_steps": state["trace_steps"]
            }
            return

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
        
        docs = state.get("retrieved_docs", [])
        full_file_doc = next((d for d in docs if d.get("metadata", {}).get("is_full_file", False)), None)
        question = state.get("original_question", state["question"])

        sources = []
        for d in docs:
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

        if full_file_doc:
            import time
            rel_p = full_file_doc.get("metadata", {}).get("rel_path", "")
            lang = full_file_doc.get("metadata", {}).get("language", "python")
            raw_content = full_file_doc.get("content", "")

            # 1. Stream opening code block
            yield {"type": "token", "content": f"Here is the complete source code for `{rel_p}`:\n\n```{lang}\n"}

            # 2. Stream code line-by-line with smooth ChatGPT typing effect
            lines = raw_content.splitlines(keepends=True)
            total_lines = len(lines)
            delay = max(0.006, min(0.020, 2.0 / max(1, total_lines)))

            for line in lines:
                yield {"type": "token", "content": line}
                time.sleep(delay)

            # 3. Close code block and start breakdown
            yield {"type": "token", "content": "\n```\n\n### 📝 Explanation & Breakdown:\n\n"}

            # 4. Stream fast breakdown
            imports_and_structure = "\n".join([
                line for line in lines[:60]
                if any(line.strip().startswith(kw) for kw in ["import ", "from ", "export ", "def ", "class ", "function ", "const "])
            ])
            summary_prompt = (
                f"File: `{rel_p}`\n"
                f"Symbols/Imports Overview:\n{imports_and_structure or raw_content[:300]}\n\n"
                f"User Question: {question}\n\n"
                f"CRITICAL INSTRUCTION: The complete source code has ALREADY been displayed to the user in the block above.\n"
                f"DO NOT repeat, re-output, or reprint any code blocks or file contents.\n"
                f"Directly start your answer with the explanation breakdown:\n"
                f"- **Summary & Purpose**: High-level overview of what this file does.\n"
                f"- **Key Components & Functions**: What each main part does.\n"
                f"- **Logic Flow**: How data and control flow through this file."
            )
            try:
                for token in self.ollama.generate_stream(prompt=summary_prompt, system_prompt=SYSTEM_PROMPT):
                    yield {"type": "token", "content": token}
            except Exception:
                yield {"type": "token", "content": f"The full file `{rel_p}` is displayed above."}
        else:
            system_prompt = SYSTEM_PROMPT
            user_prompt = self._build_user_prompt(state)
            for token in self.ollama.generate_stream(prompt=user_prompt, system_prompt=system_prompt):
                yield {"type": "token", "content": token}

        yield {
            "type": "done",
            "sources": sources,
            "trace_steps": state["trace_steps"]
        }

    async def stream_rag_async(self, initial_state: CodeRAGState):
        import asyncio
        state = dict(initial_state)

        # 1. Analyze
        state.update(self.analyze_query(state))
        yield {"type": "trace", "step": state["trace_steps"][-1]}

        # 2. Check for direct user code
        if state.get("user_code_snippet"):
            yield {"type": "trace", "step": "Generating line-by-line breakdown of provided code..."}
            code_info = state["user_code_snippet"]
            prompt = build_line_by_line_prompt(
                user_code=code_info["code"],
                user_instruction=code_info.get("user_instruction", ""),
                language=code_info.get("language", "python")
            )
            async for token in self.ollama.generate_stream_async(prompt=prompt, system_prompt=SYSTEM_PROMPT):
                yield {"type": "token", "content": token}
            yield {
                "type": "done",
                "sources": [],
                "trace_steps": state["trace_steps"]
            }
            return

        # 2. Retrieve (run in thread to prevent blocking event loop)
        retrieval_res = await asyncio.to_thread(self.retrieve_code, state)
        state.update(retrieval_res)
        yield {"type": "trace", "step": state["trace_steps"][-1]}

        # 3. Grade
        state.update(self.grade_relevance(state))
        yield {"type": "trace", "step": state["trace_steps"][-1]}

        if not state.get("is_relevant", True) and state.get("retry_count", 0) < 1:
            state.update(self.rewrite_query(state))
            yield {"type": "trace", "step": state["trace_steps"][-1]}
            retrieval_res = await asyncio.to_thread(self.retrieve_code, state)
            state.update(retrieval_res)
            yield {"type": "trace", "step": state["trace_steps"][-1]}

        # 4. Stream tokens
        yield {"type": "trace", "step": "Generating response..."}

        docs = state.get("retrieved_docs", [])
        full_file_doc = next((d for d in docs if d.get("metadata", {}).get("is_full_file", False)), None)
        question = state.get("original_question", state["question"])

        sources = []
        for d in docs:
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

        if full_file_doc:
            rel_p = full_file_doc.get("metadata", {}).get("rel_path", "")
            lang = full_file_doc.get("metadata", {}).get("language", "python")
            raw_content = full_file_doc.get("content", "")

            # 1. Stream opening code block
            yield {"type": "token", "content": f"Here is the complete source code for `{rel_p}`:\n\n```{lang}\n"}

            # 2. Stream code line-by-line with smooth ChatGPT typing effect
            lines = raw_content.splitlines(keepends=True)
            total_lines = len(lines)
            delay = max(0.006, min(0.020, 2.0 / max(1, total_lines)))

            for line in lines:
                yield {"type": "token", "content": line}
                await asyncio.sleep(delay)

            # 3. Close code block and start breakdown
            yield {"type": "token", "content": "\n```\n\n### 📝 Explanation & Breakdown:\n\n"}

            # 4. Stream fast breakdown
            imports_and_structure = "\n".join([
                line for line in lines[:60]
                if any(line.strip().startswith(kw) for kw in ["import ", "from ", "export ", "def ", "class ", "function ", "const "])
            ])
            summary_prompt = (
                f"File: `{rel_p}`\n"
                f"Symbols/Imports Overview:\n{imports_and_structure or raw_content[:300]}\n\n"
                f"User Question: {question}\n\n"
                f"CRITICAL INSTRUCTION: The complete source code has ALREADY been displayed to the user in the block above.\n"
                f"DO NOT repeat, re-output, or reprint any code blocks or file contents.\n"
                f"Directly start your answer with the explanation breakdown:\n"
                f"- **Summary & Purpose**: High-level overview of what this file does.\n"
                f"- **Key Components & Functions**: What each main part does.\n"
                f"- **Logic Flow**: How data and control flow through this file."
            )
            try:
                async for token in self.ollama.generate_stream_async(prompt=summary_prompt, system_prompt=SYSTEM_PROMPT):
                    yield {"type": "token", "content": token}
            except Exception:
                yield {"type": "token", "content": f"The full file `{rel_p}` is displayed above."}
        else:
            system_prompt = SYSTEM_PROMPT
            user_prompt = self._build_user_prompt(state)
            async for token in self.ollama.generate_stream_async(prompt=user_prompt, system_prompt=SYSTEM_PROMPT):
                yield {"type": "token", "content": token}

        yield {
            "type": "done",
            "sources": sources,
            "trace_steps": state["trace_steps"]
        }
