import re
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from src.code_rag.config import settings
from src.code_rag.rag.state import CodeRAGState
from src.code_rag.rag.retriever import HybridRetriever, detect_structure_intent
from src.code_rag.services.ollama import OllamaService
from src.code_rag.rag.export_mode import (
    detect_export_intent,
    build_export_manifest_and_tree,
    sanitize_sensitive_content,
    EXPORT_BATCH_SIZE
)

logger = logging.getLogger(__name__)

# Exact professional fallback response required when sufficient evidence cannot be found
PROFESSIONAL_FALLBACK_RESPONSE = (
    "I couldn't find sufficient information in the indexed repository to provide an accurate answer to your question. "
    "The available source code does not contain enough relevant evidence to verify the requested information.\n\n"
    "To ensure accuracy, I won't make assumptions or generate unsupported information. Please make sure the latest "
    "version of your repository has been indexed, or provide the relevant source file so I can investigate further."
)

SYSTEM_PROMPT = """You are an expert AI code assistant and educator analyzing source code.

CRITICAL RULES FOR ACCURACY & ZERO HALLUCINATION:
1. STRICT FACTUAL ACCURACY: You must NEVER invent, assume, or fabricate code, functions, classes, methods, endpoints, parameters, variables, file paths, or repository behaviors. Every repository-specific claim must be strictly grounded in the provided Repository Context.
2. CITATIONS: Reference only genuine file paths and line numbers that appear in the provided context. Never invent imaginary citations.
3. PARTIAL EVIDENCE: If the repository context only partially answers the user's question, explain ONLY what is directly verified in the code, and explicitly declare what could not be found or verified. Do not fill gaps with guesswork.
4. GENERAL PROGRAMMING QUERIES: If the user is asking a general software engineering or programming concept (e.g. syntax, algorithms, patterns) rather than asking about their specific repository, provide a clear explanation using standard programming knowledge, explicitly noting this is general programming context.
5. LINE-BY-LINE EXPLANATION: When line-by-line explanation is requested, explain the provided code sequentially with line numbers, syntax details, variable roles, logic flow, and execution traces.
6. SPECIFIC FILE RETRIEVAL: When asked for a full file, output the exact untruncated code in a fenced block, followed by an explanation breakdown.
7. REPOSITORY STRUCTURE: Output directory trees inside ```text ... ``` blocks using GitHub-style formatting (├──, └──, │). Never invent files or directories."""

def is_general_programming_query(query: str, repo_id: str = "") -> bool:
    """
    Distinguishes general software engineering and programming syntax/concepts
    from repository-specific inquiries.
    """
    q = query.strip()
    q_lower = q.lower()

    # If user explicitly references repo/project, it's repository-specific
    repo_indicators = [
        "in my project", "in this project", "in our project",
        "in my repo", "in this repo", "in our repo", "in the repo",
        "in my repository", "in this repository", "in the repository",
        "in this codebase", "in our codebase", "in the codebase",
        "in this app", "in the app", "in this code",
        "where is", "where are", "which file", "which folder",
        "does this repo", "does my project", "does our project",
        "show repo", "file tree", "directory tree", "folder structure",
        "line by line", "line-by-line"
    ]
    if any(ind in q_lower for ind in repo_indicators):
        return False

    # If user refers to a specific source file with an extension, it's repo-specific
    if re.search(r'\b[a-zA-Z0-9_\-./]+\.(?:py|js|ts|tsx|jsx|html|css|json|yaml|yml|md|sql|go|java|c|cpp|h|rs|sh)\b', q_lower):
        return False

    # Common general programming questions and concepts
    general_patterns = [
        r"^what\s+is\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z0-9_#+.\s]{2,50})\??$",
        r"^explain\s+(?:how\s+)?([a-zA-Z0-9_#+.\s]{2,50})\s+(?:works|is\s+used)\??$",
        r"^explain\s+(?:what\s+(?:is|does\s+)?\s*)?([a-zA-Z0-9_#+.\s]{2,50})\s*(?:means|is)?\??$",
        r"^how\s+does\s+([a-zA-Z0-9_#+.\s]{2,50})\s+work(?:\s+in\s+[a-zA-Z0-9_#+]{2,20})?\??$",
        r"^how\s+to\s+([a-zA-Z0-9_#+.\s]{2,50})\s+in\s+([a-zA-Z0-9_#+]{2,20})\??$",
        r"^(?:what\s+is\s+the\s+)?difference\s+between\s+([a-zA-Z0-9_#+.\s]{2,40})\s+and\s+([a-zA-Z0-9_#+.\s]{2,40})\??$",
        r"^(?:why\s+use|when\s+to\s+use)\s+([a-zA-Z0-9_#+.\s]{2,50})\??$",
    ]

    for pat in general_patterns:
        m = re.match(pat, q_lower)
        if m:
            subject = m.group(1).strip()
            # If the subject is not a specific file or symbol like main, run, cloner
            if subject not in {"main", "run", "cloner", "retriever", "graph", "state", "server"}:
                return True

    common_concepts = [
        "python", "fastapi", "django", "flask", "react", "vue", "angular", "node", "nodejs",
        "javascript", "typescript", "c++", "rust", "golang", "docker", "kubernetes",
        "database index", "b-tree", "acid", "sql vs nosql", "rest api", "graphql", "grpc",
        "binary search", "recursion", "dynamic programming", "bubble sort", "quick sort",
        "asynchronous programming", "async await", "coroutine", "event loop",
        "multithreading", "multiprocessing", "gil", "garbage collection",
        "object-oriented programming", "oop", "polymorphism", "inheritance", "encapsulation",
        "dependency injection", "solid principles", "design patterns", "singleton", "factory pattern"
    ]
    for concept in common_concepts:
        if q_lower == concept or q_lower == f"what is {concept}" or q_lower == f"explain {concept}" or f"what is {concept}" in q_lower or f"explain {concept}" in q_lower or f"how does {concept} work" in q_lower:
            return True

    return False

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

### Overview & Purpose
A concise high-level summary of what this code does, its main purpose, expected inputs, and output.

### Detailed Line-by-Line Breakdown
Walk through every line (or concise logical group of lines) in order from Line 1 to Line {len(lines)}:
- **`Line X`**: `exact line code`
  - **Syntax & Mechanism**: Explain the language constructs, keywords, operators, and functions used.
  - **Role & State**: Explain how variables change, what conditions are evaluated, or what is returned.

### Step-by-Step Logic Flow & Example Trace
Provide a concrete walkthrough with sample input values (e.g. tracing step-by-step state changes).

### Complexity & Key Takeaways
- **Time Complexity**: $O(...)$ with rationale.
- **Space Complexity**: $O(...)$ with rationale.
- **Best Practices / Key Insights**: Any edge cases, potential pitfalls, or idiomatic improvements.
"""

class CodeRAGGraph:
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        ollama_service: Optional[OllamaService] = None,
        sqlite_store: Optional[Any] = None
    ):
        self.retriever = retriever or HybridRetriever()
        self.ollama = ollama_service or OllamaService()
        if sqlite_store is not None:
            self.sqlite_store = sqlite_store
        else:
            try:
                from src.code_rag.storage.sqlite_store import SQLiteStore
                self.sqlite_store = SQLiteStore()
            except Exception:
                self.sqlite_store = None
        self._export_sessions: Dict[str, Dict[str, Any]] = {}

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
                "query_type": "repository_specific",
                "extracted_entities": [],
                "target_files": [],
                "user_code_snippet": code_info,
                "trace_steps": trace
            }

        # 2. Check for Complete Repository Code Export Mode (or continuation)
        conv_id = state.get("conversation_id", "")
        repo_id = state.get("repo_id", "")
        active_session = None
        if self.sqlite_store and conv_id and repo_id:
            try:
                active_session = self.sqlite_store.get_latest_export_session(conv_id, repo_id)
            except Exception:
                pass
        if not active_session and repo_id:
            active_session = self._export_sessions.get(f"{conv_id}_{repo_id}") or self._export_sessions.get(repo_id)

        has_active = bool(active_session and active_session.get("pending_files"))
        export_info = detect_export_intent(question, repo_id=repo_id, has_active_session=has_active)
        if export_info:
            trace.append(
                f"Detected complete repository code export request "
                f"(continuation: {export_info.get('is_continuation')}, readme_pdf_only: {export_info.get('readme_and_pdf_only')})"
            )
            return {
                "query_intent": "repo_code_export",
                "query_type": "repository_specific",
                "extracted_entities": [],
                "target_files": [],
                "export_info": export_info,
                "export_session": active_session if export_info.get("is_continuation") else None,
                "trace_steps": trace
            }

        # 3. Check for repository structure or architecture request
        struct_info = detect_structure_intent(question)
        if struct_info:
            sub = struct_info.get("subfolder", "")
            exts = struct_info.get("filter_exts", [])
            trace.append(f"Detected repository structure / architecture query (subfolder: '{sub}', filters: {exts})")
            return {
                "query_intent": "repository_structure",
                "query_type": "repository_specific",
                "extracted_entities": [],
                "target_files": [],
                "structure_info": struct_info,
                "trace_steps": trace
            }

        # 3. Target file extraction (e.g. main.py, auth_service.py)
        raw_targets = re.findall(r'[\w\-./\\]+\.[\w]+', question)
        target_files = [
            t for t in raw_targets
            if not any(t.endswith(ext) for ext in ['.com', '.org', '.net', '.io', '.gov', '.edu', '.git'])
        ]

        # 4. General programming vs Repository-specific classification
        is_general = is_general_programming_query(question, state.get("repo_id", ""))
        query_type = "general_programming" if is_general else "repository_specific"

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
        elif is_general:
            intent = "general_programming"

        trace.append(f"Query classification: type='{query_type}', intent='{intent}', target_files={target_files}")
        return {
            "query_intent": intent,
            "query_type": query_type,
            "extracted_entities": cleaned_entities,
            "target_files": target_files,
            "trace_steps": trace
        }

    def retrieve_code(self, state: CodeRAGState) -> Dict[str, Any]:
        trace = list(state.get("trace_steps", []))

        # 0. Check for complete repository code export
        if state.get("query_intent") == "repo_code_export":
            trace.append("Complete repository code export requested. Initializing repository file manifest.")
            repo_id = state.get("repo_id", "")
            repo_dir = self.retriever.cloner.get_repo_path(repo_id)
            export_info = state.get("export_info", {})
            readme_pdf_only = export_info.get("readme_and_pdf_only", False)
            is_cont = export_info.get("is_continuation", False)

            session = state.get("export_session")
            if not session or not is_cont:
                tree_text, manifest = build_export_manifest_and_tree(
                    repo_dir,
                    root_name=repo_id,
                    readme_and_pdf_only=readme_pdf_only
                )
                session = {
                    "export_id": f"exp_{int(time.time() * 1000)}",
                    "conversation_id": state.get("conversation_id", f"conv_{int(time.time() * 1000)}"),
                    "repo_id": repo_id,
                    "repository_name": repo_id,
                    "ordered_file_manifest": manifest,
                    "current_file_index": 0,
                    "total_file_count": len(manifest),
                    "completed_files": [],
                    "pending_files": [f["rel_path"] for f in manifest],
                    "failed_files": [],
                    "export_status": "in_progress",
                    "tree_text": tree_text,
                    "readme_and_pdf_only": readme_pdf_only
                }
                if self.sqlite_store:
                    try:
                        self.sqlite_store.save_export_session(session)
                    except Exception as err:
                        logger.warning("Failed to save export session to SQLite: %s", err)
                cache_key = f"{session.get('conversation_id')}_{repo_id}"
                self._export_sessions[cache_key] = session
                self._export_sessions[repo_id] = session

            return {
                "retrieved_docs": [],
                "context_str": "",
                "export_session": session,
                "trace_steps": trace
            }

        # If user provided their own code snippet, skip repository search
        if state.get("user_code_snippet"):
            trace.append("User provided direct code snippet. Skipping repository search.")
            return {
                "retrieved_docs": [],
                "context_str": "",
                "trace_steps": trace
            }

        # If it's a general programming query, skip repository code retrieval
        if state.get("query_type") == "general_programming":
            trace.append("General programming query detected. Skipping repository code retrieval.")
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

        if state.get("query_intent") == "repo_code_export":
            trace.append("Relevance grade: Passed (Complete repository code export).")
            return {
                "is_relevant": True,
                "trace_steps": trace
            }

        if state.get("query_type") == "general_programming":
            trace.append("Relevance grade: Passed (General programming query).")
            return {
                "is_relevant": True,
                "trace_steps": trace
            }

        docs = state.get("retrieved_docs", [])
        if state.get("user_code_snippet") or state.get("query_intent") == "repository_structure" or any(d.get("metadata", {}).get("is_tree") for d in docs):
            trace.append("Relevance grade: Passed (Structure or direct user code).")
            return {
                "is_relevant": True,
                "trace_steps": trace
            }

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

        rewritten = f"{' '.join(entities)} {state['question']} definition function class implementation"
        trace.append(f"Rewriting query: '{rewritten}'")

        return {
            "question": rewritten,
            "retry_count": retry,
            "trace_steps": trace
        }

    def verify_evidence(self, state: CodeRAGState) -> Dict[str, Any]:
        """
        Validates evidence strictly against the repository code.
        Detects nonexistent functions, missing files, or unsupported features,
        preventing AI hallucination and triggering the professional fallback.
        """
        trace = list(state.get("trace_steps", []))
        trace.append("Validating evidence and checking for potential hallucination...")

        # 0. Complete repository code export
        if state.get("query_intent") == "repo_code_export":
            session = state.get("export_session") or {}
            total_files = session.get("total_file_count", 0)
            repo_id = state.get("repo_id", "")
            repo_dir = self.retriever.cloner.get_repo_path(repo_id)
            if repo_dir and repo_dir.exists() and total_files > 0:
                trace.append(f"Evidence validation: Verified {total_files} eligible files in repository '{repo_id}'.")
                return {
                    "evidence_status": "sufficient",
                    "trace_steps": trace
                }
            elif repo_dir and repo_dir.exists() and total_files == 0:
                trace.append("Evidence validation: Repository contains no matching files for requested export criteria.")
                return {
                    "evidence_status": "none",
                    "trace_steps": trace
                }
            else:
                trace.append(f"Evidence validation: Repository directory '{repo_id}' not found.")
                return {
                    "evidence_status": "none",
                    "trace_steps": trace
                }

        # 1. User-provided code snippets or directory tree
        if state.get("user_code_snippet") or state.get("query_intent") in ("user_code_line_by_line", "repository_structure"):
            trace.append("Evidence validation: Direct user code or repository structure verified.")
            return {
                "evidence_status": "sufficient",
                "trace_steps": trace
            }

        # 2. General programming queries
        if state.get("query_type") == "general_programming":
            trace.append("Query type: General programming knowledge query. Skipping repository evidence check.")
            return {
                "evidence_status": "sufficient",
                "trace_steps": trace
            }

        # 3. Repository-specific queries
        docs = state.get("retrieved_docs", [])
        question = state.get("original_question") or state.get("question", "")
        target_files = state.get("target_files", [])
        repo_id = state.get("repo_id", "")

        # A. If zero documents retrieved at all
        if not docs:
            trace.append("Evidence validation: FAILED (0 documents retrieved). Insufficient repository evidence.")
            return {
                "evidence_status": "none",
                "trace_steps": trace
            }

        # B. Check if user asked for a specific file (e.g. "show billing.py", "give me auth_service.py")
        if target_files:
            file_matches = []
            for d in docs:
                rel = d.get("metadata", {}).get("rel_path", "").lower()
                for tf in target_files:
                    tf_clean = tf.lower().replace("\\", "/").strip("/")
                    if rel == tf_clean or rel.endswith("/" + tf_clean) or Path(rel).name == tf_clean:
                        file_matches.append(d)
                        break

            if not file_matches:
                disk_files = self.retriever._find_full_files_on_disk(repo_id, target_files)
                if disk_files:
                    trace.append(f"Found target file {disk_files[0]['metadata']['rel_path']} on disk via recovery scan.")
                    docs = disk_files + docs
                    context = self.retriever.build_context(docs, query=question)
                    return {
                        "retrieved_docs": docs,
                        "context_str": context,
                        "evidence_status": "sufficient",
                        "trace_steps": trace
                    }
                else:
                    trace.append(f"Evidence validation: FAILED. Target file(s) '{', '.join(target_files)}' do not exist in the repository.")
                    return {
                        "evidence_status": "none",
                        "retrieved_docs": [],
                        "trace_steps": trace
                    }

        # C. Check if user asked for a specific named function, class, or symbol
        symbol_queries = []
        fn_pattern = re.findall(r'\b([a-zA-Z_]\w*)\s*\(\s*\)', question)
        if fn_pattern:
            symbol_queries.extend(fn_pattern)
        explicit_sym = re.findall(r'\b(?:function|class|method|def)\s+([a-zA-Z_]\w*)', question, flags=re.IGNORECASE)
        if explicit_sym:
            symbol_queries.extend(explicit_sym)

        symbol_queries = [s for s in set(symbol_queries) if s.lower() not in {"function", "class", "def", "method", "print", "main"}]

        if symbol_queries:
            chunks_text = " ".join([d.get("content", "") for d in docs])
            chunks_symbols = " ".join([d.get("metadata", {}).get("symbol_name", "") for d in docs])

            found_symbols = []
            missing_symbols = []
            for sym in symbol_queries:
                if sym in chunks_symbols or sym in chunks_text:
                    found_symbols.append(sym)
                else:
                    missing_symbols.append(sym)

            if missing_symbols:
                disk_chunks = self.retriever.search_disk_for_symbols(repo_id, missing_symbols)
                if disk_chunks:
                    trace.append(f"Disk recovery found {len(disk_chunks)} chunk(s) for symbol(s): {missing_symbols}")
                    docs = disk_chunks + docs
                    for dc in disk_chunks:
                        sym_name = dc.get("metadata", {}).get("symbol_name", "")
                        for ms in list(missing_symbols):
                            if ms in sym_name or ms in dc.get("content", ""):
                                found_symbols.append(ms)
                                missing_symbols.remove(ms)

            if missing_symbols and not found_symbols:
                trace.append(f"Evidence validation: FAILED. Symbol(s) '{', '.join(missing_symbols)}' do not exist in the repository.")
                return {
                    "evidence_status": "none",
                    "retrieved_docs": [],
                    "trace_steps": trace
                }
            elif missing_symbols and found_symbols:
                trace.append(f"Evidence validation: PARTIAL. Found: {found_symbols}, Missing: {missing_symbols}")
                context = self.retriever.build_context(docs, query=question)
                return {
                    "retrieved_docs": docs,
                    "context_str": context,
                    "evidence_status": "partial",
                    "trace_steps": trace
                }

        # D. Semantic keyword and concept relevance validation
        stopwords = {
            "where", "what", "which", "how", "when", "does", "implemented",
            "handle", "works", "work", "find", "show", "tell", "explain",
            "the", "and", "for", "with", "this", "that", "from", "into",
            "have", "has", "project", "repo", "repository", "codebase", "code",
            "any", "all", "our", "their", "your", "can", "could", "would",
            "about", "inside", "support", "supported", "feature", "is", "are"
        }
        words = re.findall(r'[a-zA-Z_]{3,}', question.lower())
        core_terms = [w for w in words if w not in stopwords]

        if core_terms:
            all_content = " ".join([d.get("content", "").lower() for d in docs])
            all_paths = " ".join([d.get("metadata", {}).get("rel_path", "").lower() for d in docs])
            all_text = all_content + " " + all_paths

            matched_terms = [t for t in core_terms if t in all_text]
            match_ratio = len(matched_terms) / len(core_terms)

            if not matched_terms:
                disk_recovery = self.retriever.search_disk_for_keywords(repo_id, core_terms)
                if disk_recovery:
                    trace.append(f"Disk recovery found {len(disk_recovery)} chunk(s) for terms: {core_terms}")
                    docs = disk_recovery + docs
                    context = self.retriever.build_context(docs, query=question)
                    return {
                        "retrieved_docs": docs,
                        "context_str": context,
                        "evidence_status": "sufficient",
                        "trace_steps": trace
                    }
                else:
                    trace.append(f"Evidence validation: FAILED. No matching repository evidence found for terms: {core_terms}.")
                    return {
                        "evidence_status": "none",
                        "retrieved_docs": [],
                        "trace_steps": trace
                    }
            elif match_ratio < 0.4 and len(core_terms) >= 3:
                trace.append(f"Evidence validation: PARTIAL (Matched {len(matched_terms)} of {len(core_terms)} terms).")
                context = self.retriever.build_context(docs, query=question)
                return {
                    "retrieved_docs": docs,
                    "context_str": context,
                    "evidence_status": "partial",
                    "trace_steps": trace
                }

        trace.append("Evidence validation: PASSED. Sufficient repository evidence verified.")
        return {
            "evidence_status": "sufficient",
            "trace_steps": trace
        }

    @staticmethod
    def _build_user_prompt(state: CodeRAGState) -> str:
        docs = state.get("retrieved_docs", [])
        has_full_file = any(d.get("metadata", {}).get("is_full_file", False) for d in docs)
        question = state.get("original_question", state["question"])
        intent = state.get("query_intent", "code_explanation")
        evidence_status = state.get("evidence_status", "sufficient")
        query_type = state.get("query_type", "repository_specific")

        if query_type == "general_programming":
            return f"""The user is asking a general programming / software engineering question.
Answer clearly, accurately, and comprehensively based on standard software engineering principles and best practices.

User Question: {question}

Note: Provide a helpful general explanation. Clarify that this is general programming knowledge and not specific to the user's repository."""

        if evidence_status == "partial":
            return f"""Repository Context:
----------------------------------------
{state.get('context_str', '')}
----------------------------------------

User Question: {question}

CRITICAL INSTRUCTIONS FOR PARTIAL EVIDENCE:
1. The available repository evidence above only PARTIALLY answers the question.
2. Explain ONLY the verified implementation supported directly by the source code chunks above.
3. Explicitly and clearly state what specific aspects, functions, or features could NOT be confirmed or are missing from the codebase.
4. NEVER make assumptions, speculate, or fabricate unverified implementation details.
5. Cite genuine file paths for the verified portions."""

        instructions = "Provide a direct, precise, and helpful answer to the user's question based strictly on the repository context above. Do not fabricate functions, files, or citations not present in the context."
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

    def _generate_export_answer(self, state: CodeRAGState) -> Dict[str, Any]:
        trace = list(state.get("trace_steps", []))
        session = state.get("export_session") or {}
        repo_id = session.get("repo_id", state.get("repo_id", ""))
        repo_dir = self.retriever.cloner.get_repo_path(repo_id)
        manifest = session.get("ordered_file_manifest", [])
        start_idx = session.get("current_file_index", 0)
        total_files = session.get("total_file_count", len(manifest))
        export_info = state.get("export_info") or {}
        is_continuation = export_info.get("is_continuation", False)

        parts = []

        # 1. Output directory tree first if fresh export (not continuation)
        if not is_continuation or start_idx == 0:
            tree_text = session.get("tree_text")
            if not tree_text and repo_dir:
                tree_text, _ = build_export_manifest_and_tree(
                    repo_dir,
                    root_name=repo_id,
                    readme_and_pdf_only=session.get("readme_and_pdf_only", False)
                )
            if tree_text:
                parts.append(tree_text)
                parts.append("\n\n---\n")

        batch_size = total_files if total_files <= 6 else EXPORT_BATCH_SIZE
        end_idx = min(start_idx + batch_size, total_files)
        batch_files = manifest[start_idx:end_idx]

        completed_files = list(session.get("completed_files", []))
        failed_files = list(session.get("failed_files", []))
        sources = []

        for idx, file_meta in enumerate(batch_files, start=start_idx + 1):
            rel_p = file_meta.get("rel_path", "")
            abs_p = Path(file_meta.get("abs_path") or (repo_dir / rel_p))
            lang = file_meta.get("language", "text")
            is_pdf = file_meta.get("is_pdf", False)

            parts.append(f"\n\n**File {idx}: `{rel_p}`**\n\n")

            if not abs_p.exists():
                parts.append(f"> [!WARNING]\n> File `{rel_p}` could not be read or is missing from repository index.\n")
                failed_files.append(rel_p)
                continue

            if is_pdf:
                size_kb = round(abs_p.stat().st_size / 1024, 1)
                parts.append(f"[PDF Document: `{rel_p}` (Size: {size_kb} KB)]\n*Original PDF document is preserved in the repository at `{rel_p}`.*\n")
                completed_files.append(rel_p)
                sources.append({
                    "file_path": rel_p,
                    "language": "pdf",
                    "symbol_name": Path(rel_p).name,
                    "symbol_type": "pdf_document",
                    "start_line": 1,
                    "end_line": 1,
                    "score": 100.0,
                    "snippet": f"PDF: {rel_p}"
                })
                continue

            try:
                with open(abs_p, "r", encoding="utf-8", errors="replace") as fp:
                    content = fp.read()
                sanitized_content = sanitize_sensitive_content(rel_p, content)
                parts.append(f"```{lang}\n{sanitized_content}\n```\n")
                completed_files.append(rel_p)
                sources.append({
                    "file_path": rel_p,
                    "language": lang,
                    "symbol_name": Path(rel_p).name,
                    "symbol_type": "source_file",
                    "start_line": 1,
                    "end_line": len(content.splitlines()),
                    "score": 100.0,
                    "snippet": content[:300]
                })
            except Exception as err:
                logger.error("Failed to read file %s: %s", abs_p, err)
                parts.append(f"> [!WARNING]\n> File `{rel_p}` could not be read: {err}\n")
                failed_files.append(rel_p)

        new_pending = [f["rel_path"] for f in manifest[end_idx:]]
        session["current_file_index"] = end_idx
        session["completed_files"] = completed_files
        session["pending_files"] = new_pending
        session["failed_files"] = failed_files
        session["export_status"] = "completed" if end_idx >= total_files else "in_progress"

        if new_pending:
            parts.append(
                f"\n\n---\n**[Export Progress: {end_idx} of {total_files} files exported. "
                f"Say \"Next\", \"Continue\", or \"Baki code dao\" to export the next batch.]**\n"
            )
        else:
            parts.append(
                f"\n\n---\n**[Complete Repository Export Finished: All {total_files} files have been successfully exported.]**\n"
            )

        if self.sqlite_store:
            try:
                self.sqlite_store.save_export_session(session)
            except Exception as err:
                logger.warning("Failed to save export session to SQLite: %s", err)
        cache_key = f"{session.get('conversation_id')}_{repo_id}"
        self._export_sessions[cache_key] = session
        self._export_sessions[repo_id] = session

        trace.append(f"Exported files {start_idx + 1} to {end_idx} (total: {total_files}).")
        return {
            "answer": "".join(parts),
            "sources": sources,
            "trace_steps": trace,
            "export_session": session
        }

    def generate_answer(self, state: CodeRAGState) -> Dict[str, Any]:
        trace = list(state.get("trace_steps", []))
        trace.append("Generating response...")

        # 0. Check if evidence validation failed completely
        if state.get("evidence_status") == "none":
            trace.append("No sufficient evidence in repository. Emitting professional fallback response.")
            return {
                "answer": PROFESSIONAL_FALLBACK_RESPONSE,
                "sources": [],
                "trace_steps": trace
            }

        # Check for complete repository code export
        if state.get("query_intent") == "repo_code_export":
            return self._generate_export_answer(state)

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
        tree_doc = next((d for d in docs if d.get("metadata", {}).get("is_tree", False)), None)
        full_file_doc = next((d for d in docs if d.get("metadata", {}).get("is_full_file", False)), None)
        question = state.get("original_question", state["question"])

        if tree_doc:
            tree_content = tree_doc.get("content", "")
            wants_expl = tree_doc.get("metadata", {}).get("wants_explanation", False)
            if wants_expl:
                arch_prompt = (
                    f"Repository: {state.get('repo_id', '')}\n"
                    f"Directory Tree:\n{tree_content}\n\n"
                    f"User Question: {question}\n\n"
                    f"CRITICAL INSTRUCTION: The complete directory tree has ALREADY been displayed to the user in the block above.\n"
                    f"DO NOT reprint any tree or code blocks.\n"
                    f"Directly explain the responsibility and architectural purpose of each major directory or module shown in the tree.\n"
                    f"Base your explanation strictly on actual repository files and modules."
                )
                try:
                    expl = self.ollama.generate(prompt=arch_prompt, system_prompt=SYSTEM_PROMPT)
                    answer = f"{tree_content}\n\n### Module & Directory Responsibilities\n\n{expl}"
                except Exception:
                    answer = tree_content
            else:
                answer = tree_content
        elif full_file_doc:
            rel_p = full_file_doc.get("metadata", {}).get("rel_path", "")
            lang = full_file_doc.get("metadata", {}).get("language", "python")
            raw_content = full_file_doc.get("content", "")

            wants_expl = any(w in question.lower() for w in ["explain", "breakdown", "break down", "how does", "what does", "walk through", "line by line"])
            if wants_expl:
                answer_parts = [f"Here is the complete source code for `{rel_p}`:\n\n```{lang}\n{raw_content}\n```\n\n### Explanation & Breakdown:\n\n"]
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
                answer = f"Here is the complete source code for `{rel_p}`:\n\n```{lang}\n{raw_content}\n```"
        else:
            system_prompt = SYSTEM_PROMPT
            user_prompt = self._build_user_prompt(state)
            try:
                answer = self.ollama.generate(prompt=user_prompt, system_prompt=system_prompt)
            except Exception as err:
                logger.error("Error generating answer: %s", err)
                answer = f"Error generating answer: {err}\n\nRetrieved context:\n{state.get('context_str', '')}"

        sources = []
        if state.get("query_type") != "general_programming":
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
        workflow.add_node("verify_evidence", self.verify_evidence)
        workflow.add_node("generate_answer", self.generate_answer)

        workflow.set_entry_point("analyze_query")
        workflow.add_edge("analyze_query", "retrieve_code")
        workflow.add_edge("retrieve_code", "grade_relevance")

        def route_after_grading(s: CodeRAGState) -> str:
            if s.get("is_relevant", True) or s.get("retry_count", 0) >= 1:
                return "verify_evidence"
            return "rewrite_query"

        workflow.add_conditional_edges("grade_relevance", route_after_grading, {
            "verify_evidence": "verify_evidence",
            "rewrite_query": "rewrite_query"
        })
        workflow.add_edge("rewrite_query", "retrieve_code")
        workflow.add_edge("verify_evidence", "generate_answer")
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
                "trace_steps": state["trace_steps"],
                "query_intent": "user_code_line_by_line"
            }
            return

        # 3. Retrieve
        state.update(self.retrieve_code(state))
        yield {"type": "trace", "step": state["trace_steps"][-1]}

        # 4. Grade
        state.update(self.grade_relevance(state))
        yield {"type": "trace", "step": state["trace_steps"][-1]}

        if not state.get("is_relevant", True) and state.get("retry_count", 0) < 1:
            state.update(self.rewrite_query(state))
            yield {"type": "trace", "step": state["trace_steps"][-1]}
            state.update(self.retrieve_code(state))
            yield {"type": "trace", "step": state["trace_steps"][-1]}

        # 5. Evidence Verification
        state.update(self.verify_evidence(state))
        yield {"type": "trace", "step": state["trace_steps"][-1]}

        # Check if evidence validation failed completely
        if state.get("evidence_status") == "none":
            yield {"type": "trace", "step": "Sufficient evidence not found in repository. Emitting professional fallback response."}
            words = PROFESSIONAL_FALLBACK_RESPONSE.split(" ")
            for idx, word in enumerate(words):
                chunk = word if idx == len(words) - 1 else word + " "
                yield {"type": "token", "content": chunk}
            yield {
                "type": "done",
                "sources": [],
                "trace_steps": state["trace_steps"],
                "query_intent": "no_evidence_fallback"
            }
            return

        # Check for complete repository code export
        if state.get("query_intent") == "repo_code_export":
            yield {"type": "trace", "step": "Streaming complete repository code export..."}
            session = state.get("export_session") or {}
            repo_id = session.get("repo_id", state.get("repo_id", ""))
            repo_dir = self.retriever.cloner.get_repo_path(repo_id)
            manifest = session.get("ordered_file_manifest", [])
            start_idx = session.get("current_file_index", 0)
            total_files = session.get("total_file_count", len(manifest))
            export_info = state.get("export_info") or {}
            is_continuation = export_info.get("is_continuation", False)

            if not is_continuation or start_idx == 0:
                tree_text = session.get("tree_text")
                if not tree_text and repo_dir:
                    tree_text, _ = build_export_manifest_and_tree(
                        repo_dir,
                        root_name=repo_id,
                        readme_and_pdf_only=session.get("readme_and_pdf_only", False)
                    )
                if tree_text:
                    yield {"type": "token", "content": tree_text + "\n\n---\n"}

            batch_size = total_files if total_files <= 6 else EXPORT_BATCH_SIZE
            end_idx = min(start_idx + batch_size, total_files)
            batch_files = manifest[start_idx:end_idx]

            completed_files = list(session.get("completed_files", []))
            failed_files = list(session.get("failed_files", []))
            sources = []

            for idx, file_meta in enumerate(batch_files, start=start_idx + 1):
                rel_p = file_meta.get("rel_path", "")
                abs_p = Path(file_meta.get("abs_path") or (repo_dir / rel_p))
                lang = file_meta.get("language", "text")
                is_pdf = file_meta.get("is_pdf", False)

                yield {"type": "token", "content": f"\n\n**File {idx}: `{rel_p}`**\n\n"}

                if not abs_p.exists():
                    yield {"type": "token", "content": f"> [!WARNING]\n> File `{rel_p}` could not be read or is missing from repository index.\n"}
                    failed_files.append(rel_p)
                    continue

                if is_pdf:
                    size_kb = round(abs_p.stat().st_size / 1024, 1)
                    yield {"type": "token", "content": f"[PDF Document: `{rel_p}` (Size: {size_kb} KB)]\n*Original PDF document is preserved in the repository at `{rel_p}`.*\n"}
                    completed_files.append(rel_p)
                    sources.append({
                        "file_path": rel_p,
                        "language": "pdf",
                        "symbol_name": Path(rel_p).name,
                        "symbol_type": "pdf_document",
                        "start_line": 1,
                        "end_line": 1,
                        "score": 100.0,
                        "snippet": f"PDF: {rel_p}"
                    })
                    continue

                try:
                    with open(abs_p, "r", encoding="utf-8", errors="replace") as fp:
                        content = fp.read()
                    sanitized_content = sanitize_sensitive_content(rel_p, content)
                    yield {"type": "token", "content": f"```{lang}\n"}
                    lines = sanitized_content.splitlines(keepends=True)
                    delay = max(0.001, min(0.008, 1.0 / max(1, len(lines))))
                    for line in lines:
                        yield {"type": "token", "content": line}
                        time.sleep(delay)
                    yield {"type": "token", "content": "\n```\n"}
                    completed_files.append(rel_p)
                    sources.append({
                        "file_path": rel_p,
                        "language": lang,
                        "symbol_name": Path(rel_p).name,
                        "symbol_type": "source_file",
                        "start_line": 1,
                        "end_line": len(lines),
                        "score": 100.0,
                        "snippet": content[:300]
                    })
                except Exception as err:
                    yield {"type": "token", "content": f"> [!WARNING]\n> File `{rel_p}` could not be read: {err}\n"}
                    failed_files.append(rel_p)

            new_pending = [f["rel_path"] for f in manifest[end_idx:]]
            session["current_file_index"] = end_idx
            session["completed_files"] = completed_files
            session["pending_files"] = new_pending
            session["failed_files"] = failed_files
            session["export_status"] = "completed" if end_idx >= total_files else "in_progress"

            if new_pending:
                yield {
                    "type": "token",
                    "content": f"\n\n---\n**[Export Progress: {end_idx} of {total_files} files exported. Say \"Next\", \"Continue\", or \"Baki code dao\" to export the next batch.]**\n"
                }
            else:
                yield {
                    "type": "token",
                    "content": f"\n\n---\n**[Complete Repository Export Finished: All {total_files} files have been successfully exported.]**\n"
                }

            if self.sqlite_store:
                try:
                    self.sqlite_store.save_export_session(session)
                except Exception as err:
                    logger.warning("Failed to save export session to SQLite: %s", err)
            cache_key = f"{session.get('conversation_id')}_{repo_id}"
            self._export_sessions[cache_key] = session
            self._export_sessions[repo_id] = session

            yield {
                "type": "done",
                "sources": sources,
                "trace_steps": state["trace_steps"],
                "query_intent": "repo_code_export",
                "export_progress": {
                    "completed": end_idx,
                    "total": total_files,
                    "has_more": bool(new_pending)
                }
            }
            return

        # 6. Stream tokens
        yield {"type": "trace", "step": "Generating response..."}

        docs = state.get("retrieved_docs", [])
        tree_doc = next((d for d in docs if d.get("metadata", {}).get("is_tree", False)), None)
        full_file_doc = next((d for d in docs if d.get("metadata", {}).get("is_full_file", False)), None)
        question = state.get("original_question", state["question"])

        sources = []
        if state.get("query_type") != "general_programming":
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

        if tree_doc:
            tree_content = tree_doc.get("content", "")
            wants_expl = tree_doc.get("metadata", {}).get("wants_explanation", False)
            yield {"type": "token", "content": tree_content}
            if wants_expl:
                yield {"type": "token", "content": "\n\n### Module & Directory Responsibilities\n\n"}
                arch_prompt = (
                    f"Repository: {state.get('repo_id', '')}\n"
                    f"Directory Tree:\n{tree_content}\n\n"
                    f"User Question: {question}\n\n"
                    f"CRITICAL INSTRUCTION: The complete directory tree has ALREADY been displayed to the user in the block above.\n"
                    f"DO NOT reprint any tree or code blocks.\n"
                    f"Directly explain the responsibility and architectural purpose of each major directory or module shown in the tree.\n"
                    f"Base your explanation strictly on actual repository files and modules."
                )
                try:
                    for token in self.ollama.generate_stream(prompt=arch_prompt, system_prompt=SYSTEM_PROMPT):
                        yield {"type": "token", "content": token}
                except Exception:
                    pass
        elif full_file_doc:
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

            # 3. Close code block and start breakdown if requested
            yield {"type": "token", "content": "\n```"}

            wants_expl = any(w in question.lower() for w in ["explain", "breakdown", "break down", "how does", "what does", "walk through", "line by line"])
            if wants_expl:
                yield {"type": "token", "content": "\n\n### Explanation & Breakdown:\n\n"}
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
            "trace_steps": state["trace_steps"],
            "query_intent": state.get("query_intent", "code_explanation")
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
                "trace_steps": state["trace_steps"],
                "query_intent": "user_code_line_by_line"
            }
            return

        # 3. Retrieve (run in thread to prevent blocking event loop)
        retrieval_res = await asyncio.to_thread(self.retrieve_code, state)
        state.update(retrieval_res)
        yield {"type": "trace", "step": state["trace_steps"][-1]}

        # 4. Grade
        state.update(self.grade_relevance(state))
        yield {"type": "trace", "step": state["trace_steps"][-1]}

        if not state.get("is_relevant", True) and state.get("retry_count", 0) < 1:
            state.update(self.rewrite_query(state))
            yield {"type": "trace", "step": state["trace_steps"][-1]}
            retrieval_res = await asyncio.to_thread(self.retrieve_code, state)
            state.update(retrieval_res)
            yield {"type": "trace", "step": state["trace_steps"][-1]}

        # 5. Evidence Verification
        state.update(self.verify_evidence(state))
        yield {"type": "trace", "step": state["trace_steps"][-1]}

        # Check if evidence validation failed completely
        if state.get("evidence_status") == "none":
            yield {"type": "trace", "step": "Sufficient evidence not found in repository. Emitting professional fallback response."}
            words = PROFESSIONAL_FALLBACK_RESPONSE.split(" ")
            for idx, word in enumerate(words):
                chunk = word if idx == len(words) - 1 else word + " "
                yield {"type": "token", "content": chunk}
            yield {
                "type": "done",
                "sources": [],
                "trace_steps": state["trace_steps"],
                "query_intent": "no_evidence_fallback"
            }
            return

        # Check for complete repository code export
        if state.get("query_intent") == "repo_code_export":
            yield {"type": "trace", "step": "Streaming complete repository code export..."}
            session = state.get("export_session") or {}
            repo_id = session.get("repo_id", state.get("repo_id", ""))
            repo_dir = self.retriever.cloner.get_repo_path(repo_id)
            manifest = session.get("ordered_file_manifest", [])
            start_idx = session.get("current_file_index", 0)
            total_files = session.get("total_file_count", len(manifest))
            export_info = state.get("export_info") or {}
            is_continuation = export_info.get("is_continuation", False)

            if not is_continuation or start_idx == 0:
                tree_text = session.get("tree_text")
                if not tree_text and repo_dir:
                    tree_text, _ = build_export_manifest_and_tree(
                        repo_dir,
                        root_name=repo_id,
                        readme_and_pdf_only=session.get("readme_and_pdf_only", False)
                    )
                if tree_text:
                    yield {"type": "token", "content": tree_text + "\n\n---\n"}

            batch_size = total_files if total_files <= 6 else EXPORT_BATCH_SIZE
            end_idx = min(start_idx + batch_size, total_files)
            batch_files = manifest[start_idx:end_idx]

            completed_files = list(session.get("completed_files", []))
            failed_files = list(session.get("failed_files", []))
            sources = []

            for idx, file_meta in enumerate(batch_files, start=start_idx + 1):
                rel_p = file_meta.get("rel_path", "")
                abs_p = Path(file_meta.get("abs_path") or (repo_dir / rel_p))
                lang = file_meta.get("language", "text")
                is_pdf = file_meta.get("is_pdf", False)

                yield {"type": "token", "content": f"\n\n**File {idx}: `{rel_p}`**\n\n"}

                if not abs_p.exists():
                    yield {"type": "token", "content": f"> [!WARNING]\n> File `{rel_p}` could not be read or is missing from repository index.\n"}
                    failed_files.append(rel_p)
                    continue

                if is_pdf:
                    size_kb = round(abs_p.stat().st_size / 1024, 1)
                    yield {"type": "token", "content": f"[PDF Document: `{rel_p}` (Size: {size_kb} KB)]\n*Original PDF document is preserved in the repository at `{rel_p}`.*\n"}
                    completed_files.append(rel_p)
                    sources.append({
                        "file_path": rel_p,
                        "language": "pdf",
                        "symbol_name": Path(rel_p).name,
                        "symbol_type": "pdf_document",
                        "start_line": 1,
                        "end_line": 1,
                        "score": 100.0,
                        "snippet": f"PDF: {rel_p}"
                    })
                    continue

                try:
                    with open(abs_p, "r", encoding="utf-8", errors="replace") as fp:
                        content = fp.read()
                    sanitized_content = sanitize_sensitive_content(rel_p, content)
                    yield {"type": "token", "content": f"```{lang}\n"}
                    lines = sanitized_content.splitlines(keepends=True)
                    delay = max(0.001, min(0.008, 1.0 / max(1, len(lines))))
                    for line in lines:
                        yield {"type": "token", "content": line}
                        await asyncio.sleep(delay)
                    yield {"type": "token", "content": "\n```\n"}
                    completed_files.append(rel_p)
                    sources.append({
                        "file_path": rel_p,
                        "language": lang,
                        "symbol_name": Path(rel_p).name,
                        "symbol_type": "source_file",
                        "start_line": 1,
                        "end_line": len(lines),
                        "score": 100.0,
                        "snippet": content[:300]
                    })
                except Exception as err:
                    yield {"type": "token", "content": f"> [!WARNING]\n> File `{rel_p}` could not be read: {err}\n"}
                    failed_files.append(rel_p)

            new_pending = [f["rel_path"] for f in manifest[end_idx:]]
            session["current_file_index"] = end_idx
            session["completed_files"] = completed_files
            session["pending_files"] = new_pending
            session["failed_files"] = failed_files
            session["export_status"] = "completed" if end_idx >= total_files else "in_progress"

            if new_pending:
                yield {
                    "type": "token",
                    "content": f"\n\n---\n**[Export Progress: {end_idx} of {total_files} files exported. Say \"Next\", \"Continue\", or \"Baki code dao\" to export the next batch.]**\n"
                }
            else:
                yield {
                    "type": "token",
                    "content": f"\n\n---\n**[Complete Repository Export Finished: All {total_files} files have been successfully exported.]**\n"
                }

            if self.sqlite_store:
                try:
                    self.sqlite_store.save_export_session(session)
                except Exception as err:
                    logger.warning("Failed to save export session to SQLite: %s", err)
            cache_key = f"{session.get('conversation_id')}_{repo_id}"
            self._export_sessions[cache_key] = session
            self._export_sessions[repo_id] = session

            yield {
                "type": "done",
                "sources": sources,
                "trace_steps": state["trace_steps"],
                "query_intent": "repo_code_export",
                "export_progress": {
                    "completed": end_idx,
                    "total": total_files,
                    "has_more": bool(new_pending)
                }
            }
            return

        # 6. Stream tokens
        yield {"type": "trace", "step": "Generating response..."}

        docs = state.get("retrieved_docs", [])
        tree_doc = next((d for d in docs if d.get("metadata", {}).get("is_tree", False)), None)
        full_file_doc = next((d for d in docs if d.get("metadata", {}).get("is_full_file", False)), None)
        question = state.get("original_question", state["question"])

        sources = []
        if state.get("query_type") != "general_programming":
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

        if tree_doc:
            tree_content = tree_doc.get("content", "")
            wants_expl = tree_doc.get("metadata", {}).get("wants_explanation", False)
            yield {"type": "token", "content": tree_content}
            if wants_expl:
                yield {"type": "token", "content": "\n\n### Module & Directory Responsibilities\n\n"}
                arch_prompt = (
                    f"Repository: {state.get('repo_id', '')}\n"
                    f"Directory Tree:\n{tree_content}\n\n"
                    f"User Question: {question}\n\n"
                    f"CRITICAL INSTRUCTION: The complete directory tree has ALREADY been displayed to the user in the block above.\n"
                    f"DO NOT reprint any tree or code blocks.\n"
                    f"Directly explain the responsibility and architectural purpose of each major directory or module shown in the tree.\n"
                    f"Base your explanation strictly on actual repository files and modules."
                )
                try:
                    async for token in self.ollama.generate_stream_async(prompt=arch_prompt, system_prompt=SYSTEM_PROMPT):
                        yield {"type": "token", "content": token}
                except Exception:
                    pass
        elif full_file_doc:
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

            # 3. Close code block and start breakdown if requested
            yield {"type": "token", "content": "\n```"}

            wants_expl = any(w in question.lower() for w in ["explain", "breakdown", "break down", "how does", "what does", "walk through", "line by line"])
            if wants_expl:
                yield {"type": "token", "content": "\n\n### Explanation & Breakdown:\n\n"}
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
            "trace_steps": state["trace_steps"],
            "query_intent": state.get("query_intent", "code_explanation")
        }
