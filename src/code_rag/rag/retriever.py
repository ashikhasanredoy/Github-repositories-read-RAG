import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.code_rag.config import settings
from src.code_rag.storage.vector_store import VectorStore
from src.code_rag.storage.bm25_store import BM25Store
from src.code_rag.core.cloner import RepoCloner

logger = logging.getLogger(__name__)

LANGUAGE_MAP = {
    ".py": "python", ".ipynb": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go",
    ".java": "java", ".cpp": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".rs": "rust", ".rb": "ruby", ".php": "php", ".md": "markdown", ".markdown": "markdown", ".rst": "rst",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "ini", ".cfg": "ini",
    ".sql": "sql", ".prisma": "prisma", ".graphql": "graphql", ".gql": "graphql",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".html": "html", ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
    ".json": "json", ".jsonl": "json", ".xml": "xml", ".txt": "text", ".csv": "csv", ".tsv": "tsv",
    ".cs": "csharp", ".swift": "swift", ".kt": "kotlin", ".dart": "dart",
    ".scala": "scala", ".lua": "lua", ".env": "ini", ".dockerfile": "dockerfile",
    ".vue": "vue", ".svelte": "svelte", ".proto": "protobuf"
}

def detect_structure_intent(query: str) -> Optional[Dict[str, Any]]:
    q = query.lower().strip()

    # 1. Single file check: if user asks for a specific source file, NOT structure
    single_file_match = re.search(r"\b([a-zA-Z0-9_\-./]+\.(?:py|js|ts|html|css|json|yaml|yml|md|sql|go|java|c|cpp|h))\b", q)
    is_asking_file_content = any(phrase in q for phrase in [
        "give me", "show code", "show the code", "source code of", "get file", "view file",
        "read file", "open file", "contents of file", "code of", "line by line"
    ])
    if single_file_match and is_asking_file_content:
        return None

    # Check for extension filter
    filter_exts = []
    if "python" in q or ".py" in q:
        filter_exts.append(".py")
    if "javascript" in q or ".js" in q:
        filter_exts.append(".js")
    if "typescript" in q or ".ts" in q:
        filter_exts.append(".ts")
    if "html" in q or ".html" in q:
        filter_exts.append(".html")
    if "css" in q or ".css" in q:
        filter_exts.append(".css")

    # Check for specific folder request
    subfolder = ""
    folder_match = re.search(r"\b(?:folder|directory|dir|package|module|subdirectories|inside)\s+([a-zA-Z0-9_\-./]+)", q)
    if folder_match:
        cand = folder_match.group(1).strip("`\"'").rstrip("/")
        stopwords_folder = ["this", "the", "a", "my", "project", "repo", "repository", "codebase", "structure", "tree", "hierarchy", "layout", "all", "nested"]
        if cand not in stopwords_folder:
            subfolder = cand
    if not subfolder and "__pycache__" in q:
        subfolder = "__pycache__"
    elif not subfolder and "backend" in q and any(w in q for w in ["architecture", "structure", "hierarchy", "tree", "file"]):
        subfolder = "src/code_rag"

    structure_patterns = [
        # A. Repo structure
        r"\brepo(?:sitory)?\s+structure\b",
        r"\bcodebase\s+structure\b",
        r"\bproject\s+structure\b",
        r"\bcomplete\s+repo\s+layout\b",
        r"\bentire\s+project\s+structure\b",
        r"\bstructure\s+of\s+this\s+(?:project|repo|repository|codebase)\b",
        r"\blist\s+all\s+(?:repository\s+)?files\b",
        r"\brepo(?:sitory)?\s+contents\b",
        r"\bcodebase\s+overview\b",
        r"\bsource\s+code\s+organization\b",
        r"\bhow\s+is\s+this\s+project\s+organized\b",
        r"\bcomplete\s+directory\s+tree\b",
        r"\brepository\s+hierarchy\b",
        r"\bgithub(?:-style)?\s+(?:repository\s+)?structure\b",
        # B. Folder & directory structure
        r"\bfolder\s+structure\b",
        r"\bdirectory\s+structure\b",
        r"\bdirectory\s+hierarchy\b",
        r"\blist\s+all\s+(?:folders|directories)\b",
        r"\bshow\s+all\s+(?:folders|directories)\b",
        r"\b(?:show|display)\s+nested\s+folders\b",
        r"\bsubdirectories\b",
        r"\bfolder\s+tree\b",
        r"\bfolder\s+layout\b",
        r"\bwhat\s+folders\s+are\s+inside\b",
        r"\bcontents\s+of\s+(?:a\s+)?folder\b",
        r"\bcomplete\s+folder\s+hierarchy\b",
        r"\bnested\s+directory\s+structure\b",
        r"\blist\s+files\s+and\s+folders\b",
        r"\bparent\s+and\s+child\s+directories\b",
        # C. File structure
        r"\bfile\s+structure\b",
        r"\blist\s+project\s+files\b",
        r"\b(?:show|display|list)\s+all\s+files\b",
        r"\blist\s+source\s+files\b",
        r"\b(?:show|list|display)\s+(?:all\s+)?(?:python|javascript|html|css)\s+files\b",
        r"\bfiles\s+inside\s+a\s+directory\b",
        r"\bwhat\s+files\s+are\s+in\s+this\s+project\b",
        r"\bwhich\s+files\s+are\s+inside\b",
        r"\bfile\s+hierarchy\b",
        r"\bcomplete\s+file\s+listing\b",
        r"\bproject\s+file\s+tree\b",
        r"\bsource\s+file\s+tree\b",
        # D. Architecture & System organization
        r"\bproject\s+architecture\b",
        r"\bsystem\s+architecture\b",
        r"\bsoftware\s+architecture\b",
        r"\bapplication\s+architecture\b",
        r"\bbackend\s+(?:architecture|file\s+hierarchy|structure)\b",
        r"\bfrontend\s+(?:architecture|file\s+hierarchy|structure)\b",
        r"\bcode\s+architecture\b",
        r"\bapplication\s+structure\b",
        r"\bmodule\s+hierarchy\b",
        r"\bmodule\s+organization\b",
        r"\bcomponent\s+structure\b",
        r"\bcomponent\s+hierarchy\b",
        r"\bpackage\s+structure\b",
        r"\bhow\s+modules\s+are\s+organized\b",
        r"\binternal\s+architecture\b",
        r"\barchitectural\s+overview\b",
        r"\brelationship\s+between\s+folders\s+and\s+files\b",
        r"\bcodebase\s+organization\b",
        # E. Natural language expressions
        r"\bhow\s+does\s+the\s+project\s+look\s+internally\b",
        r"\bwhat\s+does\s+the\s+project\s+contain\b",
        r"\bwhat\s+is\s+inside\s+the\s+repository\b",
        r"\bhow\s+are\s+the\s+files\s+arranged\b",
        r"\bhow\s+is\s+the\s+code\s+organized\b",
        r"\bproject\s+layout\b",
        r"\boverview\s+of\s+the\s+directories\b",
        r"\bcomplete\s+hierarchy\b",
        r"\bmap\s+out\s+the\s+project\b",
        r"\bshow\s+me\s+everything\s+inside\s+the\s+project\b",
        r"\bprint\s+the\s+directory\s+tree\b",
        r"\bgenerate\s+a\s+tree\s+view\b",
        r"\bvisualize\s+the\s+project\s+hierarchy\b",
        r"\bmap\s+the\s+source\s+code\s+organization\b",
        r"\bwhere\s+are\s+all\s+the\s+modules\s+located\b",
        r"\btree\s+view\b",
        r"\bdirectory\s+tree\b"
    ]

    matched = any(re.search(pat, q) for pat in structure_patterns)
    if matched or (subfolder and any(w in q for w in ["show", "list", "display", "contents", "what is inside", "tree"])):
        explanation_keywords = [r"\bexplain\b", r"\boverview\b", r"\bdescribe\b", r"\bwhy\b", r"\bhow\b", r"\bresponsibilit", r"\brelationship\b"]
        wants_explanation = any(re.search(kw, q) for kw in explanation_keywords)
        return {
            "is_structure": True,
            "subfolder": subfolder,
            "filter_exts": filter_exts,
            "wants_explanation": wants_explanation
        }
    return None


class HybridRetriever:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_store: Optional[BM25Store] = None,
        cloner: Optional[RepoCloner] = None,
        rrf_k: int = settings.RRF_K
    ):
        self.vector_store = vector_store or VectorStore()
        self.bm25_store = bm25_store or BM25Store()
        self.cloner = cloner or RepoCloner()
        self.rrf_k = rrf_k

    def _find_full_files_on_disk(self, repo_id: str, target_names: List[str]) -> List[Dict[str, Any]]:
        repo_dir = self.cloner.get_repo_path(repo_id)
        if not repo_dir or not repo_dir.exists():
            return []

        full_file_chunks = []
        seen_paths = set()

        for root, dirs, files in os.walk(repo_dir):
            dirs[:] = [d for d in dirs if d not in settings.IGNORE_DIRS and not d.startswith(".")]
            for f in files:
                if f in settings.IGNORE_FILES or f.startswith("."):
                    continue
                file_p = Path(root) / f
                try:
                    rel_p = str(file_p.relative_to(repo_dir))
                except Exception:
                    rel_p = str(file_p.name)

                rel_p_lower = rel_p.lower()
                f_lower = f.lower()

                for target in target_names:
                    t_lower = target.lower().strip()
                    if not t_lower:
                        continue
                    t_stem = Path(t_lower).stem
                    f_stem = file_p.stem.lower()

                    match_score = 0
                    if rel_p_lower == t_lower:
                        match_score = 100
                    elif rel_p_lower.endswith(t_lower) or rel_p_lower.endswith("/" + t_lower):
                        match_score = 80
                    elif f_lower == t_lower:
                        match_score = 60
                    elif len(t_stem) > 2 and f_stem == t_stem:
                        match_score = 40
                    elif len(t_lower) > 3 and t_lower in rel_p_lower:
                        match_score = 20

                    if match_score > 0 and rel_p not in seen_paths:
                        seen_paths.add(rel_p)
                        try:
                            with open(file_p, "r", encoding="utf-8", errors="replace") as fp:
                                full_text = fp.read()
                            lines = full_text.splitlines()
                            total_lines = len(lines) if lines else 1
                            lang = LANGUAGE_MAP.get(file_p.suffix.lower(), "text")
                            full_file_chunks.append({
                                "chunk_id": f"{repo_id}_full_{rel_p}",
                                "content": full_text,
                                "metadata": {
                                    "rel_path": rel_p,
                                    "language": lang,
                                    "symbol_name": "full_file",
                                    "symbol_type": "full_file",
                                    "start_line": 1,
                                    "end_line": total_lines,
                                    "is_full_file": True
                                },
                                "rrf_score": 900.0 + match_score,
                                "match_score": match_score
                            })
                        except Exception as err:
                            logger.warning("Failed to read full file %s: %s", file_p, err)
                        break

        # Sort so the highest exact match comes first
        full_file_chunks.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        return full_file_chunks

    def retrieve(self, repo_id: str, query: str, top_k: int = settings.FINAL_TOP_K) -> List[Dict[str, Any]]:
        # 1. Target file extraction
        raw_targets = re.findall(r'[\w\-./\\]+\.[\w]+', query)
        target_files = [
            t for t in raw_targets
            if not any(t.endswith(ext) for ext in ['.com', '.org', '.net', '.io', '.gov', '.edu', '.git'])
        ]

        q_lower = query.lower()
        wants_code_or_file = any(
            kw in q_lower for kw in [
                "show", "code", "file", "full", "view", "give", "display",
                "read", "print", "explain", "implement", "what is inside", "lines"
            ]
        )

        # Also look for filename words without extensions if user asks to show/give a file
        if wants_code_or_file and not target_files:
            words = re.findall(r'[a-zA-Z0-9_\-]+', query)
            for w in words:
                if len(w) >= 3 and w.lower() not in {"show", "code", "file", "full", "view", "give", "read", "what", "where", "inside", "this", "that", "with", "from"}:
                    target_files.append(w)

        # Look up 100% complete files from disk if specific files are queried
        full_disk_files = []
        if target_files:
            full_disk_files = self._find_full_files_on_disk(repo_id, target_files)

        vector_results = self.vector_store.search(repo_id, query, top_k=settings.VECTOR_TOP_K)
        bm25_results = self.bm25_store.search(repo_id, query, top_k=settings.BM25_TOP_K)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(vector_results):
            cid = item["chunk_id"]
            chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        for rank, item in enumerate(bm25_results):
            cid = item["chunk_id"]
            if cid not in chunk_map:
                chunk_map[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # If user explicitly asked for specific files and we found their full content on disk, return full file chunks!
        if full_disk_files and (wants_code_or_file or len(target_files) > 0):
            return full_disk_files[:3]

        # Structure / architecture tree requests
        struct_info = detect_structure_intent(query)
        if struct_info:
            repo_dir = self.cloner.get_repo_path(repo_id)
            if repo_dir and repo_dir.exists():
                from src.code_rag.core.chunker import generate_github_style_tree
                include_hidden = bool(struct_info.get("subfolder") and "__pycache__" in struct_info["subfolder"])
                tree_text = generate_github_style_tree(
                    repo_dir=repo_dir,
                    root_name=repo_id,
                    subfolder=struct_info.get("subfolder", ""),
                    filter_exts=struct_info.get("filter_exts"),
                    include_hidden=include_hidden
                )
                return [{
                    "chunk_id": f"{repo_id}_file_tree",
                    "content": tree_text,
                    "metadata": {
                        "rel_path": "REPOSITORY_STRUCTURE.md",
                        "language": "text",
                        "symbol_name": "repository_file_tree",
                        "symbol_type": "overview",
                        "start_line": 1,
                        "end_line": 1,
                        "is_full_file": False,
                        "is_tree": True,
                        "wants_explanation": struct_info.get("wants_explanation", False)
                    },
                    "rrf_score": 999.0
                }]

            # Fallback to pre-indexed tree chunk if repo_dir not available
            tree_chunks = [
                item for cid, item in chunk_map.items()
                if "file_tree" in cid or "repository_structure" in item.get("metadata", {}).get("rel_path", "").lower()
            ]
            if tree_chunks:
                tc = dict(tree_chunks[0])
                tc["metadata"] = dict(tc.get("metadata", {}))
                tc["metadata"]["is_tree"] = True
                tc["metadata"]["is_full_file"] = False
                tc["metadata"]["wants_explanation"] = struct_info.get("wants_explanation", False)
                return [tc]

        sorted_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)[:top_k]

        final_chunks: List[Dict[str, Any]] = []
        for cid in sorted_ids:
            doc = chunk_map[cid]
            doc["rrf_score"] = round(rrf_scores[cid], 5)
            final_chunks.append(doc)

        return final_chunks

    @staticmethod
    def build_context(chunks: List[Dict[str, Any]], query: str = "") -> str:
        if not chunks:
            return "No relevant code chunks found in repository."

        blocks = []

        # Check if user asked about a specific file that is absent
        target_files = [t for t in re.findall(r'[\w\-./\\]+\.[\w]+', query) if not any(t.endswith(ext) for ext in ['.com', '.org', '.net', '.io', '.git'])]
        if target_files:
            found_any = False
            for chunk in chunks:
                rel = chunk.get("metadata", {}).get("rel_path", "").lower()
                if any(tf.lower() in rel or rel.endswith(tf.lower()) for tf in target_files):
                    found_any = True
                    break
            if not found_any:
                missing = ", ".join(target_files)
                blocks.append(
                    f"[SYSTEM NOTICE: The user explicitly asked about '{missing}', but '{missing}' DOES NOT EXIST in this repository.\n"
                    f"DO NOT attribute code from other files to '{missing}'. State clearly that '{missing}' is not present in the repository.]\n"
                )

        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            file_path = meta.get("rel_path", "unknown")
            lang = meta.get("language", "python")
            start = meta.get("start_line", 1)
            end = meta.get("end_line", 1)
            symbol = meta.get("symbol_name", "snippet")
            sym_type = meta.get("symbol_type", "code")
            is_full = meta.get("is_full_file", False)

            content = chunk.get('content', '')

            if is_full:
                blocks.append(
                    f"### [{i}] {file_path} (Lines 1–{end}) [COMPLETE UNTRUNCATED FULL FILE]\n"
                    f"**Symbol**: `full_file` (Entire Source Code)\n"
                    f"```{lang}\n"
                    f"{content}\n"
                    f"```"
                )
            else:
                blocks.append(
                    f"### [{i}] {file_path} (Lines {start}–{end}) | `{symbol}` ({sym_type})\n"
                    f"```{lang}\n"
                    f"{content}\n"
                    f"```"
                )

        return "\n\n".join(blocks)
