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

        # Boost file tree overview chunk for structure / tree queries
        is_structure_query = any(w in q_lower for w in [
            "file structure", "folder structure", "tree", "directory layout",
            "project layout", "files in this repo", "project structure", "what files",
            "give file structure", "show file structure", "show files", "list files", "structure of this repo"
        ])

        if is_structure_query:
            tree_chunks = [
                item for cid, item in chunk_map.items()
                if "file_tree" in cid or "repository_structure" in item.get("metadata", {}).get("rel_path", "").lower()
            ]
            if tree_chunks:
                return tree_chunks[:1]

            # Generate dynamically from disk if not found in chunk map
            repo_dir = self.cloner.get_repo_path(repo_id)
            if repo_dir and repo_dir.exists():
                from src.code_rag.core.parser import CodeParser
                from src.code_rag.core.chunker import generate_ascii_tree
                p = CodeParser()
                files = p.scan_repository(str(repo_dir))
                if files:
                    rel_paths = [str(f.relative_to(repo_dir)) for f in sorted(files)]
                    ascii_tree = generate_ascii_tree(rel_paths)
                    dirs_set = sorted({str(Path(p_path).parts[0]) for p_path in rel_paths if len(Path(p_path).parts) > 1})
                    root_files = sorted({p_path for p_path in rel_paths if len(Path(p_path).parts) == 1})

                    dir_breakdown = []
                    if dirs_set:
                        dir_breakdown.append("- The repository contains the following directories:")
                        for d in dirs_set:
                            sub_count = len([p_path for p_path in rel_paths if p_path.startswith(f"{d}/")])
                            dir_breakdown.append(f"  - `{d}/`: Contains {sub_count} files.")
                    if root_files:
                        dir_breakdown.append("- The repository also contains the following root files:")
                        for rf in root_files:
                            dir_breakdown.append(f"  - `{rf}`: Root configuration / entrypoint file.")
                    breakdown_str = "\n".join(dir_breakdown)

                    tree_content = (
                        f"**Repository File Structure:**\n"
                        f"```markdown\n"
                        f"{ascii_tree}\n"
                        f"```\n"
                        f"Note: The above structure is a direct representation of the repository structure as provided in the repository context.\n\n"
                        f"The repository structure is as follows:\n\n"
                        f"{breakdown_str}"
                    )
                    return [{
                        "chunk_id": f"{repo_id}_file_tree",
                        "content": tree_content,
                        "metadata": {
                            "rel_path": "REPOSITORY_STRUCTURE.md",
                            "language": "markdown",
                            "symbol_name": "repository_file_tree",
                            "symbol_type": "overview",
                            "start_line": 1,
                            "end_line": len(files),
                            "is_full_file": True
                        },
                        "rrf_score": 999.0
                    }]

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
                    f"⚠️ [SYSTEM NOTICE: The user explicitly asked about '{missing}', but '{missing}' DOES NOT EXIST in this repository.\n"
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
