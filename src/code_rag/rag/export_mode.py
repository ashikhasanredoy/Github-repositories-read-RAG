import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set

logger = logging.getLogger("code_rag.rag.export_mode")

# Language mapping for fenced code blocks
LANGUAGE_MAP = {
    ".py": "python", ".ipynb": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go",
    ".java": "java", ".cpp": "cpp", ".c": "c", ".h": "c", ".hpp": "cpp", ".cc": "cpp", ".cxx": "cpp",
    ".rs": "rust", ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".kt": "kotlin", ".kts": "kotlin", ".scala": "scala", ".dart": "dart", ".lua": "lua", ".r": "r",
    ".cs": "csharp", ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".html": "html", ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
    ".sql": "sql", ".md": "markdown", ".rst": "rst", ".txt": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".xml": "xml", ".proto": "protobuf", ".vue": "vue", ".svelte": "svelte",
    ".dockerfile": "dockerfile"
}

# Strict media file exclusions (Section 5)
EXCLUDED_MEDIA_EXTENSIONS = {
    # Images & photos
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".ico", ".tiff", ".tif", ".psd", ".ai",
    # Audio
    ".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".wma",
    # Video
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv"
}

# Directories to skip
EXCLUDED_DIRECTORIES = {
    ".git", "__pycache__", ".pytest_cache", ".venv", "venv", "node_modules",
    "dist", "build", ".mypy_cache", ".idea", ".vscode", ".tox", "target", "coverage", ".eggs"
}

# Non-code or temporary binary extensions
EXCLUDED_BINARY_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dylib", ".dll", ".exe", ".bin",
    ".zip", ".tar", ".gz", ".7z", ".rar", ".db", ".sqlite", ".sqlite3", ".DS_Store"
}

# Default batch size for pagination across responses
EXPORT_BATCH_SIZE = 5


def is_allowed_file(file_path: Path, readme_and_pdf_only: bool = False) -> bool:
    name = file_path.name
    name_lower = name.lower()
    ext = file_path.suffix.lower()

    # Never allow media or binary
    if ext in EXCLUDED_MEDIA_EXTENSIONS or ext in EXCLUDED_BINARY_EXTENSIONS:
        return False
    if name == ".DS_Store" or name.startswith("._"):
        return False

    is_readme = name_lower.startswith("readme.") or name_lower == "readme"
    is_pdf = (ext == ".pdf")

    if readme_and_pdf_only:
        return is_readme or is_pdf

    # Source code extensions
    if ext in LANGUAGE_MAP:
        return True

    # Essential configs & README / PDF
    if is_readme or is_pdf:
        return True

    essential_configs = {
        "requirements.txt", "pyproject.toml", "package.json", "package-lock.json",
        "tsconfig.json", "dockerfile", "docker-compose.yml", "docker-compose.yaml",
        "makefile", ".gitignore"
    }
    if name_lower in essential_configs:
        return True

    return False


def detect_export_intent(
    query: str,
    repo_id: str = "",
    has_active_session: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Detects if a user query intends to export the complete repository source code,
    request all files sequentially, continue an existing export, or filter README/PDF.
    """
    q = query.strip()
    q_lower = q.lower()
    q_clean = re.sub(r'[?.!,;:]+$', '', q_lower).strip()

    # 1. Continuation check (Section 7)
    continuation_patterns = [
        r"^(?:continue|next|next\s+file|continue\s+the\s+remaining\s+code|baki\s+code\s+dao|next\s+file\s+er\s+code\s+dao)$",
        r"^(?:next\s+batch|continue\s+export|resume\s+export|baki\s+gulo\s+dao|aar\s+baki\s+code)$",
        r"^continue\s+(?:the\s+)?(?:remaining|pending|next)\s*(?:code|files)?$"
    ]
    if has_active_session:
        for pat in continuation_patterns:
            if re.match(pat, q_clean):
                return {
                    "is_export": True,
                    "is_continuation": True,
                    "readme_and_pdf_only": False,
                    "exclude_media": True
                }

    # 2. Strict Negative Checks (Section 1 - CRITICAL REQUIREMENT):
    # Do NOT activate when asking for a single specific file e.g. "Give me only graph.py"
    # or "Show main.py" or "line by line" or single function explanation.
    single_file_match = re.search(r'\b([a-zA-Z0-9_\-./]+\.(?:py|js|ts|tsx|jsx|html|css|json|yaml|yml|md|sql|go|java|c|cpp|h|rs))\b', q_lower)
    has_only_single_keyword = any(w in q_lower for w in ["only", "just", "single file"])
    is_asking_all_files = any(w in q_lower for w in [
        "all the code", "all code", "all files", "complete source code", "entire project code",
        "every code file", "entire repository", "complete codebase", "whole repo", "full source code",
        "sob code", "shob code", "puro project", "prottekta file", "ek ek kore", "file wise",
        "one by one", "sequentially", "file by file"
    ])

    if single_file_match and (has_only_single_keyword or not is_asking_all_files):
        # Specific single file requested -> NOT complete export mode
        return None

    if any(w in q_lower for w in ["explain function", "what does function", "how does function", "line by line of"]):
        return None

    # Check for pure directory structure without code (e.g. "show repo structure", "file structure")
    pure_structure_keywords = ["show repo structure", "repository structure", "file structure", "folder structure", "directory tree", "show directory hierarchy"]
    is_pure_structure = any(kw in q_lower for kw in pure_structure_keywords) and not is_asking_all_files and not any(w in q_lower for w in ["then", "code", "file", "dao", "provide", "give"])
    if is_pure_structure:
        return None

    # 3. Check for Selective Export (Section 11, Test 5: "Give me only the README and PDF files")
    readme_pdf_only = False
    if ("readme" in q_lower and "pdf" in q_lower) and any(w in q_lower for w in ["only", "just"]):
        readme_pdf_only = True
    elif "only the readme and pdf" in q_lower or "only readme and pdf" in q_lower:
        readme_pdf_only = True

    # 4. Check for Complete Project Code Requests (Section 1.A, 1.B, 1.C)
    export_phrases = [
        # English
        "give me all the code one by one",
        "give me all code one by one",
        "give me all the code",
        "give me all code",
        "give me the complete source code of this repository",
        "give me the complete source code",
        "show me the entire project code",
        "show the entire project code",
        "i want all files with their complete code",
        "provide the full source code file by file",
        "provide the full source code",
        "give me every code file in this project",
        "show the complete implementation of this repository",
        "export the entire repository source code",
        "export entire repository",
        "write all project files in their correct order",
        "give me the complete project from beginning to end",
        "show every file with its full content",
        "i need the complete codebase",
        "give me the entire github repository code",
        "show all source files sequentially",
        "first show the folder structure, then provide all code",
        "first show the folder structure, then give me all the code",
        "first show the folder structure then provide all code",
        "show the directory tree and explain every file with its code",
        "give me the project structure first and then all files one by one",
        "start with the repository structure and provide the complete source code",
        "show the folder hierarchy followed by each file's content",
        "give me the complete folder structure with all source files",
        "give me the code but don't include images or audio",
        "give me the code but dont include images or audio",
        # Bangla & Banglish
        "amake sob code ek ek kore dao",
        "project er sob code file wise dao",
        "age folder structure dekhao, tarpor sob code dao",
        "age folder structure dekhao tarpor sob code dao",
        "amar repository er complete code dao",
        "prottekta file er full code dao",
        "sob file serially dekhao",
        "puro project er source code chai",
        "folder structure onujayi code gulo dao",
        "ekta ekta kore sob file er code likhe dao",
        "shob code ek ek kore dao",
        "sob code dao"
    ]

    matched = any(phrase in q_lower for phrase in export_phrases)

    if not matched:
        # Check generalized pattern matches
        has_code_word = any(w in q_lower for w in ["code", "files", "source"])
        has_all_quantifier = any(w in q_lower for w in ["all", "entire", "complete", "every", "whole", "full", "sob", "shob", "puro"])
        has_sequential_word = any(w in q_lower for w in ["one by one", "file by file", "sequentially", "serially", "ek ek kore", "file wise", "ekta ekta kore"])

        if (has_all_quantifier and has_code_word and has_sequential_word) or (
            has_all_quantifier and has_code_word and any(w in q_lower for w in ["give", "show", "export", "provide", "dao", "chai"])
        ):
            matched = True

    if matched or readme_pdf_only:
        return {
            "is_export": True,
            "is_continuation": False,
            "readme_and_pdf_only": readme_pdf_only,
            "exclude_media": True
        }

    return None


def sanitize_sensitive_content(rel_path: str, content: str) -> str:
    """
    Sanitizes API keys, passwords, and private tokens in .env or sensitive config files (Section 10).
    """
    path_lower = rel_path.lower()
    is_sensitive_file = any(kw in path_lower for kw in [".env", "secret", "credential", "private_key", "id_rsa"])

    if not is_sensitive_file:
        return content

    # Redact common secret patterns
    patterns = [
        (r'(?i)(api[_-]?key\s*=\s*)([^\s\n#]+)', r'\1[REDACTED_API_KEY]'),
        (r'(?i)(password\s*=\s*)([^\s\n#]+)', r'\1[REDACTED_PASSWORD]'),
        (r'(?i)(token\s*=\s*)([^\s\n#]+)', r'\1[REDACTED_TOKEN]'),
        (r'(?i)(secret\s*=\s*)([^\s\n#]+)', r'\1[REDACTED_SECRET]'),
        (r'(?i)(client[_-]?secret\s*=\s*)([^\s\n#]+)', r'\1[REDACTED_SECRET]'),
        (r'(?i)(database[_-]?url\s*=\s*)([^\s\n#]+)', r'\1[REDACTED_DATABASE_URL]'),
    ]
    sanitized = content
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized)
    return sanitized


def build_export_manifest_and_tree(
    repo_dir: Path,
    root_name: str = "",
    readme_and_pdf_only: bool = False
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Scans the repository using deterministic depth-first directory traversal.
    Returns:
    1. GitHub-style directory tree formatted in a ```text code block.
    2. Ordered list of eligible file dictionaries matching the exact sequence in the tree.
    """
    if not repo_dir.exists() or not repo_dir.is_dir():
        return "```text\n(repository directory not found)\n```", []

    display_root = (root_name or repo_dir.name).rstrip("/") + "/"

    # Build hierarchical tree structure of eligible files
    tree_nodes: dict = {}
    eligible_file_count = 0

    for root, dirs, files in os.walk(repo_dir):
        # Exclude blacklisted directories
        dirs[:] = sorted([
            d for d in dirs
            if d not in EXCLUDED_DIRECTORIES and not d.startswith(".")
        ], key=lambda x: x.lower())

        rel_root = os.path.relpath(root, repo_dir)
        if rel_root == ".":
            rel_root = ""

        # Filter eligible files
        valid_files = []
        for f in sorted(files, key=lambda x: x.lower()):
            if f.startswith("."):
                continue
            file_path = Path(root) / f
            if is_allowed_file(file_path, readme_and_pdf_only=readme_and_pdf_only):
                valid_files.append(f)
                eligible_file_count += 1

        parts = Path(rel_root).parts if rel_root else ()
        curr = tree_nodes
        for p in parts:
            if p not in curr:
                curr[p] = {"is_dir": True, "children": {}}
            curr = curr[p]["children"]

        for d in dirs:
            if d not in curr:
                curr[d] = {"is_dir": True, "children": {}}
        for f in valid_files:
            curr[f] = {"is_dir": False, "children": {}}

    # Prune empty directories (directories containing no eligible files)
    def prune_empty(node: dict):
        to_del = []
        for name, data in node.items():
            if data["is_dir"]:
                prune_empty(data["children"])
                if not data["children"]:
                    to_del.append(name)
        for name in to_del:
            del node[name]

    prune_empty(tree_nodes)

    # Render GitHub-style tree and simultaneously collect ordered file manifest
    lines = [display_root]
    ordered_manifest: List[Dict[str, Any]] = []

    def _render(node: dict, current_rel_dir: str = "", prefix: str = ""):
        items = list(node.items())
        # Sort directories first, then files alphabetically
        dirs_items = sorted([item for item in items if item[1]["is_dir"]], key=lambda x: x[0].lower())
        files_items = sorted([item for item in items if not item[1]["is_dir"]], key=lambda x: x[0].lower())
        sorted_items = dirs_items + files_items

        for idx, (name, data) in enumerate(sorted_items):
            is_last = (idx == len(sorted_items) - 1)
            connector = "└── " if is_last else "├── "
            display_name = f"{name}/" if data["is_dir"] else name
            lines.append(f"{prefix}{connector}{display_name}")

            rel_item_path = f"{current_rel_dir}/{name}" if current_rel_dir else name

            if data["is_dir"]:
                child_prefix = prefix + ("    " if is_last else "│   ")
                _render(data["children"], rel_item_path, child_prefix)
            else:
                abs_path = repo_dir / rel_item_path
                ext = Path(name).suffix.lower()
                lang = LANGUAGE_MAP.get(ext, "text")
                is_pdf = (ext == ".pdf")
                is_readme = name.lower().startswith("readme.") or name.lower() == "readme"
                ordered_manifest.append({
                    "rel_path": rel_item_path,
                    "abs_path": str(abs_path),
                    "filename": name,
                    "language": lang,
                    "is_pdf": is_pdf,
                    "is_readme": is_readme
                })

    _render(tree_nodes)
    tree_text = "```text\n" + "\n".join(lines) + "\n```"

    return tree_text, ordered_manifest
