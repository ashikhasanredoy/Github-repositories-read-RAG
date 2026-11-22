import hashlib
from pathlib import Path
from typing import List, Optional
from src.code_rag.config import settings
from src.code_rag.core.models import CodeChunk
from src.code_rag.core.parser import CodeParser

def generate_ascii_tree(relative_paths: List[str], root_name: str = "") -> str:
    tree: dict = {}
    for path in sorted(relative_paths):
        parts = Path(path).parts
        curr = tree
        for i, part in enumerate(parts):
            if part not in curr:
                curr[part] = {"is_dir": (i < len(parts) - 1), "children": {}}
            elif i < len(parts) - 1:
                curr[part]["is_dir"] = True
            curr = curr[part]["children"]

    lines: List[str] = []
    if root_name:
        lines.append(root_name.rstrip("/") + "/")

    def _render(node: dict, prefix: str = ""):
        # Sort directories first, then files alphabetically
        items = list(node.items())
        dirs_items = sorted([item for item in items if item[1]["is_dir"]], key=lambda x: x[0].lower())
        files_items = sorted([item for item in items if not item[1]["is_dir"]], key=lambda x: x[0].lower())
        sorted_items = dirs_items + files_items

        for idx, (name, data) in enumerate(sorted_items):
            is_last = (idx == len(sorted_items) - 1)
            connector = "└── " if is_last else "├── "
            display_name = f"{name}/" if data["is_dir"] else name
            lines.append(f"{prefix}{connector}{display_name}")
            if data["is_dir"]:
                child_prefix = prefix + ("    " if is_last else "│   ")
                _render(data["children"], child_prefix)

    _render(tree)
    return "\n".join(lines)


def generate_github_style_tree(
    repo_dir: Path,
    root_name: str = "",
    subfolder: str = "",
    filter_exts: Optional[List[str]] = None,
    include_hidden: bool = False
) -> str:
    import os

    target_dir = repo_dir
    display_root = root_name or repo_dir.name
    if subfolder:
        clean_sub = subfolder.strip("/")
        target_dir = repo_dir / clean_sub
        display_root = f"{display_root}/{clean_sub}"

    display_root = display_root.rstrip("/") + "/"
    if not target_dir.exists():
        return f"```text\n{display_root}\n(directory not found)\n```"

    filter_exts_norm = [e.lower() if e.startswith(".") else f".{e.lower()}" for e in filter_exts] if filter_exts else None

    tree_nodes: dict = {}

    for root, dirs, files in os.walk(target_dir):
        rel_root = os.path.relpath(root, target_dir)
        if rel_root == ".":
            rel_root = ""

        valid_dirs = []
        for d in dirs:
            if d.startswith(".") and not include_hidden:
                continue
            if d == "__pycache__" and not include_hidden and (not subfolder or "__pycache__" not in subfolder):
                continue
            valid_dirs.append(d)
        dirs[:] = valid_dirs

        valid_files = []
        for f in files:
            if f.startswith(".") and not include_hidden:
                continue
            if filter_exts_norm:
                if not any(f.lower().endswith(ext) for ext in filter_exts_norm):
                    continue
            valid_files.append(f)

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

    if filter_exts_norm:
        def prune_empty(node):
            to_del = []
            for name, data in node.items():
                if data["is_dir"]:
                    prune_empty(data["children"])
                    if not data["children"]:
                        to_del.append(name)
            for name in to_del:
                del node[name]
        prune_empty(tree_nodes)

    lines = [display_root]

    def _render(node: dict, prefix: str = ""):
        items = list(node.items())
        dirs_items = sorted([item for item in items if item[1]["is_dir"]], key=lambda x: x[0].lower())
        files_items = sorted([item for item in items if not item[1]["is_dir"]], key=lambda x: x[0].lower())
        sorted_items = dirs_items + files_items

        for idx, (name, data) in enumerate(sorted_items):
            is_last = (idx == len(sorted_items) - 1)
            connector = "└── " if is_last else "├── "
            display_name = f"{name}/" if data["is_dir"] else name
            lines.append(f"{prefix}{connector}{display_name}")
            if data["is_dir"]:
                child_prefix = prefix + ("    " if is_last else "│   ")
                _render(data["children"], child_prefix)

    _render(tree_nodes)
    return "```text\n" + "\n".join(lines) + "\n```"

class CodeChunker:
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

    def __init__(self, parser: Optional[CodeParser] = None, max_chunk_chars: int = settings.MAX_CHUNK_CHARS):
        self.parser = parser or CodeParser()
        self.max_chunk_chars = max_chunk_chars

    def chunk_repository(self, repo_id: str, repo_path: str) -> List[CodeChunk]:
        root = Path(repo_path).resolve()
        files = self.parser.scan_repository(str(root))
        chunks: List[CodeChunk] = []

        if not files:
            return []

        rel_paths = [str(f.relative_to(root)) for f in sorted(files)]
        ascii_tree = generate_ascii_tree(rel_paths, root_name=repo_id)

        # 1. Group directories and root files for comprehensive explanation
        dirs_set = sorted({str(Path(p).parts[0]) for p in rel_paths if len(Path(p).parts) > 1})
        root_files = sorted({p for p in rel_paths if len(Path(p).parts) == 1})

        dir_breakdown = []
        if dirs_set:
            dir_breakdown.append("- The repository contains the following directories:")
            for d in dirs_set:
                sub_count = len([p for p in rel_paths if p.startswith(f"{d}/")])
                dir_breakdown.append(f"  - `{d}/`: Contains {sub_count} files.")

        if root_files:
            dir_breakdown.append("- The repository also contains the following root files:")
            for rf in root_files:
                dir_breakdown.append(f"  - `{rf}`: Root configuration / entrypoint file.")

        breakdown_str = "\n".join(dir_breakdown)

        structure_content = (
            f"```text\n"
            f"{ascii_tree}\n"
            f"```\n\n"
            f"{breakdown_str}"
        )

        chunks.append(CodeChunk(
            chunk_id=f"{repo_id}_file_tree",
            repo_id=repo_id,
            rel_path="REPOSITORY_STRUCTURE.md",
            language="markdown",
            symbol_name="repository_file_tree",
            symbol_type="overview",
            start_line=1,
            end_line=len(files),
            code_content=structure_content,
            formatted_content=(
                f"File: REPOSITORY_STRUCTURE.md\n"
                f"Directory: /\n"
                f"Language: markdown\n"
                f"Symbol: repository_file_tree (overview)\n\n"
                f"{structure_content}"
            ),
            docstring="Complete exact list and ASCII tree of all files and folders in this repository."
        ))

        # 2. Chunk every file in root and all subdirectories
        for file_path in files:
            rel_path = str(file_path.relative_to(root))
            parent_dir = str(file_path.relative_to(root).parent)
            lang = self.LANGUAGE_MAP.get(file_path.suffix.lower(), "text")
            symbols = self.parser.parse_file(file_path, root)

            for sym in symbols:
                if not sym.content.strip():
                    continue

                clean_content = sym.content[:self.max_chunk_chars]

                header = (
                    f"File: {rel_path}\n"
                    f"Directory: {parent_dir}\n"
                    f"Language: {lang}\n"
                    f"Symbol: {sym.name} ({sym.symbol_type})\n"
                    f"Lines: {sym.start_line}-{sym.end_line}\n\n"
                )
                formatted_content = header + clean_content

                chunk_hash = hashlib.md5(
                    f"{repo_id}:{rel_path}:{sym.symbol_type}:{sym.name}:{sym.start_line}:{sym.end_line}".encode()
                ).hexdigest()[:16]

                chunks.append(CodeChunk(
                    chunk_id=f"{repo_id}_{chunk_hash}",
                    repo_id=repo_id,
                    rel_path=rel_path,
                    language=lang,
                    symbol_name=sym.name,
                    symbol_type=sym.symbol_type,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                    code_content=clean_content,
                    formatted_content=formatted_content,
                    docstring=sym.docstring or ""
                ))

        return chunks

