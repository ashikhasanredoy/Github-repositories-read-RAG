import hashlib
from pathlib import Path
from typing import List, Optional
from src.code_rag.config import settings
from src.code_rag.core.models import CodeChunk
from src.code_rag.core.parser import CodeParser

def generate_ascii_tree(relative_paths: List[str]) -> str:
    tree: dict = {}
    for path in sorted(relative_paths):
        parts = Path(path).parts
        curr = tree
        for part in parts:
            curr = curr.setdefault(part, {})

    lines: List[str] = []
    def _render(node: dict, prefix: str = ""):
        keys = sorted(node.keys(), key=lambda k: (len(node[k]) > 0, k.lower()))
        for idx, key in enumerate(keys):
            is_last = (idx == len(keys) - 1)
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{key}")
            child_prefix = prefix + ("    " if is_last else "│   ")
            _render(node[key], child_prefix)

    _render(tree)
    return "\n".join(lines)

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
        ascii_tree = generate_ascii_tree(rel_paths)

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
            f"**Repository File Structure:**\n"
            f"```markdown\n"
            f"{ascii_tree}\n"
            f"```\n"
            f"Note: The above structure is a direct representation of the repository structure as provided in the repository context.\n\n"
            f"The repository structure is as follows:\n\n"
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

