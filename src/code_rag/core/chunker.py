import hashlib
from pathlib import Path
from typing import List, Optional
from src.code_rag.config import settings
from src.code_rag.core.models import CodeChunk
from src.code_rag.core.parser import CodeParser

class CodeChunker:
    LANGUAGE_MAP = {
        ".py": "python", ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".go": "go",
        ".java": "java", ".cpp": "cpp", ".c": "c", ".rs": "rust",
        ".rb": "ruby", ".php": "php", ".md": "markdown",
        ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".sql": "sql", ".sh": "bash"
    }

    def __init__(self, parser: Optional[CodeParser] = None, max_chunk_chars: int = settings.MAX_CHUNK_CHARS):
        self.parser = parser or CodeParser()
        self.max_chunk_chars = max_chunk_chars

    def chunk_repository(self, repo_id: str, repo_path: str) -> List[CodeChunk]:
        root = Path(repo_path).resolve()
        files = self.parser.scan_repository(str(root))
        chunks: List[CodeChunk] = []

        for file_path in files:
            rel_path = str(file_path.relative_to(root))
            lang = self.LANGUAGE_MAP.get(file_path.suffix.lower(), "text")
            symbols = self.parser.parse_file(file_path, root)

            for sym in symbols:
                if not sym.content.strip():
                    continue

                # Truncate content to safe character length for embeddings
                clean_content = sym.content[:self.max_chunk_chars]

                header = (
                    f"File: {rel_path}\n"
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
