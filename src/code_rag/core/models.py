from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class CodeSymbol:
    name: str
    symbol_type: str
    start_line: int
    end_line: int
    content: str
    docstring: Optional[str] = None
    parent_name: Optional[str] = None

@dataclass
class CodeChunk:
    chunk_id: str
    repo_id: str
    rel_path: str
    language: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    code_content: str
    formatted_content: str
    docstring: str = ""

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "repo_id": self.repo_id,
            "rel_path": self.rel_path,
            "language": self.language,
            "symbol_name": self.symbol_name,
            "symbol_type": self.symbol_type,
            "start_line": int(self.start_line),
            "end_line": int(self.end_line),
            "docstring": self.docstring
        }
