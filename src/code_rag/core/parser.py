import os
import ast
import re
import logging
from pathlib import Path
from typing import List
from src.code_rag.config import settings
from src.code_rag.core.models import CodeSymbol

logger = logging.getLogger(__name__)

class CodeParser:
    def __init__(
        self,
        supported_extensions=settings.SUPPORTED_EXTENSIONS,
        ignore_dirs=settings.IGNORE_DIRS,
        ignore_files=settings.IGNORE_FILES,
        max_size_kb=settings.MAX_FILE_SIZE_KB
    ):
        self.supported_extensions = supported_extensions
        self.ignore_dirs = ignore_dirs
        self.ignore_files = ignore_files
        self.max_bytes = max_size_kb * 1024

    def scan_repository(self, repo_path: str) -> List[Path]:
        root = Path(repo_path).resolve()
        valid_files: List[Path] = []

        for parent, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs and not d.startswith('.')]

            for filename in files:
                if filename in self.ignore_files or ".min." in filename:
                    continue

                file_path = Path(parent) / filename
                if file_path.suffix.lower() in self.supported_extensions:
                    try:
                        if file_path.stat().st_size <= self.max_bytes:
                            valid_files.append(file_path)
                    except OSError:
                        continue

        return valid_files

    def parse_file(self, file_path: Path, repo_root: Path) -> List[CodeSymbol]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as err:
            logger.warning("Could not read file %s: %s", file_path, err)
            return []

        lines = content.splitlines(keepends=True)
        if not lines:
            return []

        ext = file_path.suffix.lower()
        if ext == ".py":
            return self._parse_python(content, lines)
        elif ext in {".js", ".jsx", ".ts", ".tsx"}:
            return self._parse_js_ts(lines)
        elif ext in {".go", ".java", ".cpp", ".c", ".rs"}:
            return self._parse_generic_code(lines)
        
        return self._sliding_window(lines)

    def _parse_python(self, content: str, lines: List[str]) -> List[CodeSymbol]:
        symbols: List[CodeSymbol] = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._sliding_window(lines)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, getattr(ast, 'AsyncFunctionDef', ()))):
                start = node.lineno
                end = getattr(node, 'end_lineno', start + len(node.body))
                # Bound large functions
                chunk_code = "".join(lines[start - 1:min(end, start + 50)])
                symbols.append(CodeSymbol(
                    name=node.name,
                    symbol_type="function",
                    start_line=start,
                    end_line=end,
                    content=chunk_code,
                    docstring=ast.get_docstring(node)
                ))
            elif isinstance(node, ast.ClassDef):
                start = node.lineno
                end = getattr(node, 'end_lineno', start + len(node.body))
                # Bound class overview to first 30 lines (methods are indexed individually)
                class_overview_end = min(end, start + 30)
                symbols.append(CodeSymbol(
                    name=node.name,
                    symbol_type="class",
                    start_line=start,
                    end_line=class_overview_end,
                    content="".join(lines[start - 1:class_overview_end]),
                    docstring=ast.get_docstring(node)
                ))
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, getattr(ast, 'AsyncFunctionDef', ()))):
                        sub_start = sub.lineno
                        sub_end = getattr(sub, 'end_lineno', sub_start + len(sub.body))
                        method_end = min(sub_end, sub_start + 45)
                        symbols.append(CodeSymbol(
                            name=f"{node.name}.{sub.name}",
                            symbol_type="method",
                            start_line=sub_start,
                            end_line=sub_end,
                            content="".join(lines[sub_start - 1:method_end]),
                            docstring=ast.get_docstring(sub),
                            parent_name=node.name
                        ))

        return symbols if symbols else self._sliding_window(lines)

    def _parse_js_ts(self, lines: List[str]) -> List[CodeSymbol]:
        pattern = re.compile(
            r'^(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:function|class)\s+([a-zA-Z0-9_$]+)|^(?:export\s+)?const\s+([A-Z][a-zA-Z0-9_$]+)\s*=\s*(?:async\s*)?\('
        )
        symbols: List[CodeSymbol] = []
        for i, line in enumerate(lines):
            match = pattern.match(line.strip())
            if match:
                name = match.group(1) or match.group(2)
                if not name:
                    continue
                start = i + 1
                end = min(start + 50, len(lines))
                symbols.append(CodeSymbol(
                    name=name,
                    symbol_type="function" if "function" in line or "=>" in line or "(" in line else "class",
                    start_line=start,
                    end_line=end,
                    content="".join(lines[start - 1:end])
                ))
        return symbols if symbols else self._sliding_window(lines, window_size=80, overlap=10)

    def _parse_generic_code(self, lines: List[str]) -> List[CodeSymbol]:
        pattern = re.compile(r'^(?:pub\s+|func\s+|def\s+)?(?:class|struct|interface|fn|func)\s+([a-zA-Z0-9_]+)')
        symbols: List[CodeSymbol] = []
        for i, line in enumerate(lines):
            match = pattern.match(line.strip())
            if match:
                start = i + 1
                end = min(start + 50, len(lines))
                symbols.append(CodeSymbol(
                    name=match.group(1),
                    symbol_type="block",
                    start_line=start,
                    end_line=end,
                    content="".join(lines[start - 1:end])
                ))
        return symbols if symbols else self._sliding_window(lines, window_size=80, overlap=10)

    def _sliding_window(self, lines: List[str], window_size: int = 80, overlap: int = 10) -> List[CodeSymbol]:
        total = len(lines)
        if total <= window_size:
            return [CodeSymbol(
                name="module_content",
                symbol_type="module",
                start_line=1,
                end_line=total,
                content="".join(lines)
            )]

        step = max(1, window_size - overlap)
        chunks: List[CodeSymbol] = []
        for idx, start in enumerate(range(0, total, step), 1):
            end = min(start + window_size, total)
            chunks.append(CodeSymbol(
                name=f"chunk_{idx}",
                symbol_type="block",
                start_line=start + 1,
                end_line=end,
                content="".join(lines[start:end])
            ))
            if end >= total:
                break
        return chunks
