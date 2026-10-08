import os
import ast
import json
import re
import logging
from pathlib import Path
from typing import List, Set
from src.code_rag.config import settings
from src.code_rag.core.models import CodeSymbol

logger = logging.getLogger(__name__)

class CodeParser:
    def __init__(
        self,
        supported_extensions: Set[str] = settings.SUPPORTED_EXTENSIONS,
        special_filenames: Set[str] = settings.SPECIAL_FILENAMES,
        ignore_dirs: Set[str] = settings.IGNORE_DIRS,
        ignore_files: Set[str] = settings.IGNORE_FILES,
        max_size_kb: int = settings.MAX_FILE_SIZE_KB
    ):
        self.supported_extensions = supported_extensions
        self.special_filenames = special_filenames
        self.ignore_dirs = ignore_dirs
        self.ignore_files = ignore_files
        self.max_bytes = max_size_kb * 1024

    def scan_repository(self, repo_path: str) -> List[Path]:
        root = Path(repo_path).resolve()
        valid_files: List[Path] = []

        for parent, dirs, files in os.walk(root):
            dirs[:] = [
                d for d in dirs
                if d.lower() not in self.ignore_dirs
                and not (d.startswith('.') and d not in {'.'})
            ]

            for filename in files:
                if filename in self.ignore_files or ".min." in filename:
                    continue

                file_path = Path(parent) / filename
                is_supported = (
                    file_path.suffix.lower() in self.supported_extensions
                    or filename.lower() in self.special_filenames
                )

                if is_supported:
                    try:
                        if file_path.is_file() and file_path.stat().st_size <= self.max_bytes:
                            valid_files.append(file_path)
                    except OSError:
                        continue

        return sorted(valid_files)

    def parse_file(self, file_path: Path, repo_root: Path) -> List[CodeSymbol]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as err:
            logger.warning("Could not read file %s: %s", file_path, err)
            return []

        ext = file_path.suffix.lower()
        if ext == ".ipynb":
            return self._parse_ipynb(content, file_path.name)

        lines = content.splitlines(keepends=True)
        if not lines:
            return []

        total_lines = len(lines)
        if total_lines <= 50:
            return [CodeSymbol(
                name=file_path.name,
                symbol_type="file",
                start_line=1,
                end_line=total_lines,
                content="".join(lines)
            )]

        if ext == ".py":
            return self._parse_python(content, lines, file_path.name)
        elif ext in {".js", ".jsx", ".ts", ".tsx"}:
            return self._parse_js_ts(lines, file_path.name)
        
        return self._sliding_window(lines, window_size=60, overlap=12)

    def _parse_ipynb(self, content: str, filename: str) -> List[CodeSymbol]:
        symbols: List[CodeSymbol] = []
        try:
            nb = json.loads(content)
            cells = nb.get("cells", [])
            current_line = 1
            
            for idx, cell in enumerate(cells, 1):
                cell_type = cell.get("cell_type", "code")
                raw_source = cell.get("source", [])
                if isinstance(raw_source, list):
                    source_str = "".join(raw_source)
                else:
                    source_str = str(raw_source)
                
                source_str = source_str.strip()
                if not source_str:
                    continue
                
                cell_lines = source_str.splitlines(keepends=True)
                cell_len = len(cell_lines)
                end_line = current_line + max(1, cell_len) - 1

                if cell_type == "code":
                    header = f"# Notebook Cell [{idx}] (Code)\n"
                    symbols.append(CodeSymbol(
                        name=f"cell_{idx}_code",
                        symbol_type="notebook_code",
                        start_line=current_line,
                        end_line=end_line,
                        content=header + source_str
                    ))
                else:
                    header = f"# Notebook Cell [{idx}] (Markdown/Documentation)\n"
                    symbols.append(CodeSymbol(
                        name=f"cell_{idx}_markdown",
                        symbol_type="notebook_markdown",
                        start_line=current_line,
                        end_line=end_line,
                        content=header + source_str
                    ))
                current_line = end_line + 1

        except Exception as e:
            logger.warning("Failed to parse notebook JSON %s: %s", filename, e)
            lines = content.splitlines(keepends=True)
            return self._sliding_window(lines, window_size=60, overlap=12)

        return symbols if symbols else self._sliding_window(content.splitlines(keepends=True), window_size=60, overlap=12)

    def _parse_python(self, content: str, lines: List[str], filename: str) -> List[CodeSymbol]:
        symbols: List[CodeSymbol] = []
        try:
            tree = ast.parse(content)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, getattr(ast, 'AsyncFunctionDef', ()))):
                    start = node.lineno
                    end = getattr(node, 'end_lineno', start + len(node.body))
                    symbols.append(CodeSymbol(
                        name=node.name,
                        symbol_type="function",
                        start_line=start,
                        end_line=end,
                        content="".join(lines[start - 1:end]),
                        docstring=ast.get_docstring(node)
                    ))
                elif isinstance(node, ast.ClassDef):
                    start = node.lineno
                    end = getattr(node, 'end_lineno', start + len(node.body))
                    symbols.append(CodeSymbol(
                        name=node.name,
                        symbol_type="class",
                        start_line=start,
                        end_line=end,
                        content="".join(lines[start - 1:end]),
                        docstring=ast.get_docstring(node)
                    ))
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, getattr(ast, 'AsyncFunctionDef', ()))):
                            sub_start = sub.lineno
                            sub_end = getattr(sub, 'end_lineno', sub_start + len(sub.body))
                            symbols.append(CodeSymbol(
                                name=f"{node.name}.{sub.name}",
                                symbol_type="method",
                                start_line=sub_start,
                                end_line=sub_end,
                                content="".join(lines[sub_start - 1:sub_end]),
                                docstring=ast.get_docstring(sub),
                                parent_name=node.name
                            ))
        except Exception:
            pass

        # If AST symbols found, also add module top-level overview if not already captured
        if symbols:
            # Add top of file (imports / global config) if first symbol starts after line 15
            if symbols[0].start_line > 15:
                symbols.insert(0, CodeSymbol(
                    name=f"{filename}_header",
                    symbol_type="module_header",
                    start_line=1,
                    end_line=symbols[0].start_line - 1,
                    content="".join(lines[:symbols[0].start_line - 1])
                ))
            return symbols

        # Fallback to sliding window only if no structural AST symbols found
        return self._sliding_window(lines, window_size=150, overlap=20)

    def _parse_js_ts(self, lines: List[str], filename: str) -> List[CodeSymbol]:
        symbols: List[CodeSymbol] = []
        pattern = re.compile(
            r'^(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:function|class)\s+([a-zA-Z0-9_$]+)|^(?:export\s+)?const\s+([A-Z][a-zA-Z0-9_$]+)\s*=\s*(?:async\s*)?\('
        )
        for i, line in enumerate(lines):
            match = pattern.match(line.strip())
            if match:
                name = match.group(1) or match.group(2)
                if name:
                    start = i + 1
                    end = min(start + 150, len(lines))
                    symbols.append(CodeSymbol(
                        name=name,
                        symbol_type="function" if "function" in line or "=>" in line or "(" in line else "class",
                        start_line=start,
                        end_line=end,
                        content="".join(lines[start - 1:end])
                    ))

        if symbols:
            return symbols

        return self._sliding_window(lines, window_size=150, overlap=20)

    def _sliding_window(self, lines: List[str], window_size: int = 150, overlap: int = 20) -> List[CodeSymbol]:
        total = len(lines)
        if total <= window_size:
            return [CodeSymbol(
                name="full_content",
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
                name=f"section_{idx}_L{start+1}-L{end}",
                symbol_type="block",
                start_line=start + 1,
                end_line=end,
                content="".join(lines[start:end])
            ))
            if end >= total:
                break
        return chunks

