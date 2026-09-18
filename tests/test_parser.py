import tempfile
from pathlib import Path
from src.code_rag.core.parser import CodeParser
from src.code_rag.core.chunker import CodeChunker

def test_ast_python_parsing():
    sample_code = """def calculate_average(nums: list) -> float:
    \"\"\"Calculate mean of numbers.\"\"\"
    return sum(nums) / len(nums) if nums else 0.0

class Predictor:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def predict(self, data: dict) -> dict:
        \"\"\"Inference.\"\"\"
        return {"result": 1}
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        py_file = tmp_path / "test_pipeline.py"
        py_file.write_text(sample_code)

        parser = CodeParser()
        symbols = parser.parse_file(py_file, tmp_path)

        names = [s.name for s in symbols]
        assert "calculate_average" in names
        assert "Predictor" in names
        assert "Predictor.predict" in names

        avg_sym = next(s for s in symbols if s.name == "calculate_average")
        assert avg_sym.start_line == 1
        assert avg_sym.symbol_type == "function"

def test_chunker_metadata():
    sample_code = """def health():
    return {"status": "ok"}
"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        py_file = tmp_path / "main.py"
        py_file.write_text(sample_code)

        chunker = CodeChunker()
        chunks = chunker.chunk_repository("test_repo", str(tmp_path))

        assert len(chunks) >= 1
        c = chunks[0]
        assert c.repo_id == "test_repo"
        assert c.symbol_name == "health"
        assert c.language == "python"
