import os
from pathlib import Path
from typing import Set
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    PROJECT_NAME: str = "GitHub Code RAG"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "olmo:latest"
    EMBEDDING_MODEL: str = "nomic-embed-text"
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 2048

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    REPOS_DIR: Path = DATA_DIR / "repos"
    CHROMA_DIR: Path = DATA_DIR / "chroma_db"
    BM25_DIR: Path = DATA_DIR / "bm25_indices"

    MAX_FILE_SIZE_KB: int = 300
    MAX_CHUNK_CHARS: int = 2000
    SUPPORTED_EXTENSIONS: Set[str] = {
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".java", ".cpp", ".c", ".h", ".hpp",
        ".go", ".rs", ".rb", ".php",
        ".md", ".yaml", ".yml", ".toml", ".sql", ".sh"
    }
    IGNORE_DIRS: Set[str] = {
        ".git", ".github", "node_modules", "venv", ".venv", "env",
        "__pycache__", "dist", "build", "target", ".next", ".cache",
        "vendor", "coverage", ".pytest_cache", ".idea", ".vscode"
    }
    IGNORE_FILES: Set[str] = {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "poetry.lock", "Pipfile.lock", "Cargo.lock"
    }

    VECTOR_TOP_K: int = 6
    BM25_TOP_K: int = 6
    FINAL_TOP_K: int = 3
    RRF_K: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

for folder in [settings.DATA_DIR, settings.REPOS_DIR, settings.CHROMA_DIR, settings.BM25_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
