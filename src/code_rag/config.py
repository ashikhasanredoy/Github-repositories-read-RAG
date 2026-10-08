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
    EMBEDDING_MODEL: str = "all-minilm:latest"
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 8192
    NUM_CTX: int = 16384

    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    REPOS_DIR: Path = DATA_DIR / "repos"
    CHROMA_DIR: Path = DATA_DIR / "chroma_db"
    BM25_DIR: Path = DATA_DIR / "bm25_indices"

    MAX_FILE_SIZE_KB: int = 5000
    MAX_CHUNK_CHARS: int = 4000
    EMBED_MAX_CHARS: int = 2000
    SUPPORTED_EXTENSIONS: Set[str] = {
        ".py", ".ipynb", ".js", ".ts", ".jsx", ".tsx",
        ".html", ".css", ".scss", ".sass", ".less",
        ".json", ".jsonl", ".xml", ".txt", ".csv", ".tsv",
        ".java", ".cpp", ".c", ".h", ".hpp", ".cc", ".cxx",
        ".go", ".rs", ".rb", ".php", ".cs", ".swift", ".kt", ".dart", ".scala", ".lua",
        ".md", ".markdown", ".rst", ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".sql", ".prisma", ".graphql", ".gql", ".sh", ".bash", ".zsh", ".dockerfile", ".env",
        ".vue", ".svelte", ".proto"
    }
    SPECIAL_FILENAMES: Set[str] = {
        "dockerfile", "makefile", "jenkinsfile", "requirements.txt",
        ".gitignore", ".dockerignore", "gemfile", "procfile", "caddyfile"
    }
    IGNORE_DIRS: Set[str] = {
        ".git", ".github", "node_modules", "venv", ".venv", "env",
        "__pycache__", "dist", "build", ".next", ".cache",
        "coverage", ".pytest_cache", ".idea", ".vscode", ".ipynb_checkpoints",
        "data", "chroma_db", "bm25_indices", "repos", ".mypy_cache", ".ruff_cache",
        "vendor", "out", "target", ".turbo", ".nuxt", ".svelte-kit", "site-packages"
    }
    IGNORE_FILES: Set[str] = {
        "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
        "poetry.lock", "Pipfile.lock", "Cargo.lock", "composer.lock", "bun.lockb"
    }

    VECTOR_TOP_K: int = 10
    BM25_TOP_K: int = 10
    FINAL_TOP_K: int = 4
    RRF_K: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

for folder in [settings.DATA_DIR, settings.REPOS_DIR, settings.CHROMA_DIR, settings.BM25_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
