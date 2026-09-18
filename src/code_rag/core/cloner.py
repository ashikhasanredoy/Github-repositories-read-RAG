import os
import re
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import git
from src.code_rag.config import settings

logger = logging.getLogger(__name__)

class RepoCloner:
    def __init__(self, repos_dir: Optional[Path] = None):
        self.repos_dir = repos_dir or settings.REPOS_DIR

    @staticmethod
    def sanitize_repo_name(source: str) -> str:
        clean = source.strip().rstrip("/")
        if clean.endswith(".git"):
            clean = clean[:-4]
        clean = clean.rstrip("/")
        name = clean.split("/")[-1]
        name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        name = name.strip("._-")
        return name or "repo"

    def clone_or_load(self, source: str, force_reclone: bool = False) -> Dict[str, Any]:
        is_remote = source.startswith(("http://", "https://", "git@"))
        repo_name = self.sanitize_repo_name(source)
        target_path = self.repos_dir / repo_name

        if not is_remote:
            local_path = Path(source).resolve()
            if not local_path.exists() or not local_path.is_dir():
                raise FileNotFoundError(f"Local repository directory not found: {source}")
            
            return {
                "repo_id": repo_name,
                "repo_name": repo_name,
                "source": source,
                "is_remote": False,
                "local_path": str(local_path)
            }

        if target_path.exists() and force_reclone:
            shutil.rmtree(target_path, ignore_errors=True)

        if not target_path.exists():
            target_path.mkdir(parents=True, exist_ok=True)
            try:
                git.Repo.clone_from(source, str(target_path), depth=1)
            except Exception as err:
                shutil.rmtree(target_path, ignore_errors=True)
                raise RuntimeError(f"Git clone failed for {source}: {err}") from err

        return {
            "repo_id": repo_name,
            "repo_name": repo_name,
            "source": source,
            "is_remote": True,
            "local_path": str(target_path)
        }
