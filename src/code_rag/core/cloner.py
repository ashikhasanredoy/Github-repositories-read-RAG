import os
import re
import json
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
        self.registry_file = settings.DATA_DIR / "repo_paths.json"

    def _load_registry(self) -> Dict[str, str]:
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as err:
                logger.warning("Failed to load repo_paths.json: %s", err)
        return {}

    def _save_registry(self, registry: Dict[str, str]) -> None:
        try:
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)
        except Exception as err:
            logger.warning("Failed to save repo_paths.json: %s", err)

    def save_repo_path(self, repo_id: str, local_path: str) -> None:
        registry = self._load_registry()
        registry[repo_id] = str(Path(local_path).resolve())
        self._save_registry(registry)

    def get_repo_path(self, repo_id: str) -> Optional[Path]:
        registry = self._load_registry()
        if repo_id in registry:
            p = Path(registry[repo_id])
            if p.exists() and p.is_dir():
                return p

        clean_name = self.sanitize_repo_name(repo_id)
        candidate = self.repos_dir / clean_name
        if candidate.exists() and candidate.is_dir():
            return candidate

        # Check in current workspace or parent if repo_id matches
        if Path(repo_id).exists() and Path(repo_id).is_dir():
            return Path(repo_id).resolve()

        return None

    @staticmethod
    def sanitize_repo_name(source: str) -> str:
        clean = source.strip().rstrip("/")
        if clean.endswith(".git"):
            clean = clean[:-4]
        clean = clean.rstrip("/")
        name = clean.split("/")[-1]
        name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
        name = name.strip("._-").lower()
        return name or "repo"

    def clone_or_load(self, source: str, force_reclone: bool = False) -> Dict[str, Any]:
        is_remote = source.startswith(("http://", "https://", "git@"))
        repo_name = self.sanitize_repo_name(source)
        target_path = self.repos_dir / repo_name

        if not is_remote:
            local_path = Path(source).resolve()
            if not local_path.exists() or not local_path.is_dir():
                raise FileNotFoundError(f"Local repository directory not found: {source}")
            
            self.save_repo_path(repo_name, str(local_path))
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

        self.save_repo_path(repo_name, str(target_path))
        return {
            "repo_id": repo_name,
            "repo_name": repo_name,
            "source": source,
            "is_remote": True,
            "local_path": str(target_path)
        }

    def delete_repo(self, repo_id: str) -> bool:
        clean_name = self.sanitize_repo_name(repo_id)
        target_path = self.repos_dir / clean_name
        registry = self._load_registry()
        if repo_id in registry:
            del registry[repo_id]
            self._save_registry(registry)
        if target_path.exists() and target_path.is_dir():
            try:
                shutil.rmtree(target_path, ignore_errors=True)
                logger.info("Successfully deleted cloned repo directory: %s", target_path)
                return True
            except Exception as err:
                logger.warning("Failed to remove cloned repo directory %s: %s", target_path, err)
                return False
        return False
