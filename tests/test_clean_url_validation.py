import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from code_rag.core.cloner import RepoCloner

def test_clean_github_url_validation(tmp_path):
    cloner = RepoCloner(repos_dir=tmp_path)
    
    # Invalid URLs with query params, /tree/, /blob/, or /chatgpt.com
    invalid_urls = [
        "https://github.com/ashikhasanredoy?tab=repositories.chatgpt",
        "https://github.com/ashikhasanredoy/Github-repositories-read-RAG/chatgpt.com",
        "https://github.com/ashikhasanredoy/Github-repositories-read-RAG/tree/main",
        "https://github.com/ashikhasanredoy/Github-repositories-read-RAG/tree/main/src",
        "https://github.com/ashikhasanredoy/Github-repositories-read-RAG/blob/main/README.md",
        "https://github.com/owner/repo/extra/path",
        "https://github.com/ashikhasanredoy",
    ]
    
    for url in invalid_urls:
        with pytest.raises(ValueError) as excinfo:
            cloner.clone_or_load(url)
        assert "Malformed GitHub URL" in str(excinfo.value)

def test_valid_github_url_passes_validation(tmp_path):
    cloner = RepoCloner(repos_dir=tmp_path)
    
    valid_urls = [
        "https://github.com/ashikhasanredoy/Github-repositories-read-RAG",
        "https://github.com/ashikhasanredoy/Github-repositories-read-RAG.git",
        "https://github.com/owner/repo/",
    ]
    
    with patch("git.Repo.clone_from") as mock_clone, \
         patch.object(RepoCloner, "save_repo_path"):
        mock_repo = MagicMock()
        mock_repo.head.commit.hexsha = "abcdef123456"
        mock_clone.return_value = mock_repo
        
        for url in valid_urls:
            res = cloner.clone_or_load(url, force_reclone=True)
            assert res is not None
            assert "local_path" in res
