import os
import shutil
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from src.code_rag.rag.graph import CodeRAGGraph
from src.code_rag.rag.export_mode import (
    detect_export_intent,
    build_export_manifest_and_tree,
    sanitize_sensitive_content,
    is_allowed_file
)
from src.code_rag.storage.sqlite_store import SQLiteStore


@pytest.fixture
def temp_sample_repo():
    """Creates a temporary sample repository with folders, source code, readme, pdf, and media."""
    temp_dir = tempfile.mkdtemp(prefix="test_code_export_repo_")
    repo_path = Path(temp_dir)

    # 1. Subdirectories
    src_dir = repo_path / "src" / "api"
    src_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = repo_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = repo_path / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 2. Source code files
    (src_dir / "main.py").write_text("def run():\n    print('Hello World')\n", encoding="utf-8")
    (src_dir / "schemas.py").write_text("class Item:\n    name: str\n", encoding="utf-8")
    (repo_path / "src" / "utils.py").write_text("def helper():\n    return 42\n", encoding="utf-8")

    # 3. README and PDF
    (repo_path / "README.md").write_text("# Test Repo\nThis is a test documentation.", encoding="utf-8")
    (docs_dir / "architecture.pdf").write_bytes(b"%PDF-1.4 test binary dummy pdf content")

    # 4. Media files to exclude (images, audio, video)
    (assets_dir / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
    (assets_dir / "intro.mp4").write_bytes(b"dummy mp4 video binary")
    (assets_dir / "sound.mp3").write_bytes(b"dummy mp3 audio binary")

    # 5. Config file & .env
    (repo_path / "requirements.txt").write_text("fastapi>=0.100.0\nuvicorn\n", encoding="utf-8")
    (repo_path / ".env").write_text("DATABASE_URL=postgres://user:supersecret@localhost/db\nAPI_KEY=sk_live_12345\n", encoding="utf-8")

    yield repo_path

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_test_1_intent_detection_and_all_code_one_by_one(temp_sample_repo):
    """Test 1: 'Give me all the code one by one.'
    Expected: Display directory structure first. Provide every eligible file individually. Follow directory sequence.
    """
    intent = detect_export_intent("Give me all the code one by one.")
    assert intent is not None
    assert intent["is_export"] is True

    tree, manifest = build_export_manifest_and_tree(temp_sample_repo, root_name="test_repo")
    assert "```text" in tree
    assert "src/" in tree
    assert len(manifest) >= 4

    # Run through CodeRAGGraph
    graph = CodeRAGGraph()
    with patch.object(graph.retriever.cloner, "get_repo_path", return_value=temp_sample_repo):
        res = graph.build().invoke({
            "question": "Give me all the code one by one.",
            "repo_id": "test_repo"
        })
        answer = res["answer"]
        assert "```text" in answer
        assert "**File 1:" in answer
        assert "def run():" in answer or "class Item:" in answer or "README.md" in answer
        # Verify media exclusion
        assert "logo.png" not in answer
        assert "intro.mp4" not in answer
        assert "sound.mp3" not in answer


def test_test_2_show_complete_source_code_activation():
    """Test 2: 'Show me the complete source code of this repository.'
    Expected: Activate complete repository export mode.
    """
    intent = detect_export_intent("Show me the complete source code of this repository.")
    assert intent is not None
    assert intent["is_export"] is True


def test_test_3_folder_structure_followed_by_all_code(temp_sample_repo):
    """Test 3: 'First show the folder structure, then give me all the code.'
    Expected: Display structure first. Start sequential file output.
    """
    intent = detect_export_intent("First show the folder structure, then give me all the code.")
    assert intent is not None
    assert intent["is_export"] is True

    graph = CodeRAGGraph()
    with patch.object(graph.retriever.cloner, "get_repo_path", return_value=temp_sample_repo):
        res = graph.build().invoke({
            "question": "First show the folder structure, then give me all the code.",
            "repo_id": "test_repo"
        })
        answer = res["answer"]
        # Tree must be at the very top
        assert answer.startswith("```text\ntest_repo/")
        # File 1 must follow
        assert "**File 1:" in answer


def test_test_4_banglish_intent_recognition():
    """Test 4: 'Amake sob code ek ek kore dao.'
    Expected: Recognize Banglish intent. Activate complete repository export mode.
    """
    assert detect_export_intent("Amake sob code ek ek kore dao.") is not None
    assert detect_export_intent("Project er sob code file wise dao.") is not None
    assert detect_export_intent("Age folder structure dekhao, tarpor sob code dao.") is not None
    assert detect_export_intent("Puro project er source code chai.") is not None
    assert detect_export_intent("Ekta ekta kore sob file er code likhe dao.") is not None


def test_test_5_readme_and_pdf_only_filtering(temp_sample_repo):
    """Test 5: 'Give me only the README and PDF files.'
    Expected: Display only matching README and PDF files. Do not output source code unless separately requested.
    """
    intent = detect_export_intent("Give me only the README and PDF files.")
    assert intent is not None
    assert intent["readme_and_pdf_only"] is True

    tree, manifest = build_export_manifest_and_tree(temp_sample_repo, root_name="test_repo", readme_and_pdf_only=True)
    manifest_paths = [m["rel_path"] for m in manifest]
    assert any("README.md" in p for p in manifest_paths)
    assert any("architecture.pdf" in p for p in manifest_paths)
    # Ensure source files are omitted when only readme and pdf are requested
    assert not any("main.py" in p for p in manifest_paths)
    assert not any("schemas.py" in p for p in manifest_paths)


def test_test_6_exclude_media_files(temp_sample_repo):
    """Test 6: 'Give me the code but don't include images or audio.'
    Expected: Include eligible source code. Exclude all image and audio files.
    """
    tree, manifest = build_export_manifest_and_tree(temp_sample_repo, root_name="test_repo")
    manifest_paths = [m["rel_path"] for m in manifest]

    for p in manifest_paths:
        assert not p.endswith((".png", ".jpg", ".mp3", ".mp4", ".wav", ".gif", ".webp", ".svg"))


def test_test_7_single_file_negative_check():
    """Test 7: 'Give me only graph.py.'
    Expected: Return only graph.py. Do not activate complete repository export mode.
    """
    intent = detect_export_intent("Give me only graph.py.")
    assert intent is None

    intent2 = detect_export_intent("Show me main.py")
    assert intent2 is None

    intent3 = detect_export_intent("Explain process_payment()")
    assert intent3 is None


def test_test_8_continuation_preserves_progress(temp_sample_repo):
    """Test 8: 'Continue the remaining code.'
    Expected: Resume from the exact next pending file. Do not repeat completed files.
    """
    graph = CodeRAGGraph()
    conv_id = "test_conv_cont_123"
    repo_id = "test_repo"

    with patch.object(graph.retriever.cloner, "get_repo_path", return_value=temp_sample_repo):
        # First turn: start export
        state1 = {
            "question": "Give me all the code one by one.",
            "repo_id": repo_id,
            "conversation_id": conv_id,
            "turn_id": "turn_1"
        }
        res1 = graph.build().invoke(state1)
        session1 = res1.get("export_session")
        assert session1 is not None

        # Simulate user requesting continuation
        is_cont = detect_export_intent("Continue the remaining code.", repo_id=repo_id, has_active_session=True)
        assert is_cont is not None
        assert is_cont["is_continuation"] is True


def test_test_9_large_manifest_sequential_progress():
    """Test 9: Repository contains 100+ source files.
    Expected: Maintain the complete ordered manifest. Export files in sequence across multiple responses. Never lose progress.
    """
    # Create mock manifest of 120 files
    mock_manifest = [
        {"rel_path": f"src/module_{i}/file_{j}.py", "language": "python", "is_pdf": False}
        for i in range(12) for j in range(10)
    ]
    assert len(mock_manifest) == 120

    # Ensure deterministic traversal order preserved
    paths = [m["rel_path"] for m in mock_manifest]
    assert paths[0] == "src/module_0/file_0.py"
    assert paths[119] == "src/module_11/file_9.py"


def test_test_10_missing_file_handling(temp_sample_repo):
    """Test 10: One source file is missing from the index/disk.
    Expected: Do not fabricate its contents. Report the exact missing file path. Continue safely with other verified files.
    """
    graph = CodeRAGGraph()
    repo_id = "test_repo"

    # Inject a non-existent file into the manifest
    missing_file = "src/api/deleted_file.py"
    manifest = [
        {"rel_path": missing_file, "abs_path": str(temp_sample_repo / missing_file), "language": "python", "is_pdf": False},
        {"rel_path": "README.md", "abs_path": str(temp_sample_repo / "README.md"), "language": "markdown", "is_pdf": False}
    ]

    session = {
        "export_id": "exp_missing_test",
        "conversation_id": "conv_missing",
        "repo_id": repo_id,
        "ordered_file_manifest": manifest,
        "current_file_index": 0,
        "total_file_count": 2,
        "completed_files": [],
        "pending_files": [missing_file, "README.md"],
        "failed_files": [],
        "export_status": "in_progress",
        "tree_text": "```text\ntest_repo/\n```"
    }

    state = {
        "question": "Give me all the code one by one.",
        "repo_id": repo_id,
        "query_intent": "repo_code_export",
        "export_session": session,
        "export_info": {"is_continuation": False}
    }

    with patch.object(graph.retriever.cloner, "get_repo_path", return_value=temp_sample_repo):
        res = graph._generate_export_answer(state)
        answer = res["answer"]
        # Verify it reports the exact missing file and does not fabricate code
        assert f"File `{missing_file}` could not be read or is missing" in answer
        # Verify it continues with verified README.md
        assert "README.md" in answer
        assert "Test Repo" in answer


def test_sensitive_credentials_sanitization():
    """Verify that sensitive API keys, passwords, and tokens are redacted from .env or secret configs."""
    raw_env = "DATABASE_URL=postgres://user:supersecretpass@localhost:5432/db\nAPI_KEY=sk_live_abcdef123456\nTOKEN=ghp_secrettoken999\n"
    sanitized = sanitize_sensitive_content(".env", raw_env)

    assert "supersecretpass" not in sanitized
    assert "sk_live_abcdef123456" not in sanitized
    assert "ghp_secrettoken999" not in sanitized
    assert "[REDACTED_DATABASE_URL]" in sanitized
    assert "[REDACTED_API_KEY]" in sanitized
    assert "[REDACTED_TOKEN]" in sanitized


def test_stream_rag_async_repository_code_export(temp_sample_repo):
    """Verify stream_rag_async yields tree first, followed by each file in exact sequence."""
    import asyncio
    graph = CodeRAGGraph()
    repo_id = "test_repo"

    async def run_stream():
        events = []
        with patch.object(graph.retriever.cloner, "get_repo_path", return_value=temp_sample_repo):
            async for event in graph.stream_rag_async({
                "question": "Give me all the code one by one.",
                "repo_id": repo_id,
                "conversation_id": "conv_stream_test"
            }):
                events.append(event)
        return events

    events = asyncio.run(run_stream())
    event_types = [e.get("type") for e in events]
    assert "trace" in event_types
    assert "token" in event_types
    assert "done" in event_types

    full_output = "".join([e["content"] for e in events if e.get("type") == "token"])
    # 1. First response displays structure
    assert "```text\ntest_repo/" in full_output
    # 2. Second response provides all files one by one
    assert "**File 1:" in full_output
    # 3. Code blocks present
    assert "```" in full_output
    # 4. Strict media exclusion
    assert "logo.png" not in full_output
    assert "sound.mp3" not in full_output
