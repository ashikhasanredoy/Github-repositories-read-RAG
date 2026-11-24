import os
import pytest
import tempfile
from fastapi.testclient import TestClient

from code_rag.storage.sqlite_store import SQLiteStore
from code_rag.api.main import app, sqlite_store

@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    store = SQLiteStore(db_path=db_path)
    yield store
    if os.path.exists(db_path):
        os.remove(db_path)

def test_conversation_lifecycle(temp_db):
    conv_id = "test_conv_1"
    conv = temp_db.create_or_get_conversation(conv_id, title="Test Conversation", repo_id="Repo Test")
    assert conv["conversation_id"] == conv_id
    assert conv["title"] == "Test Conversation"

    # Listing
    convs = temp_db.list_conversations()
    assert len(convs) == 1
    assert convs[0]["conversation_id"] == conv_id

    # Renaming
    renamed = temp_db.rename_conversation(conv_id, "Updated Title")
    assert renamed is True
    updated = temp_db.get_conversation(conv_id)
    assert updated["title"] == "Updated Title"

def test_turns_and_cascade_deletion(temp_db):
    conv_id = "test_conv_2"
    temp_db.create_or_get_conversation(conv_id, title="Q&A Session", repo_id="repo1")

    # Turn 1
    t1 = "turn_1"
    temp_db.create_turn(conv_id, t1)
    temp_db.add_user_message(conv_id, t1, "What is AST?")
    temp_db.create_pending_assistant_message(conv_id, t1, "q_intent")
    temp_db.update_assistant_message(conv_id, t1, "AST is Abstract Syntax Tree", [{"file": "parser.py"}], model_name="llama3.2")

    # Turn 2
    t2 = "turn_2"
    temp_db.create_turn(conv_id, t2)
    temp_db.add_user_message(conv_id, t2, "Where is main.py?")
    temp_db.create_pending_assistant_message(conv_id, t2, "q_intent")
    temp_db.update_assistant_message(conv_id, t2, "main.py is in src/api/", [{"file": "main.py"}], model_name="llama3.2")

    msgs = temp_db.get_messages(conv_id)
    assert len(msgs) == 4

    # Delete Turn 1: Cascades to delete both user question and assistant answer
    deleted, count = temp_db.delete_turn(conv_id, t1)
    assert deleted is True
    assert count == 2

    msgs_after = temp_db.get_messages(conv_id)
    assert len(msgs_after) == 2
    # Ensure remaining messages belong to Turn 2
    for m in msgs_after:
        assert m["turn_id"] == t2
        assert "AST" not in m["content"]

def test_stream_cancellation_race_no_resurrect(temp_db):
    """When a turn is deleted while an assistant response was in flight,
    calling update_assistant_message must NOT re-insert or resurrect the message."""
    conv_id = "test_conv_race"
    turn_id = "turn_race"
    temp_db.create_or_get_conversation(conv_id, title="Race Test", repo_id="repo1")
    temp_db.create_turn(conv_id, turn_id)
    temp_db.add_user_message(conv_id, turn_id, "Cancel me mid stream")
    temp_db.create_pending_assistant_message(conv_id, turn_id, "intent")

    # User cancels and deletes turn mid-stream
    temp_db.delete_turn(conv_id, turn_id)

    # Later the background stream tries to finalize/update
    res = temp_db.update_assistant_message(conv_id, turn_id, "Late content", [], model_name="llama3.2")
    assert res is False

    # Ensure no resurrected messages exist
    msgs = temp_db.get_messages(conv_id)
    assert len(msgs) == 0

def test_delete_entire_conversation(temp_db):
    conv_id = "test_conv_delete_all"
    turn_id = "turn_del_all"
    temp_db.create_or_get_conversation(conv_id, title="To Be Deleted", repo_id="repo1")
    temp_db.create_turn(conv_id, turn_id)
    temp_db.add_user_message(conv_id, turn_id, "Hello")
    temp_db.create_pending_assistant_message(conv_id, turn_id, "intent")
    temp_db.update_assistant_message(conv_id, turn_id, "Hi there", [], model_name="llama3.2")

    assert temp_db.get_conversation(conv_id) is not None
    assert len(temp_db.get_messages(conv_id)) == 2

    # Delete entire conversation
    del_ok = temp_db.delete_conversation(conv_id)
    assert del_ok is True
    assert temp_db.get_conversation(conv_id) is None
    assert len(temp_db.get_messages(conv_id)) == 0

def test_fastapi_endpoints():
    client = TestClient(app)

    conv_id = "api_test_conv_" + tempfile.mktemp()[-6:]
    turn_id = "turn_api_1"

    # Create conversation and turn directly in store
    sqlite_store.create_or_get_conversation(conv_id, title="API Conversation", repo_id="test_repo")
    sqlite_store.create_turn(conv_id, turn_id)
    sqlite_store.add_user_message(conv_id, turn_id, "What is FastAPI?")
    sqlite_store.create_pending_assistant_message(conv_id, turn_id, "code_rag")
    sqlite_store.update_assistant_message(conv_id, turn_id, "FastAPI is a modern web framework.", [], model_name="llama3.2")

    # 1. GET /api/conversations
    res = client.get("/api/conversations")
    assert res.status_code == 200
    data = res.json()
    assert any(c["conversation_id"] == conv_id for c in data["conversations"])

    # 2. GET /api/conversations/{id}
    res = client.get(f"/api/conversations/{conv_id}")
    assert res.status_code == 200
    detail = res.json()
    assert detail["conversation_id"] == conv_id
    assert len(detail["messages"]) == 2

    # 3. GET /api/conversations/{id}/messages
    res = client.get(f"/api/conversations/{conv_id}/messages")
    assert res.status_code == 200
    assert len(res.json()["messages"]) == 2

    # 4. PATCH /api/conversations/{id} (Rename)
    res = client.patch(f"/api/conversations/{conv_id}", json={"title": "Renamed API Conv"})
    assert res.status_code == 200
    assert res.json()["title"] == "Renamed API Conv"

    # 5. DELETE /api/conversations/{id}/turns/{turn_id}
    res = client.delete(f"/api/conversations/{conv_id}/turns/{turn_id}")
    assert res.status_code == 200
    assert res.json()["turn_id"] == turn_id

    # Verify messages deleted
    res = client.get(f"/api/conversations/{conv_id}/messages")
    assert len(res.json()["messages"]) == 0

    # 6. DELETE /api/conversations/{id}
    res = client.delete(f"/api/conversations/{conv_id}")
    assert res.status_code == 200
    assert res.json()["success"] is True

    # Verify conversation gone
    res = client.get(f"/api/conversations/{conv_id}")
    assert res.status_code == 404

def test_rapid_concurrent_conversation_deletion(temp_db):
    """Test rapid creation and deletion of multiple conversations to ensure no lock contention."""
    import time
    for i in range(15):
        cid = f"rapid_conv_{i}"
        tid = f"rapid_turn_{i}"
        temp_db.create_or_get_conversation(cid, title=f"Rapid {i}")
        temp_db.create_turn(cid, tid)
        temp_db.add_user_message(cid, tid, f"Question {i}")
        temp_db.create_pending_assistant_message(cid, tid, "code")
        temp_db.update_assistant_message(cid, tid, f"Answer {i}", [], model_name="llama3.2")

    start_time = time.time()
    for i in range(15):
        cid = f"rapid_conv_{i}"
        deleted = temp_db.delete_conversation(cid)
        assert deleted is True
    duration = time.time() - start_time
    # 15 deletions with cascades should execute in well under 1 second
    assert duration < 1.0
