import json
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager
from src.code_rag.config import settings

logger = logging.getLogger("code_rag.storage.sqlite")

def get_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteStore:
    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = settings.DATA_DIR / "chat_history.db"

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA busy_timeout = 30000;")
            yield conn
        finally:
            conn.close()

    def init_db(self):
        with self.get_connection() as conn:
            with conn:
                conn.execute("PRAGMA journal_mode = WAL;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute("PRAGMA busy_timeout = 30000;")
                conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT UNIQUE NOT NULL,
                    title TEXT NOT NULL,
                    repo_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """)

                conn.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                );
                """)

                conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT UNIQUE NOT NULL,
                    conversation_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    model_name TEXT,
                    query_intent TEXT,
                    sources TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
                    FOREIGN KEY (turn_id) REFERENCES turns(turn_id) ON DELETE CASCADE
                );
                """)

                conn.execute("""
                CREATE TABLE IF NOT EXISTS export_sessions (
                    export_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    repository_name TEXT NOT NULL,
                    ordered_file_manifest TEXT NOT NULL,
                    current_file_index INTEGER NOT NULL DEFAULT 0,
                    total_file_count INTEGER NOT NULL DEFAULT 0,
                    completed_files TEXT NOT NULL DEFAULT '[]',
                    pending_files TEXT NOT NULL DEFAULT '[]',
                    failed_files TEXT NOT NULL DEFAULT '[]',
                    export_status TEXT NOT NULL DEFAULT 'in_progress',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
                );
                """)

                conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(conversation_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv_created ON messages(conversation_id, created_at ASC, id ASC);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_turn ON messages(turn_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_export_sessions_conv ON export_sessions(conversation_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_export_sessions_conv_repo ON export_sessions(conversation_id, repo_id);")
        logger.info("SQLite chat history initialized at %s", self.db_path)

    # -------------------------------------------------------------
    # Conversation Operations
    # -------------------------------------------------------------

    def create_or_get_conversation(
        self,
        conversation_id: str,
        title: Optional[str] = None,
        repo_id: Optional[str] = None
    ) -> Dict[str, Any]:
        now = get_utc_now()
        with self.get_connection() as conn:
            with conn:
                cursor = conn.execute(
                    "SELECT conversation_id, title, repo_id, created_at, updated_at FROM conversations WHERE conversation_id = ?",
                    (conversation_id,)
                )
                row = cursor.fetchone()
                if row:
                    # Update repo_id if provided
                    if repo_id and repo_id != row["repo_id"]:
                        conn.execute(
                            "UPDATE conversations SET repo_id = ?, updated_at = ? WHERE conversation_id = ?",
                            (repo_id, now, conversation_id)
                        )
                    return dict(row)

                default_title = title or "New Chat"
                conn.execute(
                    """
                    INSERT INTO conversations (conversation_id, title, repo_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (conversation_id, default_title, repo_id or "", now, now)
                )
                return {
                    "conversation_id": conversation_id,
                    "title": default_title,
                    "repo_id": repo_id or "",
                    "created_at": now,
                    "updated_at": now
                }

    def list_conversations(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT c.conversation_id, c.title, c.repo_id, c.created_at, c.updated_at,
                       COUNT(DISTINCT t.turn_id) AS turn_count
                FROM conversations c
                LEFT JOIN turns t ON c.conversation_id = t.conversation_id
                GROUP BY c.conversation_id
                ORDER BY c.updated_at DESC
                """
            )
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_conversation(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            c_cursor = conn.execute(
                "SELECT conversation_id, title, repo_id, created_at, updated_at FROM conversations WHERE conversation_id = ?",
                (conversation_id,)
            )
            c_row = c_cursor.fetchone()
            if not c_row:
                return None

            conv = dict(c_row)
            if limit is not None:
                m_cursor = conn.execute(
                    """
                    SELECT message_id, conversation_id, turn_id, role, content, model_name, query_intent, sources, status, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY created_at ASC, id ASC
                    LIMIT ? OFFSET ?
                    """,
                    (conversation_id, limit, offset)
                )
            else:
                m_cursor = conn.execute(
                    """
                    SELECT message_id, conversation_id, turn_id, role, content, model_name, query_intent, sources, status, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY created_at ASC, id ASC
                    """,
                    (conversation_id,)
                )
            messages = []
            for mr in m_cursor.fetchall():
                m_dict = dict(mr)
                if m_dict.get("sources"):
                    try:
                        m_dict["sources"] = json.loads(m_dict["sources"])
                    except Exception:
                        pass
                messages.append(m_dict)

            conv["messages"] = messages
            return conv

    def rename_conversation(self, conversation_id: str, new_title: str) -> bool:
        now = get_utc_now()
        with self.get_connection() as conn:
            with conn:
                cursor = conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ?",
                    (new_title.strip() or "Untitled Chat", now, conversation_id)
                )
                return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: str) -> bool:
        with self.get_connection() as conn:
            with conn:
                cursor = conn.execute(
                    "DELETE FROM conversations WHERE conversation_id = ?",
                    (conversation_id,)
                )
                return cursor.rowcount > 0

    # -------------------------------------------------------------
    # Turn and Message Operations
    # -------------------------------------------------------------

    def create_turn(self, conversation_id: str, turn_id: str) -> Dict[str, Any]:
        now = get_utc_now()
        with self.get_connection() as conn:
            with conn:
                # Ensure conversation exists
                conn.execute(
                    """
                    INSERT OR IGNORE INTO conversations (conversation_id, title, repo_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (conversation_id, "New Chat", "", now, now)
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO turns (turn_id, conversation_id, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (turn_id, conversation_id, now)
                )
                conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                    (now, conversation_id)
                )
                return {"turn_id": turn_id, "conversation_id": conversation_id, "created_at": now}

    def add_user_message(
        self,
        conversation_id: str,
        turn_id: str,
        content: str,
        message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        now = get_utc_now()
        mid = message_id or f"msg_user_{turn_id}"
        with self.get_connection() as conn:
            with conn:
                # Ensure turn exists
                conn.execute(
                    "INSERT OR IGNORE INTO turns (turn_id, conversation_id, created_at) VALUES (?, ?, ?)",
                    (turn_id, conversation_id, now)
                )
                conn.execute(
                    """
                    INSERT INTO messages (message_id, conversation_id, turn_id, role, content, status, created_at)
                    VALUES (?, ?, ?, 'user', ?, 'completed', ?)
                    """,
                    (mid, conversation_id, turn_id, content, now)
                )

                # Auto-generate title if this is the first turn of conversation
                title_cursor = conn.execute(
                    "SELECT title FROM conversations WHERE conversation_id = ?",
                    (conversation_id,)
                )
                row = title_cursor.fetchone()
                if row and (row["title"] == "New Chat" or not row["title"]):
                    clean_title = content.strip().replace("\n", " ")
                    new_title = clean_title[:45] + ("..." if len(clean_title) > 45 else "")
                    conn.execute(
                        "UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ?",
                        (new_title, now, conversation_id)
                    )
                else:
                    conn.execute(
                        "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                        (now, conversation_id)
                    )

        return {
            "message_id": mid,
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "role": "user",
            "content": content,
            "status": "completed",
            "created_at": now
        }

    def create_pending_assistant_message(
        self,
        conversation_id: str,
        turn_id: str,
        model_name: Optional[str] = None,
        query_intent: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        now = get_utc_now()
        mid = message_id or f"msg_asst_{turn_id}"
        with self.get_connection() as conn:
            with conn:
                # Verify that the turn exists
                cursor = conn.execute("SELECT turn_id FROM turns WHERE turn_id = ? AND conversation_id = ?", (turn_id, conversation_id))
                if not cursor.fetchone():
                    return {}

                conn.execute(
                    """
                    INSERT OR REPLACE INTO messages
                    (message_id, conversation_id, turn_id, role, content, model_name, query_intent, sources, status, created_at)
                    VALUES (?, ?, ?, 'assistant', '', ?, ?, '[]', 'pending', ?)
                    """,
                    (mid, conversation_id, turn_id, model_name or "", query_intent or "", now)
                )
        return {
            "message_id": mid,
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "role": "assistant",
            "content": "",
            "status": "pending",
            "created_at": now
        }

    def update_assistant_message(
        self,
        conversation_id: str,
        turn_id: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        status: str = "completed",
        query_intent: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> bool:
        """
        Updates the pending assistant message only if the turn and message still exist.
        Never inserts or recreates a message if the turn was deleted while generating!
        """
        now = get_utc_now()
        sources_json = json.dumps(sources) if sources is not None else None
        with self.get_connection() as conn:
            with conn:
                # 1. Verify that the turn still exists
                t_cursor = conn.execute(
                    "SELECT turn_id FROM turns WHERE turn_id = ? AND conversation_id = ?",
                    (turn_id, conversation_id)
                )
                if not t_cursor.fetchone():
                    logger.warning("Turn %s was deleted during generation. Skipping assistant message update.", turn_id)
                    return False

                # 2. Update existing assistant message
                if sources_json is not None:
                    cursor = conn.execute(
                        """
                        UPDATE messages
                        SET content = ?, sources = ?, status = ?,
                            query_intent = COALESCE(?, query_intent),
                            model_name = COALESCE(?, model_name)
                        WHERE conversation_id = ? AND turn_id = ? AND role = 'assistant'
                        """,
                        (content, sources_json, status, query_intent, model_name, conversation_id, turn_id)
                    )
                else:
                    cursor = conn.execute(
                        """
                        UPDATE messages
                        SET content = ?, status = ?,
                            query_intent = COALESCE(?, query_intent),
                            model_name = COALESCE(?, model_name)
                        WHERE conversation_id = ? AND turn_id = ? AND role = 'assistant'
                        """,
                        (content, status, query_intent, model_name, conversation_id, turn_id)
                    )

                if cursor.rowcount > 0:
                    conn.execute(
                        "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                        (now, conversation_id)
                    )
                    return True
                return False

    def delete_turn(self, conversation_id: str, turn_id: str) -> Tuple[bool, int]:
        """
        Deletes a specific turn and cascades deletion to associated user and assistant messages.
        All other turns and messages are completely preserved.
        """
        now = get_utc_now()
        with self.get_connection() as conn:
            with conn:
                # Check turn existence
                check_cursor = conn.execute(
                    "SELECT turn_id FROM turns WHERE conversation_id = ? AND turn_id = ?",
                    (conversation_id, turn_id)
                )
                if not check_cursor.fetchone():
                    return False, 0

                # Count messages that will be deleted
                count_cursor = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND turn_id = ?",
                    (conversation_id, turn_id)
                )
                deleted_messages_count = count_cursor.fetchone()[0]

                # Delete the turn; ON DELETE CASCADE removes associated messages
                cursor = conn.execute(
                    """
                    DELETE FROM turns
                    WHERE conversation_id = ?
                    AND turn_id = ?
                    """,
                    (conversation_id, turn_id)
                )
                success = cursor.rowcount > 0

                if success:
                    # Update conversation timestamp
                    conn.execute(
                        "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                        (now, conversation_id)
                    )

                return success, deleted_messages_count

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT message_id, conversation_id, turn_id, role, content, model_name, query_intent, sources, status, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (conversation_id,)
            )
            messages = []
            for row in cursor.fetchall():
                m = dict(row)
                if m.get("sources"):
                    try:
                        m["sources"] = json.loads(m["sources"])
                    except Exception:
                        pass
                messages.append(m)
            return messages

    # -------------------------------------------------------------
    # Complete Repository Code Export Session Operations
    # -------------------------------------------------------------

    def save_export_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates or updates a persistent export session for sequential code export.
        """
        now = get_utc_now()
        export_id = session.get("export_id")
        conv_id = session.get("conversation_id", "")
        repo_id = session.get("repo_id", "")
        repo_name = session.get("repository_name", repo_id)

        # Ensure conversation exists to satisfy foreign key constraint
        if conv_id:
            try:
                self.create_or_get_conversation(conv_id, repo_id=repo_id)
            except Exception:
                pass

        manifest = json.dumps(session.get("ordered_file_manifest", []))
        idx = session.get("current_file_index", 0)
        total = session.get("total_file_count", 0)
        completed = json.dumps(session.get("completed_files", []))
        pending = json.dumps(session.get("pending_files", []))
        failed = json.dumps(session.get("failed_files", []))
        status = session.get("export_status", "in_progress")
        created_at = session.get("created_at", now)

        with self.get_connection() as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO export_sessions (
                        export_id, conversation_id, repo_id, repository_name,
                        ordered_file_manifest, current_file_index, total_file_count,
                        completed_files, pending_files, failed_files,
                        export_status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(export_id) DO UPDATE SET
                        current_file_index = excluded.current_file_index,
                        total_file_count = excluded.total_file_count,
                        completed_files = excluded.completed_files,
                        pending_files = excluded.pending_files,
                        failed_files = excluded.failed_files,
                        export_status = excluded.export_status,
                        updated_at = excluded.updated_at
                    """,
                    (
                        export_id, conv_id, repo_id, repo_name,
                        manifest, idx, total,
                        completed, pending, failed,
                        status, created_at, now
                    )
                )
        session["updated_at"] = now
        return session

    def get_latest_export_session(
        self,
        conversation_id: str,
        repo_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves the latest export session for the given conversation and repo.
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM export_sessions
                WHERE conversation_id = ? AND repo_id = ?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (conversation_id, repo_id)
            )
            row = cursor.fetchone()
            if not row:
                return None
            res = dict(row)
            try:
                res["ordered_file_manifest"] = json.loads(res.get("ordered_file_manifest", "[]"))
                res["completed_files"] = json.loads(res.get("completed_files", "[]"))
                res["pending_files"] = json.loads(res.get("pending_files", "[]"))
                res["failed_files"] = json.loads(res.get("failed_files", "[]"))
            except Exception:
                pass
            return res
