"""Durable chat transcripts, so a conversation can be picked up later.

Kept separate from the block store on purpose. Blocks are the memory system's
own representation -- split, tagged, summarised, and eventually purged by the
judge -- which makes them a poor source for replaying a conversation: the
user's message survives only inside a reasoning block's stimulus, truncated to
512 words and glued to the reply. This table is the plain transcript instead,
and it is deliberately never touched by the lifecycle.
"""

import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple


class ChatStore:
    # A pasted file can be megabytes. Transcripts are unbounded in count, so
    # cap what any single message can contribute to the database.
    MAX_CONTENT_CHARS = 100_000
    TITLE_CHARS = 80

    def __init__(self, path: Path):
        self.db_path = path / "chats.db"
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def open(self):
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self):
        c = self._conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT DEFAULT '',
                source TEXT DEFAULT '',
                created_at REAL,
                updated_at REAL,
                message_count INTEGER DEFAULT 0
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL
            )
        """)
        c.execute("""CREATE INDEX IF NOT EXISTS idx_messages_conv
                     ON messages(conversation_id, seq)""")
        c.execute("""CREATE INDEX IF NOT EXISTS idx_conv_updated
                     ON conversations(updated_at DESC)""")
        self._conn.commit()

    @staticmethod
    def _title_from(text: str, limit: int) -> str:
        line = " ".join((text or "").split())
        return line[:limit] + ("…" if len(line) > limit else "")

    def record_turn(self, conversation_id: str, user_text: str,
                    assistant_text: str, source: str = ""):
        """Append one exchange. Creates the conversation on first sight."""
        if not conversation_id:
            return
        user_text = (user_text or "")[:self.MAX_CONTENT_CHARS]
        assistant_text = (assistant_text or "")[:self.MAX_CONTENT_CHARS]
        if not user_text and not assistant_text:
            return
        now = time.time()
        with self._lock:
            cur = self._conn.cursor()
            row = cur.execute(
                "SELECT message_count FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO conversations (conversation_id, title, source,"
                    " created_at, updated_at, message_count) VALUES (?,?,?,?,?,0)",
                    (conversation_id, self._title_from(user_text, self.TITLE_CHARS),
                     source, now, now),
                )
                seq = 0
            else:
                seq = row[0]

            added = 0
            for role, content in (("user", user_text), ("assistant", assistant_text)):
                if not content:
                    continue
                cur.execute(
                    "INSERT INTO messages (conversation_id, seq, role, content,"
                    " created_at) VALUES (?,?,?,?,?)",
                    (conversation_id, seq + added, role, content, now),
                )
                added += 1

            cur.execute(
                "UPDATE conversations SET updated_at=?, message_count=message_count+?"
                " WHERE conversation_id=?",
                (now, added, conversation_id),
            )
            self._conn.commit()

    def list_conversations(self, limit: int = 50, offset: int = 0,
                           source: Optional[str] = None) -> Tuple[List[dict], int]:
        where, params = "1", []
        if source:
            where, params = "source=?", [source]
        with self._lock:
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM conversations WHERE {where}", params
            ).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT conversation_id, title, source, created_at, updated_at,"
                f" message_count FROM conversations WHERE {where}"
                f" ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        cols = ["conversation_id", "title", "source", "created_at",
                "updated_at", "message_count"]
        return [dict(zip(cols, r)) for r in rows], total

    def get_messages(self, conversation_id: str) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, created_at FROM messages"
                " WHERE conversation_id=? ORDER BY seq ASC, id ASC",
                (conversation_id,),
            ).fetchall()
        return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]

    def rename(self, conversation_id: str, title: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE conversations SET title=? WHERE conversation_id=?",
                (self._title_from(title, self.TITLE_CHARS), conversation_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def delete(self, conversation_id: str) -> bool:
        """Remove a transcript. Blocks derived from it are untouched --
        forgetting the conversation is not the same as forgetting what was
        learned in it, and blocks are deleted from the admin page."""
        with self._lock:
            self._conn.execute("DELETE FROM messages WHERE conversation_id=?",
                               (conversation_id,))
            cur = self._conn.execute(
                "DELETE FROM conversations WHERE conversation_id=?",
                (conversation_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def snapshot(self, dest_dir: Path):
        self._conn.execute("VACUUM INTO ?", (str(dest_dir / "chats.db"),))
        self._conn.commit()

    def restore(self, src_dir: Path):
        import shutil
        src = src_dir / "chats.db"
        if src.exists():
            if self._conn:
                self._conn.close()
            shutil.copy2(src, self.db_path)
            self.open()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
