import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import List, Tuple, Optional

import sqlite3
import sqlite_vec


class VectorIndex:
    def __init__(self, path: Path, dim: int = 1024):
        self.db_path = path / "index.db"
        self.dim = dim
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def open(self):
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self):
        c = self._conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS blocks (
                block_id TEXT PRIMARY KEY,
                type TEXT,
                status TEXT,
                created_at REAL,
                conversation_id TEXT,
                turn_index INTEGER,
                token_count INTEGER,
                verification TEXT,
                recall_count INTEGER DEFAULT 0,
                last_recalled REAL
            )
        """)
        # Migrate older DBs created before tagging existed. SQLite has no
        # "ADD COLUMN IF NOT EXISTS", so check first.
        existing_cols = {row[1] for row in c.execute("PRAGMA table_info(blocks)")}
        if "tags" not in existing_cols:
            c.execute("ALTER TABLE blocks ADD COLUMN tags TEXT DEFAULT ''")
        if "gist" not in existing_cols:
            c.execute("ALTER TABLE blocks ADD COLUMN gist TEXT DEFAULT ''")
        # If an existing block_vec was created with a different dimension,
        # drop it: a dim change invalidates stored vectors, and keeping the
        # old table makes every upsert/query raise a dim-mismatch error.
        row = c.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='block_vec'"
        ).fetchone()
        if row and row[0]:
            m = re.search(r"float\[(\d+)\]", row[0])
            if m and int(m.group(1)) != self.dim:
                c.execute("DROP TABLE block_vec")
        c.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS block_vec USING vec0(
                block_id TEXT PRIMARY KEY,
                embedding float[{self.dim}] distance_metric=cosine
            )
        """)
        self._conn.commit()

    def upsert_block_meta(self, block_id: str, type_: str, status: str,
                          created_at: float, conversation_id: str,
                          turn_index: int, token_count: int,
                          verification: str, recall_count: int,
                          last_recalled: float):
        with self._lock:
            self._conn.execute("""
                INSERT INTO blocks (block_id, type, status, created_at,
                    conversation_id, turn_index, token_count,
                    verification, recall_count, last_recalled)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(block_id) DO UPDATE SET
                    status=excluded.status,
                    verification=excluded.verification,
                    recall_count=excluded.recall_count,
                    last_recalled=excluded.last_recalled
            """, (block_id, type_, status, created_at, conversation_id,
                  turn_index, token_count, verification, recall_count,
                  last_recalled))
            self._conn.commit()

    def set_tags(self, block_id: str, tags: List[str], gist: str):
        # Stored delimiter-wrapped (",tag1,tag2,") so a LIKE '%,tag,%' filter
        # can't false-match a tag that's a substring of another (e.g. "dns"
        # inside "dns-server").
        tags_str = "," + ",".join(tags) + "," if tags else ""
        with self._lock:
            self._conn.execute(
                "UPDATE blocks SET tags=?, gist=? WHERE block_id=?",
                (tags_str, gist, block_id),
            )
            self._conn.commit()

    def tag_counts(self) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT tags FROM blocks WHERE tags != ''"
            ).fetchall()
        counts: dict = {}
        for (raw,) in rows:
            for t in raw.split(","):
                if t:
                    counts[t] = counts.get(t, 0) + 1
        return counts

    def delete_meta(self, block_id: str):
        with self._lock:
            self._conn.execute("DELETE FROM blocks WHERE block_id=?", (block_id,))
            self._conn.execute("DELETE FROM block_vec WHERE block_id=?", (block_id,))
            self._conn.commit()

    def upsert_vector(self, block_id: str, embedding: List[float]):
        if len(embedding) != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {len(embedding)}")
        with self._lock:
            # vec0 virtual tables support neither ON CONFLICT ... DO UPDATE
            # ("UPSERT not implemented for virtual table") nor INSERT OR
            # REPLACE (raises a UNIQUE constraint error instead of
            # replacing). Delete-then-insert is the only working upsert.
            packed = sqlite_vec.serialize_float32(embedding)
            self._conn.execute(
                "DELETE FROM block_vec WHERE block_id = ?", (block_id,)
            )
            self._conn.execute(
                "INSERT INTO block_vec (block_id, embedding) VALUES (?, ?)",
                (block_id, packed),
            )
            self._conn.commit()

    def query(self, embedding: List[float], k: int,
              threshold: float,
              status_filter: Tuple[str, ...] = ("shelved", "truncated")) -> List[Tuple[str, float]]:
        if len(embedding) != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {len(embedding)}")
        # sqlite-vec resolves the KNN "k" against block_vec alone, before the
        # JOIN's status filter is applied. Filtering by status in the same
        # WHERE clause as the MATCH/k constraint silently drops matches: if
        # the nearest k raw vectors are all e.g. "hot", the join yields zero
        # rows even when relevant "shelved"/"truncated" blocks exist further
        # down. Over-fetch a large candidate pool and filter by status here.
        candidate_k = max(k * 50, 500)
        with self._lock:
            rows = self._conn.execute("""
                SELECT v.block_id, v.distance, b.status
                FROM block_vec v
                JOIN blocks b ON b.block_id = v.block_id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
            """, (sqlite_vec.serialize_float32(embedding), candidate_k)).fetchall()
        results = []
        for block_id, distance, status in rows:
            if status not in status_filter:
                continue
            sim = 1.0 - distance
            if sim >= threshold:
                results.append((block_id, sim))
            if len(results) >= k:
                break
        return results

    def get_meta(self, block_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM blocks WHERE block_id=?", (block_id,)
            ).fetchone()
            if row is None:
                return None
            cols = [d[1] for d in self._conn.execute("PRAGMA table_info(blocks)")]
            meta = dict(zip(cols, row))
            meta["tags"] = [t for t in (meta.get("tags") or "").split(",") if t]
            return meta

    # Whitelist for ORDER BY. The column name cannot be bound as a parameter,
    # so it is interpolated -- only ever from this set.
    SORTABLE = {"created_at", "token_count", "recall_count", "last_recalled",
                "turn_index", "type", "status", "verification"}

    def list_meta(self, status: Optional[str] = None,
                  type_: Optional[str] = None,
                  conversation_id: Optional[str] = None,
                  tag: Optional[str] = None,
                  recall_min: Optional[int] = None,
                  recall_max: Optional[int] = None,
                  older_than: Optional[float] = None,
                  sort: str = "created_at", order: str = "desc",
                  limit: int = 100, offset: int = 0) -> Tuple[List[dict], int]:
        where = []
        params = []
        if status:
            where.append("status=?")
            params.append(status)
        if type_:
            where.append("type=?")
            params.append(type_)
        if conversation_id:
            where.append("conversation_id=?")
            params.append(conversation_id)
        if tag:
            where.append("tags LIKE ?")
            params.append(f"%,{tag},%")
        # COALESCE so blocks written before recall_count defaulted to 0 are
        # still matched by a "0 recalls" filter instead of dropping out on NULL.
        if recall_min is not None:
            where.append("COALESCE(recall_count, 0) >= ?")
            params.append(recall_min)
        if recall_max is not None:
            where.append("COALESCE(recall_count, 0) <= ?")
            params.append(recall_max)
        if older_than is not None:
            where.append("created_at < ?")
            params.append(older_than)
        clause = " AND ".join(where) if where else "1"
        sort_col = sort if sort in self.SORTABLE else "created_at"
        direction = "ASC" if str(order).lower() == "asc" else "DESC"
        with self._lock:
            count = self._conn.execute(
                f"SELECT COUNT(*) FROM blocks WHERE {clause}", params
            ).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT * FROM blocks WHERE {clause} "
                f"ORDER BY {sort_col} {direction}, block_id {direction} "
                f"LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            cols = [d[1] for d in self._conn.execute("PRAGMA table_info(blocks)")]
            items = [dict(zip(cols, row)) for row in rows]
        for item in items:
            item["tags"] = [t for t in (item.get("tags") or "").split(",") if t]
        return items, count

    def growth_by_day(self, days: int = 30) -> List[dict]:
        """Blocks and tokens created per day, split by type.

        Answers whether the judge is keeping pace with ingestion: if tokens
        added per day keeps climbing while truncated/purged counts stay flat,
        the store is growing faster than it is being pruned.
        """
        cutoff = time.time() - days * 86400
        with self._lock:
            rows = self._conn.execute("""
                SELECT date(created_at, 'unixepoch') AS day,
                       type,
                       COUNT(*) AS blocks,
                       SUM(COALESCE(token_count, 0)) AS tokens
                FROM blocks
                WHERE created_at >= ?
                GROUP BY day, type
                ORDER BY day ASC
            """, (cutoff,)).fetchall()
        return [{"day": r[0], "type": r[1], "blocks": r[2], "tokens": r[3] or 0}
                for r in rows]

    # Upper bounds; the last bucket is open-ended.
    TOKEN_BUCKETS = [64, 256, 1024, 4096, 16384]

    def token_histogram(self) -> List[dict]:
        """Block count and total tokens per size bucket.

        A median of 32 tokens against blocks of 20,000+ is the signal that
        block_tokens_reasoning is not splitting large content as expected.
        """
        edges = self.TOKEN_BUCKETS
        buckets = []
        with self._lock:
            for i, hi in enumerate(edges):
                lo = 0 if i == 0 else edges[i - 1]
                row = self._conn.execute(
                    "SELECT COUNT(*), SUM(COALESCE(token_count, 0)) FROM blocks "
                    "WHERE COALESCE(token_count, 0) > ? "
                    "AND COALESCE(token_count, 0) <= ?", (lo, hi)).fetchone()
                buckets.append({"label": f"{lo+1}–{hi}", "min": lo + 1, "max": hi,
                                "blocks": row[0], "tokens": row[1] or 0})
            row = self._conn.execute(
                "SELECT COUNT(*), SUM(COALESCE(token_count, 0)) FROM blocks "
                "WHERE COALESCE(token_count, 0) > ?", (edges[-1],)).fetchone()
            buckets.append({"label": f">{edges[-1]}", "min": edges[-1] + 1,
                            "max": None, "blocks": row[0], "tokens": row[1] or 0})
        return buckets

    def recall_effectiveness(self, top: int = 20) -> dict:
        """How much of the store is earning its keep.

        `corrected` blocks are the harmful ones: recalled often but contradicted
        in conversation. `never_recalled` tokens are dead weight -- indexed,
        embedded, and never once used.
        """
        with self._lock:
            totals = self._conn.execute("""
                SELECT COUNT(*),
                       SUM(COALESCE(token_count, 0)),
                       SUM(CASE WHEN COALESCE(recall_count,0) = 0 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN COALESCE(recall_count,0) = 0
                                THEN COALESCE(token_count,0) ELSE 0 END),
                       SUM(COALESCE(recall_count, 0))
                FROM blocks
            """).fetchone()
            by_verification = self._conn.execute("""
                SELECT COALESCE(verification, 'unknown'),
                       COUNT(*),
                       SUM(COALESCE(recall_count, 0)),
                       SUM(COALESCE(token_count, 0))
                FROM blocks GROUP BY 1
            """).fetchall()
            cols = [d[1] for d in self._conn.execute("PRAGMA table_info(blocks)")]
            top_rows = self._conn.execute("""
                SELECT * FROM blocks
                WHERE COALESCE(recall_count, 0) > 0
                ORDER BY recall_count DESC, token_count DESC LIMIT ?
            """, (top,)).fetchall()
        top_items = [dict(zip(cols, r)) for r in top_rows]
        for item in top_items:
            item["tags"] = [t for t in (item.get("tags") or "").split(",") if t]
        return {
            "blocks": totals[0] or 0,
            "tokens": totals[1] or 0,
            "never_recalled_blocks": totals[2] or 0,
            "never_recalled_tokens": totals[3] or 0,
            "total_recalls": totals[4] or 0,
            "by_verification": [
                {"verification": r[0], "blocks": r[1],
                 "recalls": r[2] or 0, "tokens": r[3] or 0}
                for r in by_verification
            ],
            "top_recalled": top_items,
        }

    def update_status(self, block_id: str, status: str):
        with self._lock:
            self._conn.execute(
                "UPDATE blocks SET status=? WHERE block_id=?", (status, block_id)
            )
            self._conn.commit()

    def update_verification(self, block_id: str, verification: str):
        with self._lock:
            self._conn.execute(
                "UPDATE blocks SET verification=? WHERE block_id=?",
                (verification, block_id),
            )
            self._conn.commit()

    def increment_recall(self, block_id: str, ts: float):
        with self._lock:
            self._conn.execute(
                "UPDATE blocks SET recall_count=recall_count+1, last_recalled=? WHERE block_id=?",
                (ts, block_id),
            )
            self._conn.commit()

    def snapshot(self, dest_dir: Path):
        dest = dest_dir / "index.db"
        self._conn.execute("VACUUM INTO ?", (str(dest),))
        self._conn.commit()

    def restore(self, src_dir: Path):
        src = src_dir / "index.db"
        if src.exists():
            if self._conn:
                self._conn.close()
            shutil.copy2(src, self.db_path)
            self.open()

    def stats(self) -> dict:
        with self._lock:
            rows = self._conn.execute("""
                SELECT status, type, COUNT(*) as cnt
                FROM blocks GROUP BY status, type
            """).fetchall()
        counts = {}
        for status, type_, cnt in rows:
            key = f"{status}_{type_}"
            counts[key] = cnt
        return counts

    def oldest_shelved_blocks(self, min_age_s: float,
                              limit: int = 100) -> List[str]:
        cutoff = __import__("time").time() - min_age_s
        with self._lock:
            rows = self._conn.execute("""
                SELECT block_id FROM blocks
                WHERE status IN ('shelved', 'truncated')
                  AND created_at < ?
                ORDER BY created_at ASC
                LIMIT ?
            """, (cutoff, limit)).fetchall()
        return [r[0] for r in rows]

    def hot_blocks_older_than(self, min_age_s: float,
                              limit: int = 200) -> List[str]:
        # A turn's blocks only shelve when the NEXT turn arrives in the same
        # conversation. A one-off message that's never followed up on would
        # otherwise stay 'hot' -- invisible to recall -- forever. This finds
        # those abandoned blocks so a background sweep can shelve them.
        cutoff = __import__("time").time() - min_age_s
        with self._lock:
            rows = self._conn.execute("""
                SELECT block_id FROM blocks
                WHERE status = 'hot' AND created_at < ?
                ORDER BY created_at ASC
                LIMIT ?
            """, (cutoff, limit)).fetchall()
        return [r[0] for r in rows]

    def close(self):
        if self._conn:
            self._conn.close()
