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
        # 0 means never judged, which sorts first in blocks_due_for_judging --
        # so an existing store starts by sweeping everything it has never
        # looked at rather than revisiting its oldest blocks again.
        if "judged_at" not in existing_cols:
            c.execute("ALTER TABLE blocks ADD COLUMN judged_at REAL DEFAULT 0")
        # How a verification was arrived at: "pattern", "model", "manual", or
        # "recalled_uncontested". A correction found by a hand-tested pattern
        # is trusted enough to delete a block outright; one guessed at by a
        # 1.5B classifier is not. See Judge._should_purge.
        if "verification_source" not in existing_cols:
            c.execute(
                "ALTER TABLE blocks ADD COLUMN verification_source TEXT DEFAULT ''"
            )
        # Set by hand, never inferred. A pinned block is skipped by both
        # decay_candidates and blocks_due_for_judging, so nothing automatic can
        # delete it or paraphrase it away. Existing rows default to 0.
        if "pinned" not in existing_cols:
            c.execute("ALTER TABLE blocks ADD COLUMN pinned INTEGER DEFAULT 0")
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
        # Which blocks recall served into which turn. This lived in a plain
        # dict on Pipeline, capped at 500 entries and lost on every restart --
        # so the one positive signal the system gathers on its own (a block was
        # put in front of the model and the user did not object) was erased by
        # a process restart, while the decay rules that consume it kept running.
        c.execute("""
            CREATE TABLE IF NOT EXISTS turn_recalls (
                conversation_id TEXT,
                turn_index INTEGER,
                block_id TEXT,
                created_at REAL,
                PRIMARY KEY (conversation_id, turn_index, block_id)
            )
        """)
        # Read by (conversation_id, turn_index) on the next turn, and swept by
        # created_at.
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_turn_recalls_age
            ON turn_recalls (created_at)
        """)
        # How many times a block was recalled into a turn and then not
        # contradicted. Distinct from recall_count, which counts being served
        # and says nothing about whether it helped.
        if "uncontested_recalls" not in existing_cols:
            c.execute("ALTER TABLE blocks ADD COLUMN "
                      "uncontested_recalls INTEGER DEFAULT 0")
        # "which blocks belong to conversation X, turn Y?" is asked up to four
        # times per user turn (shelve_previous_turn, detect_and_apply_correction,
        # verify_correction_with_model, apply_accepted_verification). Without
        # this it is a full table scan each time, and the answer is a handful of
        # rows out of thousands.
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_blocks_conversation_turn
            ON blocks (conversation_id, turn_index)
        """)
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

    def delete_vector(self, block_id: str):
        """Drop a block's embedding but keep its metadata row.

        Purging needs this: query() resolves the KNN against block_vec first
        and only then filters by status, so a purged block that keeps its
        vector still consumes candidate slots and still has to be discarded
        after the fact. The row stays so the block remains auditable.
        """
        with self._lock:
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
                "turn_index", "type", "status", "verification", "pinned"}

    def list_meta(self, status: Optional[str] = None,
                  type_: Optional[str] = None,
                  conversation_id: Optional[str] = None,
                  tag: Optional[str] = None,
                  recall_min: Optional[int] = None,
                  recall_max: Optional[int] = None,
                  older_than: Optional[float] = None,
                  pinned: Optional[bool] = None,
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
        if pinned is not None:
            where.append("COALESCE(pinned, 0) = ?")
            params.append(1 if pinned else 0)
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

    def update_verification(self, block_id: str, verification: str,
                            source: str = ""):
        with self._lock:
            self._conn.execute(
                "UPDATE blocks SET verification=?, verification_source=? "
                "WHERE block_id=?",
                (verification, source, block_id),
            )
            self._conn.commit()

    def set_pinned(self, block_id: str, pinned: bool):
        with self._lock:
            self._conn.execute(
                "UPDATE blocks SET pinned=? WHERE block_id=?",
                (1 if pinned else 0, block_id),
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

    def blocks_due_for_judging(self, min_age_s: float,
                               rejudge_interval_s: float,
                               limit: int = 200) -> List[str]:
        """Blocks the judge has not looked at recently, least recent first.

        Replaces an "oldest 50 shelved" query, which never advanced: a verdict
        of "keep" left the block shelved, so the next pass selected the same
        50. Over 142 recorded decisions it visited 82 distinct blocks out of
        395. Ordering by judged_at makes the pass a sweep -- never-judged
        blocks carry 0 and come first.
        """
        now = __import__("time").time()
        with self._lock:
            rows = self._conn.execute("""
                SELECT block_id FROM blocks
                WHERE status IN ('shelved', 'truncated')
                  AND COALESCE(pinned, 0) = 0
                  AND created_at < ?
                  AND COALESCE(judged_at, 0) < ?
                ORDER BY COALESCE(judged_at, 0) ASC, created_at ASC
                LIMIT ?
            """, (now - min_age_s, now - rejudge_interval_s, limit)).fetchall()
        return [r[0] for r in rows]

    def blocks_without_vectors(self, limit: int = 100000) -> List[str]:
        """Recallable-by-status blocks that have no embedding, oldest first.

        Blocks are embedded once, at creation, and a failure there is logged
        and dropped -- so a block written while the embedding server was
        restarting stays `shelved`, keeps its text, appears in the admin table,
        and can never be retrieved. Status cannot express that, which is why
        nothing surfaced it: 729 of 1,812 blocks in one real store were in this
        state, 57 of them holding content.
        """
        with self._lock:
            rows = self._conn.execute("""
                SELECT b.block_id FROM blocks b
                WHERE b.status IN ('shelved', 'truncated')
                  AND NOT EXISTS (SELECT 1 FROM block_vec v
                                  WHERE v.block_id = b.block_id)
                ORDER BY b.created_at ASC
                LIMIT ?
            """, (limit,)).fetchall()
        return [r[0] for r in rows]

    def record_recall(self, conversation_id: str, turn_index: int,
                      block_ids: List[str], ts: float):
        """Note that recall served these blocks into this turn.

        Durable on purpose: the next turn reads it back to decide whether the
        user objected, and that verdict feeds decay. Held in memory it was lost
        on restart and silently dropped past 500 entries.
        """
        if not block_ids:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO turn_recalls "
                "(conversation_id, turn_index, block_id, created_at) "
                "VALUES (?,?,?,?)",
                [(conversation_id, turn_index, b, ts) for b in block_ids],
            )
            self._conn.commit()

    def take_recalled_into_turn(self, conversation_id: str,
                                turn_index: int) -> List[str]:
        """The blocks recall served into one turn, consuming the record.

        Consumed rather than read, so a turn's evidence is counted once however
        many times the caller runs -- a retried or duplicated turn must not
        inflate a block's standing.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT block_id FROM turn_recalls "
                "WHERE conversation_id=? AND turn_index=?",
                (conversation_id, turn_index),
            ).fetchall()
            if rows:
                self._conn.execute(
                    "DELETE FROM turn_recalls "
                    "WHERE conversation_id=? AND turn_index=?",
                    (conversation_id, turn_index),
                )
                self._conn.commit()
        return [r[0] for r in rows]

    def prune_turn_recalls(self, older_than_s: float) -> int:
        """Drop recall records nobody will ever read back.

        Only the immediately following turn consumes one, so anything older
        than a generous window is a conversation that was abandoned mid-turn.
        Without this the table grows for the life of the store.
        """
        cutoff = time.time() - older_than_s
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM turn_recalls WHERE created_at < ?", (cutoff,)
            )
            self._conn.commit()
            return cur.rowcount or 0

    def increment_uncontested(self, block_id: str):
        with self._lock:
            self._conn.execute(
                "UPDATE blocks SET uncontested_recalls="
                "COALESCE(uncontested_recalls, 0) + 1 WHERE block_id=?",
                (block_id,),
            )
            self._conn.commit()

    def block_ids_for_turn(self, conversation_id: str,
                           turn_index: int) -> List[str]:
        """Every block written by one turn of one conversation.

        Served by idx_blocks_conversation_turn. This used to be list_meta(
        limit=10000) followed by a Python filter -- an O(store) read, several
        times per user turn, for an answer that is a handful of rows.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT block_id FROM blocks "
                "WHERE conversation_id=? AND turn_index=?",
                (conversation_id, turn_index),
            ).fetchall()
        return [r[0] for r in rows]

    def count_blocks_without_vectors(self) -> int:
        """How many recallable-by-status blocks have no embedding.

        The counting half of blocks_without_vectors, for the health signal in
        /admin/stats -- a store with tens of thousands of blocks should not have
        to materialise every id just to answer "is anything invisible?".
        """
        with self._lock:
            row = self._conn.execute("""
                SELECT COUNT(*) FROM blocks b
                WHERE b.status IN ('shelved', 'truncated')
                  AND NOT EXISTS (SELECT 1 FROM block_vec v
                                  WHERE v.block_id = b.block_id)
            """).fetchone()
        return row[0] if row else 0

    def decay_candidates(self, purge_age_s: float,
                         limit: int = 1000) -> List[str]:
        """Blocks arithmetic alone can condemn -- no model call involved.

        Deliberately NOT gated on judged_at, unlike blocks_due_for_judging.
        Forgetting costs one query, so there is no reason to make a block wait
        out the consolidation cycle first: rejudge_interval_s is longer than
        purge_age_s, so gating this the same way would mean the purge cutoff
        fired a week late, or never.

        Returns a superset -- the caller re-checks each one against the real
        rule, so this query and that rule cannot drift apart.

        No longer filtered on `recall_count = 0`. Under utility decay a block
        that was recalled can still go stale, so pre-filtering them out here
        would have made the new rule unreachable for exactly the blocks it
        exists to judge.
        """
        cutoff = time.time() - purge_age_s
        with self._lock:
            rows = self._conn.execute("""
                SELECT block_id FROM blocks
                WHERE status IN ('shelved', 'truncated')
                  AND COALESCE(pinned, 0) = 0
                  AND (verification = 'corrected' OR created_at < ?)
                ORDER BY created_at ASC
                LIMIT ?
            """, (cutoff, limit)).fetchall()
        return [r[0] for r in rows]

    def mark_judged(self, block_id: str, ts: float):
        with self._lock:
            self._conn.execute(
                "UPDATE blocks SET judged_at=? WHERE block_id=?",
                (ts, block_id),
            )
            self._conn.commit()

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
