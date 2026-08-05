import json
import os
import threading
from pathlib import Path


class WAL:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = None
        # None until open() counts the existing log; count() falls back to
        # counting the file for a WAL that was never opened for writing.
        self._count = None

    def open(self):
        self._file = open(self.path, "a", encoding="utf-8")
        # Counted once here and maintained on write, so /admin/stats does not
        # re-read a log that only grows. This process is the only writer.
        self._count = self._count_lines()

    def write(self, event: dict):
        line = json.dumps(event, default=str) + "\n"
        with self._lock:
            if self._file:
                self._file.write(line)
                self._file.flush()
                self._count += 1

    def _count_lines(self) -> int:
        if not self.path.exists():
            return 0
        n = 0
        # Bytes, not decoded text: this is a line count, and the JSON on each
        # line is nobody's business here.
        with open(self.path, "rb") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    def count(self) -> int:
        """How many events the log holds.

        The admin page polls this on a timer. read_all() answered it by
        parsing every event in the log into a dict and taking len() of the
        list -- unbounded work, growing forever, to produce one integer.
        """
        if self._count is None:
            return self._count_lines()
        return self._count

    def read_all(self) -> list[dict]:
        return list(self.iter_all())

    def iter_all(self):
        """Every event, oldest first, one at a time.

        A generator so a caller filtering for a handful of events does not
        first materialise the whole log as dicts.
        """
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    # A torn last line from a hard kill mid-write. The log is
                    # append-only diagnostics; one unreadable entry is not
                    # worth failing an admin request over.
                    continue

    def tail_events(self, limit: int, event: str = "",
                    block_id: str = "") -> list[dict]:
        """The most recent matching events, newest first.

        Scans backwards from the end of the file and stops as soon as it has
        enough, so "the last 50 recall decisions" costs the last few KB of the
        log rather than all of it.
        """
        out: list[dict] = []
        if limit <= 0:
            return out
        for raw in self._iter_lines_reverse():
            try:
                e = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if event and e.get("event") != event:
                continue
            if block_id and e.get("block_id") != block_id:
                continue
            out.append(e)
            if len(out) >= limit:
                break
        return out

    def _iter_lines_reverse(self, chunk_size: int = 65536):
        """Raw lines from the end of the file backwards.

        Reads fixed-size chunks and keeps whatever precedes the first newline
        as the (incomplete) head of the line continued in the previous chunk,
        so a line straddling a chunk boundary is still yielded whole.
        """
        if not self.path.exists():
            return
        with open(self.path, "rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()
            head = b""
            while pos > 0:
                size = min(chunk_size, pos)
                pos -= size
                f.seek(pos)
                parts = (f.read(size) + head).split(b"\n")
                head = parts.pop(0)
                for line in reversed(parts):
                    if line.strip():
                        yield line
            if head.strip():
                yield head

    def close(self):
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None
