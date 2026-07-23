import json
import threading
from pathlib import Path


class WAL:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = None

    def open(self):
        self._file = open(self.path, "a", encoding="utf-8")

    def write(self, event: dict):
        line = json.dumps(event, default=str) + "\n"
        with self._lock:
            if self._file:
                self._file.write(line)
                self._file.flush()

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def close(self):
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None
