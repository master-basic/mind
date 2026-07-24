from pathlib import Path
from typing import List
import yaml


class RecallConfig:
    def __init__(self, d: dict):
        self.k: int = d.get("k", 4)
        self.threshold: float = d.get("threshold", 0.62)
        self.budget_tokens: int = d.get("budget_tokens", 3000)


class JudgeConfig:
    def __init__(self, d: dict):
        self.interval_tokens: int = d.get("interval_tokens", 20000)
        self.min_age_s: int = d.get("min_age_s", 3600)
        self.purge_age_s: int = d.get("purge_age_s", 1209600)
        self.summary_max_tokens: int = d.get("summary_max_tokens", 400)


class ServerConfig:
    def __init__(self, d: dict):
        self.model: str = d.get("model", "")
        self.port: int = d.get("port", 0)
        self.extra_args: List[str] = d.get("extra_args", [])


class Config:
    def __init__(self, path: str | Path = "config.yaml"):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self.listen: str = raw.get("listen", "127.0.0.1:8000")
        self.reasoning_endpoint: str = raw.get("reasoning_endpoint", "http://127.0.0.1:8080")
        self.judge_endpoint: str = raw.get("judge_endpoint", "http://127.0.0.1:8081")
        self.embed_endpoint: str = raw.get("embed_endpoint", "http://127.0.0.1:8082")
        self.think_tags: List[str] = raw.get("think_tags", ["<think>", "</think>"])
        self.store_path: str = raw.get("store_path", "/mnt/ramdisk/cued_recall")
        self.snapshot_path: str = raw.get("snapshot_path", "/var/lib/cued_recall/snapshots")
        self.snapshot_interval_min: int = raw.get("snapshot_interval_min", 15)
        self.block_tokens_reasoning: int = raw.get("block_tokens_reasoning", 8000)
        self.embed_dim: int = raw.get("embed_dim", 768)
        self.recall = RecallConfig(raw.get("recall", {}))
        self.judge = JudgeConfig(raw.get("judge", {}))
        self.correction_patterns: List[str] = raw.get("correction_patterns", [
            "that's wrong", "doesn't work", "^no[,.]", "səhvdir", "işləmir",
        ])
        self.servers: dict = raw.get("servers", {})
        self.models_dir: str = raw.get("models_dir", "./models")
        self.max_context_tokens: int = raw.get("max_context_tokens", 28000)

    def get_server(self, name: str) -> ServerConfig:
        return ServerConfig(self.servers.get(name, {}))
