import shutil
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


class TaggerConfig:
    def __init__(self, d: dict):
        self.enabled: bool = d.get("enabled", True)
        # Empty = reuse judge_endpoint (tagging is a cheap call, doesn't
        # need its own model unless you want to split the load).
        self.endpoint: str = d.get("endpoint", "")
        self.max_tags: int = d.get("max_tags", 3)
        self.gist_max_chars: int = d.get("gist_max_chars", 40)


class WebSearchConfig:
    def __init__(self, d: dict):
        self.enabled: bool = d.get("enabled", True)
        # backend: "duckduckgo" (no key), "searxng", "brave", "serper"
        self.backend: str = d.get("backend", "duckduckgo")
        self.searxng_url: str = d.get("searxng_url", "")
        self.api_key: str = d.get("api_key", "")
        self.max_results: int = d.get("max_results", 5)
        self.fetch_top_n: int = d.get("fetch_top_n", 0)
        # Hard rule: if the user message matches any of these patterns, force
        # the model to call web_search on that turn (tool_choice), rather than
        # leaving it to the model's discretion.
        self.force_search: bool = d.get("force_search", True)
        self.force_patterns: List[str] = d.get("force_patterns", [
            r"\bsearch\b", r"\blatest\b", r"\bcurrent(ly)?\b", r"\bnews\b",
            r"\btoday\b", r"\brecent(ly)?\b", r"\bthis (week|month|year)\b",
            r"\bright now\b", r"\bup[- ]?to[- ]?date\b", r"\bas of\b",
            r"\bnowadays\b", r"\blook (it |this )?up\b", r"\bgoogle\b",
            # Azerbaijani
            r"\baxtar", r"\bson (xəbər|məlumat)", r"\bhal-?hazırda\b",
        ])


class ServerConfig:
    def __init__(self, d: dict):
        self.model: str = d.get("model", "")
        self.port: int = d.get("port", 0)
        self.extra_args: List[str] = d.get("extra_args", [])


class Config:
    def __init__(self, path: str | Path = "config.yaml"):
        path = Path(path)
        if not path.exists():
            # config.yaml is gitignored (it holds a machine-specific
            # store_path/snapshot_path); a fresh clone only has the generic
            # template. Bootstrap from it rather than a bare FileNotFoundError.
            example = path.parent / "config.example.yaml"
            if example.exists():
                shutil.copy2(example, path)
            else:
                raise FileNotFoundError(
                    f"{path} not found and no config.example.yaml template at {example}. "
                    "Run via run.py, or copy config.example.yaml to config.yaml yourself."
                )
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
        # A conversation left with no follow-up would otherwise stay 'hot'
        # (unrecallable) forever; this is how long to wait before treating it
        # as abandoned and shelving it anyway. Shelving is idempotent, so a
        # short value is safe -- a real follow-up a moment later just re-marks
        # already-shelved blocks, no conflict.
        self.hot_shelve_timeout_s: int = raw.get("hot_shelve_timeout_s", 15)
        self.embed_dim: int = raw.get("embed_dim", 768)
        self.recall = RecallConfig(raw.get("recall", {}))
        self.judge = JudgeConfig(raw.get("judge", {}))
        self.tagger = TaggerConfig(raw.get("tagger", {}))
        self.correction_patterns: List[str] = raw.get("correction_patterns", [
            "that's wrong", "doesn't work", "^no[,.]", "səhvdir", "işləmir",
        ])
        self.servers: dict = raw.get("servers", {})
        self.models_dir: str = raw.get("models_dir", "./models")
        self.max_context_tokens: int = raw.get("max_context_tokens", 28000)
        self.web_search = WebSearchConfig(raw.get("web_search", {}))

    def get_server(self, name: str) -> ServerConfig:
        return ServerConfig(self.servers.get(name, {}))
