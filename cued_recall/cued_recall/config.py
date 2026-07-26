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
        # Legacy single key: applied to whichever backend is selected.
        self.api_key: str = d.get("api_key", "")
        # Per-provider keys, so more than one paid backend can be configured
        # at once and used as fallbacks for each other.
        self.brave_api_key: str = d.get("brave_api_key", "")
        self.serper_api_key: str = d.get("serper_api_key", "")
        # Try other configured backends when the chosen one fails or is
        # throttled, instead of reporting "no results" to the model.
        self.fallback: bool = d.get("fallback", True)
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

    def key_for(self, backend: str) -> str:
        specific = {"brave": self.brave_api_key,
                    "serper": self.serper_api_key}.get(backend, "")
        if specific:
            return specific
        # Fall back to the shared key only for the explicitly chosen backend,
        # so a Brave key is never sent to Serper or vice versa.
        return self.api_key if (self.backend or "").lower() == backend else ""


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
        # Prompt budget in real tokens (run.py derives this from the reasoning
        # server's --ctx-size, leaving room for the reply and tool defs).
        self.max_context_tokens: int = raw.get("max_context_tokens", 26624)
        # Multiplier converting whitespace word counts into estimated tokens.
        # ~1.3 for prose; raise it for code- or CJK-heavy workloads. Kept as
        # one of two signals -- see chars_per_token.
        self.tokens_per_word: float = raw.get("tokens_per_word", 1.3)
        # Characters per token. Measured against this stack's tokenizer, this
        # ranges 2.8 (Azerbaijani) to 4.4 (Python source) where words-per-token
        # ranges 1.4 to 4.2 -- i.e. character count is roughly twice as stable
        # a predictor as word count. The budget estimator takes whichever of
        # the two signals is LARGER, because under-counting overflows the
        # server's window (a hard 400) while over-counting only trims early.
        self.chars_per_token: float = raw.get("chars_per_token", 3.2)
        # Tokens held back from the server's window for the reply. Reasoning
        # models emit long think traces before any visible answer, so this
        # needs to cover the thinking too, not just the final message.
        self.context_reserve_tokens: int = raw.get("context_reserve_tokens", 4096)
        # When the estimate lands above this fraction of the budget, stop
        # guessing and ask the server to tokenize the prompt exactly. Costs one
        # ~450 ms round trip, and only on prompts near the limit.
        self.exact_count_threshold: float = raw.get("exact_count_threshold", 0.8)
        self.web_search = WebSearchConfig(raw.get("web_search", {}))

    def get_server(self, name: str) -> ServerConfig:
        return ServerConfig(self.servers.get(name, {}))
