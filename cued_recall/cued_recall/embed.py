import httpx
from typing import List

from .utils import estimate_tokens


class EmbeddingClient:
    def __init__(self, endpoint: str, ctx_tokens: int = 2048,
                 chars_per_token: float = 3.2, tokens_per_word: float = 1.3):
        self.base = endpoint.rstrip("/")
        self.url = self.base + "/v1/embeddings"
        self._client = httpx.Client(timeout=60)
        self.usage_sink = None
        # The embedding model's context window, in tokens. Set from the
        # server's own /props at startup (detect_ctx_tokens) rather than
        # assumed: the previous guard here was a 16,000-character cap chosen
        # for an 8,192-token window, and this stack runs nomic-embed at 2,048.
        # A cap four times too generous never fires, so an oversized input
        # reached the server, came back HTTP 400, and was swallowed by
        # _embed_and_store's except -- leaving a block that is stored, listed
        # in the admin table, and permanently unrecallable.
        self.ctx_tokens = ctx_tokens
        self.chars_per_token = chars_per_token
        self.tokens_per_word = tokens_per_word

    # Held back from the window. The estimator is conservative but not exact,
    # and the server counts a couple of tokens of its own scaffolding.
    CTX_MARGIN = 0.90

    def detect_ctx_tokens(self) -> int:
        """Ask the server how big its window really is.

        Same reasoning as _detect_embed_dim and _clamp_context_budget: the
        live server is the authority, not a number in a config file that was
        written for a different model.
        """
        try:
            r = self._client.get(self.base + "/props", timeout=10)
            if r.status_code == 200:
                data = r.json()
                n_ctx = (data.get("n_ctx")
                         or (data.get("default_generation_settings") or {})
                         .get("n_ctx"))
                if isinstance(n_ctx, int) and n_ctx > 0:
                    self.ctx_tokens = n_ctx
        except (httpx.HTTPError, ValueError, AttributeError):
            pass
        return self.ctx_tokens

    def fit(self, text: str) -> str:
        """Trim text to something the server will actually accept.

        Trimming is by *characters* against a conservative token estimate,
        because the caller's truncate_tokens counts whitespace words and the
        two units diverge exactly where it hurts: 1,024 words of a code-heavy
        think trace measured 2,338 tokens on this stack -- 2.28 tokens a word
        against a 2,048-token window. Prose would have fit; code did not, so
        the failure landed only on the blocks most worth keeping.
        """
        if not text:
            return text
        budget = max(64, int(self.ctx_tokens * self.CTX_MARGIN))
        est = estimate_tokens(text, self.chars_per_token, self.tokens_per_word)
        if est <= budget:
            return text
        # estimate_tokens is monotonic in length, so scaling by the overshoot
        # lands under budget in one step for any realistic text; the loop is
        # there so "realistic" is not load-bearing.
        cut = text
        for _ in range(8):
            keep = max(1, int(len(cut) * budget / max(est, 1)))
            cut = cut[:keep]
            est = estimate_tokens(cut, self.chars_per_token,
                                  self.tokens_per_word)
            if est <= budget:
                break
        return cut

    def embed(self, text: str) -> List[float]:
        text = self.fit(text)
        resp = self._client.post(self.url, json={
            "input": text,
            "model": "default",
        })
        resp.raise_for_status()
        data = resp.json()
        vec = data["data"][0]["embedding"]
        if self.usage_sink:
            usage = data.get("usage")
            if usage:
                self.usage_sink(usage)
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def close(self):
        self._client.close()
