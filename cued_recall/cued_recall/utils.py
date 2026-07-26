import re
from typing import List, Tuple

import httpx


def estimate_tokens(text: str, chars_per_token: float = 3.2,
                    tokens_per_word: float = 1.3) -> int:
    """Conservative token estimate from character and word counts.

    Takes the larger of the two signals on purpose -- see
    Pipeline._estimate_tokens, which delegates here.
    """
    if not text:
        return 0
    by_chars = len(text) / max(chars_per_token, 0.1)
    by_words = len(text.split()) * tokens_per_word
    return int(max(by_chars, by_words))


async def count_tokens(text: str, endpoint: str, chars_per_token: float = 3.2,
                       tokens_per_word: float = 1.3, timeout: float = 10) -> int:
    """Exact token count from the model's own tokenizer.

    Block token counts were `len(text.split())` -- a word count, which measured
    36 of 36 sampled blocks low (mean 0.77x, worst 0.52x on code and markdown).
    That understated the admin `tokens` column by 42% in aggregate, and because
    the same field enforces recall.budget_tokens, a 3,000-token recall budget
    was really spending closer to 3,900.

    Falls back to the estimator when the server can't be reached, which errs
    high: a block that reads slightly large only costs some recall budget,
    whereas reading small is what overran the context in the first place.
    """
    if not text:
        return 0
    exact = await count_tokens_exact(text, endpoint, timeout)
    if exact is not None:
        return exact
    return estimate_tokens(text, chars_per_token, tokens_per_word)


async def count_tokens_exact(text: str, endpoint: str,
                             timeout: float = 10) -> "int | None":
    """Token count from the server, or None if it could not be reached.

    Separate from count_tokens so a caller can tell a real count from a
    fallback. The backfill needs that distinction: quietly writing estimator
    values across the whole store would be worse than the word counts it is
    replacing.
    """
    if not text:
        return 0
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(endpoint.rstrip("/") + "/tokenize",
                                  json={"content": text})
            if r.status_code == 200:
                tokens = r.json().get("tokens")
                if isinstance(tokens, list):
                    return len(tokens)
    except (httpx.HTTPError, ValueError, KeyError):
        pass
    return None


def truncate_tokens(text: str, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def split_paragraph_boundary(text: str, max_tokens: int) -> Tuple[str, str]:
    paragraphs = re.split(r"(\n\s*\n)", text)
    parts = []
    count = 0
    remainder_start = 0
    for i, chunk in enumerate(paragraphs):
        chunk_tokens = len(chunk.split())
        if count + chunk_tokens > max_tokens and count > 0:
            remainder_start = sum(len(p) for p in paragraphs[:i])
            break
        parts.append(chunk)
        count += chunk_tokens
    left = "".join(parts)
    right = text[len(left):]
    if not right:
        left_sentences = re.split(r"(?<=[.!?])\s+", left)
        left = ""
        count = 0
        for s in left_sentences:
            st = len(s.split())
            if count + st > max_tokens and count > 0:
                right = s + right
                break
            left += s + " "
            count += st
        left = left.strip()
    return left.strip(), right.strip()


def build_stimulus(user_message: str, result_text: str,
                   reading_excerpt: str = "") -> str:
    parts = [truncate_tokens(user_message, 512)]
    parts.append("---")
    parts.append(truncate_tokens(result_text, 512))
    if reading_excerpt:
        parts.append("---")
        parts.append(truncate_tokens(reading_excerpt, 256))
    return "\n".join(parts)


def matches_correction(text: str, patterns: List[str]) -> bool:
    for pat in patterns:
        try:
            if re.search(pat, text, re.IGNORECASE):
                return True
        except re.error:
            if pat.lower() in text.lower():
                return True
    return False
