import re
from typing import List, Tuple

import httpx


# Characters are the wrong unit for anything outside ASCII. Measured against
# this model's tokenizer, chars/token ranges from 2.54 (Azerbaijani) to 4.50
# (English prose) -- so a divisor tuned for English under-counts Azerbaijani by
# a fifth. Bytes hold much steadier, 3.11 to 4.50 over the same samples, because
# the extra UTF-8 bytes of ə/ş/ğ/ı/ö/ü/ç are exactly what the tokenizer is
# paying for. 3.0 sits just under the lowest measured figure, so the byte signal
# is an over-estimate on every sample rather than a coin flip.
BYTES_PER_TOKEN = 3.0


def estimate_tokens(text: str, chars_per_token: float = 3.2,
                    tokens_per_word: float = 1.3) -> int:
    """Conservative token estimate from character, word and byte counts.

    Takes the largest of the three signals on purpose -- see
    Pipeline._estimate_tokens, which delegates here. Under-counting hands the
    server a prompt bigger than its window; the reply is then cut off wherever
    it happened to be, which for an agent client means a tool call truncated
    mid-JSON and a turn that fails to parse. Over-counting only costs some
    trimmed history, so every tie goes to the larger number.
    """
    if not text:
        return 0
    by_chars = len(text) / max(chars_per_token, 0.1)
    by_words = len(text.split()) * tokens_per_word
    by_bytes = len(text.encode("utf-8", "ignore")) / BYTES_PER_TOKEN
    return int(max(by_chars, by_words, by_bytes))


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


# The second recall stage: whether a candidate the vector search returned
# actually applies to this question. Kept here rather than in Pipeline so
# evaluate/eval_retrieval.py scores the prompt that ships, not a copy of it
# that drifts.
RELEVANCE_SYSTEM = ("You judge whether a note helps answer a question. You "
                    "reply with exactly one word: yes or no.")


def relevance_prompt(question: str, note: str) -> str:
    """Ask the small model whether one note applies to one question.

    Both sides are bounded: the judge server's window is 8k and several
    candidates may be in flight at once. The instructions sit after the note
    for the same reason the judge's do -- on a 1.5B model, an instruction
    thousands of characters back is one it has stopped attending to.

    The three "answer no" cases are the three the embedding cannot see. On the
    evaluate/ corpus a note about phase 1 scores 0.841 against a question about
    phase 2, and one sharing only vocabulary scores 0.708; both clear the 0.62
    threshold, and both are wrong to inject.
    """
    return (
        f"Question:\n{truncate_tokens(question, 300)}\n\n"
        f"Note from the archive:\n{truncate_tokens(note, 900)}\n\n"
        "Does the note contain information that would change or improve "
        "the answer to that question?\n"
        "Answer no if the note is about a different phase, version or "
        "stage of the problem, if it only shares vocabulary with the "
        "question, or if it addresses a different issue.\n"
        "Answer yes or no."
    )


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


def embed_source_text(block, source: str = "composite") -> str:
    """The text of a block that should go into the vector index.

    One function because four callers need the same answer and a disagreement
    between them is invisible: the pipeline embeds at creation, the judge
    re-embeds after truncation, the admin re-embed endpoint and the backfill
    script repair. If any of them picked a different field, a block's vector
    would stop describing the block and nothing would report it.

    Falls back to the other field whenever the preferred one is empty, so a
    store written before embed_text existed -- or one whose composite was never
    built -- still embeds something rather than silently getting no vector.
    """
    stimulus = (getattr(block, "stimulus_text", "") or "")
    content = (getattr(block, "embed_text", "") or "")
    if source == "content":
        return content or stimulus or (getattr(block, "text", "") or "")
    return stimulus or content or (getattr(block, "text", "") or "")


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
