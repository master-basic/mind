import re
from typing import List, Tuple


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
