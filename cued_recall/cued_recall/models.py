import uuid
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional


class BlockType(str, Enum):
    reasoning = "reasoning"
    reading = "reading"
    result = "result"


class BlockStatus(str, Enum):
    hot = "hot"
    shelved = "shelved"
    truncated = "truncated"
    purged = "purged"


class Verification(str, Enum):
    accepted = "accepted"
    corrected = "corrected"
    unknown = "unknown"


@dataclass
class Block:
    block_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: BlockType = BlockType.reasoning
    status: BlockStatus = BlockStatus.hot
    created_at: float = field(default_factory=time.time)
    conversation_id: str = ""
    turn_index: int = 0
    token_count: int = 0
    text: str = ""
    original_len: int = 0
    # What text held before the judge rewrote it. Truncation replaces the
    # block's own words with a small model's paraphrase, and unlike purging
    # (which only flips a status and drops a vector) that could not be undone.
    # Written once, on the first truncation, so re-summarising a summary can
    # never lose the true original.
    original_text: str = ""
    # How many times the judge has rewritten this block. Bounded by
    # judge.max_truncate_count: each rewrite has to be 20% smaller than the
    # last, so repeated summarising converges on a topic sentence, and the
    # earlier rounds are the only ones that were ever paying for themselves.
    truncate_count: int = 0
    # What was being asked when this block was written. For a reasoning block
    # that is the question-plus-answer composite from build_stimulus; for
    # result and reading blocks it is a copy of the block's own text. Two jobs
    # historically: the vector source, and the "Question:" context in the
    # judge's consolidation prompt. Only the second is left -- see embed_text.
    stimulus_text: str = ""
    # What this block actually says, in its own words. Split out from
    # stimulus_text because the two answer different questions and only one of
    # them belongs in a search index: embedding a reasoning block as its Q+A
    # composite means the vector *is* the question that produced it, so any
    # later question sharing entities with that answer scores high against it
    # (measured 0.841 for a phase-1 block against a phase-2 question). Which of
    # the two feeds the index is config.embed_source; both are always stored,
    # so switching is a re-embed and not a re-ingest.
    embed_text: str = ""
    # The user's message that this block was written in answer to. Stored on
    # its own because it is what the relevance judge should read: shown a
    # block's own words, the judge keeps 5 of 6 trap candidates, and shown the
    # question the block was answering it keeps 0 of 6 -- with every legitimate
    # recall surviving either way. See evaluate/eval_judge_notes.py.
    #
    # For reasoning blocks this is also recoverable from stimulus_text, whose
    # first section is the question, which is what makes the change work on a
    # store written before this field existed.
    question_text: str = ""
    verification: Verification = Verification.unknown
    # User-set: exempt from decay and from consolidation, at any age. Retrieval
    # is the only evidence this system gathers on its own that a memory matters,
    # which leaves no way to say "keep this" about one that has not been needed
    # yet. This is that way.
    pinned: bool = False
    recall_count: int = 0
    last_recalled: float = 0.0
    tags: List[str] = field(default_factory=list)
    gist: str = ""

    def to_msgpack(self) -> dict:
        d = asdict(self)
        d["type"] = d["type"].value
        d["status"] = d["status"].value
        d["verification"] = d["verification"].value
        return d

    @staticmethod
    def from_msgpack(d: dict) -> "Block":
        d["type"] = BlockType(d["type"])
        d["status"] = BlockStatus(d["status"])
        d["verification"] = Verification(d["verification"])
        return Block(**d)

    @staticmethod
    def from_msgpack_opt(d: dict) -> Optional["Block"]:
        try:
            return Block.from_msgpack(d)
        except (KeyError, ValueError, TypeError):
            return None
