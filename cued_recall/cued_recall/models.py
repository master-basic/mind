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
    stimulus_text: str = ""
    verification: Verification = Verification.unknown
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
