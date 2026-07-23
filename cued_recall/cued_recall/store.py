import os
import uuid
import tempfile
import shutil
import threading
from pathlib import Path
from typing import Optional, List

import msgpack

from .models import Block, BlockStatus, BlockType, Verification


class BlockStore:
    def __init__(self, path: Path):
        self.path = path
        self.blocks_dir = path / "blocks"
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _block_path(self, block_id: str) -> Path:
        return self.blocks_dir / f"{block_id}.msgpack"

    def put(self, block: Block) -> Block:
        tmp = tempfile.NamedTemporaryFile(
            dir=self.blocks_dir,
            suffix=".tmp",
            delete=False,
        )
        try:
            data = block.to_msgpack()
            packed = msgpack.packb(data)
            tmp.write(packed)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            dest = self._block_path(block.block_id)
            os.replace(tmp.name, dest)
        except BaseException:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            raise
        return block

    def get(self, block_id: str) -> Optional[Block]:
        p = self._block_path(block_id)
        if not p.exists():
            return None
        with open(p, "rb") as f:
            data = msgpack.unpackb(f.read())
        return Block.from_msgpack_opt(data)

    def delete_file(self, block_id: str):
        p = self._block_path(block_id)
        if p.exists():
            p.unlink()

    def list_block_ids(self) -> List[str]:
        return [p.stem for p in self.blocks_dir.glob("*.msgpack")]

    def all_blocks(self) -> List[Block]:
        blocks = []
        for bid in self.list_block_ids():
            b = self.get(bid)
            if b:
                blocks.append(b)
        return blocks

    def snapshot(self, dest_dir: Path):
        dest_dir.mkdir(parents=True, exist_ok=True)
        snapshot_blocks = dest_dir / "blocks"
        if snapshot_blocks.exists():
            shutil.rmtree(snapshot_blocks)
        shutil.copytree(self.blocks_dir, snapshot_blocks)

    def restore(self, src_dir: Path):
        src_blocks = src_dir / "blocks"
        if src_blocks.exists():
            if self.blocks_dir.exists():
                shutil.rmtree(self.blocks_dir)
            shutil.copytree(src_blocks, self.blocks_dir)
