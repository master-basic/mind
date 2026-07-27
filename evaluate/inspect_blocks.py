#!/usr/bin/env python3
"""Inspect stored blocks to understand content for benchmark generation."""
import msgpack
import sys
import io
from pathlib import Path
from collections import defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

store = Path(r"S:\AI\store\blocks")
ids = [p.stem for p in store.glob("*.msgpack")]

blocks = []
for bid in ids:
    with open(store / f"{bid}.msgpack", "rb") as f:
        data = msgpack.unpackb(f.read())
    text = data.get("text", "")
    tags = data.get("tags", [])
    gist = data.get("gist", "")
    stimulus = data.get("stimulus_text", "")
    btype = data.get("type", "?")
    if len(text) > 50 and tags:
        blocks.append({
            "id": bid[:8],
            "tags": tags,
            "gist": gist,
            "text": text[:300],
            "stimulus": stimulus[:200],
            "type": btype,
        })

by_tag = defaultdict(list)
for b in blocks:
    key = tuple(sorted(b["tags"]))
    by_tag[key].append(b)

print(f"Blocks with text+tags: {len(blocks)}")
print(f"Tag groups: {len(by_tag)}")
for tag, items in sorted(by_tag.items(), key=lambda x: -len(x[1]))[:10]:
    print(f"  {tag}: {len(items)} blocks")
    for item in items[:2]:
        print(f"    gist: {item['gist']}")
        print(f"    text: {item['text'][:150]}")
        print()
