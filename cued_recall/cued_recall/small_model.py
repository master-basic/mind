"""One queue for the small CPU model.

The tagger and the correction verifier both talk to the same llama-server on
judge_endpoint. That server runs on CPU by design (-ngl 0, to leave VRAM for
the reasoning model's KV) with a single slot (-np 1), so concurrency there
buys nothing and costs latency.

Each caller throttling itself separately does not help when the thing being
protected is shared -- the tagger's own limit of 2 still let a judge pass and
two tag calls pile onto one slot, and a third of all tag calls were timing
out. Hold this instead, and the limit means what it says.
"""
import asyncio

SLOTS = asyncio.Semaphore(2)
