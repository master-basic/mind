import asyncio
import json
import re
import time
from typing import List, Optional

import httpx

from .config import Config
from .index import VectorIndex
from .models import Block, BlockStatus, BlockType, Verification
from .store import BlockStore
from .utils import count_tokens, embed_source_text, truncate_tokens
from .wal import WAL


class Judge:
    """Consolidation and decay for the archive.

    Two jobs that used to look like one. Compressing a memory means working out
    what it says, so the model does that. Deciding what to forget is age and
    recall count, which are recorded exactly -- so arithmetic does that, rather
    than a 1.5B model asked to compare numbers it was handed.

    The previous design asked the model for a three-way keep/truncate/purge
    verdict and gave it no criteria for any of the three. It answered "keep"
    142 times out of 142, and nothing in the archive was ever compressed or
    removed.
    """

    def __init__(
        self,
        config: Config,
        store: BlockStore,
        index: VectorIndex,
        wal: WAL,
        embed=None,
    ):
        self.config = config
        self.store = store
        self.index = index
        self.wal = wal
        # Truncation rewrites a block's words, which makes its vector describe
        # text the block no longer holds. Optional so an offline or test Judge
        # can run without an embedding server -- see _reembed_block.
        self.embed = embed
        self.judge_url = config.judge_endpoint.rstrip("/") + "/v1/chat/completions"
        self.usage_sink = None
        self.tps_sink = None
        self._n_ctx: Optional[int] = None
        # None = not yet known. Set False the first time the server rejects a
        # constrained response_format, so later calls stop asking for one.
        self._supports_schema: Optional[bool] = None

    # The summary alone is specified at up to 400 tokens; 600 for the whole
    # response left no room for the JSON scaffolding around it, so a verbose
    # summary ran past the cap and the closing brace never arrived. Headroom
    # without going so high that every call pays for tokens it won't use --
    # _salvage() recovers the rest if a summary still runs long.
    MAX_TOKENS = 768
    # The judge runs on CPU by design (-ngl 0, to leave VRAM for the reasoning
    # model's KV). Generating a few hundred tokens there is slow enough that
    # the old 120 s tripped on the larger blocks.
    TIMEOUT_S = 300
    # A rewrite has to earn the loss of the original wording. The whole point
    # is to spend fewer of recall.budget_tokens on this block, and a summary
    # that saves 3% does not pay for what it throws away.
    MIN_SHRINK = 0.8

    SYSTEM_PROMPT = (
        "You compress notes in a memory archive. You reply with one JSON "
        "object and nothing else."
    )

    SUMMARY_SCHEMA = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }

    async def _count_tokens(self, text: str) -> int:
        """Token count for a rewritten block.

        Measured against the reasoning model's tokenizer, not the judge's: this
        number is spent from recall.budget_tokens, which is a slice of the
        reasoning model's context.
        """
        return await count_tokens(
            text, self.config.reasoning_endpoint,
            self.config.chars_per_token, self.config.tokens_per_word,
        )

    async def _judge_n_ctx(self) -> int:
        """The judge server's context window, probed once and cached."""
        if self._n_ctx is not None:
            return self._n_ctx
        self._n_ctx = 8192
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    self.config.judge_endpoint.rstrip("/") + "/props"
                )
                if r.status_code == 200:
                    d = r.json()
                    gs = d.get("default_generation_settings", {}) or {}
                    n = gs.get("n_ctx") or d.get("n_ctx")
                    if n:
                        self._n_ctx = int(n)
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        return self._n_ctx

    @staticmethod
    def _head_tail(text: str, budget_chars: int) -> str:
        """Trim to budget, keeping both ends.

        Head-only truncation would hand the judge the opening of a document
        and ask it to summarise the whole thing. Keeping both ends at least
        shows it where the content started and where it ended up.
        """
        if len(text) <= budget_chars:
            return text
        if budget_chars <= 0:
            return ""
        half = budget_chars // 2
        omitted = len(text) - 2 * half
        return (f"{text[:half]}\n\n[... {omitted:,} characters omitted "
                f"from the middle ...]\n\n{text[-half:]}")

    async def run_pass(self, min_age: Optional[float] = None) -> dict:
        """One sweep. Returns counters, so a pass is a measurable event.

        Bounded two ways, because "it terminates" should be a property of the
        loop rather than a consequence of arithmetic elsewhere. max_per_pass
        caps how many blocks are looked at; max_pass_seconds caps how long the
        looking may take. Without the second, a store where most blocks do
        qualify for a rewrite could hold a CPU-only model for hours -- the
        blocks that skip cost two DB reads, but the ones that don't cost a
        generation each.
        """
        started = time.time()
        deadline = started + max(1, self.config.judge.max_pass_seconds)
        counts = {"purged": 0, "truncated": 0, "model_calls": 0,
                  "visited": 0, "stopped_early": False}

        # Forgetting first, and separately: it needs no model call, so it is
        # not rate-limited by the consolidation cycle. Keeping the two in one
        # loop meant a block judged at an hour old was marked and then left
        # alone for rejudge_interval_s -- longer than purge_age_s, so the
        # purge cutoff would have fired a week late.
        decayed = await self._decay_sweep()
        counts["purged"] += decayed

        # After forgetting, before rewriting: a block the decay sweep just
        # removed should not be merged into anything, and a block about to be
        # merged should not first be spent on an individual rewrite.
        merged = await self._merge_pass(deadline)
        counts.update({f"merge_{k}": v for k, v in merged.items()})

        # Manual runs pass min_age=0 to judge all shelved blocks now; the
        # automatic (idle-triggered) pass uses the configured age gate.
        if min_age is None:
            min_age = self.config.judge.min_age_s
        block_ids = await asyncio.to_thread(
            self.index.blocks_due_for_judging,
            min_age,
            self.config.judge.rejudge_interval_s,
            self.config.judge.max_per_pass,
        )
        for bid in block_ids:
            if time.time() >= deadline:
                # Not marked judged, so the next pass resumes here rather than
                # starting over: judged_at ordering makes the sweep pick up
                # where the clock cut it off.
                counts["stopped_early"] = True
                break
            block = await asyncio.to_thread(self.store.get, bid)
            if block is None:
                continue
            action, detail = await self._consider(block)
            counts["visited"] += 1
            if action == "purge":
                counts["purged"] += 1
            elif action == "truncate":
                counts["truncated"] += 1
            # _consider reports it, rather than the action name implying it:
            # "purge" covers both a decay hit that costs nothing and a model
            # that answered "nothing here worth keeping".
            if detail.pop("model_call", False):
                counts["model_calls"] += 1
            # Record the visit whatever the outcome. Without this a pass takes
            # the oldest blocks, mostly changes nothing, and the next pass
            # takes exactly the same ones -- which is why 345 of 395 blocks had
            # never been looked at.
            await asyncio.to_thread(
                self.index.mark_judged, bid, time.time()
            )
            self.wal.write({
                "event": "judge_action",
                "block_id": bid,
                "action": action,
                "timestamp": time.time(),
                **detail,
            })
        counts["elapsed_s"] = round(time.time() - started, 1)
        counts["decay_purged"] = decayed
        counts["processed"] = decayed + counts["visited"]

        # Recall records whose turn never arrived: a conversation abandoned
        # mid-turn leaves one behind, and nothing else would ever remove it.
        pruned = await asyncio.to_thread(
            self.index.prune_turn_recalls, self.config.judge.recall_record_ttl_s
        )
        if pruned:
            counts["recall_records_pruned"] = pruned

        # Once per pass, not per block: blocks that lost their embedding are
        # invisible to recall but indistinguishable from healthy ones by status,
        # so without a periodic count the condition never surfaces on its own.
        # The pass is the natural place -- it already walks the store on a timer
        # and nothing else does.
        missing = await asyncio.to_thread(self.index.count_blocks_without_vectors)
        counts["blocks_missing_vectors"] = missing
        if missing:
            self.wal.write({
                "event": "vectors_missing",
                "blocks": missing,
                "timestamp": time.time(),
            })
        return counts

    # A merge must be meaningfully smaller than the blocks it replaces,
    # measured against their combined size. Above this ratio it has not
    # generalised anything -- it has restated them and added one more
    # near-duplicate to the index this pass exists to thin out.
    MERGE_MAX_RATIO = 0.7

    # Numbers, paths and dotted identifiers -- the things a merge is most
    # likely to lose or garble, and the only part of "keep every specific"
    # that can be checked rather than requested.
    _SPECIFIC_PATTERNS = (
        re.compile(r"\d+"),                       # 30, 300, 840
        re.compile(r"/[\w./-]+"),                 # /etc/resolv-cache.conf
        re.compile(r"\b\w+(?:\.\w+)+\b"),         # dns.cache_ttl
    )

    @classmethod
    def _specifics(cls, text: str) -> set:
        out = set()
        for pattern in cls._SPECIFIC_PATTERNS:
            for m in pattern.finditer(text or ""):
                # Trailing sentence punctuation is not part of the specific.
                # Without this, "/etc/resolv-cache.conf." at the end of a
                # sentence never matches the same path written mid-sentence,
                # and every merge is rejected for losing something it kept.
                token = m.group(0).rstrip(".,;:!?)’'\"").lower()
                if token:
                    out.add(token)
        return out

    MERGE_SYSTEM = ("You combine several notes from one archive into a single "
                    "note. You reply with one JSON object and nothing else.")

    MERGE_SCHEMA = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }

    def _merge_prompt(self, notes: List[str]) -> str:
        """Ask for what the notes have in common, not a list of them.

        The instructions sit after the notes for the same reason the
        consolidation prompt's do: on a 1.5B model an instruction several
        thousand characters back is one it has stopped attending to.
        """
        body = "\n\n".join(f"Note {i + 1}:\n{n}" for i, n in enumerate(notes))
        return (
            f"Here are {len(notes)} notes from a memory archive that cover "
            "nearly the same ground.\n\n"
            f"{body}\n\n"
            "Write ONE note that a future reader could use in place of all of "
            "them.\n\n"
            "State what holds across the notes, not what each one said "
            "separately. Do not number them, do not refer to them as notes, "
            "and do not write an introduction.\n"
            # The failure this guards against is the whole risk of the pass: a
            # merge that keeps only the theme is a lossy delete wearing a
            # summary's clothes.
            "Keep every specific that appears in any of them: names, numbers, "
            "versions, file paths, commands, settings and outcomes. If two "
            "notes disagree on a detail, keep both and say they differ.\n"
            f"Do not go over {self.config.judge.summary_max_tokens} tokens.\n\n"
            'Reply with {"summary": "<the note>"}.'
        )

    async def _merge_pass(self, deadline: float) -> dict:
        """Derive what holds across near-identical blocks. Off by default.

        The system stores every episode and never reduces many into one, which
        is the single thing a semantic memory is for -- so a question asked
        three times leaves three near-identical blocks, each consuming judge
        budget and prompt tokens to say the same thing.

        Deliberately conservative, because this is the one pass that creates
        memories rather than editing them:

        - only types the judge is already trusted to rewrite,
        - only clusters of merge_min_cluster or more above merge_cluster_sim,
        - the merged block records its parents, and the originals are retired
          reversibly -- status flipped and vector dropped, file always kept,
          even when purge_deletes_file is on. Merging is not forgetting.
        """
        cfg = self.config.judge
        out = {"clusters": 0, "merged_blocks": 0, "retired": 0}
        if not cfg.merge_enabled:
            return out

        candidates = await asyncio.to_thread(
            self.index.merge_candidates, cfg.merge_min_age_s,
            tuple(cfg.consolidate_types),
        )
        # Retired into a merge this pass -- never eligible again.
        seen: set = set()
        # Tried and refused this pass. Reaching the same cluster from another
        # of its members yields the same verdict for another generation, so
        # skip it until the next pass rather than forever.
        attempted: set = set()
        for bid in candidates:
            if out["merged_blocks"] >= cfg.merge_max_per_pass:
                break
            if time.time() >= deadline:
                break
            if bid in seen or bid in attempted:
                continue
            vec = await asyncio.to_thread(self.index.get_vector, bid)
            if vec is None:
                continue
            neighbours = await asyncio.to_thread(
                self.index.query, vec, cfg.merge_min_cluster * 4,
                cfg.merge_cluster_sim,
            )
            cluster = [b for b, _sim in neighbours
                       if b not in seen and b not in attempted]
            if len(cluster) < cfg.merge_min_cluster:
                continue

            blocks = []
            for cid in cluster:
                # index.query filters on status alone, so every other rule that
                # kept a block from being a *seed* has to be applied again here
                # -- otherwise a pinned, corrected or still-warm block gets
                # pulled in as a cluster member and retired behind a merge it
                # was never eligible for. A pin especially: it means "keep this
                # exactly", and being merged is the one thing it must prevent.
                if not await self._mergeable(cid):
                    continue
                b = await asyncio.to_thread(self.store.get, cid)
                # A block whose file is gone cannot be restored later, so it
                # must not be one of the originals a merge retires.
                if b is not None and b.text:
                    blocks.append(b)
            if len(blocks) < cfg.merge_min_cluster:
                continue

            out["clusters"] += 1
            merged_text = await self._merge_notes(blocks)
            if not merged_text:
                attempted.update(b.block_id for b in blocks)
                continue
            # Checked here rather than only where the text is generated, so the
            # guard holds however the merged text was produced.
            #
            # Against the members' combined size, not the largest one: the
            # merge replaces all of them, so three 400-character notes becoming
            # one 500-character note is the win this pass exists for, even
            # though 500 is larger than any single member. Comparing against
            # the largest rejected every real merge.
            combined = sum(len(b.text or "") for b in blocks)
            lost = self._lost_specifics(blocks, merged_text)
            if len(merged_text) > combined * self.MERGE_MAX_RATIO or lost:
                self.wal.write({
                    "event": "merge_rejected",
                    "reason": ("dropped specifics" if lost else
                               "not enough smaller than the blocks it replaces"),
                    "parents": [b.block_id for b in blocks],
                    "combined_chars": combined,
                    "merged_chars": len(merged_text),
                    "lost_specifics": sorted(lost)[:20],
                    "timestamp": time.time(),
                })
                # Not retried within this pass: the same cluster reached from a
                # different seed produces the same three members and the same
                # verdict, at the cost of another generation. A later pass may
                # try again -- the blocks are untouched.
                attempted.update(b.block_id for b in blocks)
                continue

            merged = await self._store_merged_block(blocks, merged_text)
            if merged is None:
                attempted.update(b.block_id for b in blocks)
                continue
            out["merged_blocks"] += 1
            for b in blocks:
                seen.add(b.block_id)
                await self._retire_merged(b, merged.block_id)
                out["retired"] += 1
            self.wal.write({
                "event": "blocks_merged",
                "block_id": merged.block_id,
                "parents": [b.block_id for b in blocks],
                "tokens_before": sum(b.token_count for b in blocks),
                "tokens_after": merged.token_count,
                "timestamp": time.time(),
            })
        return out

    def _lost_specifics(self, blocks: List[Block], merged_text: str) -> set:
        """Numbers, paths and identifiers the merge dropped.

        The prompt asks the model to keep every specific. Asking is not enough.
        Run against the real judge on three genuine near-duplicates about DNS
        latency, the first merge produced:

            "setting dns.cache_ttl=300 ... reduces the cache TTL from 30
             seconds to 60ms"

        The originals said the TTL *was* 30s and that first-lookup *latency*
        fell from 840ms to 60ms. The model conflated two quantities and lost
        840 entirely -- a generalisation that was never true, about to have its
        evidence retired behind it.

        This is the checkable half of that instruction. It cannot catch the
        conflation, but it catches the dropped 840, and a merge that loses a
        number is not one to trust with the rest.
        """
        merged = self._specifics(merged_text)
        original = set()
        for b in blocks:
            original |= self._specifics(b.text or "")
        return original - merged

    async def _mergeable(self, block_id: str) -> bool:
        """The same eligibility test merge_candidates applies to seeds.

        Kept as one predicate because it has to hold for every block a merge
        touches, and the vector search that finds cluster members knows only
        about status.
        """
        cfg = self.config.judge
        meta = await asyncio.to_thread(self.index.get_meta, block_id)
        if not meta:
            return False
        if meta.get("pinned"):
            return False
        if meta.get("verification") == "corrected":
            return False
        if meta.get("type") not in cfg.consolidate_types:
            return False
        if (meta.get("created_at") or 0) > time.time() - cfg.merge_min_age_s:
            return False
        return True

    async def _merge_notes(self, blocks: List[Block]) -> Optional[str]:
        """One merged note, or None if the model could not produce a usable one."""
        n_ctx = await self._judge_n_ctx()
        avail_chars = int(max(512, n_ctx - self.MAX_TOKENS - 400) * 3.0)
        per_note = max(200, avail_chars // max(1, len(blocks)))
        notes = [self._head_tail(b.text or "", per_note) for b in blocks]

        body = {
            "messages": [
                {"role": "system", "content": self.MERGE_SYSTEM},
                {"role": "user", "content": self._merge_prompt(notes)},
            ],
            "temperature": 0.1,
            "max_tokens": self.MAX_TOKENS,
            "repeat_penalty": 1.1,
        }
        if self._supports_schema is not False:
            body["response_format"] = {"type": "json_object",
                                       "schema": self.MERGE_SCHEMA}
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT_S) as client:
                resp = await client.post(self.judge_url, json=body)
                if resp.status_code != 200:
                    return None
                data = resp.json()
        except (httpx.HTTPError, ValueError):
            return None
        if self.usage_sink and data.get("usage"):
            self.usage_sink(data["usage"])
        content = (data.get("choices", [{}])[0]
                   .get("message", {}).get("content", "")) or ""
        obj = self._salvage(content)
        summary = (obj or {}).get("summary") if isinstance(obj, dict) else None
        if not isinstance(summary, str):
            return None
        summary = summary.strip()
        if not summary:
            return None
        # A "merge" no shorter than one member saved nothing and added a
        # near-duplicate to the very index this pass exists to thin out.
        longest = max(len(b.text or "") for b in blocks)
        if len(summary) > longest:
            return None
        return summary

    async def _store_merged_block(self, blocks: List[Block],
                                  text: str) -> Optional[Block]:
        merged = Block(
            type=BlockType.result,
            status=BlockStatus.shelved,
            # Conversation-agnostic on purpose: the point of the block is that
            # it holds across the conversations it came from.
            conversation_id="",
            turn_index=0,
            text=text,
            token_count=await self._count_tokens(text),
            embed_text=truncate_tokens(text, self.config.embed_token_limit),
            stimulus_text=truncate_tokens(text, 1024),
            # No single originating question: judge_note_text falls back to the
            # block's own words, which for a merged block is the short,
            # generalised form rather than a full answer.
            question_text="",
            parents=[b.block_id for b in blocks],
            verification=Verification.unknown,
            created_at=time.time(),
        )
        await asyncio.to_thread(self.store.put, merged)
        await asyncio.to_thread(
            self.index.upsert_block_meta,
            merged.block_id, merged.type.value, merged.status.value,
            merged.created_at, merged.conversation_id, merged.turn_index,
            merged.token_count, merged.verification.value, 0, 0.0,
        )
        await self._reembed_block(merged)
        # A merged block nothing can retrieve is worse than no merge at all,
        # because the originals are about to be retired behind it.
        if self.embed is not None and await asyncio.to_thread(
                self.index.get_vector, merged.block_id) is None:
            await asyncio.to_thread(self.index.update_status,
                                    merged.block_id, "purged")
            self.wal.write({
                "event": "merge_abandoned",
                "block_id": merged.block_id,
                "reason": "merged block could not be embedded",
                "timestamp": time.time(),
            })
            return None
        return merged

    async def _retire_merged(self, block: Block, merged_id: str):
        """Take an original out of recall without taking it out of existence.

        The same reversible flip decay uses -- status and vector -- but never
        the file, whatever purge_deletes_file says. Decay is a judgement that a
        memory stopped being worth keeping; a merge is a judgement that it is
        better said elsewhere, and those must not share a delete.
        """
        block.status = BlockStatus.purged
        await asyncio.to_thread(self.store.put, block)
        await asyncio.to_thread(self.index.delete_vector, block.block_id)
        await asyncio.to_thread(self.index.update_status,
                                block.block_id, "purged")
        self.wal.write({
            "event": "block_retired_into_merge",
            "block_id": block.block_id,
            "merged_into": merged_id,
            "timestamp": time.time(),
        })

    async def _decay_sweep(self) -> int:
        """Purge on age and recall count alone, before anything is sent out.

        The index query returns a superset; _should_purge is the one authority
        on the rule, so the two cannot drift apart.
        """
        candidates = await asyncio.to_thread(
            self.index.decay_candidates, self.config.judge.purge_age_s
        )
        removed = 0
        for bid in candidates:
            block = await asyncio.to_thread(self.store.get, bid)
            if block is None:
                continue
            meta = await asyncio.to_thread(self.index.get_meta, bid)
            verification = meta.get("verification", "unknown") if meta else "unknown"
            recall_count = meta.get("recall_count", 0) if meta else 0
            uncontested = (meta or {}).get("uncontested_recalls") or 0
            last_recalled = (meta or {}).get("last_recalled") or 0.0
            source = meta.get("verification_source", "") if meta else ""
            pinned = bool((meta or {}).get("pinned") or block.pinned)
            age = time.time() - block.created_at
            if not self._should_purge(verification, recall_count, age,
                                      worthless=False, source=source,
                                      pinned=pinned, uncontested=uncontested,
                                      last_recalled=last_recalled):
                continue
            await self._purge_block(block)
            removed += 1
            self.wal.write({
                "event": "judge_action",
                "block_id": bid,
                "action": "purge",
                "reason": ("corrected" if verification == "corrected"
                           else "never_recalled" if recall_count == 0
                           else "utility_exhausted"),
                "recall_count": recall_count,
                "uncontested_recalls": uncontested,
                "utility": round(self._utility(recall_count, uncontested, age,
                                               last_recalled), 2),
                # Which verdict earned it, so a purge can be traced back to a
                # regex, a classifier, or a button.
                "verification_source": source,
                "age_days": round(age / 86400, 1),
                "decided_by": "decay",
                "timestamp": time.time(),
            })
        return removed

    async def _consider(self, block: Block) -> tuple[str, dict]:
        meta = await asyncio.to_thread(self.index.get_meta, block.block_id)
        verification = "unknown"
        recall_count = 0
        source = ""
        pinned = bool(block.pinned)
        if meta:
            verification = meta.get("verification", "unknown")
            recall_count = meta.get("recall_count", 0)
            source = meta.get("verification_source", "")
            pinned = bool(meta.get("pinned") or block.pinned)
        age = time.time() - block.created_at
        cfg = self.config.judge

        # The index query already excludes pinned blocks. Checked again here so
        # a caller that reaches _consider by another route cannot bypass it.
        if pinned:
            return "skip_pinned", {}

        # Forgetting first: it is arithmetic, so it costs no model call and
        # settles the blocks that were never going to be worth compressing.
        if self._should_purge(verification, recall_count, age,
                              worthless=False, source=source, pinned=pinned):
            await self._purge_block(block)
            reason = ("corrected" if verification == "corrected"
                      else "never_recalled")
            return "purge", {"reason": reason, "age_days": round(age / 86400, 1)}

        # Repeated recall is the strongest evidence a block earns its place.
        # Do not paraphrase something that keeps proving useful.
        if recall_count >= cfg.keep_recall_count:
            return "keep_recalled", {"recall_count": recall_count}

        if block.type.value not in cfg.consolidate_types:
            return "skip_type", {"type": block.type.value}

        if block.token_count < cfg.consolidate_min_tokens:
            return "skip_small", {"token_count": block.token_count}

        # Each rewrite is a paraphrase of a paraphrase, and every round costs a
        # generation whether or not it produces anything usable. Past a couple
        # of rounds there is no compression left to win -- MIN_SHRINK sees to
        # that -- so stop asking rather than re-deciding weekly forever.
        if block.truncate_count >= cfg.max_truncate_count:
            return "skip_max_truncations", {
                "truncate_count": block.truncate_count,
            }

        was = block.token_count
        summary = await self._consolidate(block)

        if summary is None:
            return "no_decision", {"model_call": True}

        if summary == "":
            # The model found nothing reusable. That is a weaker signal than a
            # correction, so it shortens the wait rather than skipping it.
            if self._should_purge(verification, recall_count, age,
                                  worthless=True, source=source, pinned=pinned):
                await self._purge_block(block)
                return "purge", {"reason": "worthless",
                                 "age_days": round(age / 86400, 1),
                                 "model_call": True}
            return "worthless_kept", {"age_days": round(age / 86400, 1),
                                      "model_call": True}

        if self._is_copied_opening(summary, block.text):
            # Seen on a 2,091-token block: the "summary" was the first two
            # sentences of the original, word for word. That is not a
            # compression of the block, it is the loss of 98% of it, and the
            # size check below waves it through precisely because it is small.
            return "summary_was_copied", {"was": was, "model_call": True}

        got = await self._count_tokens(summary)
        if got > was * self.MIN_SHRINK:
            return "summary_not_shorter", {"was": was, "got": got,
                                           "model_call": True}

        await self._truncate_block(block, summary, got)
        return "truncate", {"was": was, "got": got, "model_call": True,
                            "truncate_count": block.truncate_count}

    def _utility(self, recall_count: int, uncontested: int, age: float,
                 last_recalled: float = 0.0) -> float:
        """How much a block has earned its place, as a number rather than a flag.

        The old rule was `recall_count > 0` and nothing else: one recall, ever,
        made a block permanently exempt from age-based purging. So the system
        could express "used" and "never used" and nothing in between -- a block
        recalled once eighteen months ago outranked one recalled every week,
        because both were simply "used".

        Three terms, all in the same currency of days-of-life earned:

        - each recall is worth recall_weight days,
        - each *uncontested* recall is worth uncontested_weight more, because
          being shown to the model and not contradicted is better evidence than
          merely being shown,
        - and it decays with the time since the block was last useful, not
          since it was created -- otherwise a block that keeps being recalled
          still ages out on a fixed schedule.
        """
        cfg = self.config.judge
        earned = (recall_count * cfg.utility_recall_weight
                  + uncontested * cfg.utility_uncontested_weight)
        # Idle time, not age: a block recalled last week is not stale merely
        # because it was written a year ago. Falls back to age for a block that
        # was never recalled, where the two are the same thing.
        idle = age if not last_recalled else max(0.0, time.time() - last_recalled)
        return earned - (idle / 86400.0)

    def _should_purge(self, verification: str, recall_count: int,
                      age: float, worthless: bool,
                      source: str = "", pinned: bool = False,
                      uncontested: int = 0,
                      last_recalled: float = 0.0) -> bool:
        """Decay, as plain arithmetic.

        A corrected answer is actively harmful if it is recalled again, so it
        stops being recalled the moment it is marked -- but how fast the block
        itself goes depends on who said so, because the three sources are not
        equally reliable:

        | source    | effect                                                  |
        |-----------|---------------------------------------------------------|
        | manual    | Purge at once. The user pressed the button.             |
        | pattern   | Purge only if it was never recalled, and only after     |
        |           | corrected_grace_s. 17 regexes with no measured           |
        |           | false-positive rate should not be able to delete a       |
        |           | memory the system was actively using; dropping it from   |
        |           | recall already removes the harm, and the grace window    |
        |           | leaves time to notice and unpin/pin it.                  |
        | model     | Must clear the ordinary never-recalled-and-old bar. The  |
        |           | 1.5B classifier scores 13/14 on a hand-built set of 14,  |
        |           | which is too small a sample to license deletion, and its |
        |           | known miss reads "now do the same for the firewall" as   |
        |           | a complaint.                                             |

        Everything else is decided by utility rather than by a flag. Retrieval
        is still the only evidence this system gathers on its own that a memory
        is load-bearing, but "was it ever retrieved" is too blunt a reading of
        it: a block recalled once, long ago, was permanently exempt from
        purging, so the store could only ever grow. _utility turns recalls into
        days of life earned and spends them against idle time, so a memory that
        keeps being used keeps being kept and one that stopped being used
        eventually goes.

        A pin still exempts a block outright -- that is the mechanism for the
        memories retrieval cannot gather evidence about.
        """
        cfg = self.config.judge
        if pinned:
            return False
        if verification == "corrected":
            if source == "manual":
                return True
            if source == "pattern":
                return recall_count == 0 and age > cfg.corrected_grace_s
            # source == "model", or unrecorded: no special power, fall through.
        floor = cfg.worthless_age_s if worthless else cfg.purge_age_s
        if age <= floor:
            # Never purge a young block however unused: the age gate is what
            # keeps a quiet week from emptying the store.
            return False
        if not cfg.utility_decay:
            # The pre-2026-08-05 rule, kept switchable because this changes
            # what gets deleted and deletion is the one thing that cannot be
            # undone from the index alone.
            return recall_count == 0
        return self._utility(recall_count, uncontested, age,
                             last_recalled) <= cfg.utility_floor

    def _user_prompt(self, stimulus: str, text: str, abridged: bool) -> str:
        """The consolidation prompt.

        The instructions sit after the note, not before it. A block can run to
        thousands of characters, and on a 1.5B model an instruction that far
        back from the point of generation is one the model has largely stopped
        attending to.
        """
        note = (
            "\nThe note above was shortened to fit, so it may stop part-way "
            "through. Rewrite what is shown, and say at the end that it is "
            "only part of the original.\n"
            if abridged else ""
        )
        return (
            "Here is a note from a memory archive, and the question that "
            "produced it.\n\n"
            f"Question:\n{stimulus}\n\n"
            f"Note:\n{text}\n"
            f"{note}\n"
            "Rewrite the note so a future reader gets the same value from "
            "fewer words.\n\n"
            "Cut: restating the question, thinking out loud, attempts that "
            "were abandoned, anything said twice, and pleasantries.\n"
            "Write dense notes, not prose. Add nothing that was not already "
            "above.\n"
            # Without this the model narrates the task instead of doing it:
            # "The note describes a logic puzzle...", "The note is kept to 400
            # tokens or fewer...". Both observed on real blocks.
            "Write the note itself, not a description of it. Do not begin "
            'with "The note" or "This note", and do not mention these '
            "instructions.\n\n"
            # Last, because on this model the final instruction carries the
            # most weight, and losing the specifics is the failure that makes
            # a summary worthless. An earlier draft ended on "shorter is
            # better" and got back rewrites that had dropped every API path,
            # version and filename in the block.
            "Keep every specific: names, numbers, versions, file paths, "
            "commands, settings, error messages, and what the outcome was. "
            "If you are unsure whether a detail matters, keep it.\n"
            f"Do not go over {self.config.judge.summary_max_tokens} tokens.\n\n"
            "If the note holds nothing a future reader could use -- it is "
            "small talk, or it only repeats the question, or it stops before "
            "reaching any conclusion -- return an empty summary instead.\n\n"
            "Reply with exactly one JSON object and no other text:\n"
            '{"summary": "<the rewritten note, or an empty string>"}'
        )

    def _body(self, stimulus: str, text: str, abridged: bool) -> dict:
        body = {
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",
                 "content": self._user_prompt(stimulus, text, abridged)},
            ],
            "temperature": 0.1,
            "max_tokens": self.MAX_TOKENS,
            # Observed on a 653-token block: the model fell into a loop and
            # emitted "The note is not a summary but a continuation of the
            # puzzle's discussion." fifteen times, filling all 768 tokens and
            # taking 40 s to produce something longer than the original. The
            # shrink check rejects that, but it is cheaper not to generate it.
            "repeat_penalty": 1.1,
        }
        # Constraining the reply at the server turns "please answer in JSON"
        # into something the sampler cannot violate. Not every llama.cpp build
        # accepts it, so the first rejection turns it off for good.
        if self._supports_schema is not False:
            body["response_format"] = {
                "type": "json_object",
                "schema": self.SUMMARY_SCHEMA,
            }
        return body

    async def _consolidate(self, block: Block) -> Optional[str]:
        """The rewrite, or "" for nothing worth keeping, or None on failure."""
        # The block has to fit the judge's window, and the biggest blocks are
        # exactly the ones worth compressing. Sending them whole made the
        # request 400 (a 64 KB block is ~19,700 tokens against a window of
        # 8,192), which turned into "keep" -- so the blocks most in need of
        # truncation were the only ones that could never be truncated, and
        # every pass burned a doomed request on them.
        n_ctx = await self._judge_n_ctx()
        # Reserve the reply plus the instruction preamble, then convert the
        # remainder to characters conservatively.
        avail_tokens = max(512, n_ctx - self.MAX_TOKENS - 400)
        avail_chars = int(avail_tokens * 3.0)
        stim_chars = avail_chars // 4
        stimulus = self._head_tail(block.stimulus_text or "", stim_chars)
        text = self._head_tail(block.text or "", avail_chars - len(stimulus))
        abridged = len(text) < len(block.text or "")
        shrunk_once = False

        started = time.time()
        try:
            async with httpx.AsyncClient(timeout=self.TIMEOUT_S) as client:
                for _ in range(3):
                    resp = await client.post(
                        self.judge_url,
                        json=self._body(stimulus, text, abridged),
                    )
                    if resp.status_code == 200:
                        break
                    # The window can still be exceeded if the estimate was
                    # optimistic. The server reports the real numbers; halve
                    # the block text against them and try again. The prompt is
                    # rebuilt from the shortened text -- the old code
                    # substituted block.text into the finished prompt, which is
                    # not in it once _head_tail has trimmed, so the retry
                    # re-sent the identical oversized request.
                    if not shrunk_once and self._is_overflow(resp.text):
                        text = self._head_tail(text, len(text) // 2)
                        abridged = True
                        shrunk_once = True
                        self.wal.write({
                            "event": "judge_overflow_retry",
                            "block_id": block.block_id,
                            "retry_chars": len(text),
                            "timestamp": time.time(),
                        })
                        continue
                    if (self._supports_schema is not False
                            and self._schema_rejected(resp.text)):
                        self._supports_schema = False
                        self.wal.write({
                            "event": "judge_schema_unsupported",
                            "sample": (resp.text or "")[:200],
                            "timestamp": time.time(),
                        })
                        continue
                    resp.raise_for_status()
                resp.raise_for_status()
                data = resp.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                usage = data.get("usage")
                if usage:
                    if self.usage_sink:
                        self.usage_sink(usage)
                    if self.tps_sink:
                        self.tps_sink(
                            usage.get("completion_tokens", 0),
                            time.time() - started,
                        )
        except Exception as e:
            self.wal.write({
                "event": "judge_error",
                "block_id": block.block_id,
                # str() on an httpx timeout is the empty string, which made a
                # third of the tagger's failures unreadable in the log. Always
                # carry the exception type.
                "error": f"{type(e).__name__}: {e}",
                "timestamp": time.time(),
            })
            return None

        return self._parse_summary(content, block.block_id)

    @staticmethod
    def _salvage(content: str) -> Optional[dict]:
        """Recover the summary from JSON the model never finished writing."""
        # Take everything after the opening quote of "summary", up to a
        # closing quote that is followed by a comma or brace -- or, when the
        # output was cut off, to the end of what arrived.
        s = re.search(r'"summary"\s*:\s*"(.*?)(?:"\s*[,}]|$)', content or "",
                      re.DOTALL)
        if not s:
            return None
        text = s.group(1).strip().replace('\\"', '"').replace("\\n", "\n")
        # A salvaged summary replaces the block's text, so it should not end
        # mid-sentence. Cut back to the last sentence that finished, provided
        # that keeps most of what arrived.
        if text and text[-1] not in ".!?":
            cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
            if cut > len(text) * 0.5:
                text = text[:cut + 1]
        return {"summary": text}

    @staticmethod
    def _is_copied_opening(summary: str, text: str) -> bool:
        """True when the model returned the start of the block, not a summary.

        Compared on collapsed whitespace so a difference in line wrapping does
        not hide it.
        """
        s = " ".join((summary or "").split())
        t = " ".join((text or "").split())
        if not s or not t:
            return False
        return t.startswith(s[:200])

    @staticmethod
    def _is_overflow(raw: str) -> bool:
        try:
            err = (json.loads(raw) or {}).get("error") or {}
        except (json.JSONDecodeError, TypeError):
            return False
        return err.get("type") == "exceed_context_size_error" or bool(
            err.get("n_prompt_tokens") and err.get("n_ctx")
        )

    @staticmethod
    def _schema_rejected(raw: str) -> bool:
        low = (raw or "").lower()
        return "response_format" in low or "schema" in low

    def _parse_summary(self, content: str,
                       block_id: str = "") -> Optional[str]:
        # Unreadable output used to fall back to "keep", which was the safe
        # choice but was indistinguishable in the log from the model
        # deliberately keeping a block. Record why, so "nothing happened" can
        # be told apart from "the model is emitting garbage".
        def bail(reason):
            self.wal.write({
                "event": "judge_parse_failed",
                "block_id": block_id,
                "reason": reason,
                "sample": (content or "")[:200],
                "timestamp": time.time(),
            })
            return None

        json_match = re.search(r"\{.*\}", content or "", re.DOTALL)
        obj = None
        if json_match:
            try:
                obj = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                obj = None
        if obj is None:
            # A small model that runs past max_tokens leaves the object
            # unterminated, so `\{.*\}` matches nothing and a perfectly good
            # summary gets thrown away. The field is still readable on its own.
            obj = self._salvage(content)
        if obj is None:
            return bail("no parseable summary in output")
        if not isinstance(obj, dict) or "summary" not in obj:
            return bail("no summary field")
        summary = obj.get("summary")
        if summary is None:
            return ""
        if not isinstance(summary, str):
            return bail(f"summary was {type(summary).__name__}, not a string")
        return summary.strip()

    async def _truncate_block(self, block: Block, summary: str,
                              new_tokens: int):
        # Only on the first pass: a block that gets summarised twice must keep
        # the words it started with, not the previous summary.
        if not block.original_text:
            block.original_text = block.text
        block.text = summary
        block.original_len = block.token_count
        block.token_count = new_tokens
        block.truncate_count += 1
        block.status = BlockStatus.truncated
        # The block's own words changed, so the text that represents it has to
        # change with them. Without this a truncated block keeps a vector built
        # from words it no longer contains: recall matches on one text and
        # injects a different one, and the gap widens with each rewrite.
        block.embed_text = truncate_tokens(summary,
                                           self.config.embed_token_limit)
        # For result and reading blocks stimulus_text is a copy of the block's
        # own text (set that way at creation), so leaving it would make it a
        # copy of text that no longer exists. A reasoning block's stimulus is
        # the question that produced it, which truncation does not change.
        if block.type != BlockType.reasoning:
            block.stimulus_text = block.embed_text
        await asyncio.to_thread(self.store.put, block)
        await asyncio.to_thread(
            self.index.update_status, block.block_id, "truncated"
        )
        await self._reembed_block(block)

    async def _reembed_block(self, block: Block):
        """Rebuild a block's vector after its text changed.

        Best-effort, and deliberately non-destructive on failure: a stale
        vector still finds the block, whereas dropping it would make the block
        unrecallable and count as vector loss. The failure is recorded so it is
        not silent, which is the whole complaint about the creation-time path.
        """
        text = embed_source_text(block, self.config.embed_source)
        if self.embed is None or not text:
            return
        try:
            vec = await asyncio.to_thread(self.embed.embed, text)
            await asyncio.to_thread(
                self.index.upsert_vector, block.block_id, vec
            )
        except Exception as e:
            self.wal.write({
                "event": "reembed_error",
                "block_id": block.block_id,
                "error": str(e),
                "timestamp": time.time(),
            })

    async def _purge_block(self, block: Block):
        block.status = BlockStatus.purged
        await asyncio.to_thread(self.store.put, block)
        # Dropping the vector is what actually makes a block unreachable.
        # Leaving it behind kept purged blocks matching searches and then
        # resolving to nothing, because the status filter runs after the KNN.
        await asyncio.to_thread(self.index.delete_vector, block.block_id)
        await asyncio.to_thread(
            self.index.update_status, block.block_id, "purged"
        )
        # Status and vector removal are enough to make it unrecallable, and
        # they can be undone. Deleting the file cannot, so it is opt-in.
        if self.config.judge.purge_deletes_file:
            await asyncio.to_thread(self.store.delete_file, block.block_id)
