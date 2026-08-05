import asyncio
import json
import re
import time
from typing import Optional

import httpx

from .config import Config
from .index import VectorIndex
from .models import Block, BlockStatus
from .store import BlockStore
from .utils import count_tokens
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
    ):
        self.config = config
        self.store = store
        self.index = index
        self.wal = wal
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
            source = meta.get("verification_source", "") if meta else ""
            pinned = bool((meta or {}).get("pinned") or block.pinned)
            age = time.time() - block.created_at
            if not self._should_purge(verification, recall_count, age,
                                      worthless=False, source=source,
                                      pinned=pinned):
                continue
            await self._purge_block(block)
            removed += 1
            self.wal.write({
                "event": "judge_action",
                "block_id": bid,
                "action": "purge",
                "reason": ("corrected" if verification == "corrected"
                           else "never_recalled"),
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

    def _should_purge(self, verification: str, recall_count: int,
                      age: float, worthless: bool,
                      source: str = "", pinned: bool = False) -> bool:
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

        Everything else has to have been never once recalled -- retrieval is
        the only evidence this system gathers on its own that a memory is
        load-bearing, which is exactly why a pin exists for the memories it
        cannot gather it about.
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
        if recall_count > 0:
            return False
        if worthless:
            return age > cfg.worthless_age_s
        return age > cfg.purge_age_s

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
        await asyncio.to_thread(self.store.put, block)
        await asyncio.to_thread(
            self.index.update_status, block.block_id, "truncated"
        )

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
