import shutil
from pathlib import Path
from typing import List, Optional
import yaml


class RecallConfig:
    def __init__(self, d: dict):
        self.k: int = d.get("k", 4)
        # Lowered from 0.62 once the relevance judge below took over rejecting
        # what does not apply. The threshold used to do two jobs -- find the
        # right blocks, and suppress the wrong ones -- and it was bad at the
        # second: at 0.62 more than half the prompts that should retrieve
        # nothing retrieved something. Measured with the judge on, 0.48 gives a
        # false-fire rate of 0.00 and recalls the Azerbaijani family 6 of 6,
        # against 3 of 6 at the 0.70 the embedding alone would have needed.
        self.threshold: float = d.get("threshold", 0.48)
        self.budget_tokens: int = d.get("budget_tokens", 3000)
        # Second stage: ask the small model whether a candidate the vector
        # search returned actually helps with THIS question. Similarity alone
        # cannot separate a block about phase 1 from a question about phase 2
        # (measured 0.841 on this corpus) or vocabulary overlap from relevance
        # (0.708) -- both fire above threshold. On by default since the sweep
        # measured it: false fires go to 0.00 at every threshold, and the only
        # recall it costs is the trap family -- which it refuses on exactly the
        # grounds that family exists to test. See evaluate/benchmark.md.
        self.judge_enabled: bool = d.get("judge_enabled", True)
        # Per candidate. Short, because they run while the user waits.
        self.judge_timeout_s: float = d.get("judge_timeout_s", 5.0)
        # What the judge reads as the archived note.
        #
        #   "question" -- the user message the block was written to answer
        #   "text"     -- the block's own words (what shipped until 2026-08-05)
        #
        # Measured on the evaluate/ corpus against the real store
        # representation, at threshold 0.48 with the judge on: false-fire
        # 0.64 with "text", 0.09 with "question", and all 18 legitimate recalls
        # (exact, paraphrase, crosslingual) survive both. The 0.00 false-fire
        # recorded in benchmark.md was an artefact of a harness that showed the
        # judge a seed prompt rather than a block; shown a real block the judge
        # keeps 5 of 6 traps, which is the anchoring failure in
        # grading_traps.md.
        #
        # The risk this trades for, and it is not measured by the corpus: a
        # block whose originating question differed but whose content happens
        # to answer the new one is now refused. If recall starts missing things
        # it used to find, this is the first knob to try.
        # How many candidates to judge, as a multiple of k.
        #
        # 1 means the vector search's own top-k, which is what the judge has
        # always seen -- this reorders that set rather than enlarging it, so
        # TTFT is unchanged. Raising it widens the pool the judge ranks, and
        # costs one judge call per extra candidate: the judge server runs
        # single-slot on CPU at ~140 ms a call, so 4 turns roughly 0.6 s of
        # added latency into 2.3 s. Measure before raising it.
        self.candidate_multiplier: int = max(
            1, int(d.get("candidate_multiplier", 1))
        )
        # Below this relevance score a candidate is dropped. The score is
        # P(yes) over the judge's first token, so 0.5 is the same decision the
        # old yes/no parse made -- measured on the evaluate/ corpus, legitimate
        # recalls score 0.899-0.998 and traps 0.012-0.119, so the cut has a
        # wide empty band around it rather than sitting in a crowd.
        self.judge_score_floor: float = float(d.get("judge_score_floor", 0.5))
        self.judge_note: str = d.get("judge_note", "question")
        if self.judge_note not in ("question", "text"):
            raise ValueError(
                f"recall.judge_note must be 'question' or 'text', "
                f"got {self.judge_note!r}"
            )
        # A pin is a user's explicit "keep this", but it bought nothing at
        # recall: the budget was spent down the relevance ranking, so an
        # equally-scored unpinned block that sorted first could take the last
        # slot. This makes the pin the tie-break in the ranked fill --
        # relevance still decides what fits, a pin only decides between
        # equals. Off restores the old order exactly.
        self.pin_priority: bool = bool(d.get("pin_priority", True))
        # Cosine similarity floor for the judge stage. When the best vector
        # candidate is below this, the judge is not consulted and nothing is
        # recalled -- an off-topic turn used to pay 1.5-2.2 s of CPU calls to
        # conclude "nothing here" (evaluate/throughput.md).
        #
        # Defaults to 0.0 (off). The plan proposed 0.30, but that sits below
        # recall.threshold (0.48), so as prescribed it could never fire; and
        # the measured corpus has no safe value yet -- off-topic control
        # probes top at ~0.50 while the weakest legitimate family (cross-
        # lingual) tops at ~0.64, with trap/distractor above that, so a floor
        # wide enough to remove the tax lands right against a real recall.
        # 0.60 clears the measured live off-topic band (0.49-0.57) but needs
        # the widened corpus to confirm it cannot cut one. Raise it only with
        # a sweep.
        self.floor: float = float(d.get("floor", 0.0))
        if not (0.0 <= self.floor < 1.0):
            raise ValueError(
                f"recall.floor must be in [0, 1), got {self.floor!r}"
            )
        # The keyword channel: a query's distinctive words matched against a
        # block's gist and tags (index.keyword_query), both already in the
        # index. Used when the embed server errors, so an outage degrades
        # recall to keyword overlap rather than silently recalling nothing;
        # the relevance judge still arbitrates the keyword candidates. Phase
        # 5.1 extends this channel into the normal retrieval path, which is
        # when this knob's default matters most.
        self.tag_channel: bool = bool(d.get("tag_channel", True))
        # The keyword channel as a SECOND candidate source in the normal path
        # (Phase 5.1 / F3, F11): vector hits alone miss probes whose wording
        # shares no tokens with the stored block but whose gist/tags overlap,
        # the "relevant but semantically distant wording" miss class. Keyword
        # hits join the judge pool alongside the vector hits and the judge
        # arbitrates both -- keyword-sourced candidates are shown their
        # gist/tags in the prompt. Off by default, deliberately: the vector
        # operating point is a measured result, and the acceptance rows that
        # would justify this (overlap fires recall, content differs -> judge
        # rejects, no false-fire increase) do not exist yet.
        self.tag_second_source: bool = bool(d.get("tag_second_source", False))


class JudgeConfig:
    def __init__(self, d: dict):
        self.interval_tokens: int = d.get("interval_tokens", 20000)
        self.min_age_s: int = d.get("min_age_s", 3600)
        # Was 14 days, which is not a human scale for conversation memory.
        # Shortening it is only safe because purging no longer deletes the
        # block file by default -- see purge_deletes_file.
        self.purge_age_s: int = d.get("purge_age_s", 259200)
        self.summary_max_tokens: int = d.get("summary_max_tokens", 400)
        # Below this there is nothing to compress: a summary of a paragraph is
        # longer than the paragraph. Measured on this store the mean block is
        # 80 tokens, so the gate stops most calls from being made at all,
        # rather than made and answered "keep".
        self.consolidate_min_tokens: int = d.get("consolidate_min_tokens", 600)
        # Which block types the model is allowed to rewrite. Reasoning blocks
        # are the model's own think trace: mostly scaffolding, and compressing
        # them 90% loses nothing. Result blocks are the answer the user
        # actually received and are already dense; reading blocks are source
        # material that was pasted or fetched. Measured on this store, a 1.5B
        # model handed either of those returns a topic sentence -- a 618-token
        # status report came back as "A project status report detailing core
        # features, bugs fixed, and next steps." No wording fixed that; the
        # model cannot compress dense text without discarding it. Those types
        # are left to decay instead.
        self.consolidate_types: List[str] = d.get(
            "consolidate_types", ["reasoning"]
        )
        # A block the model reports as holding nothing reusable does not have
        # to sit out the full purge_age_s.
        self.worthless_age_s: int = d.get("worthless_age_s", 172800)
        # How long a pattern-matched correction waits before it can purge. The
        # block stops being recalled the moment it is marked, so this window
        # costs nothing but the disk it sits on, and it is the difference
        # between a false regex match hiding a memory and destroying it.
        self.corrected_grace_s: int = d.get("corrected_grace_s", 86400)
        # How many times the same block may be rewritten, ever. Each round is a
        # paraphrase of a paraphrase and costs a generation; MIN_SHRINK means
        # there is little left to win after the first two.
        self.max_truncate_count: int = d.get("max_truncate_count", 2)
        # Wall-clock ceiling on one pass. max_per_pass bounds how many blocks
        # are looked at, but not what looking costs -- a store where most
        # blocks qualify for a rewrite would otherwise hold a CPU-only model
        # for hours. A pass that runs out of time stops without marking the
        # rest judged, so the next one resumes where it stopped.
        self.max_pass_seconds: int = d.get("max_pass_seconds", 600)
        # Once judged, leave a block alone this long. Without it a pass takes
        # the oldest blocks, changes nothing, and takes the same ones again.
        self.rejudge_interval_s: int = d.get("rejudge_interval_s", 604800)
        # Repeated recall is the clearest evidence a block matters. At or above
        # this, keep it verbatim rather than compressing it.
        self.keep_recall_count: int = d.get("keep_recall_count", 3)
        # Marking a block purged and dropping its search vector already makes
        # it unrecallable. Deleting the file as well cannot be undone, so it is
        # opt-in.
        self.purge_deletes_file: bool = d.get("purge_deletes_file", False)
        # Blocks per pass. Passes run while the machine is idle, so this can be
        # far more generous than the old hard-coded 50.
        self.max_per_pass: int = d.get("max_per_pass", 200)
        # Quiet time before a pass starts. The judge shares a CPU-only model
        # with the tagger; running mid-conversation competes for CPU with the
        # reasoning model the user is waiting on.
        self.idle_trigger_s: int = d.get("idle_trigger_s", 300)
        # Run a pass this often even when no new material has arrived, so
        # decay still happens during a quiet week.
        self.sweep_interval_s: int = d.get("sweep_interval_s", 21600)
        # How long an unconsumed turn_recalls row is kept. Only the turn
        # immediately after a recall reads one back, so anything older belongs
        # to a conversation that was abandoned mid-turn. Generous, because the
        # rows are tiny and the cost of dropping one early is a block that
        # never gets credit for having been useful.
        self.recall_record_ttl_s: int = d.get("recall_record_ttl_s", 604800)
        # Decay by earned utility rather than by "was it ever recalled".
        #
        # The old rule made one recall, ever, a permanent exemption from
        # age-based purging -- the system could say "used" and "never used" and
        # nothing in between, so a block recalled once eighteen months ago
        # outranked one recalled weekly and the store could only grow. Set
        # False to restore that rule; it stays switchable because this decides
        # what gets deleted, and deletion is the one thing the index cannot
        # undo on its own.
        self.utility_decay: bool = d.get("utility_decay", True)
        # Days of life a single recall earns. 30 means a block recalled once
        # survives a month of being ignored, then goes -- against the old rule,
        # where it survived forever.
        self.utility_recall_weight: float = d.get("utility_recall_weight", 30.0)
        # Extra days for a recall the user did not contradict. Worth more than
        # a bare recall: being shown to the model and not objected to is better
        # evidence than merely being shown. See Pipeline.apply_accepted_verification.
        self.utility_uncontested_weight: float = d.get(
            "utility_uncontested_weight", 60.0
        )
        # Utility at or below this purges (once past the age gate). Zero means
        # "has spent more idle days than it earned".
        self.utility_floor: float = d.get("utility_floor", 0.0)
        # Derive one block from several near-identical ones.
        #
        # On by default since the 2026-08-05 measurement (evaluate/eval_merge.py):
        # a real pass merged a genuine near-duplicate family into a correct
        # generalization that kept every specific and fired recall, and refused
        # the family whose draft dropped one -- with the originals untouched
        # either way, because the merge is the one pass that creates a memory
        # rather than editing one. Off is still available, and it is the only
        # pass that retires evidence behind a new claim, so read the
        # blocks_merged / merge_rejected WAL events after a pass before trusting
        # a fresh store with it.
        self.merge_enabled: bool = d.get("merge_enabled", True)
        # Cosine similarity at which two blocks count as the same ground.
        # Well above recall.threshold (0.48): that one asks "is this relevant",
        # this one asks "is this the same thing said twice".
        self.merge_cluster_sim: float = d.get("merge_cluster_sim", 0.90)
        # How many near-identical blocks it takes before generalising is worth
        # a model call and the loss of the originals from recall.
        self.merge_min_cluster: int = d.get("merge_min_cluster", 3)
        # Blocks younger than this are left alone: a cluster that formed in the
        # last hour is a conversation in progress, not a settled repetition.
        self.merge_min_age_s: int = d.get("merge_min_age_s", 604800)
        # Merges per pass. Low, because each one retires several blocks.
        self.merge_max_per_pass: int = d.get("merge_max_per_pass", 5)


class VerifierConfig:
    """Second opinion on whether a user message is a correction.

    The pattern list only catches phrasings someone thought to write down.
    Across 395 stored blocks it has fired zero times, so every block in the
    archive claims to be unverified or accepted.
    """

    def __init__(self, d: dict):
        self.enabled: bool = d.get("enabled", True)
        # Empty = reuse judge_endpoint, as the tagger does.
        self.endpoint: str = d.get("endpoint", "")
        # How much of the previous answer to show the classifier.
        self.max_chars: int = d.get("max_chars", 1200)
        # Span-level corrections (Phase 4.2 / F4). On, a "yes" verdict also
        # quotes the offending part of the answer, and recall admits the
        # corrected block with that span redacted instead of suppressing the
        # whole block -- 90% of a block that is right keeps being usable.
        # Off by default: the yes/no prompt is the version eval_correction.py
        # measured, and the span quote needs its own eval before it is trusted.
        self.spans: bool = d.get("spans", False)


class TaggerConfig:
    def __init__(self, d: dict):
        self.enabled: bool = d.get("enabled", True)
        # Empty = reuse judge_endpoint (tagging is a cheap call, doesn't
        # need its own model unless you want to split the load).
        self.endpoint: str = d.get("endpoint", "")
        self.max_tags: int = d.get("max_tags", 3)
        self.gist_max_chars: int = d.get("gist_max_chars", 40)


class WebSearchConfig:
    def __init__(self, d: dict):
        self.enabled: bool = d.get("enabled", True)
        # backend: "bing" (default, keyless), "mojeek", "duckduckgo" (both
        # keyless too), or the keyed "brave" / "serper" / "searxng".
        self.backend: str = d.get("backend", "bing")
        self.searxng_url: str = d.get("searxng_url", "")
        # Legacy single key: applied to whichever backend is selected.
        self.api_key: str = d.get("api_key", "")
        # Per-provider keys, so more than one paid backend can be configured
        # at once and used as fallbacks for each other.
        self.brave_api_key: str = d.get("brave_api_key", "")
        self.serper_api_key: str = d.get("serper_api_key", "")
        # Try other configured backends when the chosen one fails or is
        # throttled, instead of reporting "no results" to the model.
        self.fallback: bool = d.get("fallback", True)
        self.max_results: int = d.get("max_results", 5)
        self.fetch_top_n: int = d.get("fetch_top_n", 0)
# When the tool query is explicitly asking for the current date/time,
        # answer against a live keyless clock (Akamai's edge servers, NTP-synced)
        # instead of a scraped snippet -- search engines serve cached snapshots
        # that are routinely weeks or months old, so the model kept answering
        # with the wrong date. Empty time_timezone = auto-detect from the
        # server's IP (ipapi.co), falling back to the machine's own local zone.
        self.time_intent: bool = d.get("time_intent", True)
        self.time_timezone: str = d.get("time_timezone", "")
        # Hard rule: if the user message matches any of these patterns, force
        # the model to call web_search on that turn (tool_choice), rather than
        # leaving it to the model's discretion.
        self.force_search: bool = d.get("force_search", True)
        self.force_patterns: List[str] = d.get("force_patterns", [
            r"\bsearch\b", r"\blatest\b", r"\bcurrent(ly)?\b", r"\bnews\b",
            r"\btoday\b", r"\brecent(ly)?\b", r"\bthis (week|month|year)\b",
            r"\bright now\b", r"\bup[- ]?to[- ]?date\b", r"\bas of\b",
            r"\bnowadays\b", r"\blook (it |this )?up\b", r"\bgoogle\b",
            # Azerbaijani
            r"\baxtar", r"\bson (xəbər|məlumat)", r"\bhal-?hazırda\b",
        ])

    def key_for(self, backend: str) -> str:
        specific = {"brave": self.brave_api_key,
                    "serper": self.serper_api_key}.get(backend, "")
        if specific:
            return specific
        # Fall back to the shared key only for the explicitly chosen backend,
        # so a Brave key is never sent to Serper or vice versa.
        return self.api_key if (self.backend or "").lower() == backend else ""


class ServerConfig:
    def __init__(self, d: dict):
        self.model: str = d.get("model", "")
        self.port: int = d.get("port", 0)
        self.extra_args: List[str] = d.get("extra_args", [])


class Config:
    def __init__(self, path: str | Path = "config.yaml"):
        path = Path(path)
        if not path.exists():
            # config.yaml is gitignored (it holds a machine-specific
            # store_path/snapshot_path); a fresh clone only has the generic
            # template. Bootstrap from it rather than a bare FileNotFoundError.
            example = path.parent / "config.example.yaml"
            if example.exists():
                shutil.copy2(example, path)
            else:
                raise FileNotFoundError(
                    f"{path} not found and no config.example.yaml template at {example}. "
                    "Run via run.py, or copy config.example.yaml to config.yaml yourself."
                )
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        self.listen: str = raw.get("listen", "127.0.0.1:8000")
        self.reasoning_endpoint: str = raw.get("reasoning_endpoint", "http://127.0.0.1:8080")
        self.judge_endpoint: str = raw.get("judge_endpoint", "http://127.0.0.1:8081")
        self.embed_endpoint: str = raw.get("embed_endpoint", "http://127.0.0.1:8082")
        self.think_tags: List[str] = raw.get("think_tags", ["<think>", "</think>"])
        self.store_path: str = raw.get("store_path", "/mnt/ramdisk/cued_recall")
        self.snapshot_path: str = raw.get("snapshot_path", "/var/lib/cued_recall/snapshots")
        self.snapshot_interval_min: int = raw.get("snapshot_interval_min", 15)
        self.block_tokens_reasoning: int = raw.get("block_tokens_reasoning", 8000)
        # A conversation left with no follow-up would otherwise stay 'hot'
        # (unrecallable) forever; this is how long to wait before treating it
        # as abandoned and shelving it anyway. Shelving is idempotent, so a
        # short value is safe -- a real follow-up a moment later just re-marks
        # already-shelved blocks, no conflict.
        self.hot_shelve_timeout_s: int = raw.get("hot_shelve_timeout_s", 15)
        self.embed_dim: int = raw.get("embed_dim", 768)
        # Which text of a block goes into the vector index.
        #
        #   "composite" -- stimulus_text: for a reasoning block, the question
        #       and the answer it produced. This is the shipped behaviour and
        #       the one the measured operating point (threshold 0.48, k=4,
        #       judge on) was tuned against, so it stays the default.
        #   "content"   -- embed_text: what the block itself says.
        #
        # The case for "content" is that the trap family scores high precisely
        # because a reasoning block's vector contains the phase-1 answer, so a
        # phase-2 question sharing those entities matches it. The case against
        # is that the query is always a question, and indexing an answer by the
        # question that produced it is why paraphrase and cross-lingual recall
        # work as well as they do. That trade is measurable and has not been
        # measured, which is exactly why this is a switch and not a rewrite:
        # both texts are stored on every block, so changing it is a re-embed
        # (backfill_reembed.py) rather than a re-ingest. Run the sweep in
        # evaluate/ before changing it.
        self.embed_source: str = raw.get("embed_source", "composite")
        if self.embed_source not in ("composite", "content"):
            raise ValueError(
                f"embed_source must be 'composite' or 'content', "
                f"got {self.embed_source!r}"
            )
        # Cap on either embed text, in whitespace words. A first pass only:
        # the binding limit is embed_ctx_tokens, enforced in tokens by
        # EmbeddingClient.fit, because words and tokens diverge worst exactly
        # where it matters (1,024 words of code measured 2,338 tokens here).
        self.embed_token_limit: int = raw.get("embed_token_limit", 1024)
        # The embedding server's context window. Overwritten at startup from
        # the server's own /props -- the config value is only the fallback for
        # a server that will not say. An input past this comes back HTTP 400,
        # and _embed_and_store swallows it, so the block is stored, listed in
        # the admin table, and unrecallable forever. The old guard was a
        # 16,000-character cap written for an 8,192-token window while this
        # stack runs nomic-embed at 2,048, so it never once fired.
        self.embed_ctx_tokens: int = raw.get("embed_ctx_tokens", 2048)
        self.recall = RecallConfig(raw.get("recall", {}))
        self.judge = JudgeConfig(raw.get("judge", {}))
        self.tagger = TaggerConfig(raw.get("tagger", {}))
        self.verifier = VerifierConfig(raw.get("verifier", {}))
        # Anchored deliberately. A false positive is expensive here: a
        # "corrected" block is dropped from recall entirely AND skips the age
        # gate before purging, so a bad match destroys a good memory. That
        # rules out bare "wrong", "mistake", "fix" and "actually", all of which
        # occur constantly in messages that correct nothing. Phrasings these
        # miss are meant to be caught by the verifier model instead.
        self.correction_patterns: List[str] = raw.get("correction_patterns", [
            # English
            r"\b(that|this|it|you)( is|'s| was| are|'re| were) "
            r"(wrong|incorrect|mistaken|false)\b",
            r"\b(isn'?t|wasn'?t|aren'?t) (right|correct|true)\b",
            # "not right now, maybe later" is a schedule, not a complaint.
            r"\bnot (right|correct|true)\b(?!\s+now)",
            r"\bnot what i (asked|meant|wanted|said)\b",
            r"\b(doesn'?t|does not|didn'?t|did not|won'?t|will not) work\b",
            r"\bstill (doesn'?t|does not|not) work",
            # Anchored to the start of the message. Unanchored, this fired on
            # "I know it doesn't exist yet, please create it", which is a
            # request, not a complaint.
            r"^\s*(that|this|it|there) (doesn'?t|does not|didn'?t|did not) exist\b",
            r"\byou made (a|an) (mistake|error)\b",
            r"\bwrong (answer|again)\b",
            # Punctuation required, as in the original pattern: allowing a
            # space here caught "no problem, carry on".
            r"^\s*(no|nope|nah)[,.!]",
            # Azerbaijani
            r"səhv",
            r"yanlış",
            r"işləmir",
            r"alınmır",
            r"düz deyil",
            r"doğru deyil",
            r"^\s*(yox|xeyr)[,.!\s]",
        ])
        self.servers: dict = raw.get("servers", {})
        self.models_dir: str = raw.get("models_dir", "./models")
        # Prompt budget in real tokens (run.py derives this from the reasoning
        # server's --ctx-size, leaving room for the reply and tool defs).
        self.max_context_tokens: int = raw.get("max_context_tokens", 26624)
        # Multiplier converting whitespace word counts into estimated tokens.
        # ~1.3 for prose; raise it for code- or CJK-heavy workloads. Kept as
        # one of two signals -- see chars_per_token.
        self.tokens_per_word: float = raw.get("tokens_per_word", 1.3)
        # Characters per token. Measured against this stack's tokenizer, this
        # ranges 2.8 (Azerbaijani) to 4.4 (Python source) where words-per-token
        # ranges 1.4 to 4.2 -- i.e. character count is roughly twice as stable
        # a predictor as word count. The budget estimator takes whichever of
        # the two signals is LARGER, because under-counting overflows the
        # server's window (a hard 400) while over-counting only trims early.
        self.chars_per_token: float = raw.get("chars_per_token", 3.2)
        # Tokens held back from the server's window for the reply. Reasoning
        # models emit long think traces before any visible answer, so this
        # needs to cover the thinking too, not just the final message.
        self.context_reserve_tokens: int = raw.get("context_reserve_tokens", 4096)
        # When the estimate lands above this fraction of the budget, stop
        # guessing and ask the server to tokenize the prompt exactly. Costs one
        # ~450 ms round trip, and only on prompts near the limit.
        #
        # 0.6 rather than 0.8 because the trigger is itself an estimate: an
        # under-count both inflates the prompt and delays the check that would
        # have caught it. At 0.8 a 24% under-count reads as 0.79 of budget
        # while the real prompt is already over it, and the exact count never
        # runs. The margin is nearly free -- 450 ms against a prefill measured
        # in tens of seconds.
        self.exact_count_threshold: float = raw.get("exact_count_threshold", 0.6)
        # Cap on tokens emitted per tool round. Reasoning models emit long
        # think traces before any visible answer; without a hard cap a stuck
        # model can stream megabytes of reasoning, never return a finish_reason
        # or a tool call, and leave the client to time out and show an empty
        # turn. Set to null to disable the cap (unbounded upstream behaviour).
        self.max_completion_tokens: Optional[int] = raw.get(
            "max_completion_tokens", 8192
        )
        # Tool-call rounds allowed per turn. When a model keeps requesting
        # tools past this budget (repeated web_search), the pipeline forces one
        # final plain-text round so the turn ends with an answer, not an empty
        # stream. Raised blindly it multiplies inference cost and context per
        # turn without fixing that, hence the forced-answer instead of 20.
        self.max_tool_rounds: int = raw.get("max_tool_rounds", 5)
        # When a forced answer round runs (see pipeline._stream_and_blockify),
        # disable the model's think phase so it writes a reply immediately
        # instead of spending the whole output budget on reasoning and being
        # cut off empty. Uses llama.cpp's chat_template_kwargs.enable_thinking.
        # Set false for a server/model whose template does not support it.
        self.forced_answer_no_think: bool = raw.get("forced_answer_no_think", True)
        self.web_search = WebSearchConfig(raw.get("web_search", {}))

    def get_server(self, name: str) -> ServerConfig:
        return ServerConfig(self.servers.get(name, {}))
