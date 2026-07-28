import time
from typing import Optional

import httpx

from .config import Config
from .small_model import SLOTS
from .wal import WAL


class CorrectionVerifier:
    """Asks the small model whether a user message is correcting the answer.

    The pattern list in config only catches phrasings someone thought to write
    down, and across 395 stored blocks it had fired exactly zero times -- so
    nothing was ever marked corrected and the judge's fastest route to
    forgetting was dead. A 1.5B model cannot do the judge's old three-way
    decision, but it can answer one yes/no question.

    Never called on the request path. The classifier costs a CPU-bound
    round trip, and verification is applied to the PREVIOUS turn's blocks
    anyway, so nothing needs it before the reply goes out.
    """

    TIMEOUT_S = 120
    MAX_TOKENS = 4
    SYSTEM_PROMPT = ("You classify one message. You reply with exactly one "
                     "word: yes or no.")

    def __init__(self, config: Config, wal: WAL):
        self.config = config
        self.wal = wal
        endpoint = config.verifier.endpoint or config.judge_endpoint
        self.url = endpoint.rstrip("/") + "/v1/chat/completions"
        self.usage_sink = None

    async def is_correction(self, previous_answer: str,
                            user_message: str) -> Optional[bool]:
        """True/False, or None when the model gave no usable answer.

        None is not False. An unreachable or confused model must not be
        recorded as "the user was happy with it".
        """
        if not self.config.verifier.enabled:
            return None
        if not (user_message or "").strip():
            return None

        limit = self.config.verifier.max_chars
        answer = (previous_answer or "")[:limit]
        # The user's message is the thing being classified, so it is never
        # trimmed as hard as the answer it responds to.
        message = (user_message or "")[:2000]

        prompt = self._prompt(answer, message)

        try:
            async with SLOTS:
                async with httpx.AsyncClient(timeout=self.TIMEOUT_S) as client:
                    resp = await client.post(
                        self.url,
                        json={
                            "messages": [
                                {"role": "system",
                                 "content": self.SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": 0,
                            "max_tokens": self.MAX_TOKENS,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    content = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    if self.usage_sink:
                        usage = data.get("usage")
                        if usage:
                            self.usage_sink(usage)
        except Exception as e:
            self.wal.write({
                "event": "verifier_error",
                # str() on an httpx timeout is "", so always carry the type.
                "error": f"{type(e).__name__}: {e}",
                "timestamp": time.time(),
            })
            return None

        return self._parse(content)

    @staticmethod
    def _prompt(answer: str, message: str) -> str:
        """The few-shot classification prompt.

        A method rather than an inline string so evaluate/eval_correction.py
        scores what ships. Few-shot because the bare instruction did not work:
        asked in plain prose this model scored 6/14 on a hand-built set --
        worse than answering "no" every time -- and it said yes to ordinary
        follow-ups like "can you also show how to change the port?", which is
        the expensive direction. The examples pin down the distinction it kept
        missing: a bad outcome versus a request for more.
        """
        return (
            "Decide whether the user is reporting that the assistant's answer "
            "was wrong or did not work.\n\n"
            "Say yes when the user reports a bad outcome: an error, a "
            "different result than expected, nothing found, something missing, "
            "or doubt that the answer is right.\n"
            "Say no when the user asks for more, asks a new or related "
            "question, or is satisfied.\n\n"
            "Examples:\n\n"
            'Answer: "Run `apt install foo`."\n'
            'Message: "that gave me: package not found"\n'
            "yes\n\n"
            'Answer: "Run `apt install foo`."\n'
            'Message: "can you also show the uninstall command?"\n'
            "no\n\n"
            'Answer: "The file is at /etc/foo.conf."\n'
            'Message: "there is no such file on my system"\n'
            "yes\n\n"
            'Answer: "The file is at /etc/foo.conf."\n'
            'Message: "does this apply to RHEL too?"\n'
            "no\n\n"
            'Answer: "Set X to 5."\n'
            'Message: "are you sure? the docs say 10"\n'
            "yes\n\n"
            'Answer: "Set X to 5."\n'
            'Message: "thanks, that worked"\n'
            "no\n\n"
            # "Now do X" is an instruction to carry on, not a complaint, and
            # was the last case the model still got wrong.
            'Answer: "Set X to 5."\n'
            'Message: "now do the same for the other file"\n'
            "no\n\n"
            "Now classify this one.\n\n"
            f'Answer: "{answer}"\n'
            f'Message: "{message}"'
        )

    @staticmethod
    def _parse(content: str) -> Optional[bool]:
        text = (content or "").strip().lower()
        if not text:
            return None
        # Match on the first word only. "no, the user is asking a follow-up"
        # starts with the answer; scanning the whole string for "yes" would
        # find it in an explanation that concluded the opposite.
        first = text.split()[0].strip('."\',:*')
        if first.startswith("yes"):
            return True
        if first.startswith("no"):
            return False
        return None
