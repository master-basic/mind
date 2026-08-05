"""Which message counts as "what the user asked".

Three places need that answer and they must all give the same one: recall
embeds it as the query, build_messages splices the recalled blocks into it,
and get_reading_content decides whether it holds pasted source material.
They did not agree -- get_last_user_message and get_reading_content took the
last role=user message flat, while _newest_user_index skipped
<tool_response>-wrapped turns as the chat template does.
"""


def body(*messages):
    return {"messages": list(messages)}


def user(text):
    return {"role": "user", "content": text}


def tool_response(text):
    # How an agentic client feeds a tool result back: role=user, with the
    # payload wrapped. The chat template does not treat it as a query.
    return {"role": "user", "content": f"<tool_response>{text}</tool_response>"}


class TestGetLastUserMessage:
    def test_plain_conversation(self, pipeline):
        b = body(user("first"), {"role": "assistant", "content": "hi"},
                 user("second"))
        assert pipeline.get_last_user_message(b) == "second"

    def test_skips_a_trailing_tool_response(self, pipeline):
        b = body(user("what is the weather in Baku?"),
                 {"role": "assistant", "content": None},
                 tool_response('{"temp": 31}'))
        assert pipeline.get_last_user_message(b) == "what is the weather in Baku?"

    def test_agrees_with_the_injection_anchor(self, pipeline):
        # The actual bug: recall embedded the tool response while the blocks it
        # found were spliced into the message before it. Query and target were
        # different texts.
        msgs = [user("build the OCR endpoint"),
                {"role": "assistant", "content": None},
                tool_response("x" * 400)]
        idx = pipeline._newest_user_index(msgs)
        assert pipeline.get_last_user_message(body(*msgs)) == msgs[idx]["content"]

    def test_skips_several_consecutive_tool_responses(self, pipeline):
        b = body(user("the question"), tool_response("a"), tool_response("b"))
        assert pipeline.get_last_user_message(b) == "the question"

    def test_content_parts_are_joined(self, pipeline):
        b = body({"role": "user", "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "..."}},
            {"type": "text", "text": "world"},
        ]})
        assert pipeline.get_last_user_message(b) == "hello world"

    def test_no_user_message_at_all(self, pipeline):
        assert pipeline.get_last_user_message(
            body({"role": "system", "content": "you are a bot"})) == ""
        assert pipeline.get_last_user_message(body()) == ""

    def test_a_conversation_of_only_tool_responses(self, pipeline):
        # _newest_user_index finds nothing, so there is no query. Better an
        # empty recall than one keyed on a tool payload.
        assert pipeline.get_last_user_message(body(tool_response("a"))) == ""


class TestGetReadingContent:
    def test_long_paste_is_reading_material(self, pipeline):
        doc = "word " * 300
        assert pipeline.get_reading_content(body(user(doc))).strip() == doc.strip()

    def test_short_message_is_not(self, pipeline):
        assert pipeline.get_reading_content(body(user("hello"))) == ""

    def test_a_long_tool_response_is_not_archived_as_a_paste(self, pipeline):
        # Tool output is not the user handing over a document, and archiving it
        # every turn is how the vector index fills with near-duplicates.
        b = body(user("short question"), tool_response("word " * 300))
        assert pipeline.get_reading_content(b) == ""

    def test_only_the_newest_turn_is_scanned(self, pipeline):
        # An earlier paste re-archived on every subsequent turn is one document
        # becoming a block per turn thereafter.
        b = body(user("word " * 300), {"role": "assistant", "content": "ok"},
                 user("thanks"))
        assert pipeline.get_reading_content(b) == ""

    def test_the_system_prompt_is_never_reading_material(self, pipeline):
        # Agent CLIs ship multi-thousand-word system prompts.
        b = body({"role": "system", "content": "prompt " * 3000},
                 user("hi"))
        assert pipeline.get_reading_content(b) == ""
