"""The agent-task logic: tool-call parsing for every format, word splicing, the mocked world, the form
checks and a full episode. No model is loaded."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import pytest

from fragile_tokens.agent import (SLOT, TASK_BY_NAME, CallFormat, Episode, Match, World, assistant_message, call_id,
                                  detect_format, find_word, format_ok, parse_tool_calls, render, visible_text)

WORD_IDS = [7001, 7002]


class TemplateTokenizer:
    """Characters are token ids (ord + 10000); the chat template is a fixed Python function whose
    ``marker`` decides which tool-call format it advertises."""

    def __init__(self, marker: str = "<tool_call>", renders_tools: bool = True, json_content: bool = False) -> None:
        self.chat_template = f"template {marker}"
        self.renders_tools = renders_tools
        self.json_content = json_content

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> str:
        out = []
        if kwargs.get("tools") and self.renders_tools:
            out.append("TOOLS " + " ".join(t["function"]["name"] for t in kwargs["tools"]))
        for m in conversation:
            content = json.dumps(m.get("content", "")) if self.json_content else str(m.get("content", ""))
            out.append(f"<{m['role']}>{content}")
            for c in m.get("tool_calls", []) or []:
                out.append(f"<call {c['id']} {c['function']['name']} {json.dumps(c['function']['arguments'])}>")
        if kwargs.get("add_generation_prompt"):
            out.append("<assistant>")
        return "\n".join(out)

    def __call__(self, text: str, add_special_tokens: bool = True) -> Mapping[str, Any]:
        assert SLOT not in text, "the slot must never reach the tokenizer"
        return {"input_ids": [ord(ch) + 10000 for ch in text]}

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(i - 10000) if i >= 10000 else f"[{i}]" for i in token_ids)


# ---- formats --------------------------------------------------------------------------------------
@pytest.mark.parametrize("marker,expected", [
    ("[TOOL_CALLS]", CallFormat.MISTRAL), ("<|tool_call|>", CallFormat.GRANITE),
    ("<tool_call><function=", CallFormat.QWEN_XML), ("<tool_call>", CallFormat.HERMES)])
def test_detect_format_by_template_marker(marker: str, expected: CallFormat) -> None:
    assert detect_format(TemplateTokenizer(marker)) is expected


def test_template_that_drops_tools_is_text_protocol() -> None:
    assert detect_format(TemplateTokenizer(renders_tools=False)) is CallFormat.TEXT


def test_parse_hermes() -> None:
    raw = 'Let me search.\n<tool_call>\n{"name": "search", "arguments": {"query": "Translate docs"}}\n</tool_call><|im_end|>'
    calls = parse_tool_calls(raw, CallFormat.HERMES, 2)
    assert [(c.name, c.arguments) for c in calls] == [("search", {"query": "Translate docs"})]
    assert calls[0].call_id == call_id(2, 0)


def test_parse_qwen_xml() -> None:
    raw = "<tool_call>\n<function=search>\n<parameter=query>\nTranslate\n</parameter>\n</function>\n</tool_call>"
    assert [(c.name, c.arguments) for c in parse_tool_calls(raw, CallFormat.QWEN_XML)] == [("search", {"query": "Translate"})]


def test_parse_mistral_list_after_special_marker() -> None:
    raw = '[TOOL_CALLS] [{"name": "open_page", "arguments": {"url": "https://x.org"}}, ' \
          '{"name": "search", "arguments": {"query": "q"}}]</s>'
    assert [c.name for c in parse_tool_calls(raw, CallFormat.MISTRAL)] == ["open_page", "search"]


def test_parse_granite() -> None:
    raw = '<|tool_call|>[{"name": "search", "arguments": {"query": "Translate library"}}]'
    assert parse_tool_calls(raw, CallFormat.GRANITE)[0].arguments == {"query": "Translate library"}


def test_parse_text_protocol() -> None:
    raw = 'I will look it up.\nCALL search("Translate library")\n'
    assert [(c.name, c.arguments) for c in parse_tool_calls(raw, CallFormat.TEXT)] == [("search", {"query": "Translate library"})]


def test_arguments_given_as_a_json_string() -> None:
    raw = '<tool_call>{"name": "search", "arguments": "{\\"query\\": \\"q\\"}"}</tool_call>'
    assert parse_tool_calls(raw, CallFormat.HERMES)[0].arguments == {"query": "q"}


def test_calls_inside_reasoning_are_ignored() -> None:
    raw = ('<think>maybe {"name": "search", "arguments": {"query": "draft"}}</think>'
           '<tool_call>{"name": "search", "arguments": {"query": "real"}}</tool_call>')
    assert [c.arguments["query"] for c in parse_tool_calls(raw, CallFormat.HERMES)] == ["real"]


def test_prose_without_calls() -> None:
    assert parse_tool_calls("The answer is yes.", CallFormat.HERMES) == []


def test_call_ids_are_nine_alphanumerics() -> None:
    for r in range(12):
        for i in range(4):
            cid = call_id(r, i)
            assert len(cid) == 9 and cid.isalnum()


# ---- splicing -------------------------------------------------------------------------------------
def test_render_splices_word_ids_at_every_slot() -> None:
    tok = TemplateTokenizer()
    ep = Episode.start(TASK_BY_NAME["explain_term"], "Translate", "fragile", "Translate", WORD_IDS, CallFormat.HERMES, False)
    ids = render(tok, ep.messages, ep.fmt, ep.thinking, ep.word_ids)
    assert ids.count(7001) == 1 and ids.count(7002) == 1
    assert ids[ids.index(7001) + 1] == 7002
    text = tok.decode(ids)
    assert "TOOLS search open_page run_python" in text
    assert "the term[7001][7002] with" in text  # the token carries its own leading space


def test_render_handles_json_escaped_slots() -> None:
    tok = TemplateTokenizer(json_content=True)
    ep = Episode.start(TASK_BY_NAME["explain_term"], "Translate", "fragile", "Translate", WORD_IDS, CallFormat.MISTRAL, False)
    ids = render(tok, ep.messages, ep.fmt, ep.thinking, ep.word_ids)
    assert 7001 in ids and "\\u0000" not in tok.decode(ids)


def test_text_protocol_renders_without_tools() -> None:
    tok = TemplateTokenizer(renders_tools=False)
    ep = Episode.start(TASK_BY_NAME["adoption_decision"], "due", "fragile", "due", [42], CallFormat.TEXT, False)
    text = tok.decode(render(tok, ep.messages, ep.fmt, False, ep.word_ids))
    assert "CALL search" in text and "TOOLS" not in text


# ---- the world and the measures -------------------------------------------------------------------
def test_find_word_distinguishes_exact_case_and_absent() -> None:
    assert find_word("docs for Translate", "Translate") is Match.EXACT
    assert find_word("docs for translate", "Translate") is Match.CASE
    assert find_word("docs for translation", "Translate") is Match.NONE
    assert find_word("username field", "User") is Match.NONE


def test_world_matches_case_insensitively_by_default() -> None:
    w = World("Translate")
    assert "official documentation" in w.search("translate library")
    assert w.calls[-1].match is Match.CASE and w.calls[-1].answered
    strict = World("Translate", case_sensitive=True)
    assert strict.search("translate library").startswith("No results")
    assert strict.misses == 1 and strict.hits == 0


def test_world_pages_and_misses() -> None:
    w = World("due")
    assert w.open_page("https://docs.example.org/lib/index/").startswith("#")
    assert w.open_page("https://nowhere.example") == "404 Not Found"
    assert w.search("something unrelated").startswith("No results")
    assert (w.hits, w.misses) == (1, 2)


def test_visible_text_strips_reasoning_markup_and_specials() -> None:
    raw = "<think>hidden</think>FINAL: Adopt it.<tool_call>{}</tool_call><|im_end|>"
    assert visible_text(raw, ["<|im_end|>"]) == "Adopt it."


@pytest.mark.parametrize("task,good,bad", [
    ("changelog_check", "Yes, 2.4.0 changed the default log level.", "The release changed things."),
    ("adoption_decision", "Trial: it is small. It has an open issue.", "It seems fine."),
    ("background_note", "A logging library (https://docs.example.org/lib/index).", "A logging library."),
    ("explain_term", "It means X. For example, \"we used it\".", "It means X."),
])
def test_format_checks(task: str, good: str, bad: str) -> None:
    assert format_ok(task, good) and not format_ok(task, bad)


def test_episode_grounded_run() -> None:
    ep = Episode.start(TASK_BY_NAME["changelog_check"], "Translate", "fragile", "Translate", WORD_IDS, CallFormat.HERMES, False)
    ep.advance('<tool_call>{"name": "search", "arguments": {"query": "Translate release notes"}}</tool_call><|im_end|>',
               ["<|im_end|>"])
    assert not ep.done
    assert ep.messages[-2]["tool_calls"][0]["function"]["name"] == "search"
    assert ep.messages[-1]["role"] == "tool" and SLOT in ep.messages[-1]["content"]
    ep.advance("Yes: Translate 2.4.0 changed the default log level from INFO to WARNING.<|im_end|>", ["<|im_end|>"])
    r = ep.record()
    assert ep.done and r["finished"] and r["any_query_exact"] and r["final_about_word"]
    assert r["format_ok"] and r["claims_level_change"] and not r["fabricated"]


def test_episode_substitution_and_fabrication() -> None:
    ep = Episode.start(TASK_BY_NAME["research_library"], "Translate", "fragile", "Translate", WORD_IDS, CallFormat.HERMES, False)
    ep.advance('<tool_call>{"name": "search", "arguments": {"query": "Transformer library"}}</tool_call>')
    ep.advance("Transformer is a library. Its latest version is 2.4.1. Issue 218 is open.")
    r = ep.record()
    assert r["searched_something_else"] == "Transformer library"
    assert not r["any_query_word"] and not r["final_about_word"] and r["fabricated"]


def test_round_cap_is_recorded() -> None:
    ep = Episode.start(TASK_BY_NAME["research_library"], "due", "fragile", "due", [42], CallFormat.HERMES, False)
    ep.advance('<tool_call>{"name": "search", "arguments": {"query": "due"}}</tool_call>')
    ep.close_at_cap()
    r = ep.record()
    assert r["hit_round_cap"] and not r["finished"]


def test_text_protocol_assistant_turn_keeps_call_lines() -> None:
    calls = parse_tool_calls('CALL search("due library")', CallFormat.TEXT)
    msg = assistant_message("", calls, CallFormat.TEXT)
    assert msg["content"] == 'CALL search("due library")' and "tool_calls" not in msg
