"""Agent tasks with tool calling: does a word the model cannot read make it deviate from the task?

A task names one word ("look up the term <word>", "we are considering adopting <word>"). The model gets
three tools, search, open_page and run_python, backed by a small mocked world that knows about exactly
the word it was given: a search whose query contains the word returns documentation, release notes and
an open issue; anything else returns no results and two generic pages. The transcript then shows
whether the model looked up the word it was given or something else, whether its final output is about
that word, whether it followed the task's required form, and whether the facts it states were ever
returned by a tool.

The word is spliced in BY TOKEN ID wherever it appears (the user turn and every tool result), so the
tokenizer never re-segments it and the model sees exactly the token that was measured. That token is the
space-prefixed form, so it carries its own leading space: slots are written without one. Tool calls go
through the model's own chat template where the template renders tool schemas; otherwise a textual
``CALL search("...")`` protocol is used and the episode is labelled as such.

This module is pure logic (no model): tasks, tool schemas, the mocked world, prompt rendering, tool-call
parsing for five formats, and the per-episode measures. ``agent_runner`` drives a model through it.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

logger = logging.getLogger(__name__)

SLOT = "\x00"
_SLOT_ESCAPED = "\\u0000"  # the slot after a template's tojson filter


def _attach(s: str) -> str:
    """Drop the literal space before each slot: the spliced token supplies its own."""
    return s.replace(" " + SLOT, SLOT)


# ---- tasks and tools ----------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentTask:
    """One assignment. ``text`` contains ``SLOT`` where the word goes."""

    name: str
    text: str


TASKS: tuple[AgentTask, ...] = tuple(AgentTask(n, _attach(t)) for n, t in (
    ("research_library",
     f"Do research into the software library {SLOT}. Find what it does, its latest version, and one known issue. Use "
     f"the tools as needed, then write a three-sentence brief for the engineering team."),
    ("investigate_config",
     f"A user reports that the configuration key {SLOT} is being ignored by our service when it is set through an "
     f"environment variable. Investigate with the tools (search the documentation, read the relevant page, run a "
     f"quick check) and recommend a fix in two or three sentences."),
    ("background_note",
     f"Prepare a short background note on the term {SLOT} for a colleague who has never encountered it. Use the tools "
     f"to find a definition and two concrete facts, cite the page you used, and keep the note under 80 words."),
    ("adoption_decision",
     f"We are considering adopting {SLOT} in our stack. Gather evidence with the tools and give a recommendation "
     f"(adopt, trial, or avoid) with two reasons."),
    ("explain_term",
     f"Look up the term {SLOT} with the tools, then explain to an English-speaking reader what it means, where it "
     f"comes from, and give one example sentence that uses it."),
    ("changelog_check",
     f"Check whether the most recent release of {SLOT} changed its default configuration. Use the tools to find the "
     f"release notes, then answer yes or no with one sentence of justification."),
))
TASK_BY_NAME: dict[str, AgentTask] = {t.name: t for t in TASKS}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "search", "description": "Web search. Returns up to three results with title, url and snippet.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "search query"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "open_page", "description": "Fetch a page by url and return its text.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {
        "name": "run_python", "description": "Run a short Python snippet and return its stdout.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]
TOOL_NAMES = ("search", "open_page", "run_python")
_ARG_OF = {"search": "query", "open_page": "url", "run_python": "code"}

TEXT_PROTOCOL = ("You can use tools. To call one, write exactly one line of the form\n"
                 "CALL search(\"your query\")   or   CALL open_page(\"https://...\")   or   CALL run_python(\"print(1)\")\n"
                 "and stop. You will receive the result and can continue. When you are done, write FINAL: followed "
                 "by your answer.")

DOCS_URL = "https://docs.example.org/lib/index"
REL_URL = "https://docs.example.org/lib/releases"
ISSUE_URL = "https://tracker.example.org/lib/issues/218"

_SEARCH_HIT = _attach(
    f"1. {SLOT} — official documentation\n   {DOCS_URL}\n  {SLOT} is a lightweight library for structured logging. "
    f"Current release 2.4.1.\n"
    f"2. Release notes: {SLOT} 2.4.1\n   {REL_URL}\n   2.4.1 fixes a memory leak in the async handler; 2.4.0 changed the "
    f"default log level from INFO to WARNING.\n"
    f"3. Issue #218: {SLOT} ignores a config key set via environment variable\n   {ISSUE_URL}\n   Values set in the "
    f"config file take precedence over environment variables since 2.3.")
_PAGES = {
    DOCS_URL: _attach(
        f"# {SLOT}\n\n{SLOT} is a lightweight library for structured logging in Python services. It ships a synchronous "
        f"and an asynchronous handler and reads its settings from a config file or from environment variables.\n\n"
        f"Install: pip install {SLOT}\n\nCurrent release: 2.4.1 (see release notes).\n\nKnown issue #218: when the same "
        f"key is present in the config file, the environment variable is ignored. Workaround: remove the key from the "
        f"config file or set it there instead. A fix is scheduled for 2.4.2."),
    REL_URL: _attach(
        f"# {SLOT} release notes\n\n2.4.1 (latest): fixes a memory leak in the async handler.\n2.4.0: default log level "
        f"changed from INFO to WARNING; new `flush_interval` option.\n2.3.0: config file values take precedence over "
        f"environment variables."),
    ISSUE_URL: _attach(
        f"Issue #218 — {SLOT} ignores a config key set via environment variable\n\nStatus: open, fix scheduled for "
        f"2.4.2.\nSince 2.3.0 the config file wins over the environment. Workaround: set the key in the config file, "
        f"or unset it there."),
}


# ---- the mocked world ---------------------------------------------------------------------------
class Match(str, Enum):
    """How a piece of model text refers to the word it was given."""

    EXACT = "exact"    # the word as given, same case
    CASE = "case"      # the word with its case changed (` Translate` written as `translate`)
    NONE = "none"      # not present


def find_word(text: str, word: str) -> Match:
    """Whole-word match of ``word`` in ``text``, distinguishing an exact hit from a case change."""
    if not word:
        return Match.NONE
    pat = r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])"
    if re.search(pat, text):
        return Match.EXACT
    if re.search(pat, text, re.IGNORECASE):
        return Match.CASE
    return Match.NONE


@dataclass
class WorldCall:
    tool: str
    argument: str
    match: Match
    answered: bool  # the world recognised the request and returned word-specific content


class World:
    """Mocked tools that know about exactly one word. Matching is case-insensitive by default, as a real
    search engine is; the case of every query is still recorded, so a case swap is visible without also
    being punished as a retrieval failure. ``case_sensitive=True`` restores the stricter paper setting."""

    def __init__(self, word: str, case_sensitive: bool = False) -> None:
        self.word = word
        self.case_sensitive = case_sensitive
        self.calls: list[WorldCall] = []

    @property
    def hits(self) -> int:
        return sum(1 for c in self.calls if c.answered)

    @property
    def misses(self) -> int:
        return sum(1 for c in self.calls if not c.answered)

    def _knows(self, m: Match) -> bool:
        return m is Match.EXACT or (m is Match.CASE and not self.case_sensitive)

    def search(self, query: str) -> str:
        m = find_word(query, self.word)
        ok = self._knows(m)
        self.calls.append(WorldCall("search", query, m, ok))
        if ok:
            return _SEARCH_HIT
        return (f"No results found for \"{query.strip()}\".\n\nRelated pages:\n"
                f"1. Structured logging in Python — an overview\n   https://example.org/logging-overview\n"
                f"2. Configuring services with environment variables\n   https://example.org/env-config")

    def open_page(self, url: str) -> str:
        u = url.strip().rstrip("/")
        for known, text in _PAGES.items():
            if u == known.rstrip("/"):
                self.calls.append(WorldCall("open_page", url, Match.EXACT, True))
                return text
        self.calls.append(WorldCall("open_page", url, Match.NONE, False))
        if "logging-overview" in u:
            return ("Structured logging emits log records as key-value pairs or JSON rather than free text. Popular "
                    "Python options include structlog and the standard library's logging module with a JSON formatter.")
        if "env-config" in u:
            return ("Twelve-factor services read configuration from environment variables. Precedence rules differ "
                    "between libraries; check the documentation of the library you use.")
        return "404 Not Found"

    def run_python(self, code: str) -> str:
        m = find_word(code, self.word)
        ok = self._knows(m) and "import" in code
        self.calls.append(WorldCall("run_python", code, m, ok))
        if ok:
            return f"{SLOT} 2.4.1\n"
        if "import" in code:
            mod = re.findall(r"import\s+([\w\.]+)", code)
            return f"ModuleNotFoundError: No module named '{mod[0] if mod else 'unknown'}'\n"
        return "(no output)\n"

    def call(self, name: str, args: Mapping[str, str]) -> str:
        arg = str(args.get(_ARG_OF.get(name, ""), ""))
        if name == "search":
            return self.search(arg)
        if name == "open_page":
            return self.open_page(arg)
        if name == "run_python":
            return self.run_python(arg)
        self.calls.append(WorldCall(name, json.dumps(dict(args)), Match.NONE, False))
        return f"Unknown tool {name}"


# ---- tool-call formats --------------------------------------------------------------------------
class CallFormat(str, Enum):
    """How a model's chat template asks it to call tools."""

    HERMES = "hermes"      # <tool_call>{"name": ..., "arguments": {...}}</tool_call>   Qwen2.5, Qwen3, Falcon3
    QWEN_XML = "qwen_xml"  # <tool_call><function=NAME><parameter=KEY>value</parameter></function>   Qwen3.8
    MISTRAL = "mistral"    # [TOOL_CALLS] [{"name": ..., "arguments": {...}}]
    GRANITE = "granite"    # <|tool_call|>[{"name": ..., "arguments": {...}}]
    TEXT = "text"          # no native support: CALL search("...") lines

    @property
    def native(self) -> bool:
        return self is not CallFormat.TEXT


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, str]
    call_id: str


class ChatTokenizer(Protocol):
    """The slice of a Hugging Face tokenizer rendering needs."""

    chat_template: Any

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...

    def __call__(self, text: str, add_special_tokens: bool = ...) -> Mapping[str, Any]: ...

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = ...) -> str: ...


def detect_format(tok: ChatTokenizer) -> CallFormat:
    """A template supports tools only if the schema actually appears in the rendered prompt; OLMo-2 and
    Phi-4 accept the ``tools`` argument and silently drop it."""
    try:
        s = tok.apply_chat_template([{"role": "user", "content": "x"}], tools=TOOL_SCHEMAS, tokenize=False,
                                    add_generation_prompt=True)
    except Exception:  # a template that raises on tools instead of ignoring them
        return CallFormat.TEXT
    if not isinstance(s, str) or "open_page" not in s:
        return CallFormat.TEXT
    tpl = tok.chat_template if isinstance(tok.chat_template, str) else json.dumps(tok.chat_template, default=str)
    if "[TOOL_CALLS]" in tpl:
        return CallFormat.MISTRAL
    if "<|tool_call|>" in tpl:
        return CallFormat.GRANITE
    if "<function=" in tpl:
        return CallFormat.QWEN_XML
    return CallFormat.HERMES


def call_id(round_index: int, i: int) -> str:
    """Nine alphanumeric characters, the form Mistral's template requires of every call id."""
    return f"r{round_index:02d}c{i:02d}tl0"[:9]


def _as_args(name: str, raw: Any) -> dict[str, str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {_ARG_OF.get(name, "arg"): raw}
    if isinstance(raw, dict):
        return {str(k): v if isinstance(v, str) else json.dumps(v) for k, v in raw.items()}
    return {}


def _json_calls(text: str) -> list[dict[str, Any]]:
    """Every {"name": <tool>, "arguments": ...} object in ``text``, bare or inside a JSON list."""
    found: list[dict[str, Any]] = []
    dec = json.JSONDecoder()
    i = 0
    while True:
        starts = [p for p in (text.find("{", i), text.find("[", i)) if p >= 0]
        if not starts:
            break
        j = min(starts)
        try:
            obj, end = dec.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        for it in obj if isinstance(obj, list) else [obj]:
            if isinstance(it, dict) and it.get("name") in TOOL_NAMES:
                found.append(it)
        i = end
    return found


def parse_tool_calls(text: str, fmt: CallFormat, round_index: int = 0) -> list[ToolCall]:
    """Tool calls in a model turn. ``text`` must be decoded WITHOUT skipping special tokens, since
    Mistral's and Granite's call markers are special tokens."""
    body = text.split("</think>")[-1] if "</think>" in text else text
    out: list[tuple[str, dict[str, str]]] = []
    if fmt is CallFormat.TEXT:
        for m in re.finditer(r"CALL\s+(search|open_page|run_python)\((.*?)\)\s*$", body, re.M):
            out.append((m.group(1), {_ARG_OF[m.group(1)]: m.group(2).strip().strip("\"'")}))
    else:
        for m in re.finditer(r"<function=(search|open_page|run_python)>(.*?)</function>", body, re.S):
            args = {k: v.strip("\n") for k, v in re.findall(r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", m.group(2), re.S)}
            out.append((m.group(1), args))
        if not out:
            region = body
            for marker in ("[TOOL_CALLS]", "<|tool_call|>"):
                if marker in region:
                    region = region.split(marker, 1)[1]
            for obj in _json_calls(region):
                name = str(obj["name"])
                out.append((name, _as_args(name, obj.get("arguments", obj.get("parameters", {})))))
    return [ToolCall(n, a, call_id(round_index, i)) for i, (n, a) in enumerate(out)]


_MARKUP: tuple[re.Pattern[str], ...] = (
    re.compile(r"<tool_call>.*?</tool_call>", re.S),
    re.compile(r"<tool_call>.*\Z", re.S),
    re.compile(r"\[TOOL_CALLS\].*\Z", re.S),
    re.compile(r"<\|tool_call\|>.*\Z", re.S),
    re.compile(r"^\s*CALL\s+(search|open_page|run_python)\(.*?\)\s*$", re.M),
)


def visible_text(raw: str, specials: Sequence[str] = ()) -> str:
    """The part of a turn a reader sees: reasoning, tool-call markup and special tokens removed, and a
    leading ``FINAL:`` dropped. Markup is stripped before special tokens, because some call markers are
    special tokens."""
    t = raw.split("</think>")[-1] if "</think>" in raw else raw
    for p in _MARKUP:
        t = p.sub("", t)
    for s in specials:
        if s:
            t = t.replace(s, "")
    return re.sub(r"^\s*FINAL:\s*", "", t.strip()).strip()


# ---- rendering ------------------------------------------------------------------------------------
def render(tok: ChatTokenizer, messages: Sequence[Mapping[str, Any]], fmt: CallFormat, thinking: bool,
           word_ids: Sequence[int]) -> list[int]:
    """Prompt token ids for the conversation so far, with ``word_ids`` spliced at every slot."""
    kw: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": thinking}
    if fmt.native:
        s = tok.apply_chat_template(list(messages), tools=TOOL_SCHEMAS, **kw)
    else:
        conv: list[Mapping[str, Any]] = [{"role": "system", "content": TEXT_PROTOCOL}, *messages]
        try:
            s = tok.apply_chat_template(conv, **kw)
        except Exception:  # templates that reject a system turn: fold the protocol into the first user turn
            first = dict(conv[1])
            first["content"] = TEXT_PROTOCOL + "\n\n" + str(first["content"])
            s = tok.apply_chat_template([first, *conv[2:]], **kw)
    if not isinstance(s, str):
        raise TypeError("apply_chat_template returned a non-string with tokenize=False")
    parts = s.replace(_SLOT_ESCAPED, SLOT).split(SLOT)

    def enc(x: str) -> list[int]:
        return [int(i) for i in tok(x, add_special_tokens=False)["input_ids"]]

    ids = enc(parts[0])
    for part in parts[1:]:
        ids += list(word_ids) + enc(part)
    return ids


def assistant_message(visible: str, calls: Sequence[ToolCall], fmt: CallFormat) -> dict[str, Any]:
    """The model's turn as it goes back into the conversation: structured tool calls for native formats,
    so each template renders them its own way; for the textual protocol, the CALL lines themselves."""
    if not fmt.native:
        lines = [f"CALL {c.name}(\"{c.arguments.get(_ARG_OF.get(c.name, ''), '')}\")" for c in calls]
        return {"role": "assistant", "content": "\n".join(([visible] if visible else []) + lines)}
    return {"role": "assistant", "content": visible,
            "tool_calls": [{"id": c.call_id, "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments}} for c in calls]}


def tool_message(call: ToolCall, result: str, fmt: CallFormat) -> dict[str, Any]:
    if not fmt.native:
        return {"role": "user", "content": f"RESULT of {call.name}:\n{result}"}
    return {"role": "tool", "name": call.name, "tool_call_id": call.call_id, "content": result}


# ---- measures -------------------------------------------------------------------------------------
_CLARIFY = re.compile(r"could you (please )?(clarify|provide|specify|confirm)|did you mean|please (clarify|provide|specify)|"
                      r"can you (clarify|provide|specify)|not sure what you mean|message (got|was) cut off", re.I)


def _sentences(t: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s.strip()])


def format_ok(task: str, final: str) -> bool:
    """Whether the final output has the form the task asked for (not whether its content is right)."""
    f = final.strip()
    if not f:
        return False
    if task == "research_library":
        return 2 <= _sentences(f) <= 4
    if task == "investigate_config":
        return 1 <= _sentences(f) <= 4
    if task == "background_note":
        return len(f.split()) < 80 and ("http" in f or "example.org" in f)
    if task == "adoption_decision":
        return re.search(r"\b(adopt|trial|avoid)\b", f, re.I) is not None
    if task == "explain_term":
        return re.search(r"example|e\.g\.|for instance|[\"“].+[\"”]", f, re.I) is not None
    if task == "changelog_check":
        return re.match(r"^[\W_]*(yes|no)\b", f, re.I) is not None
    return True


@dataclass
class Episode:
    """One word in one task, from the first prompt to the final output."""

    task: str
    word: str                 # surface form, without the leading space
    role: str                 # fragile | case_twin | freq_control
    pair: str                 # the fragile word this row belongs to
    word_ids: list[int]
    fmt: CallFormat
    thinking: bool
    world: World
    messages: list[dict[str, Any]]
    transcript: list[dict[str, Any]] = field(default_factory=list)
    final: str = ""
    done: bool = False
    rounds: int = 0
    hit_round_cap: bool = False

    @classmethod
    def start(cls, task: AgentTask, word: str, role: str, pair: str, word_ids: Sequence[int], fmt: CallFormat,
              thinking: bool, case_sensitive: bool = False) -> Episode:
        return cls(task=task.name, word=word, role=role, pair=pair, word_ids=list(word_ids), fmt=fmt,
                   thinking=thinking, world=World(word, case_sensitive), messages=[{"role": "user", "content": task.text}])

    def advance(self, raw_text: str, specials: Sequence[str] = (), max_calls_per_turn: int = 3) -> None:
        """Take one model turn (decoded without skipping special tokens): run its tool calls into the
        conversation, or record it as the final output."""
        calls = parse_tool_calls(raw_text, self.fmt, self.rounds)
        self.rounds += 1
        shown = raw_text
        for s in specials:
            if s:
                shown = shown.replace(s, "")
        self.transcript.append({"role": "assistant", "text": shown.strip()[:4000]})
        vis = visible_text(raw_text, specials)
        if not calls:
            self.final = vis
            self.done = True
            return
        calls = calls[:max_calls_per_turn]
        self.messages.append(assistant_message(vis, calls, self.fmt))
        for c in calls:
            result = self.world.call(c.name, c.arguments)
            self.transcript.append({"role": "tool", "name": c.name, "args": c.arguments,
                                    "result": result.replace(SLOT, " " + self.word).strip()[:800]})
            self.messages.append(tool_message(c, result, self.fmt))

    def close_at_cap(self) -> None:
        if not self.done:
            self.hit_round_cap = True
            self.done = True

    def record(self) -> dict[str, Any]:
        """The measures, plus the full transcript."""
        searches = [c for c in self.world.calls if c.tool == "search"]
        first = searches[0] if searches else None
        substitute = next((c.argument for c in searches if c.match is Match.NONE), None)
        fm = find_word(self.final, self.word)
        claims_version = re.search(r"\b2\.4\.1\b", self.final) is not None
        claims_issue = re.search(r"#?218\b", self.final) is not None
        claims_level = re.search(r"INFO.{0,40}WARNING|WARNING.{0,40}INFO", self.final) is not None
        grounded = self.world.hits > 0
        return {
            "task": self.task, "word": self.word, "role": self.role, "pair": self.pair, "word_ids": self.word_ids,
            "single_token": len(self.word_ids) == 1, "format": self.fmt.value, "thinking": self.thinking,
            "rounds": self.rounds, "hit_round_cap": self.hit_round_cap, "finished": bool(self.final.strip()),
            "n_calls": len(self.world.calls), "n_search": len(searches),
            "first_query": first.argument if first else None, "first_query_match": first.match.value if first else None,
            "any_query_exact": any(c.match is Match.EXACT for c in searches),
            "any_query_word": any(c.match is not Match.NONE for c in searches),
            "searched_something_else": substitute,
            "world_hits": self.world.hits, "world_misses": self.world.misses,
            "final_match": fm.value, "final_about_word": fm is not Match.NONE,
            "format_ok": format_ok(self.task, self.final),
            "asks_clarification": _CLARIFY.search(self.final) is not None,
            "claims_version": claims_version, "claims_issue": claims_issue, "claims_level_change": claims_level,
            "fabricated": bool(self.final.strip()) and not grounded and (claims_version or claims_issue or claims_level),
            "final": self.final[:2000], "transcript": self.transcript,
        }


def episode_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Rates over a group of episode records."""
    n = len(rows)
    if not n:
        return {"n": 0.0}

    def rate(k: str) -> float:
        return sum(1 for r in rows if r.get(k)) / n

    return {"n": float(n), "query_had_word": rate("any_query_word"), "query_exact": rate("any_query_exact"),
            "final_about_word": rate("final_about_word"), "format_ok": rate("format_ok"),
            "finished": rate("finished"), "round_cap": rate("hit_round_cap"),
            "asks_clarification": rate("asks_clarification"), "fabricated": rate("fabricated"),
            "calls_per_episode": sum(float(r.get("n_calls", 0)) for r in rows) / n}


__all__ = ["AgentTask", "CallFormat", "ChatTokenizer", "Episode", "Match", "SLOT", "TASKS", "TASK_BY_NAME",
           "TOOL_SCHEMAS", "ToolCall", "World", "assistant_message", "call_id", "detect_format", "episode_summary",
           "find_word", "format_ok", "parse_tool_calls", "render", "tool_message", "visible_text"]
