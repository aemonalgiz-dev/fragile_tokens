"""Tokens inside agentic tasks with mocked tools.

Open-ended assignments ("research X and write a brief") whose outcome is not a restatement
of the prompt. The model gets three tools -- search, open_page, run_python -- backed by a
small mocked world that knows about the token it was actually given: a search whose query
contains the token's string returns three relevant hits; a query for anything else returns
no results plus two generic pages. So the transcript shows whether the model looked up the
token it was given or a substitute, how it coped when the world did not recognise the
substitute, whether it fabricated the facts it never retrieved, and what it wrote up.

The token is spliced BY ID at every point it appears (user turn and tool results), under
the model's own chat template with its native tool schema where the template supports
tools (Qwen), otherwise under a textual CALL protocol (OLMo). Every episode is stored.

  python -m src.cut.agent_probe --model Qwen/Qwen3-32B --pt results/fragility_qwen3_32b.pt \
      --tasks 5 --n-per-group 10 --out results/agent_probe_qwen3_32b.json
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import torch

from .loadmodel import add_model_args, load_from_args, describe
from .task_probe import groups

T = "\x00"      # where the token goes; replaced by the token id at build time

TASKS = [
    ("research_library",
     f"Do research into the software library {T}. Find what it does, its latest version, and one known issue. "
     f"Use the tools as needed, then write a three-sentence brief for the engineering team."),
    ("investigate_config",
     f"A user reports that the configuration key {T} is being ignored by our service when it is set through an "
     f"environment variable. Investigate with the tools (search the documentation, read the relevant page, run a "
     f"quick check) and recommend a fix in two or three sentences."),
    ("background_note",
     f"Prepare a short background note on the term {T} for a colleague who has never encountered it. Use the tools "
     f"to find a definition and two concrete facts, cite the page you used, and keep the note under 80 words."),
    ("adoption_decision",
     f"We are considering adopting {T} in our stack. Gather evidence with the tools and give a recommendation "
     f"(adopt, trial, or avoid) with two reasons."),
    ("explain_term",
     f"Look up the term {T} with the tools, then explain to an English-speaking reader what it means, where it "
     f"comes from, and give one example sentence that uses it."),
    ("changelog_check",
     f"Check whether the most recent release of {T} changed its default configuration. Use the tools to find the "
     f"release notes, then answer yes or no with one sentence of justification."),
]

TOOLS = [
    {"type": "function", "function": {"name": "search", "description": "Web search. Returns up to three results with title, url and snippet.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "search query"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "open_page", "description": "Fetch a page by url and return its text.",
     "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "run_python", "description": "Run a short Python snippet and return its stdout.",
     "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]

DOCS_URL, REL_URL, ISSUE_URL = "https://docs.example.org/lib/index", "https://docs.example.org/lib/releases", "https://tracker.example.org/lib/issues/218"


class World:
    """Mocked tools that know about exactly one token (the one the model was given)."""

    def __init__(self, tok, t):
        self.t = t; self.s = tok.decode([t]).strip(); self.hits = 0; self.misses = 0; self.calls = []

    def _mentions(self, text):
        return bool(self.s) and self.s in text

    def search(self, query):
        self.calls.append(("search", query))
        if self._mentions(query):
            self.hits += 1
            return (f"1. {T} — official documentation\n   {DOCS_URL}\n   {T} is a lightweight library for structured logging. Current release 2.4.1.\n"
                    f"2. Release notes: {T} 2.4.1\n   {REL_URL}\n   2.4.1 fixes a memory leak in the async handler; 2.4.0 changed the default log level from INFO to WARNING.\n"
                    f"3. Issue #218: {T} ignores a config key set via environment variable\n   {ISSUE_URL}\n   Values set in the config file take precedence over environment variables since 2.3.")
        self.misses += 1
        return (f"No results found for \"{query.strip()}\".\n\nRelated pages:\n"
                f"1. Structured logging in Python — an overview\n   https://example.org/logging-overview\n"
                f"2. Configuring services with environment variables\n   https://example.org/env-config")

    def open_page(self, url):
        self.calls.append(("open_page", url))
        u = url.strip().rstrip("/")
        if u == DOCS_URL.rstrip("/"):
            self.hits += 1
            return (f"# {T}\n\n{T} is a lightweight library for structured logging in Python services. It ships a synchronous and an "
                    f"asynchronous handler and reads its settings from a config file or from environment variables.\n\n"
                    f"Install: pip install {T}\n\nCurrent release: 2.4.1 (see release notes).\n\n"
                    f"Known issue #218: when the same key is present in the config file, the environment variable is ignored. "
                    f"Workaround: remove the key from the config file or set it there instead. A fix is scheduled for 2.4.2.")
        if u == REL_URL.rstrip("/"):
            self.hits += 1
            return (f"# {T} release notes\n\n2.4.1 (latest): fixes a memory leak in the async handler.\n"
                    f"2.4.0: default log level changed from INFO to WARNING; new `flush_interval` option.\n2.3.0: config file values take precedence over environment variables.")
        if u == ISSUE_URL.rstrip("/"):
            self.hits += 1
            return (f"Issue #218 — {T} ignores a config key set via environment variable\n\nStatus: open, fix scheduled for 2.4.2.\n"
                    f"Since 2.3.0 the config file wins over the environment. Workaround: set the key in the config file, or unset it there.")
        if "logging-overview" in u:
            return "Structured logging emits log records as key-value pairs or JSON rather than free text. Popular Python options include structlog and the standard library's logging module with a JSON formatter."
        if "env-config" in u:
            return "Twelve-factor services read configuration from environment variables. Precedence rules differ between libraries; check the documentation of the library you use."
        self.misses += 1
        return "404 Not Found"

    def run_python(self, code):
        self.calls.append(("run_python", code))
        if self._mentions(code) and "import" in code:
            self.hits += 1
            return f"{T} 2.4.1\n"
        if "import" in code:
            mod = re.findall(r"import\s+([\w\.]+)", code)
            return f"ModuleNotFoundError: No module named '{mod[0] if mod else 'unknown'}'\n"
        return "(no output)\n"

    def call(self, name, args):
        if name == "search":
            return self.search(str(args.get("query", "")))
        if name == "open_page":
            return self.open_page(str(args.get("url", "")))
        if name == "run_python":
            return self.run_python(str(args.get("code", "")))
        return f"Unknown tool {name}"


TEXT_PROTOCOL = ("You can use tools. To call one, write exactly one line of the form\n"
                 "CALL search(\"your query\")   or   CALL open_page(\"https://...\")   or   CALL run_python(\"print(1)\")\n"
                 "and stop. You will receive the result and can continue. When you are done, write FINAL: followed by your answer.")


def supports_tools(tok):
    # a template "supports" tools only if the schema actually appears in the rendered prompt;
    # OLMo-2 accepts the kwarg and silently drops it
    try:
        s = tok.apply_chat_template([{"role": "user", "content": "x"}], tools=TOOLS, tokenize=False, add_generation_prompt=True)
        return "open_page" in s and "tool_call" in s
    except Exception:
        return False


def render(tok, msgs, t, native, thinking):
    """Chat template over messages containing \\x00 placeholders; the token id is spliced at
    every placeholder so the tokenizer never sees the token as text."""
    e = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    if native:
        s = None
        for kw in ({"enable_thinking": bool(thinking)}, {}):
            try:
                s = tok.apply_chat_template(msgs, tools=TOOLS, tokenize=False, add_generation_prompt=True, **kw); break
            except Exception:
                continue
    else:
        m2 = [{"role": "system", "content": TEXT_PROTOCOL}] + msgs
        s = None
        for kw in ({"enable_thinking": bool(thinking)}, {}):
            try:
                s = tok.apply_chat_template(m2, tokenize=False, add_generation_prompt=True, **kw); break
            except Exception:
                continue
        if s is None:                                   # no chat template at all
            s = TEXT_PROTOCOL + "\n\n" + "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs) + "\n\nASSISTANT:"
    parts = s.split(T)
    ids = e(parts[0])
    for part in parts[1:]:
        ids += [t] + e(part)
    return ids


def parse_calls(text, native):
    calls = []
    if native:
        for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S):
            try:
                j = json.loads(m.group(1)); calls.append((j.get("name", ""), j.get("arguments", {}) or {}))
            except Exception:
                continue
        if not calls:
            for m in re.finditer(r"\{\s*\"name\"\s*:\s*\"(search|open_page|run_python)\"\s*,\s*\"arguments\"\s*:\s*(\{.*?\})\s*\}", text, re.S):
                try:
                    calls.append((m.group(1), json.loads(m.group(2))))
                except Exception:
                    continue
        if not calls:
            # Qwen3.5/3.8 templates ask for <tool_call><function=NAME><parameter=KEY>value</parameter>...</function></tool_call>
            for m in re.finditer(r"<function=(search|open_page|run_python)>(.*?)</function>", text, re.S):
                args = {k: v.strip("\n") for k, v in
                        re.findall(r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", m.group(2), re.S)}
                calls.append((m.group(1), args))
    else:
        for m in re.finditer(r"CALL\s+(search|open_page|run_python)\((.*?)\)\s*$", text, re.M):
            arg = m.group(2).strip().strip("\"'")
            key = {"search": "query", "open_page": "url", "run_python": "code"}[m.group(1)]
            calls.append((m.group(1), {key: arg}))
    return calls


@torch.no_grad()
def generate(model, tok, dev, ids, max_new):
    p = torch.tensor(ids, device=dev).unsqueeze(0)
    out = model.generate(p, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id or 0,
                         return_dict_in_generate=True, output_scores=True)
    gen = out.sequences[0, p.shape[1]:].tolist()
    lp = torch.log_softmax(out.scores[0][0].float(), -1)
    return gen, float(-(lp.exp() * lp).sum() / np.log(2))


def episode(model, tok, dev, t, task_text, native, thinking, max_new, max_rounds):
    world = World(tok, t)
    msgs = [{"role": "user", "content": task_text}]
    transcript, ent1, final = [], None, ""
    for rnd in range(max_rounds):
        ids = render(tok, msgs, t, native, thinking)
        gen, e1 = generate(model, tok, dev, ids, max_new)
        if ent1 is None:
            ent1 = e1
        text = tok.decode(gen, skip_special_tokens=True)
        visible = text.split("</think>")[-1] if "</think>" in text else text
        calls = parse_calls(visible, native)
        transcript.append({"role": "assistant", "text": text[:1500]})
        if not calls:
            final = visible
            break
        msgs.append({"role": "assistant", "content": text if native else visible})
        for name, args in calls[:3]:
            result = world.call(name, args)
            transcript.append({"role": "tool", "name": name, "args": args, "result": result.replace(T, world.s)[:600]})
            if native:
                msgs.append({"role": "tool", "content": result})
            else:
                msgs.append({"role": "user", "content": f"RESULT of {name}:\n{result}"})
    else:
        final = ""
    return world, transcript, final, ent1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--n-per-group", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    add_model_args(ap)
    a = ap.parse_args()

    d = torch.load(a.pt, weights_only=False, map_location="cpu")
    model, tok, dev = load_from_args(a)
    native = supports_tools(tok)
    grp = groups(d, tok, a.n_per_group, a.seed)
    rng = np.random.default_rng(a.seed)
    tasks = [TASKS[i] for i in sorted(rng.choice(len(TASKS), min(a.tasks, len(TASKS)), replace=False))]
    print(f"{a.model}: native tool template={native}; thinking={a.thinking}; tasks {[n for n, _ in tasks]}; groups "
          + ", ".join(f"{k} {len(v)}" for k, v in grp.items()), flush=True)

    records = []
    for name, text in tasks:
        for k, toks in grp.items():
            for x in toks:
                world, transcript, final, ent1 = episode(model, tok, dev, x["id"], text, native, a.thinking, a.max_new, a.max_rounds)
                s = x["string"].strip()
                queries = [q for n, q in world.calls if n == "search"]
                # what the model looked for instead: the first search query with the token's string removed
                looked_up = queries[0].strip() if queries else None
                m_term = re.search(r"[`\"']([^`\"']{1,60})[`\"']", final)
                records.append({
                    "task": name, "group": k, "id": x["id"], "string": x["string"], "fragility": x["fragility"],
                    "n_tool_calls": len(world.calls), "n_search": len(queries),
                    "first_query": looked_up, "first_query_has_token": bool(queries) and s in queries[0],
                    "any_query_has_token": any(s in q for q in queries),
                    "world_hits": world.hits, "world_misses": world.misses,
                    "final_mentions_token": bool(s) and s in final, "final_term": m_term.group(1) if m_term else None,
                    "final_nonempty": bool(final.strip()), "finished": bool(final.strip()),
                    "claims_version": bool(re.search(r"2\.4\.1", final)), "claims_issue": "218" in final or "environment" in final.lower(),
                    "fabricated": bool(final.strip()) and world.hits == 0 and bool(re.search(r"\d+\.\d+(\.\d+)?", final)),
                    "entropy_first_step_bits": ent1, "final": final[:800], "transcript": transcript})
        print(f"  task {name}: {len(records)} episodes so far", flush=True)

    def rate(rs, key):
        return float(np.mean([bool(r[key]) for r in rs])) if rs else float("nan")
    summary = {}
    print("\n" + "=" * 110)
    print(f"{'group':>8} | {'n':>3} | {'1st query has token':>19} {'any query':>9} {'world hits>0':>12} {'final mentions':>14} "
          f"{'finished':>8} {'fabricated':>10} {'calls/ep':>8}")
    for k in grp:
        rs = [r for r in records if r["group"] == k]
        summary[k] = {"n": len(rs), "first_query_has_token": rate(rs, "first_query_has_token"),
                      "any_query_has_token": rate(rs, "any_query_has_token"),
                      "world_hits_any": float(np.mean([r["world_hits"] > 0 for r in rs])) if rs else float("nan"),
                      "final_mentions_token": rate(rs, "final_mentions_token"), "finished": rate(rs, "finished"),
                      "fabricated": rate(rs, "fabricated"),
                      "calls_per_episode": float(np.mean([r["n_tool_calls"] for r in rs])) if rs else float("nan")}
        v = summary[k]
        print(f"{k:>8} | {v['n']:3d} | {v['first_query_has_token']:19.2f} {v['any_query_has_token']:9.2f} {v['world_hits_any']:12.2f} "
              f"{v['final_mentions_token']:14.2f} {v['finished']:8.2f} {v['fabricated']:10.2f} {v['calls_per_episode']:8.2f}")
    print("\nfragile / glitch episodes where the first search did not contain the token:")
    for r in [r for r in records if r["group"] in ("fragile", "glitch") and r["n_search"] and not r["first_query_has_token"]][:14]:
        print(f"  [{r['task']:>18}] {r['string']!r:>16} searched for {r['first_query']!r:<40} final about {r['final_term']!r}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "model_info": describe(model, tok, a.model), "pt": a.pt, "thinking": a.thinking,
               "native_tools": native, "tasks": [{"name": n, "text": t.replace(T, "<TOKEN>")} for n, t in tasks],
               "groups": grp, "summary": summary, "records": records}, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
