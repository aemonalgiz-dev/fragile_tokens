"""Tokens inside real tasks, not random filler.

For one model: take three token groups from its fragility matrix (fragile clean-looking,
glitch, stable controls), draw K tasks from a bank, splice each token BY ID into the user
turn of each task (the chat template is applied to a placeholder and the id inserted, as in
reasoning_drift.build, so the tokenizer never sees the token as text), generate greedily,
and record whether the token comes back, what the model put where the token should be,
and whether the task was still completed. Every generation is stored.

  python -m src.cut.task_probe --model Qwen/Qwen3-32B --pt results/fragility_qwen3_32b.pt \
      --tasks 5 --n-per-group 10 --out results/task_probe_qwen3_32b.json
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import torch

from .loadmodel import add_model_args, load_from_args, describe

GREEDY = float(np.log(0.5))

# (name, user turn with \x00 where the token goes, how to read the model's version of the token back)
TASKS = [
    ("translate", "Translate the following sentence into French. Keep any word you do not recognise exactly as written.\n\n"
                  "Sentence: The report mentioned \x00 twice before the meeting ended.", None),
    ("json_extract", "Extract every product name from the text below into a JSON list of strings, exactly as written. "
                     "Output only the JSON.\n\nText: Our catalogue lists Aurora, \x00 and Meridian as this season's releases.",
     r"\[(.*?)\]"),
    ("code_name", "Write a Python function whose name is exactly \x00 and which returns the sum of a list of numbers. "
                  "Output only the code.", r"def\s+([^\s(]+)"),
    ("quote_title", "Quote the title of the document exactly, in double quotes, and write nothing else.\n\n"
                    "Document title: Notes on \x00 and related matters\n\nBody: This document collects observations made during the review.",
     r"\"(.*?)\""),
    ("list_item", "Here is a list of items:\n1. lantern\n2. \x00\n3. harbour\n4. compass\n\n"
                  "What is the second item? Answer with the item only.", r"^\s*(.+?)\s*$"),
    ("fix_spelling", "Correct any spelling mistakes in the sentence below and return the corrected sentence. "
                     "If there are none, return it unchanged.\n\nSentence: We agreed to file the \x00 form before Friday.",
     r"file the (.+?) form"),
    ("replace_word", "In the text below, replace the word 'blue' with 'green' and return the full text otherwise unchanged.\n\n"
                     "Text: The blue folder labelled \x00 was left on the desk.", r"labelled (.+?) was"),
    ("summarize", "Summarize the following in one sentence that keeps every proper noun exactly as written.\n\n"
                  "Text: The committee met on Tuesday. \x00 presented the budget, which was approved without changes, "
                  "and the next meeting was scheduled for May.", None),
    ("to_csv", "Convert this record to CSV with the header line name,value and one data line. Output only the CSV.\n\n"
               "name: \x00\nvalue: 42", r"name,value\s*\n\s*(.+?),\s*42"),
    ("count_chars", "How many characters are in the word \x00? Write the word back in double quotes first, then the count.",
     r"\"(.*?)\""),
]


def build(tok, t, text, thinking):
    """Prompt ids for one task with token id t spliced in where \\x00 stands."""
    e = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    t_str = tok.decode([t])
    # the templates put a space before the placeholder; a token that carries its own
    # leading space must not get a second one
    if t_str.startswith(" "):
        text = text.replace(" \x00", "\x00")
    msgs = [{"role": "user", "content": text}]
    for kw in ({"enable_thinking": bool(thinking)}, {}):
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            a, b = s.split("\x00")
            return e(a) + [t] + e(b)
        except Exception:
            continue
    a, b = text.split("\x00")                      # no chat template: bare instruction
    return e("Instruction: " + a) + [t] + e(b + "\nAnswer:")


def groups(d, tok, n, seed):
    tokens = np.asarray(d["tokens"]); s = np.asarray(d["single"], np.float64)
    M = np.asarray(d["M"], np.float64); thr = float(d["fail_thr"]); g = np.asarray(d["is_glitch"])
    frag = (M < thr).mean(1); clean = (s > GREEDY) & (g == 0)
    printable = lambda i: bool(tok.decode([int(tokens[i])]).strip()) and tok.decode([int(tokens[i])]).isprintable()
    fragile = [i for i in np.argsort(-frag) if clean[i] and frag[i] >= 0.10 and printable(i)][:n]
    glitch = [i for i in np.where(g == 1)[0] if printable(i)][:n]
    if not glitch:                                 # label-free: fails alone and in >= 90% of contexts
        glitch = [i for i in np.argsort(s) if frag[i] >= 0.9 and printable(i)][:n]
    rng = np.random.default_rng(seed)
    pool = [i for i in np.where(clean & (frag == 0))[0] if printable(i)]
    stable = list(rng.choice(pool, min(n, len(pool)), replace=False)) if pool else []
    out = {}
    for name, idx in (("fragile", fragile), ("glitch", glitch), ("stable", stable)):
        out[name] = [{"id": int(tokens[i]), "string": tok.decode([int(tokens[i])]),
                      "fragility": float(frag[i]), "single_lp": float(s[i])} for i in idx]
    return out


def readback(pattern, text):
    if pattern is None:
        return None
    m = re.search(pattern, text, re.S | re.M)
    if not m:
        return None
    return m.group(1).strip().strip("\"'").strip()


def task_done(name, text):
    t = text.strip()
    if not t:
        return False
    if name == "json_extract":
        m = re.search(r"\[.*?\]", t, re.S)
        try:
            return m is not None and isinstance(json.loads(m.group(0)), list)
        except Exception:
            return False
    if name == "code_name":
        return "def " in t and "return" in t
    if name == "quote_title":
        return '"' in t
    if name == "to_csv":
        return "name,value" in t.replace(" ", "")
    if name == "count_chars":
        return re.search(r"\d", t) is not None
    return True


@torch.no_grad()
def generate(model, tok, dev, ids, max_new):
    p = torch.tensor(ids, device=dev).unsqueeze(0)
    out = model.generate(p, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id or 0,
                         return_dict_in_generate=True, output_scores=True)
    gen = out.sequences[0, p.shape[1]:].tolist()
    lp = torch.log_softmax(out.scores[0][0].float(), -1)
    ent1 = float(-(lp.exp() * lp).sum() / np.log(2))
    return gen, ent1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--n-per-group", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--thinking", action="store_true", help="Qwen3-style enable_thinking=True (raise --max-new)")
    ap.add_argument("--seed", type=int, default=0)
    add_model_args(ap)
    a = ap.parse_args()

    d = torch.load(a.pt, weights_only=False, map_location="cpu")
    model, tok, dev = load_from_args(a)
    grp = groups(d, tok, a.n_per_group, a.seed)
    rng = np.random.default_rng(a.seed)
    task_idx = sorted(rng.choice(len(TASKS), min(a.tasks, len(TASKS)), replace=False))
    tasks = [TASKS[i] for i in task_idx]
    print(f"{a.model}: tasks {[t[0] for t in tasks]}; groups " +
          ", ".join(f"{k} {len(v)}" for k, v in grp.items()) + f"; thinking={a.thinking}", flush=True)
    for k, v in grp.items():
        print(f"  {k:>8}: " + ", ".join(repr(x["string"]) for x in v[:10]))

    records = []
    for name, text, pat in tasks:
        for k, toks in grp.items():
            for x in toks:
                ids = build(tok, x["id"], text, a.thinking)
                gen, ent1 = generate(model, tok, dev, ids, a.max_new)
                out_text = tok.decode(gen, skip_special_tokens=True)
                # for thinking models, judge the final answer after the thinking block if present
                answer = out_text.split("</think>")[-1] if "</think>" in out_text else out_text
                records.append({"task": name, "group": k, "id": x["id"], "string": x["string"],
                                "fragility": x["fragility"], "single_lp": x["single_lp"],
                                "string_in_output": bool(x["string"].strip()) and x["string"].strip() in answer,
                                "id_in_output": x["id"] in gen,
                                "readback": readback(pat, answer),
                                "task_done": task_done(name, answer),
                                "entropy_first_step_bits": ent1,
                                "output": out_text[:600]})
        done = len(records)
        print(f"  task {name}: {done} generations so far", flush=True)

    def rate(rs, key):
        return float(np.mean([r[key] for r in rs])) if rs else float("nan")
    summary = {}
    print("\n" + "=" * 96)
    print(f"{'group':>8} | {'n':>3} | {'token string in output':>22} {'exact id':>9} {'task done':>10} {'H first step':>12}")
    for k in grp:
        rs = [r for r in records if r["group"] == k]
        summary[k] = {"n": len(rs), "string_in_output": rate(rs, "string_in_output"), "id_in_output": rate(rs, "id_in_output"),
                      "task_done": rate(rs, "task_done"), "entropy_first_step": rate(rs, "entropy_first_step_bits")}
        print(f"{k:>8} | {len(rs):3d} | {summary[k]['string_in_output']:22.3f} {summary[k]['id_in_output']:9.3f} "
              f"{summary[k]['task_done']:10.3f} {summary[k]['entropy_first_step']:12.2f}")
    print("\nby task (token string in output):")
    by_task = {}
    for name, _, _ in tasks:
        row = {k: rate([r for r in records if r["task"] == name and r["group"] == k], "string_in_output") for k in grp}
        by_task[name] = row
        print(f"  {name:>13}: " + "  ".join(f"{k} {v:.2f}" for k, v in row.items()))
    print("\nsample: fragile tokens that did not come back")
    for r in [r for r in records if r["group"] == "fragile" and not r["string_in_output"]][:12]:
        print(f"  [{r['task']:>12}] {r['string']!r:>14} -> readback {r['readback']!r}: {r['output'][:140]!r}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "model_info": describe(model, tok, a.model), "pt": a.pt, "thinking": a.thinking,
               "tasks": [{"name": n, "text": t} for n, t, _ in tasks], "groups": grp, "summary": summary,
               "by_task": by_task, "records": records}, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
