"""Store the confident substitutions as examples, with model-level provenance.

The fragility matrix records only the target token's log-probability per cell;
the phenomenon is what the model emitted INSTEAD, how confidently, and whether
the identity was still present when it did. This collects that, per specimen:

  token      id, string, canonical (decode->encode round-trips), NFC-normal,
             Magikarp category, single-probe copy lp, fragility, mean lp
  context    the exact filler context and slot, decoded and as ids
  emission   top-k emitted tokens with probabilities, entropy at the position,
             the target's probability, an 8-token greedy continuation
  mode       deletion (top-1 is the NEXT context token: the model skipped it),
             truncation (top-1 is a proper prefix of the target), substitution
             (anything else), or correct (a best-context control cell)
  retention  cosine between the last-layer Copy-span slot state and the token's
             own unembedding row -- whether the identity was still pointing at
             its readout when the readout chose something else

and a header with everything needed to reproduce: model id and commit hash,
dtype, layers, hidden size, vocabulary size, tying, tokenizer class, torch and
transformers versions, the copy-prompt head, and the failure threshold.

Also folds in the reasoning-mode substitutions (reasoning_drift.json) and the
hallucination-bridge near-misses (hallucination_bridge.json) when present, so
docs/specimens.md is the single place every example lives.
"""
from __future__ import annotations
import argparse, gzip, json, platform, unicodedata
from pathlib import Path
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .interaction_screen import FEWSHOT
from .embed_predict import final_norm_weight
from .loadmodel import add_model_args, load_from_args, describe
from .copyprompt import copy_prompt_parts


def _fmt(v):
    return 'n/a' if v is None else f'{v:.3f}'


def load_cat(path):
    cat = {}
    for line in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(line); cat[int(r["i"])] = r.get("category", "?")
    return cat


def model_info(model, tok, name):
    cfg = model.config
    sha = getattr(cfg, "_commit_hash", None)
    if sha is None:
        try:
            from huggingface_hub import model_info as hf_info
            sha = hf_info(name).sha
        except Exception:
            sha = "unknown"
    return {"model": name, "commit": sha, "dtype": str(next(model.parameters()).dtype),
            "num_layers": cfg.num_hidden_layers, "hidden_size": cfg.hidden_size,
            "vocab_size": model.get_input_embeddings().weight.shape[0],
            "tie_word_embeddings": bool(cfg.tie_word_embeddings),
            "tokenizer_class": type(tok).__name__,
            "architecture": (cfg.architectures or ["?"])[0],
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "device": str(next(model.parameters()).device),
            "decoding": "greedy", "prompt_head": "Repeat the text exactly.\\n" + "".join(
                f"Text:{d}\\nCopy:{d}\\n" for d in FEWSHOT) + "Text:<context>\\nCopy:<context-prefix>"}


NO_CONT = False   # --no-continuation: skip the greedy continuation (FP8 MoE generation is ~7 s/token)


@torch.no_grad()
def emit(model, tok, dev, ctx, slot, t, topk):
    e = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    head = e("Repeat the text exactly.\n")
    for dm in FEWSHOT:
        head += e("Text:") + e(dm) + e("\nCopy:") + e(dm) + e("\n")
    pre_t, pre_c = e("Text:"), e("\nCopy:")
    head, pre_t, pre_c = copy_prompt_parts(tok, head, FEWSHOT)   # raw or chat framing (GLITCH_PROMPT_STYLE)
    K = len(ctx); ts = len(head) + len(pre_t); cs = ts + K + len(pre_c)
    q = head + pre_t + list(ctx) + pre_c + list(ctx)
    q[ts + slot] = t; q[cs + slot] = t
    ids = torch.tensor(q, device=dev).unsqueeze(0)
    lg = model(input_ids=ids).logits[0, cs + slot - 1].float()
    lp = torch.log_softmax(lg, -1); p = lp.exp()
    top = torch.topk(p, topk)
    if NO_CONT:
        cont = ""
    else:
        gen = model.generate(ids[:, :cs + slot], max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        cont = tok.decode(gen[0, cs + slot:].tolist(), skip_special_tokens=True)
    return {"top": [(int(j), tok.decode([int(j)]), round(float(v), 4))
                    for v, j in zip(top.values, top.indices)],
            "entropy_bits": float(-(p * lp).sum() / np.log(2)),
            "p_target": float(p[t]), "lp_target": float(lp[t]),
            "greedy_continuation": cont}


def mode(rec, ctx, slot, t, tok):
    top1_id, top1_s, _ = rec["top"][0]
    if top1_id == t:
        return "correct"
    if slot + 1 < len(ctx) and top1_id == ctx[slot + 1]:
        return "deletion"
    ts, tt = top1_s.strip(), tok.decode([t]).strip()
    if ts and tt.startswith(ts) and len(ts) < len(tt):
        return "truncation"
    return "substitution"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--ext", default="external/allenai_OLMo_2_1124_7B.jsonl.gz")
    ap.add_argument("--pt", default="results/fragility_L.pt")
    ap.add_argument("--min-frag", type=float, default=0.05)
    ap.add_argument("--gate-lp", type=float, default=-0.1,
                    help="clean-looking gate on the single probe (paper: -0.1; greedy gate: -0.693)")
    ap.add_argument("--max-fragile", type=int, default=80)
    ap.add_argument("--n-glitch", type=int, default=20)
    ap.add_argument("--worst-per-token", type=int, default=3)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--reasoning", default="results/reasoning_drift.json")
    ap.add_argument("--bridge", default="results/hallucination_bridge.json")
    ap.add_argument("--out", default="results/specimens_confident_substitution.jsonl")
    ap.add_argument("--md", default="docs/specimens.md")
    ap.add_argument("--no-continuation", action="store_true",
                    help="skip the 8-token greedy continuation per cell (teacher-forced emission only)")
    add_model_args(ap)
    a = ap.parse_args()
    global NO_CONT
    NO_CONT = bool(a.no_continuation)

    d = torch.load(a.pt, weights_only=False)
    tokens = np.array(d["tokens"]); is_glitch = d["is_glitch"]; single = d["single"]
    M = d["M"]; bank = d["bank"]; thr = d["fail_thr"]; states = d.get("states")
    frag = (M < thr).mean(1)
    cat = load_cat(a.ext) if (a.ext and Path(a.ext).exists()) else {}

    model, tok, dev = load_from_args(a)
    info = model_info(model, tok, a.model)
    info.update(describe(model, tok, a.model))
    info["fail_threshold_lp"] = float(thr); info["n_contexts"] = int(M.shape[1])
    E_out = model.get_output_embeddings().weight.detach().float()
    g = final_norm_weight(model, E_out.shape[1]).to(E_out.device)
    Eo = (E_out * g.unsqueeze(0)).cpu()

    clean = (single > a.gate_lp) & (is_glitch == 0)
    info["clean_gate_lp"] = float(a.gate_lp)
    frag_idx = [i for i in np.argsort(-frag) if clean[i] and frag[i] >= a.min_frag][:a.max_fragile]
    gl_idx = [i for i in np.where(is_glitch == 1)[0]][:a.n_glitch]
    if not gl_idx:
        # no labelled reference class: use the worst tokens by the single probe
        gl_idx = [int(i) for i in np.argsort(single)
                  if tok.decode([int(tokens[i])]).strip()][:a.n_glitch]
    print(f"{len(frag_idx)} fragile clean-looking tokens, {len(gl_idx)} glitch tokens", flush=True)

    records = []
    for group, idxs in (("fragile_clean_looking", frag_idx), ("verified_glitch", gl_idx)):
        for i in idxs:
            t = int(tokens[i]); s = tok.decode([t])
            worst = list(np.argsort(M[i])[:a.worst_per_token]); best = int(np.argmax(M[i]))
            cells = [(int(c), "worst") for c in worst] + [(best, "best")]
            tinfo = {"group": group, "id": t, "string": s,
                     "canonical": tok(s, add_special_tokens=False)["input_ids"] == [t],
                     "nfc_normal": unicodedata.normalize("NFC", s) == s,
                     "magikarp_category": cat.get(t, "?"),
                     "single_probe_lp": float(single[i]), "fragility": float(frag[i]),
                     "mean_lp": float(M[i].mean())}
            for c, kind in cells:
                ctx, slot = bank[c]
                rec = emit(model, tok, dev, ctx, slot, t, a.topk)
                if states is not None:      # a compact .pt (no hidden states) still yields specimens
                    h = states["copy_last"][i, c].float()
                    ret = float(torch.nn.functional.cosine_similarity(h, Eo[t], dim=0))
                else:
                    ret = None
                rec.update({"cell": kind, "context_index": c,
                            "context_ids": [int(x) for x in ctx], "slot": int(slot),
                            "context_text": tok.decode(ctx), "context_length": len(ctx),
                            "left": tok.decode(ctx[max(0, slot - 4):slot]),
                            "right": tok.decode(ctx[slot + 1:slot + 4]),
                            "identity_retention_last_layer": ret,
                            "mode": mode(rec, ctx, slot, t, tok)})
                records.append({**tinfo, **rec})
            print(f"  {s!r:>22} frag {frag[i]:.2f}  modes: "
                  + ", ".join(r["mode"] for r in records[-len(cells):]), flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_model_info": info}) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------- markdown ----------
    L = ["# Confident-substitution specimens", "",
         "Every example in this file was produced by greedy decoding; `p` is the model's "
         "probability for the token it emitted, `H` the entropy in bits at that position, "
         "`p(target)` its probability for the correct token, `ret` the cosine between the "
         "last-layer slot state and the correct token's unembedding row (identity still "
         "pointing at its readout while the readout chose otherwise).", "",
         "## Model", ""]
    for k, v in info.items():
        L.append(f"- **{k}**: `{v}`")
    L += ["", "## Fragile clean-looking tokens (pass the single probe; fail in some contexts)", ""]
    cur = None
    for r in records:
        if r["group"] != "fragile_clean_looking":
            continue
        if r["id"] != cur:
            cur = r["id"]
            L += ["", f"### `{r['string']!r}` — id {r['id']}", "",
                  f"single-probe lp {r['single_probe_lp']:+.3f} · fragility {r['fragility']:.2f} · "
                  f"mean lp {r['mean_lp']:+.2f} · canonical {r['canonical']} · "
                  f"Magikarp `{r['magikarp_category']}`", "",
                  "| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |",
                  "|---|---|---|---|---|---|---|---|---|"]
        top1 = r["top"][0]
        L.append(f"| {r['cell']} | `{r['left']!r} [·] {r['right']!r}` | {r['context_length']} | "
                 f"`{top1[1]!r}` ({top1[2]:.3f}) | **{r['mode']}** | {r['entropy_bits']:.2f} | "
                 f"{r['p_target']:.3f} | {_fmt(r['identity_retention_last_layer'])} | "
                 f"`{r['greedy_continuation'][:40]!r}` |")
    L += ["", "## Verified glitch tokens (reference)", ""]
    cur = None
    for r in records:
        if r["group"] != "verified_glitch":
            continue
        if r["id"] != cur:
            cur = r["id"]
            L += ["", f"### `{r['string']!r}` — id {r['id']}  (single-probe lp "
                  f"{r['single_probe_lp']:+.2f}, fragility {r['fragility']:.2f})", "",
                  "| cell | context | emitted (p) | mode | H bits | p(target) | ret |",
                  "|---|---|---|---|---|---|---|"]
        top1 = r["top"][0]
        L.append(f"| {r['cell']} | `{r['left']!r} [·] {r['right']!r}` | `{top1[1]!r}` "
                 f"({top1[2]:.3f}) | {r['mode']} | {r['entropy_bits']:.2f} | "
                 f"{r['p_target']:.3f} | {_fmt(r['identity_retention_last_layer'])} |")

    if Path(a.reasoning).exists():
        rd = json.load(open(a.reasoning))
        L += ["", "## Reasoning-mode substitutions (chat template, 256-token greedy chains)", "",
              f"model `{rd.get('model')}`; seed reproduced: glitch "
              f"{rd['reasoning']['self'][0]:.3f}, healthy {rd['reasoning']['self'][1]:.3f}", "",
              "| group | seed token | response (first 200 chars) |", "|---|---|---|"]
        for s in rd["reasoning"]["samples"]:
            L.append(f"| {s['group']} | `{s['tok']!r}` | {s['gen'][:200].replace('|', '/')!r} |")

    if Path(a.bridge).exists():
        bd = json.load(open(a.bridge))
        near = [r for r in bd["rows"] if r["cls"] == 1]
        L += ["", f"## Hallucination-bridge near-misses ({len(near)} of {bd['n']} entities)", "",
              "| entity | type | model produced | frag_max | familiarity |", "|---|---|---|---|---|"]
        for r in sorted(near, key=lambda r: -r["frag_max"]):
            L.append(f"| `{r['entity']}` | {r['type']} | `{r['span']}` | {r['frag_max']:.2f} | "
                     f"{r['familiarity']:+.2f} |")
    Path(a.md).parent.mkdir(parents=True, exist_ok=True)
    open(a.md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\n{len(records)} specimen records -> {a.out}\nmarkdown -> {a.md}")


if __name__ == "__main__":
    main()
