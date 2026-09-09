"""Is glitch-token damage MASKED in short completions and AMPLIFIED under long
self-conditioned generation?

Field reports say the effect shows up in reasoning models and in reasoning mode,
and is easy to miss otherwise. Everything measured in this project so far --
here and in the published detectors -- lives in the masked regime: one forward
pass, one token, a handful of predicted tokens. forward_geom.py found the
token's identity linearly recoverable at every layer for verified glitch tokens,
which is a real null and exactly what "masked in a standard completion" predicts.

Long generation is a different object. The model conditions on its own output,
so a small perturbation at step 0 is re-encoded at every subsequent step. Three
things could happen, and they are distinguishable:

  DAMPED       the chain recovers; per-step entropy converges to the healthy
               chain. Then reasoning mode is irrelevant and the field reports
               are about prompt phrasing.
  AMPLIFIED    the glitch/healthy entropy gap WIDENS with generated length.
  ATTRACTOR    chains seeded by DIFFERENT glitch tokens converge on each other
               in representation space, while healthy-seeded chains stay apart.
               This is the geometric version of the claim, and it is what
               "spews word fragments in a wide variety of languages" would look
               like from the inside: the chain has fallen into a common
               high-entropy basin that carries no seed identity.

The design holds the model fixed and varies only the regime, because comparing a
reasoning model to a non-reasoning model confounds regime with everything else
about the two models. Same weights, same tokens, same seeds:

  COMPLETION   short continuation, 8 tokens.
  REASONING    a chat-formatted prompt that asks the model to identify and
               reason about the token, 256 tokens.

Seed token ids are spliced in DIRECTLY. segmentation.py shows a leading space
re-segments ~90% of tokens regardless of health, so building these prompts as
strings would silently replace the token under test.

Convergence is measured on centred last-layer states: the residual stream has a
large shared component, and without centring every pair of chains looks
identical at every step.
"""
from __future__ import annotations
import argparse, gzip, json, unicodedata
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .loadmodel import add_model_args, load_from_args, describe

STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 192, 255, 383, 511]
THINKING = False
INSTRUCTION = False   # --instruction: the reference-model user turn (Table 6) instead of the token alone


def load_verified(path, n):
    ver = np.zeros(n, int); tested = np.zeros(n, bool)
    for line in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(line); i = int(r["i"])
        if i < n and "magikarp" in r:
            tested[i] = True
            if "verified" in r["magikarp"]:
                ver[i] = 1
    return ver, tested


def build(tok, t, regime):
    """Prompt ids with token id t spliced in directly (never via encode(str))."""
    e = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    if regime == "completion":
        return e("The word") + [t]
    head = ("<|user|>\nWhat is this token? Identify it exactly, then explain "
            "step by step what it means:")
    tail = "\n<|assistant|>\nLet me think about this carefully."
    content = ("What is this token? Identify it exactly, then explain step by step what it means: \x00"
               if INSTRUCTION else "\x00")
    msgs = [{"role": "user", "content": content}]
    # Qwen3-style templates default to thinking ON, so the flag must be passed
    # explicitly in BOTH states or the "plain" run silently thinks too. Templates
    # that do not know the kwarg reject it, and the second attempt omits it.
    for kw in ({"enable_thinking": bool(THINKING)}, {}):
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
            a, b = s.split("\x00")
            return e(a) + [t] + e(b + "Let me think about this carefully.")
        except Exception:
            continue
    return e(head) + [t] + e(tail)


@torch.no_grad()
def run(model, tok, dev, ids_list, max_new):
    """Greedy-generate, then one forward pass to recover per-step entropy and
    last-layer states. Greedy keeps the comparison deterministic; sampling noise
    at 256 steps would swamp a group difference this size.

    GLITCH_GEN_BATCH=k generates k chains at once, left-padded, with position ids
    that make every chain see the same positions it would see alone. The default
    of 1 is the original loop. On the FP8 mixture-of-experts models a step costs
    about the same for one chain as for sixty-four, so batching is what makes
    their reasoning-mode measurement possible at all."""
    import os
    batch = max(1, int(os.environ.get("GLITCH_GEN_BATCH", "1")))
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    suppress = [tok.eos_token_id] if tok.eos_token_id is not None else None
    gens, ents, Hs = [], [], []
    for b0 in range(0, len(ids_list), batch):
        chunk = ids_list[b0:b0 + batch]
        L = max(len(x) for x in chunk)
        p = torch.full((len(chunk), L), pad, dtype=torch.long, device=dev)
        m = torch.zeros((len(chunk), L), dtype=torch.long, device=dev)
        for k, x in enumerate(chunk):
            p[k, L - len(x):] = torch.tensor(x, device=dev)
            m[k, L - len(x):] = 1
        # min_new_tokens forces every chain to the same length. Without it a
        # chain that emits EOS at step 1 is silently one step long, and the
        # per-step comparison is between chains of different ages. EOS is
        # suppressed rather than merely un-stopped so it cannot dominate the
        # distribution and flatten the entropy curve.
        out = model.generate(p, attention_mask=m, max_new_tokens=max_new, min_new_tokens=max_new,
                             do_sample=False, pad_token_id=pad, suppress_tokens=suppress)
        full = out[:, :L + max_new]
        mask = torch.cat([m, torch.ones((len(chunk), full.shape[1] - L), dtype=torch.long, device=dev)], 1)
        pos_ids = (mask.cumsum(-1) - 1).clamp(min=0)
        o = model(input_ids=full, attention_mask=mask, position_ids=pos_ids, output_hidden_states=True)
        for k in range(len(chunk)):
            gens.append(full[k, L:].tolist())
            lg = o.logits[k, L - 1:-1].float()
            lp = torch.log_softmax(lg, -1)
            ents.append((-(lp.exp() * lp).sum(-1)).cpu().numpy())
            Hs.append(o.hidden_states[-1][k, L:].float().cpu().half())
        del o, out, full
    return gens, ents, Hs


def pcos(H):
    """Mean pairwise cosine of centred vectors, minus the -1/(n-1) floor that a
    centred set has by construction. 0 = no more aligned than chance."""
    n = H.shape[0]
    if n < 3:
        return float("nan")
    X = (H - H.mean(0, keepdim=True)).float()
    X = X / (X.norm(dim=-1, keepdim=True) + 1e-9)
    G = X @ X.T
    return float((G.sum() - n) / (n * (n - 1))) + 1.0 / (n - 1)


def scripts(s):
    out = set()
    for c in s:
        if c.isspace() or not c.isalpha():
            continue
        try:
            out.add(unicodedata.name(c).split()[0])
        except ValueError:
            out.add("UNNAMED")
    return out


def rep_rate(g):
    if len(g) < 4:
        return float("nan")
    return 1.0 - len(set(g)) / len(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--ext", default="external/allenai_OLMo_2_1124_7B.jsonl.gz")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--pt", default=None,
                    help="fragility .pt to derive label-free glitch/clean sets from")
    ap.add_argument("--thinking", action="store_true",
                    help="enable the model's thinking mode in the chat template "
                         "(Qwen3-style enable_thinking=True); raise --max-new to cover it")
    ap.add_argument("--instruction", action="store_true",
                    help="user turn = 'What is this token? Identify it exactly, then explain step by step "
                         "what it means:' followed by the token (the reference-model prompt), instead of the token alone")
    ap.add_argument("--fragile", action="store_true",
                    help="seed with the fragile clean-looking tokens of the matrix (greedy gate, "
                         "fragility >= 0.10, most fragile first) instead of the worst tokens by the probe; "
                         "controls are then stable clean-looking tokens")
    ap.add_argument("--no-printable", dest="printable", action="store_false",
                    help="admit undecodable/format tokens into the label-free glitch set")
    ap.add_argument("--out", default="results/reasoning_drift.json")
    add_model_args(ap)
    a = ap.parse_args()
    global THINKING, INSTRUCTION
    THINKING = a.thinking
    INSTRUCTION = a.instruction

    model, tok, dev = load_from_args(a)
    V = model.get_input_embeddings().weight.shape[0]
    rng = np.random.default_rng(0)
    if a.pt:
        # label-free: the glitch set is the worst tokens by the single probe in a
        # fragility matrix, the controls a random draw from those it copies
        # perfectly. This is what lets the ladder run on models nobody has
        # labelled; where labels exist the two sets overlap heavily.
        d = torch.load(a.pt, weights_only=False)
        toks = np.array(d["tokens"]); single = np.asarray(d["single"])

        def usable(t):
            s = tok.decode([int(t)])
            if s.strip() == "":
                return False
            if a.printable:
                # the very worst tokens by the probe are undecodable byte pieces
                # and format characters, for which "did you make a typo?" is a
                # sensible answer; the verified sets are printable identifiers
                # and words, so the label-free set is held to the same standard
                if not s.isprintable() or "�" in s or not any(c.isalnum() for c in s):
                    return False
            return True
        keep = np.array([usable(t) for t in toks])
        if a.fragile:
            M = np.asarray(d["M"], dtype=float); thr = float(d["fail_thr"])
            frag = (M < thr).mean(1); gate = single > np.log(0.5)
            order = [int(toks[i]) for i in np.argsort(-frag) if keep[i] and gate[i] and frag[i] >= 0.10]
            pos = order[:a.n]
            cleanpool = [int(toks[i]) for i in range(len(toks)) if keep[i] and gate[i] and frag[i] == 0]
            neg = [int(x) for x in rng.choice(cleanpool, size=len(pos), replace=False)]
            print(f"fragile sets from {a.pt}: {len(pos)} most fragile clean-looking tokens "
                  f"(fragility {frag[np.argsort(-frag)[len(pos)-1]]:.2f}..{frag.max():.2f}) vs {len(neg)} stable")
        else:
            order = [int(toks[i]) for i in np.argsort(single) if keep[i]]
            pos = order[:a.n]
            cleanpool = [int(toks[i]) for i in range(len(toks)) if keep[i] and single[i] > -0.05]
            neg = [int(x) for x in rng.choice(cleanpool, size=len(pos), replace=False)]
            print(f"label-free sets from {a.pt}: worst-{len(pos)} by single probe "
                  f"(lp {single.min():.2f}..{np.sort(single)[len(pos)-1]:.2f}) vs {len(neg)} clean")
    else:
        ver, tested = load_verified(a.ext, V)
        pos = [int(i) for i in np.where(ver == 1)[0]]
        rng.shuffle(pos); pos = pos[:a.n]
        negpool = np.where((ver == 0) & tested)[0]
        neg = [int(i) for i in rng.choice(negpool, size=len(pos), replace=False)]
    allids = pos + neg
    grp = {"glitch": np.arange(len(pos)),
           "healthy": np.arange(len(pos), len(allids))}
    print(f"{a.model}: {len(pos)} glitch vs {len(neg)} healthy, {dev}")
    print("  glitch:", [repr(tok.decode([t]))[:14] for t in pos[:6]])
    print("  healthy:", [repr(tok.decode([t]))[:14] for t in neg[:6]])

    report = {"model": a.model, "n": len(pos), "thinking": bool(a.thinking),
              "model_info": describe(model, tok, a.model),
              "glitch_ids": pos, "control_ids": neg}
    for regime, mx in [("completion", 8), ("reasoning", a.max_new)]:
        print()
        print("=" * 74)
        print(f"REGIME: {regime}   ({mx} generated tokens)")
        print("=" * 74)
        seqs = [build(tok, t, regime) for t in allids]
        gens, ents, Hs = run(model, tok, dev, seqs, mx)

        print(f"{'step':>6} | {'entropy gl':>10} {'entropy he':>10} {'gap':>7} | "
              f"{'pcos gl':>8} {'pcos he':>8}")
        print("-" * 60)
        rows = []
        for s in [x for x in STEPS if x <= mx]:
            eg = np.mean([ents[i][s - 1] for i in grp["glitch"]])
            eh = np.mean([ents[i][s - 1] for i in grp["healthy"]])
            Hg = torch.stack([Hs[i][s - 1] for i in grp["glitch"]])
            Hh = torch.stack([Hs[i][s - 1] for i in grp["healthy"]])
            # centre across ALL chains at this step, then measure within-group
            allH = torch.cat([Hg, Hh]); mu = allH.float().mean(0, keepdim=True)
            r = {"step": s, "ent_g": float(eg), "ent_h": float(eh),
                 "gap": float(eg - eh),
                 "pcos_g": pcos(Hg.float() - mu), "pcos_h": pcos(Hh.float() - mu)}
            rows.append(r)
            print(f"{s:6d} | {eg:10.3f} {eh:10.3f} {r['gap']:7.3f} | "
                  f"{r['pcos_g']:8.4f} {r['pcos_h']:8.4f}")

        txt = [tok.decode(g) for g in gens]
        sg = np.mean([len(scripts(txt[i])) for i in grp["glitch"]])
        sh = np.mean([len(scripts(txt[i])) for i in grp["healthy"]])
        rg = np.mean([rep_rate(gens[i]) for i in grp["glitch"]])
        rh = np.mean([rep_rate(gens[i]) for i in grp["healthy"]])
        selfg = np.mean([allids[i] in gens[i] for i in grp["glitch"]])
        selfh = np.mean([allids[i] in gens[i] for i in grp["healthy"]])
        # string-level: the seed's text appears in the output even if re-segmented into other
        # ids (a quoted "_ghost" is "_" + "ghost"); id-level alone would call that a failure
        strs = [tok.decode([int(t)]).strip() for t in allids]
        selfg_s = np.mean([bool(strs[i]) and strs[i] in txt[i] for i in grp["glitch"]])
        selfh_s = np.mean([bool(strs[i]) and strs[i] in txt[i] for i in grp["healthy"]])
        print(f"\n  distinct scripts in output : glitch {sg:.2f}  healthy {sh:.2f}")
        print(f"  token-repetition rate      : glitch {rg:.3f}  healthy {rh:.3f}")
        print(f"  reproduced the seed token  : glitch {selfg:.3f}  healthy {selfh:.3f}   (exact id)")
        print(f"  seed string in output      : glitch {selfg_s:.3f}  healthy {selfh_s:.3f}   (any segmentation)")
        report[regime] = {"rows": rows, "scripts": [sg, sh], "rep": [rg, rh],
                          "self": [float(selfg), float(selfh)],
                          "self_str": [float(selfg_s), float(selfh_s)],
                          "gens_all": [{"group": "glitch" if i in grp["glitch"] else "healthy",
                                        "id": int(allids[i]), "tok": tok.decode([int(allids[i])]),
                                        "gen": txt[i][:400]} for i in range(len(allids))]}

        print(f"\n  --- sample generations ({regime}) ---")
        for lab, idx in [("GLITCH", grp["glitch"][:3]), ("HEALTHY", grp["healthy"][:2])]:
            for i in idx:
                print(f"  [{lab}] {tok.decode([allids[i]])!r} -> "
                      f"{txt[i][:160].encode('unicode_escape').decode()!r}")
        report[regime]["samples"] = [
            {"group": "glitch" if i in grp["glitch"] else "healthy",
             "tok": tok.decode([allids[i]]), "gen": txt[i][:400]}
            for i in list(grp["glitch"][:8]) + list(grp["healthy"][:8])]

    print()
    print("=" * 74)
    print("MASKING TEST: does the glitch/healthy gap grow from completion to")
    print("reasoning, and does the glitch group converge as the chain runs?")
    print("=" * 74)
    c, r = report["completion"]["rows"], report["reasoning"]["rows"]
    print(f"  entropy gap, completion  @step8   {c[-1]['gap']:+.3f}")
    print(f"  entropy gap, reasoning   @step8   "
          f"{[x for x in r if x['step']==8][0]['gap']:+.3f}")
    print(f"  entropy gap, reasoning   @step{r[-1]['step']} {r[-1]['gap']:+.3f}")
    print(f"  glitch chain alignment,  step1 -> step{r[-1]['step']}   "
          f"{r[0]['pcos_g']:.4f} -> {r[-1]['pcos_g']:.4f}")
    print(f"  healthy chain alignment, step1 -> step{r[-1]['step']}   "
          f"{r[0]['pcos_h']:.4f} -> {r[-1]['pcos_h']:.4f}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
