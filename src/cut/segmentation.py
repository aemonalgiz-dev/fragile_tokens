"""Is the "you cannot put a space/quote/bracket before it" pattern a model
property, or a tokenizer property?

Field reports of glitch tokens describe a usage pattern: the token behaves when
it stands alone, after a newline, after a tab, or repeated back-to-back, but
breaks when preceded by a space, a quote, or a bracket.

That set is suspicious. In GPT-family BPE the pre-tokenizer regex SPLITS on
newline and tab, so whatever follows is encoded independently and the intended
id reaches the model. Space, quote and bracket do not split -- they can merge
rightward into the following text. So the "breaks it" set is precisely the set
of prefixes that can change the segmentation.

If that is the whole story, the phenomenon is re-segmentation: the user types a
string containing the glitch token, the tokenizer emits a DIFFERENT id sequence,
and the model never sees the token at all. No embedding geometry required.

This measures that directly, and separates two failure modes:

  DROPPED   -- prefix+s does not encode to anything ending in t. The token is
               gone; whatever the model saw, it was not this token.
  INTACT    -- t is still the final id. Any misbehaviour here is the model's,
               not the tokenizer's, and is what the forward-pass analysis in
               forward_geom.py is for.

The comparison that matters is glitch vs healthy: if healthy tokens survive a
leading space at the same rate glitch tokens do, re-segmentation is not what
distinguishes them.
"""
from __future__ import annotations
import argparse, gzip, json, sys, unicodedata
from collections import Counter
from pathlib import Path
import numpy as np

# newline/tab are pre-tokenizer boundaries in GPT-family BPE; the rest are not.
PREFIXES = [
    ("none",      "",   "boundary"),
    ("newline",   "\n", "boundary"),
    ("tab",       "\t", "boundary"),
    ("space",     " ",  "merging"),
    ("dquote",    '"',  "merging"),
    ("squote",    "'",  "merging"),
    ("lparen",    "(",  "merging"),
    ("lbracket",  "[",  "merging"),
    ("lbrace",    "{",  "merging"),
    ("period",    ".",  "merging"),
    ("comma",     ",",  "merging"),
]


def survives(enc, dec, t, prefix):
    """Does token id t still arrive, as the final id, after prepending prefix?"""
    s = dec(t)
    if s == "":
        return None, 0
    try:
        ids = enc(prefix + s)
    except Exception:
        return None, 0
    return (len(ids) > 0 and ids[-1] == t), len(ids)


def doubled(enc, dec, t):
    s = dec(t)
    if s == "":
        return None
    try:
        ids = enc(s + s)
    except Exception:
        return None
    return ids == [t, t]


def profile(enc, dec, ids, name):
    """Survival rate per prefix over a set of token ids."""
    out = {}
    for pname, p, cls in PREFIXES:
        ok = [survives(enc, dec, t, p)[0] for t in ids]
        ok = [b for b in ok if b is not None]
        out[pname] = (float(np.mean(ok)) if ok else float("nan"), cls)
    d = [doubled(enc, dec, t) for t in ids]
    d = [b for b in d if b is not None]
    out["self-doubled"] = (float(np.mean(d)) if d else float("nan"), "boundary")
    return out


def show(tab, groups):
    print(f"{'prefix':>13} | {'class':>9} | " + " | ".join(f"{g:>9}" for g in groups))
    print("-" * (26 + 12 * len(groups)))
    for pname, _, cls in PREFIXES + [("self-doubled", "", "boundary")]:
        row = []
        for g in groups:
            v = tab[g].get(pname, (float("nan"), ""))[0]
            row.append(f"{v:9.3f}")
        print(f"{pname:>13} | {cls:>9} | " + " | ".join(row))


def load_verified(path, n):
    ver = np.zeros(n, int); tested = np.zeros(n, bool)
    for line in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(line); i = int(r["i"])
        if i < n and "magikarp" in r:
            tested[i] = True
            if "verified" in r["magikarp"]:
                ver[i] = 1
    return ver, tested


def hf_part(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    V = tok.vocab_size
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    dec = lambda t: tok.decode([t])
    ver, tested = load_verified(args.ext, V)
    rng = np.random.default_rng(0)
    pos = [int(i) for i in np.where(ver == 1)[0]]
    negpool = np.where((ver == 0) & tested)[0]
    neg = [int(i) for i in rng.choice(negpool, size=min(600, len(negpool)), replace=False)]
    print(f"{args.model}: {len(pos)} verified glitch, {len(neg)} tested-and-rejected\n")
    tab = {"glitch": profile(enc, dec, pos, "glitch"),
           "healthy": profile(enc, dec, neg, "healthy")}
    show(tab, ["glitch", "healthy"])

    print("\nWhen a leading space DOES break it, what does the model receive instead?")
    shown = 0
    for t in pos:
        s = dec(t)
        if not s:
            continue
        ids = enc(" " + s)
        if ids and ids[-1] != t:
            print(f"  {s!r:20s} (id {t:6d})  ->  {[repr(dec(i)) for i in ids]}")
            shown += 1
            if shown >= 12:
                break
    if shown == 0:
        print("  (a leading space never re-segments a verified glitch token here)")
    return tab


def tiktoken_part(args):
    """The field reports name specific ids in the o200k vocabulary. That vocab is
    public, so the segmentation claim can be checked against the actual tokenizer
    the reports are about, even though the weights are not available."""
    try:
        import tiktoken
    except ImportError:
        print("\n(tiktoken not installed; skipping o200k check)")
        return None
    for encname in ("o200k_base", "cl100k_base"):
        try:
            e = tiktoken.get_encoding(encname)
        except Exception as ex:
            print(f"{encname}: {ex}"); continue
        V = e.n_vocab
        enc = lambda s: e.encode(s, allowed_special=set(), disallowed_special=set())
        def dec(t):
            try:
                return e.decode([t])
            except Exception:
                return ""
        print(f"\n{'='*74}\n{encname}  (n_vocab {V})\n{'='*74}")
        for t in args.ids:
            if t >= V:
                print(f"  id {t}: out of range"); continue
            s = dec(t)
            cats = Counter(unicodedata.category(c) for c in s)
            names = []
            for c in s:
                try:
                    names.append(unicodedata.name(c))
                except ValueError:
                    names.append(f"U+{ord(c):04X} <unnamed>")
            print(f"  id {t}: {s!r}  cats={dict(cats)}")
            for nm in names:
                print(f"        {nm}")
            for pname, p, cls in PREFIXES:
                ok, n = survives(enc, dec, t, p)
                mark = "INTACT " if ok else "DROPPED"
                print(f"        prefix {pname:9s} ({cls:8s}) -> {mark} ({n} ids)")
            print(f"        doubled -> {'INTACT' if doubled(enc,dec,t) else 'DROPPED'}")

        # aggregate: private-use / unassigned codepoints vs ordinary tokens
        rng = np.random.default_rng(0)
        pua, ordinary = [], []
        for t in range(V):
            s = dec(int(t))
            if not s:
                continue
            if any(unicodedata.category(c) in ("Co", "Cn") for c in s):
                pua.append(int(t))
            elif s.isascii() and s.strip():
                ordinary.append(int(t))
        rng.shuffle(ordinary); rng.shuffle(pua)
        ordinary = ordinary[:800]; pua = pua[:800]
        print(f"\n  private-use/unassigned tokens: {len(pua)}, ordinary ascii: {len(ordinary)}")
        if pua and ordinary:
            tab = {"pua": profile(enc, dec, pua, "pua"),
                   "ascii": profile(enc, dec, ordinary, "ascii")}
            show(tab, ["pua", "ascii"])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--ext", default="external/pythia_6_9b.jsonl.gz")
    ap.add_argument("--ids", type=int, nargs="*", default=[128188, 152383],
                    help="specific token ids to profile in the tiktoken vocabs")
    ap.add_argument("--out", default="results/segmentation.json")
    ap.add_argument("--skip-hf", action="store_true")
    a = ap.parse_args()

    print("=" * 74)
    print("RE-SEGMENTATION AUDIT")
    print("fraction of tokens that still arrive as the final id after a prefix")
    print("=" * 74)
    tab = None
    if not a.skip_hf:
        tab = hf_part(a)
    tiktoken_part(a)

    if tab:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({k: {p: v[0] for p, v in d.items()} for k, d in tab.items()},
                  open(a.out, "w"), indent=1)
        print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
