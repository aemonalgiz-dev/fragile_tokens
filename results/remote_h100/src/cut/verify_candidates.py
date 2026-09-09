"""Behaviourally verify candidate glitch tokens -- the actual test, not an indicator.

Three groups, run identically:
  KNOWN     -- Magikarp's verified under-trained tokens. Positive control: if
               these do not fail, the test is broken and no result means anything.
  CANDIDATE -- tokens our detector flags that Magikarp never verified.
  CONTROL   -- length-matched ordinary tokens. Negative control: if these fail
               too, we are measuring model incompetence, not glitchiness.

A candidate is only "real" if KNOWN fails, CONTROL passes, and CANDIDATE
patterns with KNOWN. Reporting the middle group alone would prove nothing.

Three prompt templates, because a single phrasing can fail for reasons that have
nothing to do with the token.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TEMPLATES = {
    "fewshot": ([(" apple pie", " apple pie"), (" quick brown fox", " quick brown fox")],
                "Text:{t}\nCopy:"),
    "quoted":  ([("hello world", "hello world"), ("blue sky", "blue sky")],
                "Repeat exactly: \"{t}\"\nAnswer: \""),
    "qa":      ([(" mountain", " mountain"), (" river", " river")],
                "Q: Repeat this string: {t}\nA:"),
}


def build(tok, name, tok_id):
    demos, tail = TEMPLATES[name]
    s = ""
    for a, b in demos:
        s += tail.format(t=a) + b + ("\"\n" if name == "quoted" else "\n")
    ids = tok(s, add_special_tokens=False).input_ids
    head, sep = tail.split("{t}")
    ids += tok(head, add_special_tokens=False).input_ids + [tok_id] + \
           tok(sep, add_special_tokens=False).input_ids
    return ids


@torch.no_grad()
def test_group(model, tok, device, ids_list, name, max_new=6):
    hits, gens = [], []
    for t in ids_list:
        prompt = torch.tensor(build(tok, name, t), device=device).unsqueeze(0)
        out = model.generate(prompt, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        gen = out[0, prompt.shape[1]:].tolist()
        target = tok.decode([t])
        text = tok.decode(gen)
        # Magikarp's criterion: does the reproduction start with the target string?
        ok = text.strip().startswith(target.strip()) and len(target.strip()) > 0
        hits.append(ok); gens.append(text)
    return hits, gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-1.4b")
    ap.add_argument("--cands", default="results/candidates.json")
    ap.add_argument("--out", default="results/verification.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev)
    model.eval()
    C = json.load(open(a.cands))

    groups = {"KNOWN (magikarp-verified)": C["known_verified"],
              "CANDIDATE (ours, new)": C["candidates"],
              "CONTROL (matched normal)": C["controls"]}
    res = {}
    print(f"model: {a.model}\n")
    print(f"{'group':>26} | " + " | ".join(f"{t:>9}" for t in TEMPLATES) + " | overall")
    print("-" * 70)
    for gname, ids in groups.items():
        per, allhits = {}, []
        for tname in TEMPLATES:
            h, g = test_group(model, tok, dev, ids, tname)
            per[tname] = sum(h) / len(h)
            allhits += h
            res.setdefault(gname, {})[tname] = {"rate": per[tname], "gens": g, "hits": h}
        print(f"{gname:>26} | " + " | ".join(f"{per[t]:8.1%}" for t in TEMPLATES)
              + f" | {sum(allhits)/len(allhits):7.1%}")

    print("\n--- actual generations (fewshot template) ---")
    for gname, ids in groups.items():
        print(f"\n{gname}:")
        gens = res[gname]["fewshot"]["gens"]
        for t, g in list(zip(ids, gens))[:8]:
            tgt = tok.decode([t])
            mark = "OK " if g.strip().startswith(tgt.strip()) else "FAIL"
            print(f"  {mark}  want {tgt!r:22s} got {g[:34]!r}")

    print()
    print("--- per-token verdict (passes out of 3 templates) ---")
    for gname, ids in groups.items():
        hits = [res[gname][t]["hits"] for t in TEMPLATES]
        tot = [sum(h[i] for h in hits) for i in range(len(ids))]
        bad = [(tok.decode([t]), c) for t, c in zip(ids, tot) if c <= 1]
        print(f"  {gname:>26}: {sum(1 for c in tot if c<=1)}/{len(ids)} fail >=2 of 3 templates")
        if gname.startswith("CANDIDATE"):
            print("     failing candidates:", [b[0] for b in bad][:16])
            print("     passing candidates:",
                  [tok.decode([t]) for t, c in zip(ids, tot) if c >= 2][:10])

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "results": res}, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
