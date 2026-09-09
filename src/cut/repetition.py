"""Stage 5: behavioral glitch metric = can the model copy the sequence back?

Everything is done at the TOKEN-ID level. We never decode-and-re-encode the
target, because retokenization would silently canonicalise the very sequences
we are trying to test (and would confound this with the non-canonical
tokenization literature).

Primary metric  : mean teacher-forced log p of the copied span (continuous).
Secondary metric: greedy exact-match of the copied span (interpretable).

Both share one prompt builder. `build_prefix` stops right after "Copy:" -- the
scorer appends the target and masks it; the generator is given the prefix ALONE
and must produce the target itself.
"""
from __future__ import annotations
import torch
from .modelio import seq_logprobs

FEWSHOT = [(" apple pie is good", " apple pie is good"),
           (" the quick brown fox", " the quick brown fox")]
HEAD = "Repeat the text exactly.\n"


def build_prefix(tok, target_ids: list[int]) -> list[int]:
    """head + demos + 'Text:'<target> '\nCopy:'  -- ends where the copy begins."""
    def enc(s):
        return tok(s, add_special_tokens=False).input_ids
    ids: list[int] = enc(HEAD)
    for a, b in FEWSHOT:
        ids += enc("Text:") + enc(a) + enc("\nCopy:") + enc(b) + enc("\n")
    ids += enc("Text:") + list(target_ids) + enc("\nCopy:")
    return ids


def score_batch(model, tok, device, targets: list[list[int]], batch_size: int = 8):
    """Mean teacher-forced logprob of copying each target. Returns tensor (N,)."""
    out = []
    for s in range(0, len(targets), batch_size):
        chunk = targets[s:s + batch_size]
        built = []
        for t in chunk:
            pre = build_prefix(tok, t)
            built.append((pre + list(t), [False] * len(pre) + [True] * len(t)))
        L = max(len(i) for i, _ in built)
        pad = tok.eos_token_id or 0
        ids = torch.full((len(built), L), pad, dtype=torch.long)
        msk = torch.zeros((len(built), L), dtype=torch.bool)
        for r, (i, m) in enumerate(built):      # RIGHT-pad: causal attention means
            ids[r, :len(i)] = torch.tensor(i)   # trailing pads cannot influence the
            msk[r, :len(i)] = torch.tensor(m)   # scored span, so no mask is needed
        lp = seq_logprobs(model, ids.to(device), msk.to(device))
        out.append(lp.cpu())
    return torch.cat(out)


@torch.no_grad()
def exact_match(model, tok, device, targets: list[list[int]]):
    """Greedy-decode the copy span from the prefix ALONE and compare token ids."""
    hits, gens = [], []
    for t in targets:
        prompt = torch.tensor(build_prefix(tok, t), device=device).unsqueeze(0)
        gen = model.generate(prompt, max_new_tokens=len(t), do_sample=False,
                             pad_token_id=tok.eos_token_id or 0)
        produced = gen[0, prompt.shape[1]:].tolist()
        hits.append(produced == list(t))
        gens.append(produced)
    return torch.tensor(hits, dtype=torch.float), gens
