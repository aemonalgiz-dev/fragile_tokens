"""Stage 2+3: extract welded chains, then construct the four experimental arms.

Arms (all length-k token sequences):
  A  weld_intact   : the greedy chain itself           -> maximally on-manifold
  B  seam_violation: one committed position swapped to its 2nd-argmax  [TREATMENT]
  C  perm_control  : arm B's exact token multiset, shuffled            [zero-count, perfectly freq-matched]
  D  random_control: random tokens matched on unembedding-norm decile  [zero-count, loosely freq-matched]

Arms C and D are what "Broken Tokens" says should be fine. If B is also fine,
compositional under-training is not a real phenomenon and we stop.
"""
from __future__ import annotations
import torch, random
from .modelio import next_logits


@torch.no_grad()
def extract_chain(model, tok, device, seed_tok: int, max_len: int = 6,
                  ent_ceiling: float = 1.5, prefix: str = ""):
    """Greedily extend from seed_tok while the model stays committed.

    Returns (tokens, entropies, second_choices). tokens[0] is the seed.
    """
    pre = tok(prefix, return_tensors="pt").input_ids[0].to(device) if prefix else \
          torch.empty(0, dtype=torch.long, device=device)
    seq = [seed_tok]
    ents, seconds = [], []
    for _ in range(max_len - 1):
        inp = torch.cat([pre, torch.tensor(seq, device=device)]).unsqueeze(0)
        logits = next_logits(model, inp)
        logp = torch.log_softmax(logits, dim=-1)
        H = -(logp.exp() * logp).sum(-1).item()
        t2 = logits.topk(2, dim=-1).indices[0]
        seq.append(int(t2[0]))
        ents.append(H)
        seconds.append(int(t2[1]))
    return seq, ents, seconds


def build_arms(chain, ents, seconds, unemb_norm, rng: random.Random, n_random_pool=None):
    """Given a chain, its per-step entropies and 2nd choices, emit one item per arm.

    Returns dict arm -> list[int] token ids, or None if the chain is too short.
    """
    k = len(chain)
    if k < 3 or not seconds:
        return None

    A = list(chain)

    # B: violate the single most-committed seam (the step with lowest entropy is
    # position argmin over steps; steps[i] produced chain[i+1]).
    j = min(range(len(seconds)), key=lambda i: ents[i])  # most-committed step
    B = list(chain)
    B[j + 1] = seconds[j]

    # C: same multiset as B, shuffled into an arrangement that is not B or A
    for _ in range(20):
        C = B[:]
        rng.shuffle(C)
        if C != B and C != A:
            break

    # D: random tokens matched to B on unembedding-norm decile
    D = []
    if n_random_pool is not None:
        for t in B:
            band = n_random_pool.get(_decile(unemb_norm, t), None)
            D.append(rng.choice(band) if band else int(rng.randrange(len(unemb_norm))))
    else:
        D = [int(rng.randrange(len(unemb_norm))) for _ in B]

    return {"A_weld_intact": A, "B_seam_violation": B,
            "C_perm_control": C, "D_random_control": D}


_DEC_CACHE = {}


def _decile(unemb_norm, t):
    key = id(unemb_norm)
    if key not in _DEC_CACHE:
        q = torch.quantile(unemb_norm.float(),
                           torch.linspace(0, 1, 11)[1:-1])
        _DEC_CACHE[key] = q
    return int(torch.bucketize(unemb_norm[t], _DEC_CACHE[key]).item())


def norm_pool(unemb_norm, exclude: set[int] | None = None):
    """Bucket all vocab ids by unembedding-norm decile, for arm D sampling."""
    exclude = exclude or set()
    pool = {}
    for t in range(len(unemb_norm)):
        if t in exclude:
            continue
        pool.setdefault(_decile(unemb_norm, t), []).append(t)
    return pool
