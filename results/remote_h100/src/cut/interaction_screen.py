"""Do NON-CONTIGUOUS subsets of a context cause glitch-like behaviour, and how
does that depend on how far apart they are?

Every published detector, and every earlier experiment in this repo, scores a
token in isolation or tests ADJACENT pairs and CONTIGUOUS n-grams. None of that
addresses the actual claim, which is that a subset of a context -- far apart,
individually healthy -- jointly breaks the model:

    [1, 231, 885, 9911, 1112, 231]         the REPETITION of 231 at distance
    [1, 231, 885, 9911, 9999, 1922, 1013]  885 and 1013 CO-OCCURRING at distance

DISTANCE IS THE VARIABLE. Two long neutral filler sequences; for each distance d
the carrier is filler[:d+4] with slots i=2, j=2+d. Carriers at different
distances share a prefix, so distance is not confounded with filler content, and
two fillers let the residual filler effect be checked. The window is limited only
by compute (cost is linear in d; context is 4096), so --distances is a flag.

For an ordered pair (a, b) at slots (i, j), per replicate r = (filler, d):

    S(00)  c1 at i, c2 at j            controls in both slots
    S(10)  a  at i, c2 at j
    S(01)  c1 at i, b  at j
    S(11)  a  at i, b  at j
    I(a,b) = S11 - S10 - S01 + S00     damage NOT attributable to either alone

The "absent" condition is a substituted control token, not a deletion, so the
length and every other position are held fixed. S(00) uses two DIFFERENT
controls -- a repeated control would itself be a repetition cell. The diagonal
a == b is the repetition case; its two single cells are kept separate.

S is teacher-forced copy logprob (the ngram_glitch.copy_lp construction). The
PRIMARY S is the mean over the two SLOT positions in the copy span -- does the
pair fail to copy itself? Over a 130-token carrier a whole-context mean would
dilute a two-position effect ~65x. Whole-context and filler-only means are kept
as secondary scores; all three are fixed linear functionals of the per-position
vector, so the 2x2 contrast cancels main effects identically for each.

The pool is INDIVIDUALLY CLEAN: only tokens the model reproduces perfectly
alone, so any interaction cannot be inherited from a member.

THE NULL. N^2 ordered pairs is ~4000 simultaneous tests and the interaction is a
difference of four noisy cells. The test statistic is the additive-model
residual, standardised per row/column (see stats.perm_null_additive for why),
permuted within replicate, BH-corrected. The VERDICT is not a fixed threshold:
20 no-interaction datasets are simulated from the additive fit to the real data
with its real heteroscedastic noise and pushed through the identical pipeline;
"detected" means more rejections than the 95th percentile of those AND a
family-wise max-statistic p < 0.05. A live power estimate (recovery of an
injected effect of known size, in units of real cell noise) is printed so a
null can be read as "null at this size", not "null".

Hidden states at the slot positions (layers L/2 and L) are recorded in the same
forward pass for interaction_geometry.py. With causal attention slot i in the
Text span cannot see slot j, so the recorded states are slot j in the Text span
and both slots in the Copy span.
"""
from __future__ import annotations
import argparse, gzip, json, random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ngram_identify import reproduce
from .stagewise import copy_logprob_ids
from .stats import interaction_test, additive_fit, simulate_additive
from .loadmodel import add_model_args, load_from_args, describe

FEWSHOT = [" apple pie is good", " the quick brown fox"]
CONTROL_WORDS = [(" table", " green"), (" music", " seven"), (" window", " paper")]
STATE_POS = ("text_j", "copy_i", "copy_j")
BANDS = {"short": (2, 4, 8), "mid": (16, 32), "long": (64, 128, 256, 512)}


def load_verified(path, n):
    ver = np.zeros(n, int); tested = np.zeros(n, bool)
    for line in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(line); i = int(r["i"])
        if i < n and "magikarp" in r:
            tested[i] = True
            if "verified" in r["magikarp"]:
                ver[i] = 1
    return ver, tested


def single_id(tok, w):
    ids = tok(w, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise SystemExit(f"control word {w!r} is not a single token: {ids}")
    return ids[0]


def common_word_ids(tok, V, limit=2000):
    """Low-id tokens of the form ' lowercaseword' -- in BPE the earliest merges
    are the most frequent pieces, so these are the common English words."""
    out = []
    for i in range(V):
        s = tok.decode([i])
        if len(s) >= 3 and s[0] == " " and s[1:].isalpha() and s[1:].islower():
            out.append(i)
            if len(out) >= limit:
                break
    return out


@torch.no_grad()
def score_cells(model, tok, dev, ctxs, slot, layers, batch=16):
    """All ctxs have the same length K (one replicate), so every built sequence
    has identical length and no padding is needed. Returns per-position copy
    logprob (n, K) and slot hidden states."""
    e = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    head = e("Repeat the text exactly.\n")
    for d in FEWSHOT:
        head += e("Text:") + e(d) + e("\nCopy:") + e(d) + e("\n")
    pre_t, pre_c = e("Text:"), e("\nCopy:")
    K = len(ctxs[0])
    text_start = len(head) + len(pre_t)
    n_total = text_start + K + len(pre_c) + K
    copy_start = n_total - K
    i, j = slot
    lp_out = np.zeros((len(ctxs), K), np.float32)
    states = {f"{nm}_{pos}": [] for nm in layers for pos in STATE_POS}
    for s in range(0, len(ctxs), batch):
        ch = ctxs[s:s + batch]
        ids = torch.tensor([head + pre_t + list(c) + pre_c + list(c) for c in ch],
                           device=dev)
        out = model(input_ids=ids, output_hidden_states=True)
        lp = torch.log_softmax(out.logits[:, copy_start - 1:n_total - 1, :].float(), -1)
        tgt = ids[:, copy_start:n_total]
        lp_out[s:s + len(ch)] = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        for nm, l in layers.items():
            H = out.hidden_states[l]
            states[f"{nm}_text_j"].append(H[:, text_start + j].half().cpu())
            states[f"{nm}_copy_i"].append(H[:, copy_start + i].half().cpu())
            states[f"{nm}_copy_j"].append(H[:, copy_start + j].half().cpu())
    return lp_out, {k: torch.cat(v) for k, v in states.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    ap.add_argument("--ext", default="external/allenai_OLMo_2_1124_7B.jsonl.gz")
    ap.add_argument("--n", type=int, default=64, help="pool size")
    ap.add_argument("--candidates", type=int, default=600)
    ap.add_argument("--pool-mode", choices=["clean", "mixed"], default="clean",
                    help="'clean': only tokens reproduced perfectly alone (the first "
                         "run showed this pool sits at ceiling, p~0.98, leaving no room "
                         "for a combination to hurt). 'mixed': half clean, half WEAK -- "
                         "single-token copy logprob in (--weak-lo, --weak-hi), tokens that "
                         "copy sometimes -- so a combination can push them over the edge.")
    ap.add_argument("--weak-lo", type=float, default=-3.0)
    ap.add_argument("--weak-hi", type=float, default=-0.3)
    ap.add_argument("--distances", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 128],
                    help="slot separations; limited only by compute and context")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--n-sim", type=int, default=20, help="matched-null simulations")
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/interaction_screen.json")
    ap.add_argument("--pt", default="results/interaction_screen.pt")
    add_model_args(ap)
    a = ap.parse_args()

    model, tok, dev = load_from_args(a)
    V = model.get_input_embeddings().weight.shape[0]
    L = model.config.num_hidden_layers
    layers = {"mid": L // 2, "last": L}
    if a.ext and Path(a.ext).exists():
        ver, tested = load_verified(a.ext, V)
    else:
        print(f"  no label file ({a.ext}); no verified-glitch positive controls this run")
        ver, tested = np.zeros(V, int), np.zeros(V, bool)
    rng = random.Random(a.seed)
    special = set(tok.all_special_ids)
    maxd = max(a.distances)
    batch = a.batch or 16

    # ---------- controls ----------
    ctrl_pairs = [(single_id(tok, c1), single_id(tok, c2)) for c1, c2 in CONTROL_WORDS]
    ctrl_ids = {t for p in ctrl_pairs for t in p}

    # ---------- individually-clean pool ----------
    cand = []
    for t in rng.sample(range(V), min(a.candidates * 2, V)):
        s = tok.decode([t])
        if t in ctrl_ids or t in special or not s.strip() or not s.isprintable():
            continue
        cand.append(t)
        if len(cand) >= a.candidates:
            break
    print(f"{a.model}: {len(cand)} candidates -> single-token reproduction filter...",
          flush=True)
    ex, _, _ = reproduce(model, tok, dev, [[t] for t in cand])
    clean = [t for t, ok in zip(cand, ex) if ok]
    rng.shuffle(clean)
    if a.pool_mode == "mixed":
        lp1 = copy_logprob_ids(model, tok, dev, np.array(cand))
        lpmap = dict(zip(cand, lp1))
        strong = [t for t in clean if lpmap[t] > -0.05]
        weak = [t for t in cand if a.weak_lo < lpmap[t] < a.weak_hi]
        rng.shuffle(weak)
        half = a.n // 2
        if len(weak) < half:
            print(f"  WARNING: only {len(weak)} weak tokens in ({a.weak_lo}, {a.weak_hi}); "
                  f"pool will be {len(weak)} weak + {a.n - len(weak)} clean")
        wsel = weak[:half]
        pool = sorted(wsel + strong[:a.n - len(wsel)])
        kind = {t: ("weak" if t in set(wsel) else "clean") for t in pool}
        print(f"  individually clean: {len(clean)}/{len(cand)}; weak (copy lp in "
              f"({a.weak_lo},{a.weak_hi})): {len(weak)}; pool N={len(pool)} = "
              f"{len(wsel)} weak + {len(pool) - len(wsel)} clean")
        print("  weak sample:", [f"{tok.decode([t])!r}:{lpmap[t]:.2f}" for t in wsel[:6]])
    else:
        pool = sorted(clean[:a.n])
        kind = {t: "clean" for t in pool}
        print(f"  individually clean: {len(clean)}/{len(cand)}; pool N={len(pool)}")
    N = len(pool); pool_set = set(pool)
    print("  sample:", [repr(tok.decode([t]))[:14] for t in pool[:8]])

    # ---------- filler: common clean words, three disjoint sequences ----------
    need = 3 * (maxd + 4)
    cw = [t for t in common_word_ids(tok, V) if t not in pool_set and t not in ctrl_ids]
    exw, _, _ = reproduce(model, tok, dev, [[t] for t in cw])
    fill = [t for t, ok in zip(cw, exw) if ok]
    if len(fill) < need:
        raise SystemExit(f"only {len(fill)} clean common words, need {need} for "
                         f"3 fillers of length {maxd + 4}; lower --distances")
    rng.shuffle(fill)
    F = [fill[k * (maxd + 4):(k + 1) * (maxd + 4)] for k in range(3)]
    print(f"  filler: {len(fill)} clean common words -> 3 sequences of {maxd + 4}")
    print("  F1 starts:", "".join(tok.decode([t]) for t in F[0][:8]))

    # positive control for the scorer: known-bad tokens, single cells only
    glitch = [int(i) for i in np.where(ver == 1)[0]
              if tok.decode([int(i)]).strip() and int(i) not in pool_set]
    rng.shuffle(glitch)
    pos_ctrl = glitch[:3]
    print("  positive-control glitch tokens:", [repr(tok.decode([t])) for t in pos_ctrl])

    # ---------- replicates and cells ----------
    reps = [(f, d) for d in a.distances for f in (0, 1)]
    R, C = len(reps), len(ctrl_pairs)
    dists = [d for _, d in reps]
    meta, ctxs, rep_of = [], [], []

    def add(kind, ctx, r, **kw):
        meta.append(dict(kind=kind, r=r, **kw)); ctxs.append(ctx); rep_of.append(r)

    for r, (f, d) in enumerate(reps):
        base = F[f][:d + 4]; i, j = 2, 2 + d
        for cp, (c1, c2) in enumerate(ctrl_pairs):
            c = list(base); c[i] = c1; c[j] = c2; add("00", c, r, cp=cp)
            for ai, t in enumerate(pool):
                c = list(base); c[i] = t; c[j] = c2; add("10", c, r, cp=cp, a=ai)
                c = list(base); c[i] = c1; c[j] = t; add("01", c, r, cp=cp, b=ai)
            for g in pos_ctrl:
                c = list(base); c[i] = g; c[j] = c2; add("pos10", c, r, cp=cp, g=g)
                c = list(base); c[i] = c1; c[j] = g; add("pos01", c, r, cp=cp, g=g)
        for ai, ta in enumerate(pool):
            for bi, tb in enumerate(pool):
                c = list(base); c[i] = ta; c[j] = tb; add("11", c, r, a=ai, b=bi)
    n_cells = len(ctxs)
    print(f"  replicates R={R} (distances {a.distances} x 2 fillers), cells {n_cells}",
          flush=True)

    # ---------- score, one replicate (= one length) at a time ----------
    S_slot = np.zeros(n_cells, np.float32); S_full = np.zeros(n_cells, np.float32)
    S_fill = np.zeros(n_cells, np.float32); lp_all = [None] * n_cells
    states_all = {f"{nm}_{pos}": [None] * n_cells for nm in layers for pos in STATE_POS}
    rep_of = np.array(rep_of)
    for r, (f, d) in enumerate(reps):
        idx = np.where(rep_of == r)[0]
        i, j = 2, 2 + d
        print(f"  scoring replicate {r + 1}/{R}  d={d} filler={f}  ({len(idx)} cells)",
              flush=True)
        lp, st = score_cells(model, tok, dev, [ctxs[k] for k in idx], (i, j), layers,
                             batch=batch)
        K = lp.shape[1]; keep = [p for p in range(K) if p not in (i, j)]
        S_slot[idx] = lp[:, [i, j]].mean(1)
        S_full[idx] = lp.mean(1)
        S_fill[idx] = lp[:, keep].mean(1)
        for n_, k in enumerate(idx):
            lp_all[k] = lp[n_]
            for key in states_all:
                states_all[key][k] = st[key][n_]
    states_all = {k: torch.stack(v) for k, v in states_all.items()}

    def tables(S):
        S00 = np.zeros((R, C)); S10 = np.zeros((N, R, C)); S01 = np.zeros((N, R, C))
        S11 = np.zeros((N, N, R)); pos = {"10": {}, "01": {}}
        for k, m in enumerate(meta):
            kd = m["kind"]
            if kd == "00": S00[m["r"], m["cp"]] = S[k]
            elif kd == "10": S10[m["a"], m["r"], m["cp"]] = S[k]
            elif kd == "01": S01[m["b"], m["r"], m["cp"]] = S[k]
            elif kd == "11": S11[m["a"], m["b"], m["r"]] = S[k]
            else: pos[kd[3:]].setdefault(m["g"], []).append(S[k])
        I_r = (S11 - S10.mean(2)[:, None, :] - S01.mean(2)[None, :, :]
               + S00.mean(1)[None, None, :])
        return S00, S10, S01, S11, I_r, pos

    S00, S10, S01, S11, I_r, pos = tables(S_slot)
    _, _, _, _, I_full_r, _ = tables(S_full)
    _, _, _, _, I_fill_r, _ = tables(S_fill)
    I = I_r.mean(2); off = ~np.eye(N, dtype=bool)

    print()
    print("=" * 78)
    print("SCORER SANITY -- positive control (slot-position copy logprob)")
    print("=" * 78)
    print(f"  clean pool singles: S10 {S10.mean():.3f}  S01 {S01.mean():.3f}   "
          f"controls S00 {S00.mean():.3f} (replicate sd {S00.std():.3f})")
    for g in pos_ctrl:
        print(f"  glitch {tok.decode([g])!r:>18}: S10 {np.mean(pos['10'][g]):.3f}   "
              f"S01 {np.mean(pos['01'][g]):.3f}")
    scorer_ok = (all(np.mean(pos["10"][g]) < S10.mean() - 2 * S10.std() for g in pos_ctrl)
                 if pos_ctrl else None)
    print(f"  -> positive controls fall >2 sd below the clean pool: {scorer_ok}"
          + ("  (no labelled controls this run)" if not pos_ctrl else ""))

    # ---------- pooled test ----------
    M = np.transpose(S11, (2, 0, 1))                        # (R, N, N)
    T = interaction_test(M, n_perm=a.n_perm, q=a.q, seed=a.seed)
    n_sig_off = int(T["rej"][off].sum()); n_sig_diag = int(T["rej"][~off].sum())
    corr = float(np.corrcoef(I[off], T["obs"][off])[0, 1])

    # ---------- matched-null simulation -> empirical verdict threshold ----------
    print("\n  simulating matched no-interaction datasets for the verdict threshold...",
          flush=True)
    mu, al, be, rs, cs = additive_fit(M)
    srng = np.random.default_rng(a.seed + 7)
    sim_counts, sim_fwer = [], []
    for k in range(a.n_sim):
        Ms = simulate_additive(mu, al, be, rs, cs, srng)
        Ts = interaction_test(Ms, n_perm=300, q=a.q, seed=1000 + k)
        sim_counts.append(int(Ts["rej"][off].sum())); sim_fwer.append(Ts["fwer_p"])
    n_null95 = float(np.percentile(sim_counts, 95))
    # live power: inject ONE known effect in units of the median cell noise. A
    # single true positive among ~4000 tests is the Bonferroni-like worst case
    # for BH; if many pairs truly interact the threshold relaxes and power is
    # far higher, so this is a floor, not the screen's sensitivity in general.
    cell_sd = float(np.median(np.sqrt(0.5 * (rs[:, None] ** 2 + cs[None, :] ** 2))))
    power = {}
    for mult in (1.0, 1.5, 2.0, 3.0):
        hit = 0
        for k in range(5):
            Ms = simulate_additive(mu, al, be, rs, cs, srng, inject=(3, 17, -mult * cell_sd))
            Ts = interaction_test(Ms, n_perm=300, q=a.q, seed=2000 + k)
            hit += int(Ts["rej"][3, 17])
        power[mult] = hit / 5

    print()
    print("=" * 78)
    print("INTERACTION SCREEN  (pooled over all replicates)")
    print("=" * 78)
    print(f"  pool N={N}, replicates R={R}, control pairs {C}, permutations {a.n_perm}")
    print(f"  corr(control-based I, additive residual): {corr:.3f}")
    print(f"  I off-diagonal: mean {I[off].mean():+.4f}  sd {I[off].std():.4f}    "
          f"diagonal (repetition): mean {I[~off].mean():+.4f}  sd {I[~off].std():.4f}")
    print(f"  median cell noise sd {cell_sd:.4f} logprob")
    print(f"\n  BH-FDR q={a.q}: {n_sig_off} significant off-diagonal of {off.sum()}, "
          f"{n_sig_diag} diagonal of {N}")
    print(f"  matched-null simulations ({a.n_sim}): rejections {sorted(sim_counts)}  "
          f"-> 95th pct {n_null95:.0f}")
    print(f"  family-wise: max|z| {T['max_obs']:.3f} vs 95% threshold {T['fwer_thr']:.3f}"
          f"   p = {T['fwer_p']:.3f}")
    print(f"  power (recovery of an injected effect, 5 sims each): "
          + "  ".join(f"{m:.1f}x noise -> {p:.1f}" for m, p in power.items()))
    diag_mean = float(T["z"][~off].mean())
    diag_p = float((np.abs(T["diag_null_means"]) >= abs(diag_mean)).mean())
    print(f"\n  repetition diagonal as a group: mean z {diag_mean:+.3f}, "
          f"mean I {I[~off].mean():+.4f}, permutation p {diag_p:.3f}")

    # ---------- by distance ----------
    print()
    print("=" * 78)
    print("INTERACTION BY DISTANCE")
    print("=" * 78)
    print(f"{'d':>5} | {'I off-diag mean':>15} {'sd':>8} | {'I repetition mean':>17} "
          f"{'sd':>8} | {'filler agreement':>16}")
    print("-" * 82)
    by_d = {}
    for d in a.distances:
        rr = [r for r, dd in enumerate(dists) if dd == d]
        Id = I_r[:, :, rr].mean(2)
        fa = float(np.corrcoef(I_r[:, :, rr[0]][off], I_r[:, :, rr[1]][off])[0, 1]) \
            if len(rr) == 2 else float("nan")
        by_d[d] = Id
        print(f"{d:5d} | {Id[off].mean():+15.4f} {Id[off].std():8.4f} | "
              f"{Id[~off].mean():+17.4f} {Id[~off].std():8.4f} | {fa:16.3f}")

    print("\n  banded tests (each band its own permutation null):")
    print(f"{'band':>6} | {'R':>3} | {'sig off':>7} | {'sig diag':>8} | {'fwer p':>7}")
    print("-" * 46)
    band_res = {}
    for bname, bds in BANDS.items():
        rr = [r for r, dd in enumerate(dists) if dd in bds]
        if len(rr) < 2:
            continue
        Tb = interaction_test(M[rr], n_perm=500, q=a.q, seed=a.seed + 3)
        band_res[bname] = {"R": len(rr), "sig_off": int(Tb["rej"][off].sum()),
                           "sig_diag": int(Tb["rej"][~off].sum()), "fwer_p": Tb["fwer_p"],
                           "rej": Tb["rej"]}
        print(f"{bname:>6} | {len(rr):3d} | {band_res[bname]['sig_off']:7d} | "
              f"{band_res[bname]['sig_diag']:8d} | {Tb['fwer_p']:7.3f}")

    # ---------- verdict ----------
    detected = (n_sig_off > n_null95) and (T["fwer_p"] < 0.05)
    print()
    print("=" * 78)
    print("PRE-REGISTERED VERDICT")
    print("=" * 78)
    if detected:
        print(f"  DETECTED: {n_sig_off} significant pairs > matched-null 95th pct "
              f"{n_null95:.0f}, family-wise p {T['fwer_p']:.3f} < 0.05")
    else:
        print(f"  NO detectable non-contiguous interaction at this power:")
        print(f"    significant pairs {n_sig_off} vs matched-null 95th pct {n_null95:.0f};"
              f"  family-wise p {T['fwer_p']:.3f}")
        print(f"    power: effects of {[m for m, p in power.items() if p >= 0.8]}x cell "
              f"noise ({cell_sd:.3f} logprob) would have been recovered")

    dec = [tok.decode([t]) for t in pool]
    order = np.argsort(I.ravel())
    print("\n  most negative I (pair worse than its parts):")
    for k in order[:10]:
        ai, bi = divmod(int(k), N)
        bd = min(by_d, key=lambda d: by_d[d][ai, bi])
        print(f"    {dec[ai]!r:>14} @i + {dec[bi]!r:<14} @j  I={I[ai,bi]:+.4f}  "
              f"p_adj={T['padj'][ai,bi]:.3f}{' *' if T['rej'][ai,bi] else ''}"
              f"  worst d={bd}{'  (repeat)' if ai == bi else ''}")

    # ---------- fragility: the structure the first run actually had ----------
    # Individually-clean tokens still failed to copy in ~1% of cells, but the
    # failures concentrated in a handful of tokens and did not recur when the
    # filler changed. That is a token main effect times a context effect -- the
    # additive model absorbs it, and it is not a pair interaction. Report it so
    # the null is read correctly.
    kind_arr = np.array([kind[t] for t in pool])
    bad = S11 < -0.5
    row_bad = bad.sum((1, 2)); col_bad = bad.sum((0, 2))
    print()
    print("=" * 78)
    print("FRAGILITY -- tokens that fail to copy in SOME contexts (S11 slot lp < -0.5)")
    print("=" * 78)
    print(f"  failing cells: {int(bad.sum())} of {bad.size} ({bad.mean():.4f})")
    top = np.argsort(-(row_bad + col_bad))[:8]
    for t in top:
        print(f"    {dec[t]!r:>16}  as a: {row_bad[t]:4d}   as b: {col_bad[t]:4d}   "
              f"({kind_arr[t]})")
    share = (row_bad[top[:5]].sum() + col_bad[top[:5]].sum()) / max(1, 2 * bad.sum())
    print(f"  share of failing cells in the 5 most fragile tokens: {share:.3f}")
    ov = []
    for dd in a.distances:
        rr = [r for r, x in enumerate(dists) if x == dd]
        if len(rr) == 2:
            A_, B_ = bad[:, :, rr[0]], bad[:, :, rr[1]]
            ov.append((dd, int(A_.sum()), int(B_.sum()), int((A_ & B_).sum())))
    print("  same-d failures across fillers (d, f0, f1, both): "
          + "  ".join(str(x) for x in ov))
    celltype = {}
    if a.pool_mode == "mixed":
        print("\n  cell-type breakdown (a-kind x b-kind):")
        print(f"    {'type':>13} | {'n':>5} | {'mean I':>8} | {'mean S11':>8} | {'sig':>4}")
        for ka in ("clean", "weak"):
            for kb in ("clean", "weak"):
                m = np.outer(kind_arr == ka, kind_arr == kb) & off
                if m.sum() == 0:
                    continue
                celltype[f"{ka}x{kb}"] = {
                    "n": int(m.sum()), "mean_I": float(I[m].mean()),
                    "mean_S11": float(S11.mean(2)[m].mean()),
                    "n_sig": int(T["rej"][m].sum())}
                c_ = celltype[f"{ka}x{kb}"]
                print(f"    {ka + ' x ' + kb:>13} | {c_['n']:5d} | {c_['mean_I']:+8.4f} | "
                      f"{c_['mean_S11']:+8.3f} | {c_['n_sig']:4d}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "rep_of": rep_of, "reps": reps, "dists": dists,
                "kind": list(kind_arr),
                "lp": lp_all, "states": states_all, "layers": layers,
                "pool": pool, "ctrl_pairs": ctrl_pairs, "fillers": F,
                "S_slot": S_slot, "S_full": S_full, "S_fill": S_fill,
                "S00": S00, "S10": S10, "S01": S01, "S11": S11,
                "I_r": I_r, "I": I, "I_full": I_full_r.mean(2), "I_fill": I_fill_r.mean(2),
                "z": T["z"], "p": T["p"], "padj": T["padj"], "rej": T["rej"],
                "band_rej": {b: v["rej"] for b, v in band_res.items()}}, a.pt)
    json.dump({"model": a.model, "model_info": describe(model, tok, a.model),
               "N": N, "pool": pool, "decoded": dec,
               "pool_mode": a.pool_mode, "pool_kind": list(kind_arr),
               "fragility": {"n_bad": int(bad.sum()), "frac_bad": float(bad.mean()),
                             "as_a": row_bad.tolist(), "as_b": col_bad.tolist(),
                             "top5_share": float(share), "filler_overlap": ov},
               "celltype": celltype,
               "distances": a.distances, "controls": CONTROL_WORDS,
               "filler3": F[2], "scorer_ok": bool(scorer_ok), "cell_sd": cell_sd,
               "corr_ctrl_resid": corr,
               "n_sig_offdiag": n_sig_off, "n_sig_diag": n_sig_diag,
               "sim_counts": sim_counts, "n_null95": n_null95,
               "fwer_p": T["fwer_p"], "fwer_thr": T["fwer_thr"], "max_obs": T["max_obs"],
               "power": power, "diag_mean_z": diag_mean, "diag_p": diag_p,
               "detected": bool(detected),
               "by_distance": {str(d): {"off_mean": float(v[off].mean()),
                                        "diag_mean": float(v[~off].mean())}
                               for d, v in by_d.items()},
               "I_by_dist": {str(d): v.tolist() for d, v in by_d.items()},
               "bands": {b: {k: v for k, v in r_.items() if k != "rej"}
                         for b, r_ in band_res.items()},
               "I": I.tolist(), "padj": T["padj"].tolist(),
               "rej": T["rej"].astype(int).tolist()}, open(a.out, "w"), indent=1)
    print(f"\nsaved -> {a.out}, {a.pt}")


if __name__ == "__main__":
    main()
