# Fragility: context-dependent copy failure on ordinary tokens, and whether geometry predicts it

## Why this object

The non-contiguous interaction screen (`findings_interaction.md`) showed that
copy failures among individually-clean tokens are not pairwise: they factor as
a fragile token × a hostile surrounding context, and which contexts are hostile
does not reproduce across filler sequences. Every existing glitch detector — and
the standard single probe in this repo — measures a token in **one** context. A
token that copies perfectly there and fails in 15% of ordinary contexts is
invisible to all of them.

So: a fixed bank of 48 random contexts (common clean filler words; lengths
8/16/32/64 × 12; random interior slot), 2,098 tokens (1,998 id-stratified random
+ 100 Magikarp-verified glitch as reference) placed in every context, cell =
teacher-forced copy logprob at the slot. N×C, not N². Model
`allenai/OLMo-2-1124-7B-Instruct`. Code: `src/cut/fragility.py`,
`fragility_predict.py`, `fragility_diag.py`; results `results/fragility*.json`,
`results/fragility.pt` (3.4 GB incl. slot hidden states).

## 1. Context dependence, quantified

Two-way decomposition `lp(t,c) = μ + token(t) + context(c) + residual(t,c)`:

| population / measure | token | context | **token × context** |
|---|---|---|---|
| all tokens, logprob | 0.952 | 0.001 | 0.047 |
| all tokens, fail indicator | 0.743 | 0.004 | 0.254 |
| clean-looking tokens, logprob | 0.717 | **0.000** | **0.282** |

**No context is systematically hostile** (context share ≈ 0; hostility ranges
only 0.10–0.19 across the 48 contexts; clean-looking fail rate is flat across
lengths 8→64: 0.005–0.008). The dependence is entirely **token-specific**: a
given token fails in *its* particular contexts. Among clean-looking tokens that
residual is 28% of variance — with the caveat that a sparse matrix of rare large
drops inflates a residual share, so this bounds rather than measures structure.

The cell-level test agrees: predicting `fail(t,c)` on held-out contexts,
token fragility (from the other contexts) gives AUC 0.940, context hostility
(from other tokens) 0.559, and neither static token–context similarity (0.720
alone; **adds nothing** over main effects, 0.942 → 0.930) nor per-cell identity
retention (0.340) predicts *which* context breaks a token.

## 2. How common is fragility?

`frag(t)` = fraction of 48 contexts with slot logprob < −0.5 (p < 0.61).

| population | n | mean frag | frag ≥ 0.10 | frag ≥ 0.25 | never fails |
|---|---|---|---|---|---|
| random tokens | 1998 | 0.090 | 17.5% | 11.9% | 67.0% |
| verified glitch (reference) | 100 | **0.993** | 100% | 100% | 0% |
| **clean-looking** (single probe > −0.1) | 873 | 0.007 | **1.6%** | 0.8% | 92.1% |

The scorer is right (glitch tokens fail everywhere; none of the 100 passed the
single probe). The hazard is real but narrow: about **1 in 60 tokens that pass
the standard probe fails in ≥10% of ordinary contexts**; 1 in 12 fails at least
once in 48. Fragility is a stable token property — fragility on even-numbered
contexts predicts odd-numbered contexts at AUC 0.996.

## 3. The specimens — and the failure modes

All 14 fragile clean-looking tokens are **canonical** (decode→encode
round-trips to the same id), NFC-normal, Magikarp category `OK`. The
duplicate-encoding explanation is dead; these are ordinary reachable tokens.

| token | id | single probe | frag | worst context → model emits (p) | best context |
|---|---|---|---|---|---|
| `' according'` | 4184 | −0.001 | **0.92** | `…deliver rel behind [·] sent floor` → **`' sent'` (1.000)** — word deleted, copy continues | 0.999 |
| `' pueden'` | 41604 | −0.030 | 0.62 | `…became financial contains [·]` → `' pena'` (0.775) | 0.992 |
| `' abbiamo'` | 95396 | −0.020 | 0.60 | `…nullptr park popular [·]` → `' abdom'`→"abdominoplasty" | 0.972 |
| `' sólo'` | 53288 | −0.032 | 0.48 | `…vot [·] ext gr` → `' só'` (0.892), truncated | 0.957 |
| `' getSystemService'` | 92476 | −0.053 | 0.38 | `…hope where sales [·]` → `'SystemService'` (0.30) | 0.961 |
| `' será'` | 33998 | −0.048 | 0.29 | `…includ music inter [·]` → `' sé'`+`'ra'` = "séra" | 0.994 |

Three modes, all confident (0.6–1.0), all reversed by changing the filler words:
**deletion** (the token is skipped and the copy proceeds from the next word),
**substitution** with an orthographic or same-language neighbour, and
**truncation** to a sub-token. These are the reasoning-mode behaviours seen all
day (`' Hexatrigesimal'`→"Calculus", `'ance'+'led'`→"cancelled") now
reproduced on tokens that every detector would pass, in a copy task, with the
switch being nothing but the surrounding ordinary words.

`' according'` is the striking one: a common English word, low id, perfect in
isolation and in 4 of 48 contexts, deleted with probability 1.000 in the other
44. The composition of the fragile set otherwise is the rare tail of an
English-dominant instruct mix — Spanish, Italian, Portuguese, German, Chinese,
Russian words and Android/Java identifiers.

## 4. Is fragility predictable from geometry?

Target: fragile (`frag_test ≥ 0.10` on the held-out context half) among the 873
clean-looking tokens — **13 positives** (18 at ≥ 0.05). That number governs
everything below.

Single features (AUC, bootstrap 95% CI), clean-looking tokens:

| tier | feature | AUC | CI |
|---|---|---|---|
| surface | id (rarity) | 0.614 | — |
| surface | ascii | 0.649 | — |
| static | **cdist_out** (unembedding row → centroid) | **0.752** | [0.57, 0.90] |
| static | **dp_self** (E_in[t]·E_out[t], LN-folded) | **0.755** | [0.65, 0.85] |
| static | norm_in, cdist_in | 0.56 | — |
| static | nn_cos, knn density, inout_cos, glitch_dir | 0.65–0.73 (reversed) | wide |
| dynamic | text_norm_mean (slot state norm, train contexts) | 0.857 | [0.70, 0.96] |
| dynamic | identity_retention_text / copy | 0.71 / 0.67 | [0.49, 0.90] |
| behaviour | frag on train half | 0.996 | [0.99, 1.00] |

Cross-validated L2 logistic (5-fold out-of-fold AUC):

| tier | frag ≥ 0.10 (13 pos) | frag ≥ 0.05 (18 pos) |
|---|---|---|
| surface only | 0.768 [0.62, 0.90] | 0.751 [0.61, 0.88] |
| **static geometry** | **0.845 [0.67, 0.96]** | **0.838 [0.70, 0.94]** |
| dynamic (train-half representation) | 0.934 [0.88, 0.98] | 0.892 [0.81, 0.96] |
| behavioural (train-half copy) | 0.994 | 0.905 |

**Pre-registered bar** (CV AUC ≥ 0.65, CI excludes 0.5, beats surface): static
geometry meets it on point estimates — and I do not consider the "beats surface"
half established. The static and surface CIs overlap heavily at 13–18 positives,
and the same feature (cdist_out) swings 0.752 → 0.668 when the target is
redefined from the test half to all contexts. Against the looser "any failure"
target (69 positives) the single output-row features fall to 0.53–0.69.

What the geometry says, stated at its actual strength: **fragility is
moderately predictable with no forward pass (~0.84), and the signal lives in the
output embedding** — how far the token's unembedding row sits from the centroid
and how weakly the input row excites its own readout. Those are rarity proxies
with a mechanism attached (an output row that is seldom the target moves less),
and they are not clearly better than knowing the token is a rare non-English
word. The evolving representation on a handful of contexts does better (~0.9),
and simply measuring copy on a handful of contexts does best (~0.9–1.0).

## 5. What this establishes

1. **Context-dependent glitch behaviour on ordinary tokens is real, isolated,
   and quantified.** Not pairwise, not a property of hostile contexts — a
   token-specific sensitivity that flips confident copying to confident
   deletion/substitution/truncation depending on nothing but surrounding words.
   ~1.6% of tokens that pass the standard probe carry it at ≥10% of contexts.
2. **It is a stable token property** (0.996 across context halves), so it is
   measurable once and cheaply: N×C forward passes with C ≈ 24 suffices.
3. **Static geometry predicts it moderately (CV AUC ~0.84) via the output
   embedding**, but with 13–18 positives the advantage over surface rarity is
   not established. Doubling the token count would settle it; the machinery is
   built.
4. Which *context* breaks a given token is **not** predictable from
   token–context similarity or per-cell representation — only from the token's
   own fragility.

## Replication, the geometry verdict, and the hallucination bridge (second run)

**Replication bank**: 6,273 tokens (4,200 id-stratified random + 1,973 constituent
tokens of the real entities below + 100 verified glitch) × 24 new contexts
(`results/fragility_L.*`). Everything reproduces: context share of variance 0.001;
clean-looking (n = 1,888) frag ≥ 0.10 = **1.5%** (28 tokens), never-fail 95.0%;
glitch reference 0.987; `' according'` frag 0.92, mean lp −13.3. New specimens:
`' due'` (single −0.000, frag 0.42, worst −23.8 — same deletion signature; both
`according` and `due` are "X to" words, as are several fragile connectives:
`mentre`, `allerdings`, `może`, `который` — untested hypothesis: fragility tracks
the peakedness of the token's continuation prior); a **translation** mode
(`' 查询'` → `' QUERY'` p = 0.970; `' который'` → `' что'`).

**Geometry — established, and corrected upward.** 28 positives. A bug found
during the scale ladder (the `glitch_dir` feature indexed E_in by list
position, not vocab id) was fixed and the replication re-scored
(`results/fragility_predict_L_fixed.json`): CV AUC surface 0.711 [0.58, 0.83];
static **0.911 [0.86, 0.95]** (was 0.862); dynamic 0.823 [0.72, 0.92];
static+dynamic **0.930 [0.89, 0.96]**; behavioural 0.996. **Paired bootstrap vs
surface**: static **+0.202 [+0.096, +0.320]**; dynamic +0.112 [+0.024, +0.216];
static+dynamic +0.218 [+0.110, +0.334]. All exclude zero. The corrected glitch
direction is the strongest single static feature; the others remain modest
(cdist_out 0.678, dp_self 0.655). Dynamic slot-state norm alone: 0.864.

**Hallucination bridge — negative.** 1,255 real entities (1,100 stdlib/torch/
transformers identifiers, 155 proper nouns), natural-use prompts, scored on the
entity's final component (a first scorer counted `np.` aliases and from-imports
as near-misses — inflated to 23.5%; rescored with `bridge_rescore.py`). Code:
exact 82.5%, near-miss **1.7%** (19), miss 15.7% (all 173 are bare `import`
lines — avoidance, not hallucination); nouns exact 99.4%. Adding constituent
fragility to rarity + single-probe + familiarity: **−0.001 [−0.008, +0.006]**
(any failure, 193 pos); −0.001 [−0.042, +0.042] (near-miss, 19 pos — too few for
anything; controls themselves 0.59 [0.46, 0.73]). Fragility alone 0.55–0.57.
Genuine near-misses exist (`curses.BUTTON5_PRESSED` → `C_BUTTON5_PRESSED`,
`transformers.EncodecModel` → `EncodeModel`, `torch.BoolTensor` → `bool_tensor`)
but token fragility does not predict them. The §3 hallucination framing is an
analogy, not a mechanism.

**Specimen store**: `results/specimens_confident_substitution.jsonl` (228
records: token, context, top-5 emissions with p, entropy, p(target), mode label,
greedy continuation) with a model header (commit `470b1fba…`, fp16, 32 layers,
4096 hidden, vocab 100352, untied, torch 2.7.0, transformers 5.16.1);
human-readable `docs/specimens.md`. The `ret` cosine column there is
uninformative as computed; `p(target)` is the identity measure.

## Honest ledger for this line of work

Falsified today, each by a control run after the claim: BPE-unreachable
adjacency; long-generation entropy amplification; a shared attractor basin;
direct-path substitution prediction; orthographic completion as the general
mechanism; trajectory homology (underpowered null); stage-relative abandonment
across families; non-contiguous pairwise interaction (both pools, pre-registered);
and the Unicode-variant explanation of fragility. What survived is §5 above.
