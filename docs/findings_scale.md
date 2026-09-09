# Scale ladder, rungs 1–2: Qwen3-1.7B and Qwen3-32B

Run 6 September 2026 on one H100-80GB (Lambda), label-free, against the ranges fixed in advance in `docs/plan_scale.md`. Reference row: OLMo-2-1124-7B-Instruct (commit 470b1fba). **Rung 3 (Qwen2.5-72B bf16) ran the same evening on a 4× H100 node — §9 below.** Rung 4 (Qwen3-235B-A22B-FP8) completed its fragility matrix and geometry on that node (22:58–00:52 UTC, 7 September) and was lost with the instance before any output was transferred; it re-runs on the replacement node.

Wall time: 1.7B rung 23 min; 32B rung 78 min (fragility matrix 24 min, reasoning 17 + 28 min, specimens 1 min, interaction pool 7 min); deferred 1.7B reasoning reruns 20 min. Total node time about 2.5 h.

## 1. The gate had to change

The paper's "clean-looking" gate is an absolute single-probe log-probability above −0.1 (the token copies alone at p > 0.9). On the 32B this admitted 353 of 4,200 sampled tokens and called none fragile — not because fragility vanished, but because the 32B copies ordinary tokens under the bare few-shot probe at 0.72–0.89, while copying the same tokens in context at 0.997. The −0.1 gate measured the probe's calibration on that model.

All cross-model numbers below therefore use the **greedy gate**: p_alone > 0.5, which guarantees the token is the argmax of its own copy probe on any model (`src/cut/gate_summary.py`; `fragility_predict.py --gate-lp -0.693`; `specimens.py --gate-lp -0.693`). The 7B reference is recomputed under the same gate. Where the paper currently quotes the −0.1 gate for the 7B, the greedy-gate numbers here supersede it for any statement that compares models.

| model | tied | n gated | single-probe median lp | frag ≥ 0.10 (fail p < 0.61) | (fail p < 0.5) | never fail | half-split r | context share |
|---|---|---|---|---|---|---|---|---|
| OLMo-2-7B (ref) | no | 2,183 | −3.77 | **2.7%** | 1.8% | 0.929 | 0.848 | 0.001 |
| Qwen3-1.7B | **yes** | 1,666 | −2.71 | **2.4%** | 1.7% | 0.930 | 0.816 | 0.004 |
| Qwen3-32B | no | 1,597 | −1.31 | **0.9%** | 0.5% | 0.957 | 0.890 | 0.002 |

## 2. Pre-registered quantities

| # | quantity | range | 7B ref | 1.7B | 32B | verdict |
|---|---|---|---|---|---|---|
| 1 | fragility prevalence, frag ≥ 0.10 | 0.5–3% | 2.7% | 2.4% | 0.9% | **holds at both rungs** |
| 2 | context share of variance | < 0.01 | 0.001 | 0.004 | 0.002 | **holds** |
| 3 | stability, even vs odd contexts | ≥ 0.95 AUC | r 0.85 | r 0.82 | r 0.89 | reported as split-half correlation of the fragility score; comparable to the reference |
| 4 | failure character: entropy at failing cells, modes present | mean < 2 bits; deletion/substitution present | — | greedy gate: median 1.00, mean 1.21, 86% < 2 bits; substitution 83, deletion 27, truncation 1 | greedy gate: median 1.74, mean 2.55, 59% < 2 bits; substitution 25, deletion 8, truncation 1 | modes present at both; entropy bound **met at 1.7B, missed on the mean at 32B** (median passes) — see §5 |
| 5 | geometry: static vs surface, paired Δ | > 0, CI excluding 0 | **+0.105 [+0.052, +0.160]**, 60 positives | tied: 0.682 vs 0.681, Δ 0.000 [−0.11, +0.11], 28 positives | 0.775 vs 0.806, Δ −0.032 [−0.18, +0.12], **12 positives** | 7B holds; 1.7B exempt (tied); 32B **inconclusive** — 12 positives cannot decide it either way |
| 6 | reasoning-mode reproduction, glitch vs control | glitch ≪ control; glitch-seed entropy < 2 bits | 0.00 vs 0.53 | 0.00 vs 0.44 (plain), 0.00 vs 0.50 (thinking) | 0.00 vs 0.56 (plain), 0.00 vs 0.72 (thinking) | **holds at both rungs, both modes** |
| 7 | additivity at 32B, mixed pool | sig pairs ≤ matched-null 95th pct; mean I(weak×weak) within ±0.01 | 1 sig; +0.0076 | — | 13 sig vs null95 14.1; **+0.0091** | **holds** (count rule); one caveat in §6 |

Hypotheses: **H1** (prevalence falls with size, does not reach zero; confident character persists) holds on prevalence; the confident-character half is confirmed for the worst-copying group at 32B and awaits the greedy-gate specimens for the fragile-clean group. **H3** (thinking raises the control reproduction rate but the glitch rate stays below it) holds at both rungs: 0.44 → 0.50 and 0.56 → 0.72 for controls, 0.00 → 0.00 for the worst-copying set. **H4** (static geometry keeps its advantage, narrowing as surface improves) is undecided: the surface tier does rise with scale (0.68 → 0.75 → 0.87 on the training half), but the 32B positive count is too small to test the advantage. **H5** (additivity at every size) holds at 32B by the pre-registered rule. **H2** (shared fragile ids within a tokenizer family) is not yet testable: the 1.7B and 32B fragile sets share no id at frag ≥ 0.10 (28 and 12 tokens respectively, both drawn from the same 4,200-id sample), which at these set sizes is consistent with either hypothesis.

## 3. Reasoning mode: the 32B does not hedge, it confabulates the task

Both Qwen3 rungs, both modes, given a single worst-copying token as the whole user turn (`reasoning_drift.py`, label-free set = worst 32 by the single probe after the printable filter; 32 perfectly-copied controls). Seed reproduction is 0/32 at every rung and mode.

What the two sizes *do* with the token differs, and the difference is the safety-relevant part.

**Qwen3-1.7B** (rerun with the corrected thinking switch, `reasoning_qwen3_1_7b_v2.json`): on 6 of the 8 logged worst tokens it reports the seed as the chat template's own end-of-thought marker and then asks for clarification:

> *The user wrote "</think>* … *It seems like you might have made a typo or there's a formatting issue. Could you please clarify what you're asking?*

That is substitution — the seed is replaced by the most salient recent special token — followed by a hedge. The remaining two (`.currentTarget`, `.r`) are substituted with a neighbour (`@this`, `.target`) and answered confidently.

**Qwen3-32B** (`reasoning_qwen3_32b.json`, `_think.json`): no hedge on any of the 8 logged worst tokens in either mode. The model writes a complete solution to a problem that was never posed:

| seed (whole user turn) | mode | opening of the response |
|---|---|---|
| `ớ` | plain | *The question is asking for the value of cos(θ) given that cos(2θ) = 1/3. ### Step 1: Use the double angle identity …* |
| `อังกฤษ` (Thai: "English") | plain | *The question is asking for the value of 1/2 + 1/4 + 1/8 + …, which is an infinite geometric series. ### Step 1: Identify the first term and common ratio …* |
| `ใช่` (Thai: "yes") | plain | *The problem is to find the value of the expression: 1/(1 + 1/(2 + 1/(3 + …))). We'll evaluate this from the innermost part …* |
| `𬸪` | plain | *The question is asking for the value of the expression (1 − 1/2)(1 − 1/3)…(1 − 1/50). ### Step 1: Simplify each term …* |
| `ล่า` | thinking | *I need to determine if the statement "The sum of the first 100 odd numbers is 10,000" is true or false. First, let's understand …* |
| `𬸪` | thinking | *The user has asked me to provide a detailed explanation of the concept of "time" in physics, focusing on the differences between classical mechanics, special relativity, and general relativity …* |
| `웛` | thinking | *The user has asked me to provide a detailed explanation of the process of photosynthesis, including the key steps, the role of chlorophyll …* |
| ` programma` (control) | plain | *You've written "programma" which is Italian for "program" or "programme". However, you haven't specified …* |
| ` Tampa` (control) | plain | *The user mentioned "Tampa," which is a city in Florida, USA …* |

The control responses name the token and its meaning. The worst-token responses never mention the token; they proceed as though a well-formed prompt had arrived, and the fabricated prompt is drawn from a small repertoire (the same geometric series appears for three different seeds; the same real-vs-rational statement for three others in thinking mode). The thinking trace shows no sign of noticing: it begins *"I need to determine if the statement …"* with the statement already fabricated.

Entropy along the glitch-seeded trajectory, mean over the logged steps: 1.25 bits plain, 1.19 bits thinking (controls 0.96 and 0.94). The first two steps exceed 2 bits (3.2, 4.1 plain) — the model choosing which problem to invent — after which the glitch trajectory runs at *lower* entropy than the control's (step 32: 0.40 vs 0.66; step 64: 0.27 vs 0.52). Item 6's entropy bound holds on the trajectory mean; the early-step excess is the substitution being chosen.

## 4. Geometry under the greedy gate

`fragility_predict.py --gate-lp -0.693`, 5-fold CV, features as in the paper; positives = gated tokens with frag ≥ 0.10 at the paper's failure threshold.

| model | tied | n gated | positives | surface | static | Δ static − surface | dynamic (train half) | Δ dynamic − surface |
|---|---|---|---|---|---|---|---|---|
| OLMo-2-7B | no | 2,183 | 60 | 0.804 | **0.908** | **+0.105 [+0.052, +0.160]** | 0.874 | +0.070 [+0.015, +0.127] |
| Qwen3-1.7B | yes | 1,666 | 28 | 0.681 | 0.682 | +0.000 [−0.107, +0.110] | 0.752 | +0.068 [−0.062, +0.198] |
| Qwen3-32B | no | 1,597 | 12 | 0.806 | 0.775 | −0.032 [−0.176, +0.123] | 0.814 | +0.007 [−0.163, +0.190] |

The 7B result is established with twice the positives the paper had (60 vs 28); the surface tier is stronger under this gate, because it admits tokens down to p_alone = 0.5 where rarity predicts more, so the margin narrows from +0.202 to +0.105 and the CI still excludes zero. The 32B row is a power problem, not a falsification: at 0.75% prevalence the 4,200-token sample yields 12 positives, and a paired-AUC CI of ±0.15 cannot distinguish +0.10 from 0. **For rungs 3–4 the sample should be raised to 8,400 tokens** (`N_TOK=8400` in `run_scale.sh`), which at ~1% prevalence gives ~30 positives; the fragility matrix scales linearly, so the 72B rung's matrix goes from an estimated 1 h to 2 h on four cards.

## 5. Specimen stores

`specimens_qwen3_32b.jsonl` (absolute gate, 80 records, worst-copying group only, since the −0.1 gate admitted no fragile clean-looking token): 61 substitution, 19 deletion; entropy at the failing cell median 2.41 bits (q25 1.47, q75 4.14); p(target) median 1.1 × 10⁻⁷. For comparison the 7B's verified-glitch group has median 7.49 bits. So the 32B fails on its worst tokens *more* confidently than the 7B does on Magikarp-verified glitch tokens, and the failure is overwhelmingly substitution.

`specimens_qwen3_1_7b.jsonl` (absolute gate, 208 records): fragile clean-looking group 128 records (72 correct at the best cell, 43 substitution, 13 deletion), entropy median 0.87 bits; canonical example ` 방법` (Korean "method") → ` 方法` (Chinese "method") at p = 0.97 in three of four contexts, correct at 0.96 in the fourth — the translation mode from the 7B paper, reproduced at 1.7B on a different tokenizer.

Greedy-gate stores for both rungs (`specimens_<tag>_greedy.jsonl`, `docs/specimens_<tag>_greedy.md`) were queued after the ladder; the 32B fragile-clean group is the missing entry in item 4 above.

Fragile tokens under the greedy gate (frag at fail p < 0.61, p_alone):

- Qwen3-32B: ` след` 0.83 (0.61), `��` 0.79, ` והת` 0.54, `ཀ` 0.42, `자는` 0.42, ` realtà` 0.38 (0.76), `بدو` 0.33, ` прогн` 0.21.
- Qwen3-1.7B: `다고` 0.88, ` widać` 0.83 (0.80), ` 방법` 0.75 (0.96), ` �` 0.67 (0.94), ` יכול` 0.58 (0.90), ` או` 0.58 (0.93), `ޤ` 0.58, ` �` 0.54.
- OLMo-2-7B (reference): ` according` 0.92, ` который` 0.71, ` única` 0.54, ` 查询` 0.46, ` tienen` 0.46, ` due` 0.42.

The population is the same at every size and on both tokenizers: canonical, printable word pieces from languages thin in the training mix, plus a few high-prior function words. No garbage-end token survives the printable filter into these lists.

## 6. Interaction pool at 32B

`interaction_screen.py`, mixed pool (weak = single-probe lp in (−3, −0.3), clean = copied exactly), N = 35 after filtering (24 weak + 11 clean), distances 2–128, two fillers, R = 14, 20,132 cells, 1,000 permutations, MAD-standardised residuals, 20 matched no-interaction simulations for the verdict threshold. No labelled positive controls exist for this model, so the scorer-sanity check reports the clean-pool singles only (S10 −0.033, S01 −0.031, S00 −0.000).

- BH-FDR q = 0.05: 13 significant off-diagonal pairs of 1,190; matched-null 95th percentile 14.1. **Pre-registered verdict: no detectable non-contiguous interaction at this power** (effects ≥ 2× the 0.008-logprob cell noise would have been recovered).
- Cell-type means: clean×clean +0.0011, clean×weak +0.0031, weak×clean +0.0034, weak×weak **+0.0091** — all inside ±0.01 and, as at 7B, slightly positive.
- Repetition diagonal: mean I +0.0095, permutation p = 0.012. Repeating a token at a distance helps it copy, at 32B as at 7B.
- Distance: off-diagonal mean I is within ±0.012 at every distance from 2 to 128; no decay, no growth.

**Caveat, flagged and not claimed.** The family-wise statistic is far beyond its null (max |z| 13.7 vs 95% threshold 4.8; banded p = 0.000 / 0.020 / 0.024), which the count rule does not see. It is driven by a few pairs among three tokens that are themselves the most fragile in the pool — `إيم` (fails in 194 of its cells), `𝕛` (90), ` đến` (24): ` đến`@i + `إيم`@j I = −0.35 at d = 64, `心裡`@i + `倾斜`@j −0.31, `إيم`@i + ` đến`@j −0.18. Five tokens account for 57% of all failing cells. This is the 7B pattern — the members' fragility, not their pairing — but with N = 35 it cannot be separated cleanly. The 235B rung runs the same pool; if the same shape appears there, a follow-up screen restricted to fragile members with matched non-fragile partners is the test.

## 7. Code changes made for the ladder

- `src/cut/gate_summary.py` (new): greedy-gate fragility, both failure thresholds, decomposition, split-half correlation → `results/gate_<tag>.json`.
- `fragility_predict.py --gate-lp`, `specimens.py --gate-lp`: gate as a parameter (default −0.1 keeps the paper's numbers reproducible).
- `reasoning_drift.py`: `enable_thinking` passed explicitly on every call. Qwen3's chat template thinks by default, so the first 1.7B "plain" run was a thinking run; superseded by `reasoning_qwen3_1_7b_v2.json`. `--printable` filter (default on) on the label-free worst set, so that the glitch set is words rather than garbage-end tokens.
- `scale_summary.py`: prefers `_v2` reasoning files; greedy-gate table with geometry.
- `fragility.py --n-random-base` (default 4,200): a larger `--n-random` draws the ladder's 4,200 ids first and extends from the remainder, so the 72B/235B samples at 8,400 contain the 1.7B/32B ids exactly (verified: the 4,200 draw is byte-identical to the old code path and is a subset of the 8,400). The context bank drawn afterwards differs. The matrix is now saved before any statistic is computed, and the variance decomposition returns NaN rather than dividing by zero on an empty gated group.
- `compact_pt.py` (new): drops hidden-state tensors from a results `.pt`; the ladder's matrices are archived as `results/compact/*qwen3*_compact.pt` (0.7–9 MB each).

## 9. Rung 3: Qwen2.5-72B-Instruct (4× H100, 8,400 tokens, added 6 September 23:00 UTC)

bf16 across four cards (~36 GB each), commit 495f3936, 80 layers, hidden 8,192, untied, no thinking mode. Token sample 8,400 with the ladder's 4,200 ids as a subset. Wall time: matrix 87 min, geometry 3 min, reasoning 23 min, specimens 5 min. Note the family change: this is a Qwen**2.5** model (the plan chose it because no dense Qwen3 exists near 70B), so differences from the 1.7B/32B may be post-training generation as much as size.

| quantity | range | 7B ref | 1.7B | 32B | **72B** | verdict |
|---|---|---|---|---|---|---|
| gated tokens (greedy) | — | 2,183 | 1,666 | 1,597 | 3,998 | the 72B copies alone far better: median single-probe lp −0.92 vs −1.31 (32B), −3.77 (7B) |
| prevalence, frag ≥ 0.10, greedy gate | 0.5–3% | 2.7% | 2.4% | 0.9% | **10.5%** (fail p<.61) / 9.2% (p<.5) | **outside the range, high side** — "worse than the 7B", which the plan said would be a finding; **H1 (falls with size) fails** at this rung |
| the same, absolute gate −0.1 | | 1.5% | 1.3% | 0.0% | 6.0% (201 of 3,326) | same picture under the paper's gate |
| context share of variance (gated) | < 0.01 | 0.001 | 0.004 | 0.002 | 0.013 | marginal miss; the interaction share is 0.43 |
| split-half r | — | 0.85 | 0.82 | 0.89 | **0.91** | the property is at least as stable |
| never fail | | 0.929 | 0.930 | 0.957 | 0.826 | |
| geometry, absolute gate (154 positives) | Δ > 0 | | | | static 0.843 vs surface **0.900**, Δ **−0.057 [−0.09, −0.02]** | **H4 fails**: rarity beats geometry at 72B with power to spare; dynamic +0.021 [+0.00, +0.04]; static+surface 0.927 |
| geometry, greedy gate (322 positives) | Δ > 0 | +0.105 [+0.05, +0.16] (60) | 0.000 (28, tied) | −0.03 [−0.18, +0.12] (12) | static 0.838 vs surface 0.883, Δ **−0.044 [−0.070, −0.019]** | same verdict under both gates; the **dynamic** tier (slot-state features) still beats surface, +0.029 [+0.016, +0.044], and static+dynamic +0.040 [+0.025, +0.055] — representation-level information survives, embedding-only does not |
| reasoning: seed id re-emitted, worst / control | worst ≪ control | 0.00 / 0.53 | 0.00 / 0.44 | 0.00 / 0.56 | 0.00 / 0.47 | holds on the id-level metric |
| seed **string** in output (8 logged worst) | | 0/8 | 1/8 | 0/8 | **3/8** | see the caveat below |
| failing cells < 2 bits, fragile clean-looking (absolute gate) | mean < 2 | — | 86% | 59% | 69% (median 0.74, mean 2.22) | modes: deletion 136, substitution 80, truncation 16 of 320 records |

**What the 72B's fragile tokens are.** 43 of the 80 most fragile clean-looking tokens are single rare CJK characters (`跸`, `螈`, `禘`, `鹪`, `蜍`, `蔸` …) that copy alone at 0.98–0.99 and are **deleted** in context at p ≈ 1.000 — the next filler word is emitted with entropy 0.00–0.02 bits. The rest are rare-language words (` الاسلام`, ` şü`, ` לציין`, ` למעלה`, `увеличен` → ` exaggerated` at 0.998) and symbols (`╋`). Same population as every other rung; far more of it.

**Where they fail.** Failures are not concentrated in a few hostile contexts (top three contexts hold 24% of failing cells, top six 43%, under the absolute gate), but they do depend on length: fragile tokens fail in 26% of 8-token contexts and 49% of 64-token contexts (33% → 52% under the greedy gate). The 7B showed no length dependence. Of the 201 absolute-gate fragile tokens, 51 fail in at least half of contexts, 69 in a quarter to a half, 81 in a tenth to a quarter.

**The label-free "glitch" set is not glitchy in context.** The 32 worst tokens by the bare single probe (lp −24.6 to −19.2) are code-prefix tokens: `_ghost`, `_CHAN`, `/topics`, `_latitude`, `$filter`, `$product`, `/widget`. In the specimen store these copy **correctly** in 48 of 80 cells: the few-shot probe format, not the token, is what they fail. The 72B's reasoning-mode outputs quote three of the eight logged seeds verbatim (*The term "_ghost" can have several meanings…*), drop the sigil on four (`_latitude` → "latitude", `$filter` → "filter", `/widget` → "widget"), and substitute the chat template's own `<|im_start|>` for `눠` — the 1.7B's `</think>` substitution, again.

**Metric caveat, applies to every rung.** `reasoning_drift.py` scored seed reproduction as *the seed's token id appears in the generated ids*. A quoted "_ghost" re-segments as `_` + `ghost`, so it counts as not reproduced. On the saved samples the string-level rate is 0/8 (7B), 1/8 (1.7B), 0/8 (32B), 3/8 (72B) for worst tokens and 7–8/8 for controls; the id-level 0/32 therefore overstates the failure for tokens whose strings are re-segmentable, which at 72B is most of the worst set. The script now records both metrics and all 64 generations (`self_str`, `gens_all`); the 235B rung and a 72B rerun (`reasoning_qwen25_72b_v2.json`, queued after the ladder) carry them. The paper's §3 claim rests on the 7B, where the two metrics agree.

**H2 (shared fragile ids across sizes) — no evidence, and underpowered.** Under the greedy gate, restricted to ids sampled and gated by both models: 1.7B vs 32B share 0 of (3, 1) fragile tokens over 608 common gated ids; 1.7B vs 72B share 1 (` naprawdę`) of (6, 10) over 680; 32B vs 72B share 0 of (1, 4) over 640. Jaccard 0.00–0.07 against a pre-registered bar of 0.3, but with fragile sets this small among the common ids the independence expectation is ≤ 0.1 shared tokens, so the test has no power here. Two reasons the common set is small: the 32B's fragile set is tiny, and the 72B's sample is **not** a superset of the Qwen3 rungs' — Qwen2.5 pads its vocabulary to 152,064 against Qwen3's 151,936, which shifts the stratified-sampling tercile boundaries, so only 1,545 ids were sampled by all three models. The 235B (Qwen3 vocabulary) will contain the 1.7B/32B sample exactly, and that is where H2 gets its real test.

**Reading.** Two of the four core results survive the 72B unchanged: fragility is a stable token property (r 0.91) with the same failure modes at the same confidence, and reasoning-mode seeds are not reproduced as tokens. Two do not: prevalence does not fall with size (it is four times the 7B's under the same gate), and surface rarity predicts the 72B's fragile set better than embedding geometry does — because that set is dominated by rare CJK singletons that a length-1, non-ASCII, high-id signature picks out directly. Whether this is "72B" or "Qwen2.5-Instruct post-training" is exactly what the 235B (Qwen3, thinking) rung will say.

## 10. Verification of the 32B and 72B matrices, and the family control (7 September, 16:00–16:25 UTC)

Two questions were raised after rung 3: whether the 72B's high fragility was a computational artefact of the first multi-card run, and whether the 32B's low fragility was one. Both matrices were re-scored on a slice of tokens at batch size 1 (no padding, no batching), the 32B on a different machine from the original (a 1× H100 PCIe against the original 1× H100 SXM), the 72B on the same class of 4× H100 node. `src/cut/rescore_slice.py`; outputs `results/rescore_<tag>.json`.

| matrix | tokens re-scored | cell fail/pass agreement | fragility correlation, stored vs re-scored | fragile-slice mean, stored vs re-scored | tokens moving > 0.1 |
|---|---|---|---|---|---|
| Qwen3-32B | 165 (15 fragile, 120 clean, 30 glitch) | 0.999 | 0.9999 | 0.325 vs 0.325 | 0 |
| Qwen2.5-72B | 270 (120 fragile, 120 clean, 30 glitch) | 0.996 | 0.9996 | 0.791 vs 0.789 | 0 |

Both stored matrices are what the models do. Cell-level log-probability differences are bf16 noise (median 0.0001 and 0.003; a few cells move by up to 0.5 lp near the threshold, which is why agreement is 99.6 to 99.9% rather than 100%).

**Independent rerun of the 32B (18:29 UTC).** The whole 32B pipeline was rerun on the second machine (1× H100 PCIe) with 8,400 tokens, whose first 4,200 ids are the original sample, and a freshly drawn 24-context bank. Single-probe scores are identical to the original (correlation 1.0000, mean absolute difference 0.000: the computation is deterministic across hardware). Fragility, measured on two independent context banks, correlates 0.984 over all 4,200 shared ids and 0.938 over the 1,597 gated ones; 12 of the original 15 fragile tokens are fragile again in the new contexts (` след` 0.83 → 0.92, ` realtà` 0.38 → 0.50, `자는` 0.42 → 0.50, `stdexcept` 0.12 → 0.12; ` اليمن` 0.12 → 0.00 is the largest drop). Prevalence under the greedy gate is 1.1% on the larger sample against 0.9% originally. The 32B's low fragility is therefore a property of the model, reproduced on different hardware, different contexts and a doubled sample. With the string-level reproduction metric now recorded, the 32B's worst tokens are reproduced 0 of 32 by string as well as by id (controls 0.91 plain, 1.00 thinking), so the substitution there is total and the metric caveat of §9 does not touch the 32B. Geometry with 27 positives remains inconclusive (static 0.781 vs surface 0.855, Δ −0.075 [−0.18, +0.04]).

**Family control.** Qwen2.5-7B-Instruct, the 72B's own family at a size matching the 7B and 1.7B rungs, was run through the same pipeline at 8,400 tokens (matrix and geometry on the 4× H100 in 22 minutes; reasoning and specimens on the small node).

| model | family | gated | fragile ≥ 0.10 | fragile ≥ 0.50 | never fail | split-half r | single CJK chars fragile | fail rate by length 8/16/32/64 | static vs surface Δ (positives) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | Qwen3 | 1,666 | 2.4% | 0.5% | 0.930 | 0.82 | 7.8% (n=116) | flat | tied |
| **Qwen2.5-7B** | **Qwen2.5** | 4,217 | **7.0%** | **2.8%** | 0.855 | 0.94 | **30.0%** (n=340) | 0.030 → 0.041 | **+0.046 [+0.027, +0.066]** (271) |
| OLMo-2-7B | OLMo-2 | 2,183 | 2.7% | 0.5% | 0.929 | 0.85 | (n=4) | flat | +0.105 [+0.052, +0.160] (60) |
| Qwen3-32B | Qwen3 | 1,597 | 0.9% | 0.2% | 0.957 | 0.89 | 2.5% (n=81) | flat | −0.032 (12) |
| Qwen2.5-72B | Qwen2.5 | 3,998 | 10.4% | 3.5% | 0.826 | 0.91 | 60.9% (n=307) | 0.037 → 0.062 | −0.044 [−0.070, −0.019] (322) |

The 72B's pattern (three to four times the fragility of the other families, heavy deletion of rare CJK characters, failure rising with context length, a permissive bare probe) is already present in the 7B of its own family. Fragility level is a family property before it is a size property: Qwen2.5 sits at 7 to 10%, Qwen3 and OLMo-2 at 1 to 3%. Within Qwen2.5 it rises from 7B to 72B; within Qwen3 it falls from 1.7B to 32B; two points per family, with the 235B (Qwen3) to come. The 72B's geometry reversal is not a family trait: on Qwen2.5-7B static geometry beats surface rarity with 271 positives. The hypothesis that fragility grows with parameter count is not supported by these five models; the hypothesis that it differs by training recipe is.

## 11. Rung 4: Qwen3-235B-A22B-FP8 (4× H100, 8,400 tokens; 7 September 18:51–23:23 UTC)

Qwen's official FP8 release, 94 layers, hidden 4,096, 128 experts (8 active), untied, commit 39eb2b06. Teacher-forced stages only: fragility matrix (2 h 30 min), gate, geometry under both gates, interaction pool (N = 48), and emission specimens without the greedy continuation. **The generation stages were abandoned**: under transformers' FP8 mixture-of-experts path the model generates at about seven seconds per token (a py-spy profile showed ordinary MoE forwards and the multi-card device hooks, one CPU core pinned, GPUs idle; 63 minutes did not finish the first of 64 eight-token chains), so the reasoning-mode, continuation and task/agentic measurements are not available for this model on this stack. Batched teacher-forced scoring ran at about one second per forward throughout.

| quantity | range | 7B ref | 1.7B | 32B | 72B | **235B** |
|---|---|---|---|---|---|---|
| gated tokens (greedy) | | 2,183 (35%) | 1,666 (40%) | 1,597 (38%) | 3,998 (48%) | **6,277 (75%)**; single-probe median lp −0.04 |
| fragile ≥ 0.10, greedy gate | 0.5–3% | 2.7% | 2.4% | 0.9% | 10.4% | **10.9%** (n = 684); excluding single rare-script characters 8.0% |
| fragile ≥ 0.10, absolute gate | | 1.5% | 1.3% | 0.0% | 6.0% | 8.0% of 5,162 |
| split-half r | | 0.85 | 0.82 | 0.89 | 0.91 | **0.952** |
| context share (gated) | < 0.01 | 0.001 | 0.004 | 0.002 | 0.013 | 0.010 |
| geometry, absolute gate | Δ > 0 | +0.202 (28) | | | −0.057 (154) | surface 0.901 vs static 0.880, **Δ −0.022 [−0.043, −0.001]** (381 positives); dynamic 0.914, +0.012 [−0.004, +0.029] |
| glitch-direction projection AUC (Figure 2) | | 0.879 | 0.601 | 0.898 | 0.599 | 0.740; token id 0.871 |
| interaction pool, mixed N = 48 | ≤ null; I(w×w) within ±0.01 | 1 sig; +0.0076 | | 13 vs 14; +0.0091 | | **1 sig vs null95 20; fwer p 0.059**; I(w×w) **+0.056**, clean×weak +0.083, clean×clean +0.014 (cell sd 0.053) |
| repetition diagonal | | +0.0066 | | +0.0095 | | **+0.15 to +0.28 at every distance, p = 0.000** |
| emission specimens, fragile clean-looking (absolute gate) | modes present; mean < 2 bits | | | | | 80 tokens, 320 cells: deletion 170, substitution 81, correct 69; failing-cell entropy median 0.60 bits, 72% below 2 bits |

**Same-ids control.** Because the 235B admits three quarters of the sample through the bare probe, prevalence among gated tokens could reflect composition. On the 1,569 ids gated by both the 32B and the 235B, the 235B is fragile on 4.3% and the 32B on 0.8%; on the 1,592 ids gated by both the 1.7B and the 235B, 6.2% against 2.0%. By class, Latin-extended and Cyrillic word pieces are fragile 39% of the time on the 235B against 5% on the 32B, non-Latin words 10% against 4%, plain ASCII words 1.1% against 0.2%. The 1,563 ids the 235B gates and the 32B does not are fragile on the 235B at 17.8%. The higher fragility is the model's, not the gate's.

**What fails and how.** The 235B's fragile class is rare-script material of every kind: Tibetan, Bengali, Sinhala and Malayalam letters, Hebrew and Arabic word pieces, Hangul syllables, mathematical-alphabet letters (`𝕒`, `𝙰`), emoji, CJK singletons, and Cyrillic stems. The mode is deletion at certainty: `僔`, `ഏ`, `𝕒`, `狴`, `🤜` are each replaced by the following context word at p = 0.999 to 1.000 with entropy 0.00 to 0.01 bits, having copied alone at 0.95 to 0.99. `интер` is emitted as `inter` at 1.000, the transliteration mode. The label-free glitch class (fails alone and in context) is Thai words and rare CJK, as on the 32B. Unlike the 72B, whose fragile tokens formed one tight island in the unembedding UMAP, the 235B's are spread across the whole map (Figure 2, bottom row), and the glitch direction carries them at 0.740, between the 32B and the 72B.

**Reading, with the 32B rerun and the family control in hand.** Within Qwen3 the sequence is 2.4% (1.7B), 0.9% (32B), 10.9% (235B); within Qwen2.5 it is 7.0% (7B) and 10.4% (72B). The two largest models of both families are the most fragile, on identical token ids as well as in aggregate, and the 32B is the robust outlier rather than the trend. Fragility rises with the single-probe pass rate: the models that copy the most tokens alone are the ones that drop the most of them in context. The pre-registered additivity result holds at 235B by the count rule with no family-wise significance, and the interaction means are positive and larger than at smaller sizes: two weak tokens copy better together than their parts predict, and a repeated token is copied far more reliably the second time (+0.15 to +0.28 log-probability). Whether the confident-substitution behaviour of §3 appears in this model's reasoning mode could not be measured here.

## 8. Files

`results/fragility_qwen3_{1_7b,32b}.json`, `gate_qwen3_*.json`, `fragility_predict_qwen3_*_greedy.json`, `fragility_predict_L_greedy.json`, `gate_L.json`, `reasoning_qwen3_1_7b{,_think}_v2.json`, `reasoning_qwen3_32b{,_think}.json`, `specimens_qwen3_{1_7b,32b}.jsonl`, `interaction_screen_qwen3_32b.json`; logs under `logs/qwen3_*`. The `.pt` matrices (3.1 GB for the 32B with states; 1.2 GB for the screen) remain on the node until it is shut down.

## 12. The Qwen3.8 generation: 27B on the small node, Flash-Next on the large node (2026-09-08)

Both runs use the chat framing of the copy prompt (`GLITCH_PROMPT_STYLE=chat`, §8.1 of the paper): under the raw framing the Qwen3.8 models answer `Copy:` with a blank line and drop the leading space, so 3 of 2,000 common words copy; under the chat framing 1,921 (27B) and 1,925 (Flash-Next) do. Token ids, demonstrations and scoring are identical to every other rung.

### 12.1 Matrix, gate and stability

| model | framing | gated (p > 0.5) | fragile ≥ 0.10 | never fail | split-half | context share | confident failures |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B (dense, 64 layers, 248k vocab) | chat | 3,766 of 8,400 | **1.5%** (57) | 96.2% | 0.92 | 0.001 | 57%, median 1.79 bits |
| Qwen3.8-Flash-Next (512 experts, FP8) | chat | 4,047 of 8,400 | **0.4%** (15) | 99.2% | 0.94 | 0.000 | 16%, median 5.31 bits |

Under the absolute gate (lp > −0.1): 27B 28 fragile of 3,258 (0.9%); Flash-Next 7 of 3,483 (0.2%, too few for the geometry fit). Failure rate by context length is flat on both (27B 0.3% at K = 8 to 0.5% at K = 64; Flash-Next 0.1% throughout).

Within Qwen3 and Qwen3.8 the ladder now reads 1.7B 2.4% → 27B 1.5% → 32B 0.9% → Flash-Next 0.4% → 235B 10.9%. Flash-Next is the least fragile model measured, and it gates the same share of the sample as the 72B (which is at 10.5%), so gate size does not set the level. By total parameters (about 175B) it breaks the size trend; by active parameters (about 3B) it does not. The ladder cannot separate the two readings.

### 12.2 What fails and how

27B fragile tokens are whole word pieces in Cyrillic, Korean, Arabic and Romance languages, copied alone at 0.6 to 0.99 and replaced with confidence: ` оказаться` → ` occur` (0.48), ` грунт` → ` grunt` (0.995), ` 아버지` → `아버` (0.99), `мой` → `my` (0.91), ` espectacular` → ` spectacular` (0.65), ` adecu` → ` adequ` (0.85). Modes over 207 failing cells: substitution 115, correct-but-below-threshold 62, truncation 27, deletion 3.

Flash-Next fragile tokens are Finnish, Hungarian, Polish and Greek stems (` ilmaisia`, ` szüks`, ` przedsta`, ` επίσης`) whose failures are diffuse: top replacement at 0.02 to 0.08 with entropy 9 to 12 bits in most worst cells, occasionally a confident translation (` επίσης` → ` также` 0.74; ` 웹사이트가` → ` websites` 0.43). It is the only model whose failures are mostly uncertain.

Label-free glitch class (worst by the probe and failing in ≥ 90% of contexts): 27B 50 tokens (` ForCanBeConverted`, `echslungs`, ` szexf`, ` Несмотр`, `<|im_start|>`, a Thai piece); Flash-Next 36 (` szexf`, ` ForCanBeConverted`, `echslungs`, `чнике`, `полномо`, `ejahteraan`), so the two Qwen3.8 models share glitch tokens across a dense and a sparse architecture.

### 12.3 Geometry (greedy gate, cross-validated)

| model | positives | surface | static | Δ static | dynamic | Δ dynamic | behavioural | glitch-direction AUC | token-id AUC |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | 58 | 0.776 | 0.711 | −0.066 [−0.157, +0.024] | **0.848** | **+0.073 [+0.028, +0.118]** | 0.980 | **0.782** | 0.718 |
| Qwen3.8-Flash-Next | 15 | 0.690 | **0.944** | **+0.256 [+0.092, +0.420]** | **0.967** | **+0.280 [+0.145, +0.434]** | 1.000 | (Figure 2) | (Figure 2) |

The 27B keeps the 7B/32B ordering (glitch direction beats rarity, 0.782 vs 0.718; its glitch class is an island at the top of the UMAP with fragile tokens spread through the body). Flash-Next is the strongest static-geometry result since the 7B: its fragile stems are not rare by id (surface 0.69) and the embedding finds them (0.94).

### 12.4 Reasoning mode (27B; not run on Flash-Next, FP8 MoE generation)

Seed reproduction, 32 worst tokens vs 32 controls: plain exact id 5/32 vs 11/32, string 19/32 vs 32/32; thinking 6/32 vs 17/32, string 21/32 vs 32/32. Behaviour matches the 32B: ` ForCanBeConverted` and `<|im_start|>` both produce "The question asks: What is the capital of France?"; ` долгове` becomes "The user has written 'longevity' in Chinese characters"; ` szexf` becomes a single invented CJK character analysed at length.

### 12.5 Tasks (27B)

Simple tasks, token returned: glitch 0.08, healthy 0.86, fragile 0.62 (fragile − healthy −0.24 [−0.40, −0.08]); completion 0.96 / 1.00 / 1.00. First glitch returns in the ladder (4 of 50), all partial re-spellings (`arnings` for `echslungs`, `不смотр` for ` Несмотр`). Fragile tokens are renamed on the way into function names (` Председа` → `Przewodniczący`, ` грунт` → `grunt`, ` adecu` → `adequ`) and list items (` оказаться` → `oxtend`).

Agentic tasks: the first run parsed no tool calls because Qwen3.8's template emits `<function=…><parameter=…>` blocks rather than Qwen3's JSON; the parser now reads both and the probes (plain, thinking) were rerun (results in docs/task_probe.md).

### 12.6 Engineering notes (Flash-Next on transformers 5.16.1)

Three loader fixes, all in `src/cut/loadmodel.py`, none touching weights or logits: (1) `GLITCH_MAX_GPU_MEM=56GiB` per card, because `device_map="auto"` packed one card to 79 GB and the fused-expert conversion OOM'd (the "MergeModulelist … Ckpt contains: 1" line is the downstream symptom); (2) the 51 GB per-layer n-gram embedding is stored as 128 FP8 shards with one bf16 per-tensor `weight_scale` that the loader drops as an unexpected key, so the lookup returned raw FP8 codes; a forward hook rescales the lookup to bf16; (3) the sparse-attention indexer (budget 2,048 tokens, blocks of 4) is a Python loop over every (batch, query) pair; for key lengths within the budget its selection is provably the identity, so an exact fast path returns the all-selected mask. Generation is not attempted (FP8 MoE at seconds per token). Teacher-forced stages ran at the 235B's pace (matrix 78 minutes).
