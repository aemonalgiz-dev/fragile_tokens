# Fragile Tokens

### Ordinary tokens that pass every glitch probe fail to copy in some contexts, confidently, and the output embedding says which ones

**Jeffrey R. Gordon** · draft, 6 September 2026

---

## Abstract

Glitch tokens are found one token at a time: a vocabulary entry is shown to a model in isolation, the model is asked to repeat it, and tokens that fail are labelled. We show that this misses a class of failure that matters for any system assembling prompts programmatically. Placing 2,098 and then 6,273 tokens into banks of random ordinary contexts, **1.5–1.6% of tokens that pass the standard single-token probe fail to copy in at least 10% of contexts.** They are canonical, common tokens — the English word `according` copies perfectly alone and is deleted with probability 1.000, at entropy 0.00 bits, in 44 of 48 contexts. The failure is a property of the token, not the context: a two-way decomposition puts the context's share of variance at 0.000, fragility is stable across independent context banks (AUC 0.996), and which contexts break a token is unpredictable while whether it is fragile is not. Failures take four confident forms — deletion, substitution by a neighbour, truncation, translation — and in a reasoning-style prompt the same substitution appears for verified glitch tokens (0 of 32 reproduced) at low entropy with the downstream task completed every time, while the token's identity remains linearly recoverable from the final layer: the readout fails, not the representation. Fragility is predictable with no forward pass from the geometry of the output embedding — cross-validated AUC 0.862 against 0.711 for surface rarity, a paired-bootstrap improvement of +0.15 [+0.06, +0.26] — and better still from the token's representation on a dozen contexts. A pre-registered factorial screen over 8,064 token pairs at distances 2–128 finds no interaction: two tokens' effects add, and repetition of a healthy token helps. The hazard is a fragile token in a hostile context, not a conspiracy of tokens. We release every failure as a specimen with model provenance.

---

## 1. Introduction

Every detector of glitch tokens scores a token by itself. Magikarp ranks vocabulary entries by unembedding-matrix indicators and verifies candidates by asking the model to repeat each one; GlitchHunter clusters, GlitchProber classifies activations, GlitchMiner optimises next-token entropy, GlitchQuiz runs a battery of templates. All of them answer the question *is this token broken?* with the token alone in the prompt.

Practitioners report something that question cannot see. Reasoning models "struggle to identify" certain tokens, "conflate" them with other scripts, and behave differently depending on what surrounds them — a single instance often does nothing, while the same token in a longer input produces confident nonsense. Two readings are available. Either glitch behaviour is *compositional* — caused by particular combinations of tokens, possibly individually healthy and far apart — or it is *contextual*: a property of a token that only some surroundings reveal.

We test both on an open 7B instruction-tuned model with published glitch labels. The compositional reading fails a pre-registered, calibrated test: across 8,064 ordered pairs and separations of 2 to 128 tokens, the effect of two tokens is additive to within 0.008 log-probability in every cell type, including pairs of individually weak tokens with ample room to fail. The contextual reading holds, and it is the paper. We define **fragility** — the fraction of ordinary contexts in which a token fails to copy — measure it for thousands of tokens, show that it is real, rare, stable, canonical, and invisible to every existing probe, characterise its failure modes, and show that the output embedding predicts it better than rarity does.

**Contributions.**
1. **Fragility** as a measured token property: prevalence among probe-passing tokens (1.5–1.6%), a variance decomposition placing it in the token rather than the context, stability across independent context banks, replication, and specimens (§4).
2. **The failure mode**: confident substitution — deletion, neighbour substitution, truncation, translation — at low entropy with tasks completed, while the token's identity is still linearly present at the final layer (§3, §4.4).
3. **Prediction from geometry**: the output-embedding row predicts fragility with no forward pass, better than surface rarity by a paired test; the token's representation on a dozen contexts predicts it better still (§5).
4. **Non-interaction**: a pre-registered factorial screen with a calibrated permutation null showing co-occurrence at a distance is additive and repetition of healthy tokens helps (§6).

Code, per-cell results, the pre-registration document, and a specimen store with model provenance accompany the paper.

---

## 2. Related work

*Glitch tokens.* Anomalous tokens were documented in GPT-2/3 by Rumbelow and Watkins (2023); Fell (2023) coined "unspeakable" for tokens a model cannot repeat, so the bare fact of substitution is not new. Land and Bartolo (2024, *Fishing for Magikarp*) traced the cause to tokenizer–corpus mismatch, detected candidates from unembedding geometry, and verified them with a repetition probe; we use their labels. GlitchHunter (Li et al., 2024) gave a taxonomy; GlitchProber (Zhang et al., 2024) showed glitch tokens produce distinguishable intermediate activations; GlitchMiner (Wu et al., AAAI 2026) used next-token entropy; GlitchQuiz used eight templates. Several note that verification depends on the template. None treats that dependence as a quantity, which is what fragility is.

*Static geometry.* Detecting glitch tokens from the output (`lm_head`) matrix without inference is Magikarp's method and was recently extended by an SVD-based score in "Check Your LLM's Secret Dictionary" (May 2026). We do not claim the idea. Our target is different: failure *among tokens the repetition probe already passes*, and a paired test against surface rarity.

*Gradient starvation of rare tokens.* Yu et al. (ACL 2022) showed rare-token embeddings receive gradient mostly as softmax negatives, in a shared direction; Leviathan (Jan 2026) and TIDE (May 2026, the "Rare Token Problem") build architectures around it. We do not claim the mechanism; §7 credits it and reports what our checkpoint measurements add.

*Copying circuits.* Induction heads (Olsson et al., 2022) copy; copy-suppression heads (McDougall et al., 2023) suppress re-emission of an earlier token when the prediction is confident. The latter is the natural candidate for the deletion mode we observe.

*Hallucination.* Hidden states carrying information the output contradicts is established (Kadavath et al., 2022; Azaria and Mitchell, 2023); hallucination tracking fact rarity is argued by Kalai and Vempala (2024). §7 reports a direct test of the token-level version of that link, which fails.

---

## 3. Setup, and what the failure looks like

**Model and labels.** `allenai/OLMo-2-1124-7B-Instruct` (untied embeddings, vocabulary 100,352; commit `470b1fba`, fp16, greedy decoding throughout). Magikarp's published run on the base model tested 3,171 tokens and verified 442 as glitch; "healthy" below always means *tested and rejected by that probe*.

**The copy probe.** A few-shot prompt — `Repeat the text exactly. Text: … Copy: …` — with the target spliced in **as token ids**. This matters: a leading space re-segments about 90% of tokens regardless of health (glitch 0.083 survival, healthy 0.093), so prompts built as strings measure the tokenizer. The score is the teacher-forced log-probability of the target at its copy position.

**Clean-looking.** A token is *clean-looking* if its single-token copy log-probability exceeds −0.1 — copied with probability above 0.9 in isolation. None of the verified glitch tokens we sampled meets this; 873 of 1,998 random tokens do (1,888 of 6,173 in the replication).

### 3.1 Confident substitution

Thirty-two verified glitch tokens and thirty-two controls were spliced into the model's chat template after *"What is this token? Identify it exactly, then explain step by step what it means:"* and 256 tokens were generated. Two hypotheses from the field reports — that damage amplifies with length, and that glitch-seeded chains converge on a common basin — both failed: the glitch/healthy entropy gap is +0.27 bits at step 8 and −0.29 at step 255, and chain alignment decays to 0.0001 for glitch and 0.0002 for healthy seeds. What happens instead:

| seed | model's answer |
|---|---|
| `.XtraLayout` | *"The token **"layui"** refers to a lightweight, modular JavaScript framework…"* |
| ` Hexatrigesimal` | *"The token **"Calculus"** refers to a branch of mathematics…"* |
| `")){\r\n` | *"The token you've provided, **`){`**, is a closing curly brace…"* |
| ` woman` (control) | *"The token "woman" is a noun in the English language…"* |

**Seed reproduced: glitch 0 / 32, controls 17 / 32.** The model names a plausible different token and reasons about it fluently at 0.85–1.3 bits of entropy. In the factorial experiment of §6, where a trivially answerable question follows the context, the task-completion rate was 1.000 in every arm. The model does not stop; it does not hedge; it proceeds on substituted content. This is the correct statement of the hazard, and it is why entropy-based detectors do not see it.

### 3.2 The identity is still there

On Pythia-1.4b (36 verified glitch, 1,088 controls, eight contexts), we asked whether a token can be identified from its hidden state at each layer by nearest-neighbour matching across contexts within a 1,124-candidate pool (states centred per context):

| layer | glitch | healthy |
|---|---|---|
| 0–12 | 1.000 | 0.988–1.000 |
| 18 | 0.979 | 0.965 |
| 24 (last) | 0.880 | 0.874 |

Glitch tokens are as identifiable as healthy ones at every layer including the last. A logit lens shows the model *continuing* glitch tokens correctly — `idepress` → `ant` at every layer — while unable to reproduce them. Continuation survives; self-reference breaks. The information reaches the readout; the readout substitutes.

---

## 4. Fragility

### 4.1 Design

A fixed bank of random contexts — sequences of common, individually clean filler words of length 8, 16, 32 and 64, with a random interior slot — and a set of tokens. Every token is placed in every context; the cell is the copy log-probability at the slot; a cell **fails** if it is below −0.5 (probability < 0.61). **Fragility** `frag(t)` is the fraction of contexts a token fails in. Contexts are shared across tokens, so any difference in fragility is a token difference.

Pilot: 1,998 id-stratified random tokens + 100 verified glitch tokens × 48 contexts. Replication: 6,173 random and entity-constituent tokens + 100 glitch × 24 new contexts, drawn independently.

### 4.2 The dependence is on the token, not the context

Two-way decomposition of `lp(t,c) = μ + token(t) + context(c) + residual(t,c)`:

| population, measure | token | context | token × context |
|---|---|---|---|
| all tokens, log-probability | 0.952 | 0.001 | 0.047 |
| all tokens, fail indicator | 0.743 | 0.004 | 0.254 |
| clean-looking, log-probability | 0.717 | **0.000** | **0.282** |

No context is systematically hostile: its share is at most 0.007 in either bank, hostility ranges only 0.06–0.19 across contexts, and the clean-looking fail rate is flat from 8 to 64 words. What varies is which token fails in *its* particular contexts. Predicting a held-out cell, the token's fragility from other contexts gives AUC 0.85–0.94; context hostility from other tokens gives 0.56–0.61; token–context similarity adds nothing consistent. The sensitivity is real and token-specific; its trigger is not something we can read off.

### 4.3 Prevalence and stability

| population | n | mean frag | frag ≥ 0.10 | never fails |
|---|---|---|---|---|
| random tokens | 1,998 | 0.090 | 17.5% | 67.0% |
| verified glitch (reference) | 100 | **0.993** | 100% | 0% |
| **clean-looking** | 873 | 0.007 | **1.6%** | 92.1% |
| *replication:* clean-looking | 1,888 | 0.006 | **1.5%** | 95.0% |

The reference row validates the measure — glitch tokens fail everywhere and none passes the single probe. The hazard is narrow but real: **about one token in sixty that every existing detector would pass fails in a tenth or more of ordinary contexts.** It is a stable property: fragility on even-numbered contexts predicts fragility on odd-numbered contexts at AUC 0.996, and `according` scores 0.92 in both banks.

### 4.4 The specimens

All fragile clean-looking tokens are **canonical** — their decoded string re-encodes to the same id, they are NFC-normal, and Magikarp classifies them `OK`. They are not encoding variants. They are ordinary tokens.

| token | id | alone | frag | worst context → emitted (p) | best |
|---|---|---|---|---|---|
| `' according'` | 4184 | −0.001 | **0.92** | `…deliver rel behind [·] sent floor` → **`' sent'` (1.000)** — deleted, copy continues | 0.999 |
| `' due'` | — | −0.000 | 0.42 | `…dif signific while [·] money tre` → deleted; worst cell −23.8 | — |
| `' pueden'` | 41604 | −0.030 | 0.62 | `…became financial contains [·]` → `' pena'` (0.775) | 0.992 |
| `' abbiamo'` | 95396 | −0.020 | 0.60 | `…nullptr park popular [·]` → `' abdom'`→"abdominoplasty" | 0.972 |
| `' sólo'` | 53288 | −0.032 | 0.48 | `…vot [·] ext gr` → `' só'` (0.892), truncated | 0.957 |
| `' 查询'` | 81628 | −0.015 | 0.46 | `…comment began influ [·]` → **`' QUERY'` (0.970)** — translated | 0.877 |
| `' который'` | 98851 | −0.017 | 0.71 | `…whether more redu [·]` → `' что'` | 0.936 |
| `' getSystemService'` | 92476 | −0.053 | 0.38 | `…hope where sales [·]` → `'SystemService'` (0.30) | 0.961 |

Four modes, all confident, all reversed by changing the filler words: **deletion** (the emitted token is the *next* context word — the model skipped the target and kept copying), **substitution** by an orthographic or same-language neighbour, **truncation** to a sub-token, and **translation**. In the specimen store, `according` is deleted at p = 1.000 and entropy **0.00 bits** in three separate contexts while copying at 1.000 in a fourth.

Two regularities we record without having tested: the two catastrophic English cases, `according` and `due`, are words whose next token is almost always `to`, as are several fragile connectives (`mentre`, `allerdings`, `może`, `который`) — fragility may track the *peakedness of a token's continuation prior*, a word the model will not copy into a slot where its obligatory successor is absent; and a deletion at entropy zero is what a copy-suppression head firing on a confident prediction would look like. The remainder of the fragile set is the rare tail of an English-dominant instruction mix: Spanish, Italian, Portuguese, German, Polish, Chinese and Russian words, and Android/Java identifiers.

Every failing and control cell for every fragile and glitch token — context, top-5 emissions with probabilities, entropy, the target's probability, greedy continuation, mode — is stored with a model header (commit, dtype, layers, hidden size, vocabulary, tying, library versions): 228 records (`results/specimens_confident_substitution.jsonl`; `docs/specimens.md`).

---

## 5. Predicting fragility from geometry

Target: fragile (`frag ≥ 0.10` on a held-out half of the contexts) among clean-looking tokens — 13 positives in the pilot, **28 in the replication**, which decides the claim. Three tiers, kept apart because they cost different things: **static** features from the two embedding matrices alone; **dynamic** features of the token's representation on the *training* half of the contexts, predicting the *test* half; and the **behavioural** baseline of measuring copy on the training half. Every geometric feature must beat surface features — token id (rarity), character length, leading space, alphabetic, ASCII.

Single features on the replication (AUC, bootstrap CI): distance of the unembedding row from the centroid **0.678** [0.55, 0.79]; direct-path self-score E_in[t]·E_out[t] **0.655** [0.56, 0.75]; input-row norm 0.60; slot-state norm on training contexts **0.864** [0.78, 0.92]; copy on the training half 0.997.

Cross-validated L2 logistic regression, out-of-fold AUC, with a **paired bootstrap of the improvement over the surface tier**:

| tier | pilot (13 pos) | replication (28 pos) | Δ vs surface, paired |
|---|---|---|---|
| surface only | 0.768 | 0.711 [0.58, 0.83] | — |
| **static geometry** | 0.845 | **0.862 [0.77, 0.94]** | **+0.151 [+0.055, +0.258]** |
| dynamic (representation, train half) | 0.934 | 0.823 [0.72, 0.92] | +0.112 [+0.024, +0.216] |
| static + dynamic | 0.857 | **0.899 [0.82, 0.96]** | **+0.188 [+0.086, +0.302]** |
| behavioural (copy, train half) | 0.994 | 0.996 | — |

Every geometric tier beats surface rarity with an interval that excludes zero, and the pilot's 0.845 reproduces at 0.862 on twice the positives. The signal is concentrated in the **output embedding** — how far the unembedding row sits from the centroid, how weakly the input row excites its own readout — though no single feature exceeds 0.68; the joint model carries it. These are rarity proxies with a mechanism (§7): a row that is seldom the training target moves little.

The practical ladder: **geometry for free** (pre-screen from the weights), **representation for a few forward passes** (the slot-state norm alone reaches 0.86), **measurement for a few more** (near-perfect).

---

## 6. Combinations of tokens do not interact

The compositional hypothesis, in its sharpest form: in `[1, 231, 885, 9911, 1112, 231]` the *repetition* of 231 causes glitch behaviour; in `[1, 231, 885, 9911, 9999, 1922, 1013]` the *co-occurrence* of 885 and 1013 does — members individually healthy, arbitrarily far apart.

**Design.** A carrier context with two slots (i, j); for an ordered pair (a, b), the four cells S(00), S(10), S(01), S(11) with control tokens filling absent slots (so length and every other position are fixed; two *different* controls in S(00) so it is not itself a repetition), and `I = S11 − S10 − S01 + S00`, the damage not attributable to either token alone. Distance is the variable: two long filler sequences, slots at 2 and 2 + d for d ∈ {2, 4, 8, 16, 32, 64, 128}, sharing a prefix so distance is not confounded with content. Pool of 64 tokens, first all individually clean, then 32 clean + 32 *weak* (tokens that copy sometimes). 63,014 cells per run.

**Null.** ~4,000 simultaneous tests on a difference of four noisy cells. Per replicate the S11 matrix is fitted additively, residuals are standardised per row and column, permuted, and averaged; BH-FDR at q = 0.05 with a max-statistic family-wise test; the verdict threshold is taken from twenty no-interaction datasets simulated from the additive fit with the observed heteroscedastic noise. Standardisation was forced by calibration — without it a heteroscedastic null gave 5–18 false rejections and family-wise p ≈ 0.003; with it, 0 in 10/10 null runs. Design, null and failure criteria were fixed before the screen ran.

**Result.** Clean pool: 1 significant pair of 4,032 (matched null 95th percentile 4). Mixed pool: 1 of 4,032 (matched null 14). Neither passes. The averages, which do not depend on the per-pair test's power, settle it:

| a × b (mixed pool) | n | mean S11 | **mean I** | se |
|---|---|---|---|---|
| clean × clean | 992 | −0.027 | +0.0011 | 0.0004 |
| clean × weak | 1024 | −0.146 | +0.0030 | 0.0016 |
| weak × clean | 1024 | −0.195 | +0.0033 | 0.0014 |
| **weak × weak** | 992 | **−0.313** | **+0.0076** | 0.0027 |

Weak tokens copy twelve times worse — additively. The interaction is slightly *positive* everywhere: two weak tokens at any distance copy marginally better together than their parts predict. Off-diagonal mean I by distance is −0.0007 to +0.0055 at every d. Positive-control glitch tokens dropped to −3 to −6.5 against a clean baseline of −0.02, so the scorer sees real effects. **Repetition of a healthy token helps** (diagonal I = +0.0066, p = 0.000), growing with distance from +0.005 at d = 2 to +0.018 at d = 128 — induction working. Where a repeated token *did* fail, it failed in one filler sequence and not the other: context, not pair.

The 453 clean-pool cells that did fail were concentrated in five tokens and did not recur across filler sequences (overlap 3, 0, 1, 0, 0, 0, 2). That structure — fragile token × hostile context — is what sent us to §4.

---

## 7. Discussion

**Where the geometry comes from.** The mechanism is known: rare tokens' rows are starved of token-specific gradient and receive mostly the shared component (Yu et al., 2022; Leviathan; TIDE). Our checkpoint measurements on seven pretraining lineages confirm it for behaviourally labelled glitch tokens — token-specific residual suppressed to 0.15–0.74 of healthy while shared drift is comparable or larger, replicating in a LLaMA-lineage model with no shared ancestry (0.573), and reversing under weight tying as predicted (shared 1.17–1.23×) — and add one consequence: post-training deltas have essentially no shared drift (total and residual ratios agree to three decimals in eight families), so the up-to-35% of rows post-training never touches are preserved, not damaged. The output row that was rarely a *target* is the one §5 reads fragility from.

**Fragility is not a mechanism for hallucination.** Confident substitution on retained information looks like a hallucination in miniature, so we tested the token-level version directly: 1,255 real entities (stdlib, torch and transformers identifiers from the live interpreter; proper nouns in several scripts) used in a natural task, scored on the entity's final component as exact / near-miss / miss. The model reproduces real API names exactly 82.5% of the time, mangles them 1.7% (`curses.BUTTON5_PRESSED` → `C_BUTTON5_PRESSED`, `torch.BoolTensor` → `bool_tensor`), and dodges 15.7% with a bare `import` line; proper nouns 99.4% exact. Constituent fragility adds **−0.001 [−0.008, +0.006]** to rarity, the single probe and the model's own familiarity as a predictor of getting the name wrong. The resemblance is descriptive, not causal.

**What this means for deployment.** Systems that assemble prompts programmatically — retrieval, tool output, templated code — put tokens into contexts no one typed. A token audit that runs each token alone certifies `according`. Fragility costs N × C forward passes with C ≈ 24, is stable enough to measure once per model, can be pre-screened from the embedding matrices, and emits its failures with no uncertainty signal a guardrail could catch — which is the reason to measure it rather than wait for it.

**Limitations.** Behavioural results are on one model (plus Pythia-1.4b for §3.2); the interaction and fragility code runs unchanged on any model with a label set. Copy log-probability is a proxy for the reasoning-mode behaviour users care about; it agrees with the generations we inspected, and it is what made an N² screen and an N × C matrix affordable. The §5 paired intervals exclude zero but are wide (+0.06 to +0.26); a second family would narrow them. The fragility contexts are random common words, not natural text; "no context is hostile" is a claim about this bank. Which contexts break a given token remains unexplained; the continuation-prior and copy-suppression hypotheses in §4.4 are the obvious next tests.

---

## 8. Conclusion

Glitch behaviour is not compositional and it is not an information loss. Two tokens at a distance do not conspire; a glitch token's identity reaches the last layer and the readout substitutes anyway, confidently, and the model finishes the task on the wrong content. What varies with surroundings is a property of the token: about one in sixty tokens that pass every single-context probe fails to copy in a tenth or more of ordinary contexts — deleted, swapped, truncated or translated — with the trigger unpredictable but the fact stable, replicable, and readable from the output embedding better than from rarity alone. Detectors that score tokens one at a time are measuring the wrong object for systems that assemble text by machine. Fragility is the right one, and it is cheap.

---

## Appendix A. Pre-registration and calibration (§6)

Design, null model, verdict criterion and failure criteria were written before the screen ran. Two amendments followed synthetic calibration and preceded any real data: residuals standardised per row/column before permutation (heteroscedastic null otherwise gave 5–18 false rejections, p ≈ 0.003; 0 after), and the fixed "≤ 5% of pairs" threshold replaced by the matched-simulation threshold (a true null yields 0–2 rejections, not ~200). Distance was made the design variable on a collaborator's correction. Power, corrected: a single isolated effect is recovered 4/5 at 1.5× cell noise with R = 14 — the Bonferroni-like worst case; with many true effects BH's threshold relaxes.

## Appendix B. Negative results, one line each

| hypothesis | test | result |
|---|---|---|
| "no space/bracket before it" is a glitch property | prefix survival, real o200k ids + Pythia | tokenizer boundary; glitch 0.083 vs healthy 0.093 — not glitch-specific |
| BPE-unreachable adjacency causes damage | 200 within-pair order swaps | unreachable *easier*: +0.94 [+0.61, +1.26] |
| contiguous n-grams interact | reproduction vs n, individually perfect tokens | falls 1.00 → 0.65 at n = 8 but *sub*-multiplicatively (independence predicts 0.35) |
| direct path E_in·E_out predicts the substitute | top-1 agreement | 0.003 |
| orthographic completion is the mechanism | completable-fragment feature | AUC 0.545; 11% of failures |
| generation length amplifies damage; glitch chains share a basin | 256-step chains | gap +0.27 → −0.29; alignment → 0.0001 both groups |
| trajectory persistent homology discriminates | H₁ of 64 chains | best AUC 0.597, underpowered |
| post-training creates new glitch tokens | 8 stage series, boundary control | real movement only in OLMo-2; jitter elsewhere |
| fragile tokens are Unicode/duplicate variants | canonicality of all fragile tokens | 100% canonical, NFC, Magikarp `OK` |
| fragility explains entity near-miss hallucination | 1,255 entities, three control tiers | Δ −0.001 [−0.008, +0.006] |

## Appendix C. Data

`docs/DATA_MANIFEST.md` maps every section to its files. Key files: `results/fragility.json`, `fragility_L.json` and `results/compact/*_compact.pt` (matrices, contexts, ids); `results/fragility_L.pt` (with slot hidden states); `results/fragility_predict_L.json` (§5, incl. paired bootstraps); `results/interaction_screen*.json` and compacts (§6); `results/reasoning_drift.json`, `forward_geom.json` (§3); `results/hallucination_bridge_rescored.json`, `entities.json` (§7); `results/specimens_confident_substitution.jsonl`, `docs/specimens.md`. Code in `src/cut/`. All runs used one A100-40GB; the largest is 25 minutes.
