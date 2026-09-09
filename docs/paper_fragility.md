# Fragile Tokens: Glitch Behaviour Is Confident Substitution, and It Depends on Context

**Jeffrey R. Gordon** · draft v1, 5 September 2026

---

## Abstract

Glitch tokens are detected one token at a time: a vocabulary entry is scored in isolation, usually by asking a model to repeat it, and tokens that fail are labelled. Field reports of glitch-like behaviour in reasoning models describe something the isolated probe cannot see — behaviour that appears only for particular combinations of tokens, or only in particular surroundings. We ask whether glitch behaviour is a property of a token, of a combination of tokens, or of a token *in a context*, and we answer with a series of pre-registered experiments on an open 7B instruction-tuned model with published glitch labels.

Three findings survive. **(1) The failure mode is confident substitution, not degradation.** Asked to identify a glitch token, the model names a different token and reasons fluently about it: 0 of 32 verified glitch tokens were reproduced against 17 of 32 controls, at next-token entropy below 1.3 bits, with the downstream task completed in every case. The token's identity is still linearly recoverable from the final residual-stream layer; the readout substitutes. **(2) Combinations of tokens at a distance do not interact.** In a factorial screen over 8,064 ordered pairs, distances 2–128, with a calibrated permutation null, the effect of two tokens is additive to within 0.008 logprob in every cell type — including pairs of individually weak tokens with ample room to fail. Repeating a healthy token at a distance *helps*. **(3) A previously unmeasured hazard, fragility, is real and rare.** Placing tokens in banks of random ordinary contexts (2,098 × 48, then an independent 6,273 × 24), 1.5–1.6% of tokens that pass the standard single probe fail to copy in at least 10% of contexts. They are canonical, ordinary tokens: the common English word `according` copies perfectly in isolation and is deleted — probability 1.000, entropy 0.00 bits — in 44 of 48 contexts, and again in the second bank. Fragility is a stable property of the token (AUC 0.996 across context halves), not of the context (context share of variance ≈ 0.000), and it is predictable from the geometry of the output embedding alone: cross-validated AUC 0.86 against 0.71 for surface rarity, a paired-bootstrap improvement of +0.15 [+0.06, +0.26] on 28 positives. Fragility does **not**, however, predict near-miss hallucination on 1,255 real named entities beyond rarity and the model's own familiarity with the entity; the structural resemblance between confident substitution and hallucination is an analogy we tested and could not turn into a mechanism.

We connect the surviving findings to training dynamics: during pretraining, under-trained rows receive shared drift but almost no token-specific update, and the output row that is rarely a target is the one that later fails to be read out. We report ten hypotheses that did not survive their controls, including several that would have been the headline had the controls not been run, and we release every confident substitution as a specimen with full model provenance.

---

## 1. Introduction

A glitch token is a vocabulary entry a language model cannot handle: asked to repeat it, spell it, or say what it is, the model produces something else. The canonical example is `SolidGoldMagikarp` in GPT-3; every model family since has its own set. The existing detectors — Magikarp's embedding indicators followed by a repetition probe [Land & Bartolo 2024], GlitchMiner's entropy criterion, GlitchHunter/GlitchProber's clustering and activation classifiers, GlitchQuiz's template battery — share one design choice. **They score a token by itself.**

The field reports that motivated this work do not fit that design. Users of reasoning models describe two tokens in the o200k vocabulary (ids 128188 and 152383) that a reasoning model "struggles to identify as anything," "commonly conflates as Arabic script," while a non-reasoning model classifies them as Chinese or Gujarati and an unauthenticated model "spews out a bunch of word fragments in a wide variety of language." They report that the tokens can be used alone or after a newline but break when preceded by a space or bracket, and that they can be placed back-to-back. And they observe that a single instance often does nothing, while the behaviour emerges in longer, multi-token settings.

This paper takes those reports as hypotheses and tests them on open weights. The questions are:

- **Q1.** What does the model actually do when it hits a glitch token in a reasoning-style prompt? Degrade, derail, or something else?
- **Q2.** Is glitch behaviour compositional — caused by *combinations* of tokens, possibly individually healthy, possibly far apart?
- **Q3.** Is it *contextual* — the same token fine in one surrounding and broken in another — and if so, is that predictable from the geometry of the embeddings or of the representation as it evolves through the network?

The answers, in order: substitution; no; yes and partly.

### Contributions

1. A behavioural characterisation of glitch failure as **confident, low-entropy substitution with intact task completion**, and evidence that the token's identity survives to the last layer — locating the failure at readout. (§3)
2. A **pre-registered factorial screen** for non-contiguous token interaction with a calibrated permutation null, showing that co-occurrence at distances 2–128 is additive in every cell type and that repetition helps healthy tokens. (§4)
3. **Fragility**: a token-level quantity — the fraction of ordinary contexts in which a token fails to copy — that is invisible to single-context probes, affects ~1.5% of tokens that pass them, is stable, replicates across context banks, and is predictable from the output embedding better than from surface rarity. (§5–6)
4. A mechanistic connection to pretraining dynamics, a **specimen store** of every confident substitution with model provenance, and a documented set of **negative results** with the controls that produced them — including a direct test of whether fragility explains entity-level hallucination, which it does not. (§7–8)

All code, per-cell results, and the pre-registration document are in the accompanying repository.

---

## 2. Setting

**Model.** Behavioural experiments use `allenai/OLMo-2-1124-7B-Instruct` (untied embeddings, vocabulary 100,352). It is the largest fully open instruction-tuned model for which an independent set of behaviourally verified glitch labels exists: Magikarp's published run on the base model tested 3,171 tokens and verified 442. We use those labels for the positive class where labels are needed and note that "healthy" always means *tested by Magikarp and rejected*, so the control tokens have passed the same probe the glitch tokens failed.

**The copy probe.** Where a continuous score is needed we use teacher-forced copy log-probability: a few-shot prompt (`Repeat the text exactly. Text: … Copy: …`) with the target spliced in **as token ids**, never through the tokenizer on a concatenated string. The distinction matters (§8.1): a leading space re-segments about 90% of tokens regardless of health, so prompts built as text would measure the tokenizer.

**Verification threshold for "clean."** A token is *clean-looking* if its single-token copy log-probability exceeds −0.1 (it is copied with probability > 0.9 in isolation). None of the 100 verified glitch tokens we sampled meets this; 873 of 1,998 random tokens do.

### 2.1 Related work, and what is and is not new here

*Glitch tokens.* Anomalous tokens were documented in GPT-2/3 by Rumbelow and Watkins (2023) and Fell (2023), who coined "unspeakable" for tokens a model cannot repeat; the substitution behaviour itself is therefore not new. Land and Bartolo (2024, *Fishing for Magikarp*) traced the cause to tokenizer–corpus mismatch, detected candidates from unembedding-matrix indicators, and verified them with a repetition probe — the labels we use. Li et al. (2024, GlitchHunter) gave a taxonomy and clustering detector; Zhang et al. (2024, GlitchProber) showed glitch tokens produce distinguishable intermediate-layer activations and attention; Wu et al. (AAAI 2026, GlitchMiner) used next-token entropy as the objective; GlitchQuiz used a battery of templates. All score a token by itself. Several note that verification depends on the template; none measures that dependence as a quantity.

*Static geometry.* Using the output (`lm_head`) matrix to find glitch tokens without inference is Magikarp's core idea and was recently extended by "Check Your LLM's Secret Dictionary" (May 2026), whose SVD-based Weighted Projection Score recovers known glitch tokens from `lm_head` alone. Our §6 does not claim that idea. What is new is the *target*: predicting failure among tokens that the repetition probe already passes, and showing with a paired test that output-row geometry beats surface rarity for that target.

*Gradient starvation of rare tokens.* The mechanism in §7 has clear precedent. Yu et al. (ACL 2022, "Rare Tokens Degenerate All Tokens") showed that rare-token embeddings receive gradient predominantly as softmax negatives, pushing them in a common direction; Leviathan (January 2026) decouples input and output representations because a lookup table receives too little gradient for rare tokens; TIDE (May 2026) names the "Rare Token Problem" — Zipfian frequency leaves rare rows chronically under-trained — and re-injects token identity at every layer to counter it. We do not claim the mechanism. Our contribution there is measurement: the shared-drift / token-specific decomposition applied across pretraining checkpoints of seven model lineages with behaviourally labelled glitch tokens, the tied/untied contrast, and the post-training result that follows from it.

*Copying and its suppression.* Induction heads (Olsson et al., 2022) are the copying circuit; McDougall et al. (2023) describe copy-suppression heads that actively suppress a token's re-emission when the model's prediction is confident. Copy suppression is a natural candidate mechanism for the deletion mode in §5.4 and we flag it as such; we have not tested it.

*Hallucination.* That hidden states carry information the output contradicts is established (Kadavath et al., 2022; Azaria and Mitchell, 2023); that hallucination rate tracks how rarely facts were seen is argued by Kalai and Vempala (2024). §3.4 and §8.9 test, and reject, the token-level version of that link.

*What we believe is new.* (i) Fragility as a measured, stable, per-token quantity on tokens every detector passes — its prevalence, its variance decomposition (token, not context), its four modes, and specimens such as `according`. (ii) A pre-registered, calibrated factorial test showing non-contiguous co-occurrence is additive. (iii) The quantified characterisation of glitch failure as low-entropy substitution with identity retained to the last layer, in reasoning-style prompts. (iv) The cross-family post-training result and its explanation. (v) The specimen store with model provenance. Our literature check covers work through May 2026 plus a search of later postings; a 2026 paper measuring context-dependent copy failure could exist without our having found it, and a full related-work pass should precede submission.

---

## 3. The failure mode is confident substitution

### 3.1 Design

Thirty-two verified glitch tokens and thirty-two Magikarp-rejected controls were each spliced into the model's own chat template after *"What is this token? Identify it exactly, then explain step by step what it means:"*, and the model generated 256 tokens greedily (EOS suppressed so every chain has equal length; a chain that emits EOS at step 1 would otherwise be silently one step long). At every step we recorded next-token entropy and the last-layer residual state. Two hypotheses from the field reports were pre-specified: that damage **amplifies** with generation length, and that chains seeded by different glitch tokens **converge** on a common high-entropy basin.

### 3.2 Result

Both hypotheses failed. The glitch/healthy entropy gap is +0.27 bits at step 8 and −0.29 at step 255 — it wobbles around zero. Chain alignment (mean pairwise cosine of centred last-layer states across seeds) decays from 0.016 to 0.0001 for glitch seeds and from 0.017 to 0.0002 for healthy seeds; there is no attractor.

What happens instead:

| seed token | model's answer |
|---|---|
| `.XtraLayout` | *"The token **"layui"** refers to a lightweight, modular, and flexible JavaScript framework primarily used for building web interfaces…"* |
| ` Hexatrigesimal` | *"The token **"Calculus"** refers to a branch of mathematics that deals with the study of change, motion, and accumulation…"* |
| `")){\r\n` | *"The token you've provided, **`){`**, is a closing curly brace…"* |
| ` woman` (control) | *"The token "woman" is a noun in the English language. Here's a step-by-step explanation…"* |

**Reproduction of the seed token: glitch 0 / 32, healthy 17 / 32.** The model names a different, plausible token and reasons about it fluently. Entropy while doing so is 0.85–1.3 bits — it is not uncertain. In the pair experiment of §4, where a trivially answerable question follows the context, the task-completion rate was **1.000 in every arm**: the model never stops working; it works on substituted content.

This is why entropy-based detectors miss the phenomenon, and it is the correct statement of the agentic hazard: no uncertainty signal is emitted for a guardrail to catch.

### 3.3 The identity is still there

Is the substitution an information loss or a readout failure? On Pythia-1.4b (36 verified glitch tokens, 1,088 controls), we placed each token at the end of eight different contexts, took the hidden state at the token's position at every layer, and asked whether the token could be identified by nearest-neighbour matching across contexts within a pool of 1,124 candidates (states centred within context to remove the shared component).

| layer | glitch top-1 | healthy top-1 |
|---|---|---|
| 0–12 | 1.000 | 0.988–1.000 |
| 18 | 0.979 | 0.965 |
| 24 (last) | 0.880 | 0.874 |

Glitch tokens are as identifiable as healthy ones at every layer, including the last. The information reaches the top of the network; the unembedding readout substitutes. This is consistent with GlitchProber's finding that glitch tokens leave distinguishable intermediate activations, and it is the empirical counterpart of the design intuition TIDE (2026) builds on — that token identity persists beneath the context at every layer — here measured for the tokens where persistence and readout come apart. A logit lens at the same positions shows why the standard framing of "untrained" is incomplete: `idepress` — a verified glitch token — is *continued* correctly to `ant` at every layer, as `ccording` is to `ly` / ` to`. Continuation survives; **self-reference** (copying, quoting, spelling, reasoning-about-the-token) is what breaks. Pretraining on running text rewards only the former.

### 3.4 Relation to hallucination — an analogy, tested

Stripped to its structure, confident substitution is a hallucination in miniature: fluent, confident, wrong, a plausible neighbour of the truth, with the correct information present internally. That resemblance invites the hypothesis that token-level fragility (§5) is a *mechanism* for one class of hallucination — the plausible near-miss on a rare entity: the fabricated API that almost exists, the name with a syllable swapped. We tested it directly (§8.9). It did not hold: on 1,255 real entities in a natural-use task, constituent fragility adds nothing to rarity and the model's own familiarity as a predictor of getting the name wrong. The resemblance stands as a description of *what the failure looks like*, not as an account of why models hallucinate.

---

## 4. Combinations at a distance do not interact

### 4.1 The claim and the design

The compositional claim, stated by our collaborator in its sharpest form: in a context such as `[1, 231, 885, 9911, 1112, 231]` the *repetition* of 231 causes glitch behaviour; in `[1, 231, 885, 9911, 9999, 1922, 1013]` the *co-occurrence* of 885 and 1013 does. Members may be individually healthy and arbitrarily far apart. Earlier in this project we had tested adjacent pairs and contiguous n-grams — the wrong object.

A factorial screen answers it. A carrier context has two designated slots (i, j); for an ordered pair (a, b):

```
S(00)  c1 at i, c2 at j            controls in both slots
S(10)  a  at i, c2 at j
S(01)  c1 at i, b  at j
S(11)  a  at i, b  at j
I(a,b) = S11 − S10 − S01 + S00     damage not attributable to either token alone
```

The absent condition substitutes a control token rather than deleting, so length and every other position are fixed; S(00) uses two *different* controls so it is not itself a repetition cell. The diagonal a = b is the repetition case. **Distance is the variable**: two long neutral filler sequences of common words; for each d ∈ {2, 4, 8, 16, 32, 64, 128} the carrier is `filler[:d+4]` with slots at 2 and 2+d; carriers at different distances share a prefix so distance is not confounded with filler content. Pool: 64 tokens, in the first run all individually clean (so any interaction cannot be inherited from a member), in a second run 32 clean + 32 *weak* (single-token copy lp in (−3, −0.3): tokens with room to fail). S is the copy log-probability at the two slot positions. 63,014 cells per run.

**The null.** N² ordered pairs is ~4,000 simultaneous tests and the interaction is a difference of four noisy cells. Per replicate, the S11 matrix is fitted with an additive model (row and column means), residuals are standardised per row and column by a robust scale, permuted within replicate, and averaged; per-pair p-values go through Benjamini–Hochberg at q = 0.05, with a max-statistic family-wise test alongside. The verdict threshold is *empirical*: twenty no-interaction datasets are simulated from the additive fit with the observed heteroscedastic noise and pushed through the identical pipeline. **Detected** means more rejections than their 95th percentile *and* family-wise p < 0.05. All of this, and the failure criteria, were fixed before the screen ran.

Calibration on synthetic data mattered: without standardisation, a null in which one-eighth of tokens are 3× noisier produced 5–18 false rejections and family-wise p ≈ 0.003 in every run; with it, 0 rejections in 10/10 null runs.

### 4.2 Result: additive

| pool | BH-significant pairs / 4,032 | matched-null 95th pct | family-wise p |
|---|---|---|---|
| clean | 1 | 4 | 0.035 |
| clean + weak | 1 | 14 | 0.005 |

Neither run passes the pre-registered criterion. The averages are more informative than the test, because they do not depend on its power:

| a-kind × b-kind (mixed pool) | n | mean S11 | **mean I** | se |
|---|---|---|---|---|
| clean × clean | 992 | −0.027 | +0.0011 | 0.0004 |
| clean × weak | 1024 | −0.146 | +0.0030 | 0.0016 |
| weak × clean | 1024 | −0.195 | +0.0033 | 0.0014 |
| **weak × weak** | 992 | **−0.313** | **+0.0076** | 0.0027 |

Weak tokens copy twelve times worse than clean ones — the *additive* main effect. The interaction, the damage beyond what each token does alone, is slightly **positive** in every cell type. Two weak tokens at any distance from 2 to 128 copy marginally *better* together than additivity predicts. Off-diagonal mean I by distance is −0.0007 to +0.0055 at every d. Across both pools, 8,064 ordered pairs, and 14 replicates each, there is no compositional damage from co-occurrence at a distance. The scorer is not the problem: positive-control glitch tokens dropped to −3 to −6.5 logprob against a clean baseline of −0.02.

### 4.3 Repetition helps healthy tokens

On the clean diagonal, mean I = +0.0066 (permutation p = 0.000 against the off-diagonal null), and the benefit **grows with distance**: +0.005 at d = 2 → +0.018 at d = 128. A repeated healthy token copies better because its first occurrence is in context — induction working as designed. The mixed pool's diagonal is negative on average (−0.013), but that is driven by a handful of tokens in *one* filler sequence at d = 128 (`Tools`: I = −1.39 in filler 0, −0.000 in filler 1). The other filler at the same distance shows nothing. Repetition-specific failure exists — the 2×2 controls for single-occurrence fragility — but it is a property of (token, context), not of (token, token).

### 4.4 What the failures that do exist look like

Even in the clean pool, 453 of 57,344 cells (0.8%) fell below −0.5. They were concentrated in five tokens (47% of failing cells: `.Ver` 111 failures in slot i, `Sentence` 118 in slot j, ` Patriots` 91) and they **did not reproduce across fillers**: same-distance overlap of failing cells between the two filler sequences was 3, 0, 1, 0, 0, 0, 2. At d = 4 one filler produced 68 failures and the other 1. The failures factor as **fragile token × hostile context**, which is exactly what an additive-model residual absorbs. That observation defines the next experiment.

---

## 5. Fragility: the hazard single-context probes miss

### 5.1 Design

A fixed bank of 48 random contexts — sequences of common, individually clean filler words of length 8, 16, 32, 64 (twelve each), with a random interior slot — and 2,098 tokens: 1,998 sampled with stratification over vocabulary-id terciles (uniform sampling skews to the rare tail) plus 100 verified glitch tokens as a reference. Every token in every context; the cell is copy log-probability at the slot. **Fragility** `frag(t)` is the fraction of contexts with slot log-probability below −0.5 (probability < 0.61). The same contexts are used for every token, so any difference in fragility is a token difference.

### 5.2 Context dependence, quantified

Two-way decomposition of `lp(t, c) = μ + token(t) + context(c) + residual(t, c)`:

| population / measure | token | context | **token × context** |
|---|---|---|---|
| all tokens, log-probability | 0.952 | 0.001 | 0.047 |
| all tokens, fail indicator | 0.743 | 0.004 | 0.254 |
| clean-looking tokens, log-probability | 0.717 | **0.000** | **0.282** |

**No context is systematically hostile.** The context main effect is essentially zero, hostility ranges only 0.10–0.19 across the 48 contexts, and the clean-looking fail rate is flat across lengths 8 → 64 (0.005–0.008). The dependence is entirely token-specific: a given token fails in *its* contexts. At the cell level, predicting `fail(t, c)` on held-out contexts, the token's fragility from other contexts gives AUC 0.940, context hostility from other tokens 0.559, and token–context similarity adds nothing over the two main effects (0.942 → 0.930). The sensitivity is real; *which* context triggers it is not predictable from anything we measured.

### 5.3 Prevalence

| population | n | mean frag | frag ≥ 0.10 | frag ≥ 0.25 | never fails |
|---|---|---|---|---|---|
| random tokens | 1,998 | 0.090 | 17.5% | 11.9% | 67.0% |
| verified glitch (reference) | 100 | **0.993** | 100% | 100% | 0% |
| **clean-looking** (pass the single probe) | 873 | 0.007 | **1.6%** | 0.8% | 92.1% |

The reference row validates the measure — verified glitch tokens fail in 99.3% of contexts, and none of them passes the single probe. The hazard is narrow but real: about **one in sixty tokens that every existing detector would pass fails in at least 10% of ordinary contexts**, and one in twelve fails at least once in 48. Fragility is stable — fragility on even-numbered contexts predicts fragility on odd-numbered contexts at AUC 0.996 — so it is a property of the token that can be measured once.

**Replication.** An independent bank — 6,273 tokens (1,973 of them the constituent tokens of the real entities used in §8.9) × 24 new contexts — reproduces every number: clean-looking tokens (n = 1,888) at frag ≥ 0.10: **1.5%** (28 tokens); never failing: 95.0% (fewer contexts, fewer chances); glitch reference 0.987; context share of variance 0.001; `according` fragility 0.92 again, now with mean log-probability −13.3.

### 5.4 The specimens

All fourteen fragile clean-looking tokens are **canonical**: their decoded string re-encodes to the same id, they are NFC-normal, and Magikarp classifies them `OK`. They are not Unicode variants or duplicate encodings. They are ordinary tokens.

| token | id | single probe | frag | worst context → model emits | best context |
|---|---|---|---|---|---|
| `' according'` | 4184 | −0.001 | **0.92** | `…deliver rel behind [·] sent floor` → **`' sent'` p = 1.000** — the word is deleted and the copy continues | p = 0.999 |
| `' pueden'` | 41604 | −0.030 | 0.62 | `…became financial contains [·]` → `' pena'` (0.78) | 0.992 |
| `' abbiamo'` | 95396 | −0.020 | 0.60 | `…nullptr park popular [·]` → `' abdom'` → "abdominoplasty" | 0.972 |
| `' sólo'` | 53288 | −0.032 | 0.48 | `…vot [·] ext gr` → `' só'` (0.89), truncated | 0.957 |
| `' getSystemService'` | 92476 | −0.053 | 0.38 | `…hope where sales [·]` → `'SystemService'` (0.30) | 0.961 |
| `' será'` | 33998 | −0.048 | 0.29 | `…includ music inter [·]` → `' sé'`+`'ra'` = "séra" | 0.994 |

Four modes, all confident (0.6–1.0), all reversed by changing the filler words: **deletion** (the token is skipped and copying resumes from the next word), **substitution** with an orthographic or same-language neighbour, **truncation** to a sub-token, and — seen in the second bank — **translation**: `' 查询'` ("query") is emitted as `' QUERY'` at p = 0.970 and `' который'` ("which") as `' что'`. They are the reasoning-mode behaviours of §3 — `Hexatrigesimal` → "Calculus" — now reproduced on tokens that pass every probe, in a copy task, with the switch being nothing but the surrounding ordinary words.

`according` is the specimen to remember. It is the canonical token for a common English word, id 4184, low enough to be among the most frequent pieces in the vocabulary. In isolation and in 4 of 48 contexts it copies at 0.999. In the other 44 the model deletes it with probability 1.000 — entropy 0.00 bits — and continues the copy correctly without it. The second bank adds `due` (single probe −0.000, fragility 0.42, worst −23.8) with the same deletion signature. Both are words whose next token is almost always `to`; so are several of the other fragile connectives (`mentre`, `allerdings`, `może`, `который`). We record, without having tested it, the hypothesis that fragility tracks the *peakedness of a token's continuation prior*: a word that strongly predicts its successor is the one the model refuses to copy into a position where that successor is absent. Copy-suppression heads (McDougall et al., 2023), which suppress re-emission of an earlier token when the model is confident, are the obvious circuit-level candidate for the deletion mode; the deletions here occur at probability 1.000 and entropy 0.00 bits, which is what a suppression head firing on a confident prediction would look like. The rest of the fragile set is the rare tail of an English-dominant instruction mix: Spanish, Italian, Portuguese, German, Polish, Chinese and Russian words, and Android/Java identifiers.

Every failing and control cell for every fragile and glitch token — the exact context, the top-5 emitted tokens with probabilities, entropy, the target's probability, the greedy continuation, a mode label, and a model header with commit hash, dtype, layer count, vocabulary and library versions — is stored as a specimen (`results/specimens_confident_substitution.jsonl`, 228 records; `docs/specimens.md`).

---

## 6. Predicting fragility from geometry

The target is fragile (`frag ≥ 0.10` on the held-out half of contexts) among clean-looking tokens. The pilot bank gave 873 clean-looking tokens and **13 positives**; the replication bank gives 1,888 and **28 positives**, and it is the replication that decides the claim.

Three tiers, kept separate because they cost different things: **static** features computed from the two embedding matrices with no forward pass; **dynamic** features of the token's representation on the *training* half of the contexts (spread of the slot state across contexts, retention of its own identity as cosine to its unembedding row, state norm), predicting failures on the *test* half; and the **behavioural** baseline of simply measuring copy on the training half. Every geometric feature must beat surface features — token id (rarity), character length, leading space, alphabetic, ASCII.

Single features on the replication bank (28 positives), AUC with bootstrap 95% CI:

| tier | feature | AUC |
|---|---|---|
| surface | id (rarity); ascii | 0.67; 0.67 (reversed) |
| static | **distance of the unembedding row from the centroid** | **0.678** [0.55, 0.79] |
| static | **direct-path self-score E_in[t]·E_out[t]** (final-LN folded) | **0.655** [0.56, 0.75] |
| static | input-row norm, input centroid distance | 0.59–0.60 |
| dynamic | slot-state norm (train contexts) | **0.864** [0.78, 0.92] |
| dynamic | identity retention (cos to own unembedding) | 0.61 |
| behaviour | mean copy lp on train half | 0.997 |

Cross-validated L2 logistic regression, five-fold out-of-fold AUC, with a **paired bootstrap of the improvement over the surface tier** — the statistic that decides whether geometry adds anything:

| tier | pilot (13 pos) | **replication (28 pos)** | **Δ vs surface, paired** |
|---|---|---|---|
| surface only (id, length, space, alpha, ascii) | 0.768 | 0.711 [0.58, 0.83] | — |
| **static geometry** | 0.845 | **0.862 [0.77, 0.94]** | **+0.151 [+0.055, +0.258]** |
| dynamic (train-half representation) | 0.934 | 0.823 [0.72, 0.92] | +0.112 [+0.024, +0.216] |
| static + dynamic | 0.857 | **0.899 [0.82, 0.96]** | **+0.188 [+0.086, +0.302]** |
| behavioural (train-half copy) | 0.994 | 0.996 | — |

The pre-registered bar — CV AUC ≥ 0.65, CI excluding 0.5, above the surface baseline — is met, and on the replication the "above surface" clause is established: every geometric tier beats the surface tier with a paired-bootstrap interval that excludes zero. The pilot's 0.845 was not a fluke of thirteen tokens; it reproduces at 0.862 on twice the positives.

What the geometry says: **fragility is predictable with no forward pass, and the signal is concentrated in the output embedding** — how far the token's unembedding row sits from the centroid, and how weakly its input row excites its own readout — with the caveat that no single static feature exceeds 0.68; it is the joint model that reaches 0.86. Those features are rarity proxies with a mechanism attached (§7): a row that is seldom the training target moves little and stays near initialisation. That output-row geometry finds *glitch* tokens is Magikarp's method and the Secret Dictionary paper's; that it still carries signal *within the set Magikarp's probe passes* is the addition here. The token's evolving representation on a dozen contexts (its slot-state norm alone reaches 0.86) adds further, and simply measuring copy on a dozen contexts is near-perfect. So the practical ladder is: geometry for free, representation for a few forward passes, measurement for a few more.

---

## 7. Connection to training dynamics

The mechanism here is not ours. That rare tokens' embedding rows are starved of gradient — receiving it mostly as softmax negatives, in a direction shared with every other rare row — was shown by Yu et al. (2022) and is the premise of Leviathan and TIDE (2026). What this section adds is measurement on real pretraining runs with behaviourally labelled glitch tokens: two facts from the checkpoint-level phase of this project that explain where the §6 signal comes from and why it lives in the output row.

**During pretraining, under-trained rows are dragged, not skipped.** Decomposing each checkpoint-to-checkpoint update of the embedding matrix into a *shared drift* (the component along the mean update direction) and a *token-specific residual*, verified glitch rows receive shared drift comparable to or exceeding healthy rows while their token-specific residual is suppressed to 0.15–0.74 of the healthy value. This replicates in six GPT-NeoX-lineage models across Pile, Dolma and OLMo-2 data, dense and MoE, and in LLM360/Amber — a LLaMA-architecture, SentencePiece, untied model sharing no lineage with the others (early-half residual ratio 0.573; 0.575 with whitespace tokens excluded). The mechanism is gradient sparsity: an input-embedding row is updated with token-specific signal only on steps where its token is in the batch; the shared drift comes from everything else. The row ends up displaced along a direction that carries no information about the token.

**Weight tying changes the mechanism in the predicted direction.** In four HuggingFaceFW ablation models with identical architecture and tokenizer but tied embeddings, glitch rows' shared drift rises *above* healthy (1.17–1.23×) and residual suppression weakens (0.61–0.85): a tied row is a softmax negative on every step and receives dense gradient it would not otherwise get. The one tied series in our post-training sweep (Qwen2.5-1.5B) shows the same: glitch rows move 4.85× *more* than healthy, against < 1.0 in every untied family.

The output row of an untied model is the one that is *rarely the target*, and it is the one whose geometry predicts fragility in §6. That is the connection, and it is also why post-training does not manufacture new glitch tokens even though it leaves up to 35% of vocabulary rows with zero token-specific update (Tulu-3 DPO → RLVR; 23% in OLMo-2-7B): post-training deltas have essentially **no shared drift** — total and residual ratios agree to three decimals in all eight families we measured (Amber 0.550/0.550, OLMoE 0.277/0.277, Qwen2.5-7B 0.030/0.030). A row that post-training never touches is preserved at whatever pretraining made it, not damaged. Zero update is not damage; the drift is.

---

## 8. Negative results

Each of the following was asserted, then killed by a control. They are reported because several would have been the headline otherwise.

**8.1 The prefix rule is the tokenizer.** Both reported o200k ids resolve — 128188 is U+4E50A + `app`, 152383 is U+7968A + `app` — and the reported usage rule reproduces exactly on the real vocabulary with no model: intact after nothing / newline / tab / itself, dropped after space, quote, bracket, period, comma. It is the BPE pre-tokenizer boundary. But it is **not a glitch property**: in Pythia a leading space re-segments verified glitch tokens and healthy tokens at the same rate (survival 0.083 vs 0.093). The rule is explained and it is a red herring for glitchiness.

**8.2 BPE-unreachable adjacency does not predict damage.** Pairs (a, b) such that `encode(decode(a)+decode(b)) ≠ [a, b]` — adjacencies training essentially never produces — were *easier* to copy than the same two tokens in the reachable order: paired difference +0.94 [+0.61, +1.26], with 33.5% of pairs worse. A within-pair control with identical tokens.

**8.3 Reproduction of contiguous n-grams falls with n, sub-multiplicatively.** Among individually perfect tokens: 1.000, 0.864, 0.780, 0.728, 0.704, 0.648 for n = 1…8 (rare tail); 1.000 → 0.892 for common English. Independence from the n = 2 rate predicts 0.354 at n = 8; observed 0.648. More tokens, more chances — no interaction. Every structural predictor scored 0.47–0.51. The failures are the same modes as §5.4 (`' haven numberWithInt see really another…'` → `' havenventario see realmente otro…'`).

**8.4 The direct-path readout does not predict the substitute.** Predicting *which* token a model emits from `argmax_j E_in[t]·E_out[j]`: top-1 agreement 0.003, top-5 0.006. The margin's AUC for failure (0.546) was below the row-norm baseline (0.581).

**8.5 Orthographic completion is a minority mechanism.** The substitutions `ction` → `tion`, `ience` → `science`, `osition` → `position` suggested a spelling-corrector prior; being a completable fragment gives AUC 0.545 and the emitted string is an exact lexicon completion in 11% of failures.

**8.6 Trajectory homology: underpowered null.** Persistent H₁ of the per-step residual trajectory (64 chains, 192 steps): best feature AUC 0.597, ~1.3 SE from chance. The one informative number: H₁ is *anti*-correlated with repetition (−0.49 to −0.75), so it is not a repetition detector in disguise.

**8.7 Stage-relative abandonment does not replicate.** The worst-1% copy set turns over 74% across OLMo-2-1B's post-training, and the newly-broken tokens (`'{\r\n'`, `'\tString'`, `'*/\r\n'`) invited the reading that instruction tuning abandons CRLF and tab-indented code. The control — where the newly-broken tokens sat in the *base* model's own ranking relative to the 1% cut — showed real movement only in OLMo-2 (6.0× and 4.75× the threshold rank); Amber, OLMoE, Tulu-3, Qwen2.5 and Qwen3 sit at 1.2–2.1×, i.e. boundary jitter. The 7B's movers were not even the same kind of token as the 1B's. §7 gives the reason it had to fail.

**8.8 The Unicode-variant explanation of fragility.** Dead on arrival: all 14 fragile clean-looking tokens are canonical (§5.4).

**8.9 Fragility does not explain near-miss hallucination on real entities.** The bridge test of §3.4. 1,255 real multi-token entities — 1,100 Python standard-library, torch and transformers identifiers taken from the live interpreter, stratified over 2–6 tokens, plus 155 real proper nouns in several scripts — were used in a natural task ("write one line of Python that imports or calls `X`"; "write one sentence that mentions X"), and the output scored on the entity's final component as *exact*, *near-miss* (a fuzzy match ≥ 0.75 — the fabricated-API hallucination), or *miss*. Every constituent token's fragility came from the replication bank, into which those tokens were deliberately placed. Controls, fitted jointly: rarity (token ids, length, type), the standard single probe on the constituents, and **familiarity** — the model's own teacher-forced log-probability of the whole entity after a neutral prefix, the fairest "how well does it know this string" control available.

Results: code entities exact 82.5%, near-miss **1.7%** (19 cases), miss 15.7% — and all 173 misses are bare `import` lines that never use the identifier, task avoidance rather than hallucination; proper nouns exact 99.4%. The genuine near-misses are the phenomenon we hoped to explain — `curses.BUTTON5_PRESSED` → `C_BUTTON5_PRESSED`, `transformers.EncodecModel` → `EncodeModel`, `torch.BoolTensor` → `bool_tensor`, `tokenize.SOFT_KEYWORD` → `SOFT_KEYWORDS` — but constituent fragility does not predict them: adding it to the controls changes out-of-fold AUC by **−0.001 [−0.008, +0.006]** for any failure (193 positives), and by −0.001 [−0.042, +0.042] for near-miss, where 19 positives cannot support any predictor (the controls themselves reach only 0.59 [0.46, 0.73]). Fragility alone scores 0.55–0.57. A first version of this scorer compared the whole dotted path and counted `numpy.X` → `np.X` and `from transformers import X` as near-misses; that inflated near-miss to 23.5% and is why the analysis was rerun on the final component. Both versions reach the same verdict on fragility. The two phenomena share a cause — rarity — and nothing more that we can measure.

---

## 9. Limitations

- **One model for the behavioural results.** §3–6 are on OLMo-2-1124-7B-Instruct (plus Pythia-1.4b for §3.3). The compositional null and the fragility measurement should be replicated on a second family; the code runs unchanged given a label set.
- **Copy log-probability as the proxy.** It is continuous, deterministic and cheap, which an N² screen needs, and it agrees with the generation-based failures we inspected; but the reasoning-mode measure in §3 is what users care about, and the exploratory reasoning-mode confirmation of the screen's top pairs was confounded (three tokens dominated the selected set; half the matched controls dropped the anchor token). We report it as inconclusive.
- **Twenty-eight positives** in §6 on the replication (thirteen on the pilot). The paired-bootstrap interval excludes zero, but it is wide (+0.06 to +0.26); the effect's size is uncertain even though its sign is not. A third bank on a second model family is the natural next step.
- **The bridge scorer** (§8.9) treats a bare `import` line as a miss; those 173 cases are task avoidance and are excluded from the near-miss class, but they do mean the entity task elicited the name in only 84% of prompts. A prompt that forces use of the identifier would give a cleaner denominator.
- **Power of the interaction screen** for *sparse* large effects is low in the weak pool (0/5 recovery of a single injected effect at 2× cell noise). The additivity claim rests on the cell-type averages, which are tight, not on the per-pair test.
- The fragility contexts are random sequences of common words, not natural text. Natural text may have a different hostility distribution; the finding that *no* context is systematically hostile is specific to this bank.

---

## 10. Conclusion

Glitch behaviour is not an information loss and it is not compositional. A glitch token's identity survives to the last layer; the readout substitutes a plausible neighbour, confidently, and the model completes whatever task it was given on the substituted content. Two tokens at a distance do not conspire — their effects add — and a repeated healthy token copies better, not worse. What *does* vary with surroundings is a token-level property we call fragility: about one in sixty tokens that pass every single-context probe fails to copy in a tenth or more of ordinary contexts, by deletion, neighbour substitution, truncation or translation, with which contexts trigger it unpredictable but the fact of it stable, replicable, and readable from the geometry of the output embedding better than from rarity alone. That geometry is where pretraining's gradient sparsity leaves its mark, and it is the same reason post-training, which drifts nothing, breaks nothing. What fragility is not is a mechanism for hallucination at large: on real named entities it adds nothing to rarity and familiarity, and we say so.

The practical recommendation is simple. Detectors that score a token in one context are measuring the wrong thing for systems that assemble prompts programmatically. Fragility costs N × C forward passes with C ≈ 24, is stable enough to measure once per model, can be pre-screened from the embedding matrices alone, and finds the `according`s.

---

## Appendix A. Pre-registration and calibration

The design, null model, verdict criterion and failure criteria for §4 were written before the screen ran (`whimsical-greeting-lerdorf.md`). Two amendments were made after synthetic calibration and before any real data: (i) residuals are standardised per row/column before permutation, because a heteroscedastic null otherwise produced family-wise false positives (5–18 rejections, p ≈ 0.003; 0 after the fix); (ii) the fixed "≤ 5% of pairs" null threshold was replaced by the matched-simulation threshold, because a true null yields 0–2 rejections, not ~200, and the original criterion would have called a spurious 150 pairs "null". A third amendment, on the collaborator's correction, made distance the design variable (2–128, with the cap set by compute rather than design). The corrected power statement — a single isolated effect is recovered 4/5 at 1.5× cell noise with R = 14, the Bonferroni-like worst case — replaced an earlier one that had injected raw log-probability while labelling it noise units and overstated power threefold.

## Appendix B. Reproducibility

`src/cut/reasoning_drift.py` (§3.1–3.2), `forward_geom.py` (§3.3), `interaction_screen.py`, `interaction_confirm.py`, `interaction_geometry.py`, `stats.py` (§4), `fragility.py`, `fragility_predict.py`, `fragility_diag.py`, `run_fragility.sh` (§5–6; replication via `run_bridge.sh` with `--extra-tokens`), `specimens.py` (specimen store), `entities.py`, `hallucination_bridge.py`, `bridge_rescore.py` (§8.9), `harvest_trajectory.py`, `harvest_family2.py`, `stagewise.py`, `run_stages.sh` (§7), `segmentation.py`, `ngram_glitch.py`, `ngram_identify.py`, `embed_predict.py`, `orthographic.py`, `traj_topology.py` (§8). Per-cell results are in `results/` (`fragility.pt`, `fragility_L.pt` hold the full matrices and slot states; `specimens_confident_substitution.jsonl` the examples; `hallucination_bridge*.json` every entity prompt and output); findings notes with full tables in `docs/findings_*.md`. Behavioural runs used a single A100-40GB; the largest (the 6,273 × 24 fragility matrix) is 25 minutes.
