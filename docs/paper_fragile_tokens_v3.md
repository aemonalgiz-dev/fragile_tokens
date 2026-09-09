# Fragile Tokens

### Context-dependent copy failure on tokens that pass single-token glitch probes

Jeffrey R. Gordon · September 2026

---

## Abstract

Glitch-token detectors evaluate each vocabulary entry in isolation, typically by asking the model to repeat it. Practitioners, however, report failures that depend on the surrounding text: a token that behaves alone and breaks in a sentence, or that a reasoning model confidently misidentifies. This paper examines whether such behaviour is compositional — a property of which tokens are combined — or contextual — a property of a token that only some surroundings reveal — using an open 7B instruction-tuned model with independently published glitch labels. A pre-registered factorial screen over 8,064 ordered token pairs at separations of 2 to 128 positions finds no interaction: the effect of two tokens is additive to within 0.008 log-probability in every condition, and repetition of a healthy token improves copying rather than degrading it. The contextual reading holds. Placing 2,098 tokens, and then an independent 6,273, into banks of random ordinary contexts, 1.5–1.6% of tokens that pass the single-token probe fail to copy in at least 10% of contexts. These are canonical, common tokens; the English word `according` copies at 0.999 in isolation and is deleted with probability 1.000 in 44 of 48 contexts. The dependence is on the token rather than the context (context share of variance ≤ 0.007; stability across independent banks AUC 0.996), and the failures take four confident forms — deletion, neighbour substitution, truncation, and translation — consistent with the model's behaviour on verified glitch tokens, whose identity remains linearly recoverable from the final layer while the readout substitutes. Fragility is predictable from the embedding matrices without a forward pass (cross-validated AUC 0.911 against 0.711 for surface rarity, paired-bootstrap improvement +0.20, 95% CI [+0.10, +0.32]). A direct test finds that fragility does not, beyond rarity and familiarity, predict near-miss errors on 1,255 real named entities. All failing cases are released with model provenance. On three further models sharing one tokenizer — Qwen3-1.7B, Qwen3-32B and Qwen2.5-72B — run label-free against ranges fixed in advance, fragility is present and token-intrinsic at every size (2.4%, 0.9% and 10.5% of gated tokens; split-half correlation 0.82–0.91), is additive across positions at 32B, and produces the same failure modes; given a single such token as the whole prompt, the 32B does not hedge but solves, in plain and in thinking mode, a problem it was never asked. Prevalence is not monotone in size, and at 72B, where the fragile set is dominated by rare CJK characters, surface rarity predicts it better than embedding geometry does (paired Δ −0.057 [−0.09, −0.02]); the geometric advantage is a property of the 7B's fragile population, not a law.

---

## 1. Introduction

Detection of glitch tokens has, since the first catalogue of anomalous strings in GPT-2, proceeded one token at a time. Magikarp ranks vocabulary entries by indicators computed from the unembedding matrix and verifies candidates with a repetition prompt; GlitchHunter clusters; GlitchProber classifies intermediate activations; GlitchMiner optimises next-token entropy; GlitchQuiz applies a battery of templates. Each answers the question *is this token defective* with the token alone in the prompt, and each is accurate on its own terms.

Field reports describe behaviour that this question does not address. Users of reasoning models describe tokens that a model "struggles to identify," "conflates" with another script, or handles correctly in isolation and incorrectly in a longer input. Two accounts are consistent with such reports. On the first, glitch behaviour is compositional: particular combinations of tokens, possibly individually healthy and possibly widely separated, produce it. On the second, it is contextual: a property of a token that some surroundings expose and others do not. The two accounts imply different detectors, so it matters which is correct.

Both are tested here on `allenai/OLMo-2-1124-7B-Instruct`, an untied model with a vocabulary of 100,352 for whose base model an independent group published behavioural labels — 3,171 tokens tested by repetition probe, 442 verified as failing. Throughout, a token described as *healthy* is one that group tested and passed. All generation is greedy at commit `470b1fba` in half precision, and every failing case reported is stored with its context and emission probabilities.

The findings are as follows. Combinations of tokens do not interact (§4). The contextual account holds, and the paper defines and measures the quantity it implies: a token's **fragility**, the fraction of ordinary contexts in which it fails to copy (§5). Fragility is rare among tokens that pass the single probe, stable, canonical, and invisible to existing detectors; its failure modes are those of verified glitch tokens (§3); and on this model it is predictable from the geometry of the output embedding more accurately than from surface rarity (§6). The mechanism that produces the geometric signal is credited to prior work and confirmed on labelled tokens (§7), and a direct test of whether fragility explains a class of hallucination is reported as negative (§7). On three further models of 1.7B to 72B parameters, run label-free against ranges fixed in advance, fragility and its failure modes replicate at every size while its prevalence does not fall with size and the geometric advantage does not survive a fragile population dominated by rare CJK characters (§8).

---

## 2. What a single-token probe measures

The English word `according` is token 4184 in the vocabulary, placing it among the most frequent pieces, and it is canonical: decoding the id and re-encoding the string returns the same id. Under the repetition probe used by every detector — *Repeat the text exactly. Text: according. Copy:* — the log-probability of the correct continuation is −0.001. By the standard in use since 2023, the token is healthy.

The same token was then placed in sequences of ordinary common words drawn at random, so that the surroundings were not chosen, and the model was asked to repeat the sequence. In the context *deliver rel behind according sent floor wrong*, at the position where the copy should produce `according`, the model produced `sent` with probability 1.000 and entropy 0.00 bits, then continued the copy correctly from that word. Across 48 such contexts the token was deleted in 44. In the remaining four it was copied at 0.999.

The token did not change between the two measurements. The first was made in a context where the copying mechanism has one candidate; the second in the contexts a token ordinarily occupies. A score on a token in isolation measures whether the model can reproduce it when nothing competes. Whether that reproduction survives surrounding text is a distinct question, and it is the question a system that assembles prompts programmatically poses on every call.

The probe requires one methodological fixture. A prompt assembled as a string passes through the tokenizer, and a leading space re-segments approximately nine tokens in ten into different ids regardless of their status: verified glitch tokens survive a preceding space in 8.3% of cases and healthy tokens in 9.3%. A string-built prompt therefore measures the tokenizer. Every prompt in this paper is assembled from token ids, and the score is the teacher-forced log-probability of the target id at its copy position. The reported rule that certain tokens "cannot be preceded by a space or bracket but can follow a newline" is real; it is the pre-tokenizer's boundary set, it applies to nearly the whole vocabulary, and it carries no information about which tokens are defective.

The scope of this section is narrow. A single word failing in 44 contexts is a specimen, and a specimen selected for being striking says nothing about prevalence, which §5 measures. It does establish that the single-token probe and the in-context probe can disagree completely on a common canonical token.

---

## 3. The failure mode is confident substitution

Before measuring how often in-context failure occurs, its character should be fixed, because the reports describe it as confusion and the evidence does not.

### 3.1 Reasoning-style prompts

Thirty-two verified glitch tokens and thirty-two healthy controls were spliced into the model's chat template after *What is this token? Identify it exactly, then explain step by step what it means*, and 256 tokens were generated for each. Two predictions drawn from the field reports were tested first. Neither held. The difference in next-token entropy between glitch-seeded and control-seeded responses is +0.27 bits at step 8 and −0.29 bits at step 255; it does not grow with length. Responses seeded by different glitch tokens do not converge on a common state: the mean pairwise cosine of their centred final-layer states decays from 0.016 to 0.0001 for glitch seeds and from 0.017 to 0.0002 for controls.

The model instead answers about a different token. Table 1 gives the token supplied, whether the model reproduced it, the token the model said it had been given, and the opening of the explanation that followed, for eight glitch seeds and five controls; the full 256-token responses are in `results/reasoning_drift.json`.

**Table 1.** Reasoning-style prompt: token supplied and the model's account of it. *Reproduced* is by exact token id within the response.

| token supplied | group | reproduced | model said it was given | opening of the explanation |
|---|---|---|---|---|
| `.XtraLayout` | glitch | no | `"layui"` | a lightweight, modular, and flexible JavaScript framework primarily used for building web interfaces |
| ` Hexatrigesimal` | glitch | no | `"Calculus"` | a branch of mathematics that deals with the study of change, motion, and accumulation |
| `雅黑` | glitch | no | `"턱"` (Korean Hangul) | represents the consonant sound "t" or "d" … combined with the vowel "u" to form the syllable "ttuk" |
| ` didn` | glitch | no | `"don"` | a short form or abbreviation of "do not" |
| `.AddModelError` | glitch | no | `ErrorDictionary` | is not a standard token in programming languages … it might be a misunderstanding or a typo |
| `.JpaRepository` | glitch | no | `"IpaRepository"` | does not directly correspond to a widely recognized term or acronym |
| `")){\r\n` | glitch | no | `){` | a closing curly brace (`}`) in programming syntax |
| `.ContextCompat` | glitch | no (leading `.` dropped) | `ContextCompat` | a static utility class in the Android SDK — correct apart from the punctuation |
| ` woman` | control | yes | `"woman"` | a noun in the English language |
| ` actually` | control | yes | `"actually"` | an adverb in the English language |
| ` life` | control | yes | `"life"` | a concept that has been extensively explored across various disciplines |
| `,\r\n` | control | no | `",` | a quotation mark or double quote character in ASCII |
| `.onOptionsItemSelected` | control | no (leading `.` dropped) | `"OptionsItemSelected"` | not a standalone token with a universally recognized meaning |

Across the 32 glitch seeds the model reproduced the token it was shown zero times by exact id; across the 32 controls, seventeen times. The table qualifies the count in two ways. One glitch seed, `.ContextCompat`, was identified correctly apart from its leading punctuation, so the substitution is not universal in kind; and two controls failed, one of them likewise by dropping a leading dot. The characteristic glitch response is nonetheless a fluent account of a plausible neighbour — a JavaScript framework for a UI-library token, a branch of mathematics for a base-36 numeral token, a Korean syllable for a Chinese font name — delivered at 0.85–1.3 bits of entropy and followed in every case by a step-by-step explanation of the substituted token. In the short-completion regime, by contrast, the same seeds were simply continued as though a definition were being written — `.XtraLayout` and ` Hexatrigesimal` were both continued with *is a portmanteau of "*, and `雅黑` with *is a palindrome* — and no seed in either group was reproduced. That regime is where a single-token failure is masked; the reasoning-style prompt is where it is exposed, because the task requires the model to refer to the token rather than to continue past it. In the factorial experiment of §4, where a trivially answerable question follows the token, the task-completion rate was 1.000 in every arm. The model neither halts nor signals uncertainty; it completes the task on substituted content. This is the appropriate statement of the hazard, and it accounts for why detectors built on entropy do not observe it.

### 3.2 The identity is retained

Whether the model has lost the token or failed to emit it determines the remedy. On Pythia-1.4b, for which labels of the same kind exist, each of 36 verified glitch tokens and 1,088 controls was placed at the end of eight contexts and its hidden state read at every layer. At each layer the token was matched from its state in one context to its state in another within a pool of 1,124 candidates, after removing each context's shared component. From the embedding layer through layer 12, glitch tokens were identified in 1.000 of cases and controls in 0.988–1.000; at the final layer, 0.880 against 0.874. The identity of a glitch token reaches the top of the network as intact as that of a healthy token. A logit lens at the same positions shows the model *continuing* glitch tokens correctly — `idepress` is followed by `ant` at every layer — while unable to reproduce them. Continuation, which pretraining on running text rewards, survives; self-reference, which it does not reward, fails. The failure is at the readout rather than in the representation.

---

## 4. Combinations of tokens do not interact

The compositional account was stated in its sharpest form by a collaborator: in `[1, 231, 885, 9911, 1112, 231]` the repetition of 231 produces glitch behaviour, and in `[1, 231, 885, 9911, 9999, 1922, 1013]` the co-occurrence of 885 and 1013 does, with members individually healthy and arbitrarily separated. Earlier work in this project had examined adjacent pairs and contiguous runs, in which separation was not a variable; the present design makes it one.

### 4.1 Design

A carrier is a sequence of ordinary filler words with slots at positions 2 and 2 + d. For an ordered pair (a, b), four carriers are scored: control words in both slots, a with a control, a control with b, and a with b. Absent tokens are replaced by controls rather than removed, so that length and all other positions are fixed, and the two controls differ so that the all-control carrier is not itself a repetition. The interaction I is the fourth score minus the second and third plus the first — the damage attributable to the pair beyond that attributable to its members. Separation d took the values 2, 4, 8, 16, 32, 64 and 128, with two independent filler sequences at each; carriers at different separations share a prefix so that separation is not confounded with filler content. The pool contained 64 tokens, in the first run all individually clean and in the second 32 clean and 32 *weak* (tokens copied in isolation only intermittently, leaving room for a combination to degrade them). Each run scored 63,014 carriers.

### 4.2 Null model

Four thousand ordered pairs constitute four thousand simultaneous tests on a difference of four noisy quantities, and the null was therefore fixed before real data were scored. Within each carrier the matrix of pair scores is fitted additively; the residual is the interaction; residuals are permuted within carrier and averaged across carriers; per-pair p-values are corrected by Benjamini–Hochberg at q = 0.05, with a maximum-statistic family-wise test alongside. The criterion for a positive screen is not a fixed count but the 95th percentile of rejections across twenty datasets simulated from the additive fit with the observed noise structure.

Synthetic calibration corrected the first version of this null. On matrices in which one token in eight was three times noisier than the rest, a pooled permutation produced 5–18 false rejections and family-wise p ≈ 0.003 in every run, because the pairs of a noisy token have large residuals against a null calibrated to the average. Standardising each residual by the robust scale of its row and column before permutation produced zero false rejections in ten of ten null runs. The standardised version was used.

### 4.3 Results

The screen was negative in both pools. In the clean pool one pair of 4,032 was significant against a matched-null 95th percentile of 4; in the mixed pool one of 4,032 against 14. The condition means, which do not depend on the per-pair test's power, are the more informative statistic:

| a × b (mixed pool) | n | mean joint score | mean I | s.e. |
|---|---|---|---|---|
| clean × clean | 992 | −0.027 | +0.0011 | 0.0004 |
| clean × weak | 1024 | −0.146 | +0.0030 | 0.0016 |
| weak × clean | 1024 | −0.195 | +0.0033 | 0.0014 |
| weak × weak | 992 | −0.313 | +0.0076 | 0.0027 |

Weak tokens copy twelve times worse than clean ones, and the whole of that difference is the additive main effect. The interaction is slightly positive in every condition and at every separation (off-diagonal means by separation range from −0.0007 to +0.0055): two weak tokens copy marginally better together than the sum of their separate effects predicts. Positive-control glitch tokens placed in single slots scored between −3 and −6.5 against a clean baseline of −0.02, so the scorer detects effects where they exist.

Repetition operates in the opposite direction from the compositional account. Along the diagonal of the clean pool the mean interaction was +0.0066 (permutation p = 0.000), increasing with separation from +0.005 at d = 2 to +0.018 at d = 128; a healthy token repeated at a distance is copied more reliably the second time because its first occurrence is in context, which is induction operating as intended. In the mixed pool a small number of tokens did fail when repeated — `Tools` scored −1.39 in one filler sequence at d = 128 — and the same token in the other filler sequence at the same separation scored −0.000. Where repetition degraded copying, it did so in one context and not in another.

### 4.4 Structure of the failures that did occur

In the clean pool 453 of 57,344 carriers failed outright (slot log-probability below −0.5). These were concentrated in five tokens, which together accounted for 47% of failures — `.Ver` failed 111 times in the first slot and `Sentence` 118 times in the second — and they did not recur across filler sequences: at each separation, the number of carriers failing in both sequences was 3, 0, 1, 0, 0, 0 and 2. At d = 4 one filler sequence produced 68 failures and the other one. The failures factor as a susceptible token in an unfavourable context, which is the structure an additive model absorbs and an interaction term cannot represent. The compositional account is not supported. The object to measure is the token, and the quantity is its sensitivity to context.

---

## 5. Fragility

### 5.1 Design

Forty-eight contexts were constructed by drawing common, individually clean words at random: twelve each of lengths 8, 16, 32 and 64, with an interior slot at a random position. Into every context were placed 2,098 tokens — 1,998 sampled with the id range stratified into thirds, since uniform sampling of ids falls mostly in the rare tail, and 100 verified glitch tokens as a reference class. A cell is the copy log-probability at the slot; a cell fails below −0.5 (probability 0.61); a token's fragility is the fraction of contexts in which it fails. Because every token meets the same contexts, differences in fragility are differences between tokens. An independent replication bank of 6,273 tokens in 24 new contexts was constructed afterwards.

### 5.2 The dependence is on the token

Decomposing each cell as a grand mean, a token effect, a context effect and a residual, the shares of variance over all tokens are 0.952 (token), 0.001 (context) and 0.047 (residual) on the log-probability scale; over clean-looking tokens alone, 0.717, 0.000 and 0.282. No context is generally hostile: the most severe of the 48 fails 19% of tokens and the least severe 10%, and the failure rate of clean-looking tokens does not vary with context length (0.005, 0.008, 0.008, 0.008 across lengths 8 to 64). The residual share represents tokens failing in particular contexts. A held-out analysis confirms the asymmetry: predicting whether a token fails in a context it was not scored in, the token's fragility on other contexts gives AUC 0.94, the context's hostility on other tokens gives 0.56, and a measure of similarity between the token and its surrounding words adds nothing to either. The sensitivity is a property of the token; which context elicits it is not predicted by the features examined.

### 5.3 Prevalence and stability

| population | n | mean fragility | fragility ≥ 0.10 | never fails |
|---|---|---|---|---|
| random tokens | 1,998 | 0.090 | 17.5% | 67.0% |
| verified glitch (reference) | 100 | 0.993 | 100% | 0% |
| clean-looking (single probe > −0.1) | 873 | 0.007 | **1.6%** | 92.1% |
| clean-looking, replication bank | 1,888 | 0.006 | **1.5%** | 95.0% |

The reference row validates the measure: verified glitch tokens fail in nearly every context, and none passes the single probe. Among tokens that do pass it, approximately one in sixty fails in one context in ten. Fragility is stable rather than an artefact of a particular bank: fragility on even-numbered contexts predicts fragility on odd-numbered contexts with AUC 0.996, and `according`, at 0.92 in the first bank, scored 0.92 in the second.

### 5.4 The fragile tokens are canonical

The fourteen fragile clean-looking tokens in the first bank were tested against the most economical explanation — that they are non-canonical encodings, such as a decomposed-accent variant that copies after *Copy:* but loses to its composed twin in running text. All fourteen round-trip from id to string to the same id; all fourteen are in Unicode composed form; all fourteen are classed `OK` in the labelling group's taxonomy. They are ordinary tokens.

| token | id | isolated lp | fragility | worst-context emission |
|---|---|---|---|---|
| `' according'` | 4184 | −0.001 | 0.92 | `' sent'` (the following word) at 1.000; token deleted, copy continues |
| `' который'` | 98851 | −0.017 | 0.71 | `' что'` |
| `' pueden'` | 41604 | −0.030 | 0.62 | `' pena'` at 0.775 |
| `' abbiamo'` | 95396 | −0.020 | 0.60 | `' abdom'`, continued to "abdominoplasty" |
| `' sólo'` | 53288 | −0.032 | 0.48 | `' só'` at 0.892 |
| `' 查询'` | 81628 | −0.015 | 0.46 | `' QUERY'` at 0.970 |
| `' due'` | — | −0.000 | 0.42 | deleted; worst cell −23.8 |
| `' getSystemService'` | 92476 | −0.053 | 0.38 | `'SystemService'` at 0.30 |

Four failure modes occur, and they correspond to those of §3. The token is **deleted**: the model emits the following context word and continues as though the slot were empty. It is **substituted** with a neighbour — Spanish for Spanish, one identifier for a similar identifier. It is **truncated** to a prefix of itself. Or it is **translated**: the Chinese token for *query* is emitted as the English `QUERY` at 0.970, and the Russian for *which* as the Russian for *what*. Each is confident, and each reverses when the filler words change; `according` is deleted at 1.000 in three stored contexts and copied at 1.000 in a fourth.

Two regularities in the fragile set are noted without having been tested. The two English cases with catastrophic scores, `according` and `due`, are words whose following token is almost invariably `to`, and several of the fragile connectives — `mentre`, `allerdings`, `może`, `который` — are likewise words that strongly constrain their successor. Fragility may therefore track the concentration of a token's continuation distribution: a token the model declines to copy into a position where its expected successor is absent. Separately, a deletion at probability 1.000 and entropy zero is the observable signature one would expect of a copy-suppression head, the circuit described by McDougall et al. (2023) that suppresses re-emission of an earlier token when the model's prediction is confident. The remainder of the fragile set is the rare tail of an English-dominant instruction mix — Spanish, Italian, Portuguese, German, Polish, Chinese and Russian words, and Android/Java identifiers — as would be expected if rarity accounted for most of the effect; §6 tests how much of it rarity does account for.

### 5.5 Specimen store

For each fragile token and for twenty verified glitch tokens, the three worst cells and one control cell are stored with the exact context and slot, the five most probable emissions and their probabilities, the entropy at the position, the probability assigned to the correct token, the greedy continuation, and a label for the failure mode, under a header recording the model, commit, precision, depth, width, vocabulary size, tying, and library versions. There are 228 records (`results/specimens_confident_substitution.jsonl`; rendered in `docs/specimens.md`). The claim that the failures are confident is verifiable from them directly.

---

## 6. Predicting fragility from the embedding

If fragility is a property of the token, the practical question is whether it can be estimated without running the model. Three tiers of predictor were evaluated, separated by cost.

**Static** features are computed from the two embedding matrices alone: the norm of the input row and its distance from the centroid; the same for the output row; the cosine to the nearest neighbour and the mean cosine to the ten nearest; the direct-path self-score E_in[t]·E_out[t] with the final layer norm folded in, and its margin over the best competing row. **Dynamic** features are computed from the token's representation on half of the contexts — the dispersion of its slot state across contexts, the norm of that state, and its cosine to its own unembedding row — and predict failure on the other half, so that no feature predicts itself. The **behavioural** tier measures copy on half the contexts and predicts the other half. Every geometric feature is compared with **surface** features available without any model access: token id (which in a byte-pair vocabulary approximates frequency rank), character length, leading space, alphabetic, and ASCII.

The target is failure in at least 10% of held-out contexts among clean-looking tokens: 13 positives in the first bank and 28 in the replication bank. The replication bank decides the claim. On it, the single static features are modest — output-row centroid distance AUC 0.678 [0.55, 0.79], direct-path self-score 0.655 [0.56, 0.75], input-row norm and centroid distance approximately 0.60 — while the strongest dynamic feature, the norm of the slot state across training contexts, reaches 0.864 [0.78, 0.92], and copy on the training half reaches 0.997.

Combined by cross-validated L2 logistic regression and scored out of fold:

| tier | replication (28 pos.) | improvement over surface, paired bootstrap |
|---|---|---|
| surface only | 0.711 [0.58, 0.83] | — |
| static geometry | **0.911 [0.86, 0.95]** | **+0.202 [+0.096, +0.320]** |
| dynamic (train-half representation) | 0.823 [0.72, 0.92] | +0.112 [+0.024, +0.216] |
| static + dynamic | **0.930 [0.89, 0.96]** | **+0.218 [+0.110, +0.334]** |
| behavioural (train-half copy) | 0.996 | — |

Because overlapping marginal intervals do not bear on a difference, the improvement of each geometric tier over the surface tier was bootstrapped in pairs over the same tokens. Every interval excludes zero.

One correction is recorded here rather than silently. A first version of the static tier contained a *glitch direction* feature — each token's projection onto the mean embedding of the verified glitch tokens minus the global mean — that was computed from the wrong rows through an indexing error, so that it contributed noise; with that feature computed correctly the static tier's AUC rises from 0.862 to 0.911 and its advantage over surface from +0.151 to +0.202. The correction strengthened the result. The glitch-direction projection is, once correct, by some distance the strongest single static feature — AUC 0.915 [0.88, 0.95] on its own, against 0.678 for the output-row centroid distance and 0.655 for the direct-path self-score — so the static tier's accuracy is substantially this one feature. It is also the only static feature that requires labels: a model with no verified glitch set can substitute the mean embedding of its worst tokens by the single probe, as the label-free runs of §8 do.

Under the gate of §8 — the token is the greedy output of its own copy probe, p > 0.5, rather than above 0.9 — the same tier on the same bank has 60 positives instead of 28: static 0.908 against surface 0.804, paired improvement +0.105 [+0.052, +0.160]. The margin narrows, because the wider gate admits tokens for which rarity is itself informative, and the direction is established with twice the positives.

Two qualifications apply. The intervals on the improvements are wide, so the sign of the advantage over rarity is established with more confidence than its magnitude; and the finding is now, in effect, that fragility among probe-passing tokens lies along the same embedding direction as verified glitch tokens — which is a specific and testable statement about geometry, but a narrower one than "several independent geometric features each predict it". The features carrying the signal are rarity with a mechanism attached: the output row that lies far from the centroid and weakly excites its own readout, and the input row that lies in the direction glitch rows share, are the rows that were seldom the training target and so were seldom updated in a token-specific way, which §7 addresses. What the tiers establish is a graded procedure. From the weights alone, fragility can be pre-screened at 0.91; with a dozen contexts' forward passes the token's representation raises this to 0.93; with a dozen contexts' copy scores it reaches 0.99.

---

## 7. Mechanism, and a negative result

### 7.1 Origin of the geometric signal

The mechanism is established in prior work. Yu et al. (2022) showed that the embedding rows of rare tokens receive gradient chiefly as softmax negatives, in a direction shared with other rare rows; two architectures published in 2026 — one decoupling input from output representations, one re-injecting token identity at every layer under the designation "Rare Token Problem" — are built to counter it. The checkpoint phase of this project adds measurement on real pretraining runs with behaviourally labelled tokens, and the measurement agrees. Decomposing each update to the embedding matrix into its component along the mean update direction and the remainder, verified glitch rows receive the shared component in comparable or greater measure than healthy rows while their token-specific remainder is suppressed to 0.15–0.74 of the healthy value, across seven model lineages on three corpora and in a LLaMA-architecture model of independent ancestry (0.573). Under weight tying the relationship reverses as the mechanism requires, the shared component rising to 1.17–1.23 times the healthy value, since a tied row is a softmax negative at every step. The output row from which §6 reads fragility is the row that was rarely the target.

The same decomposition explains an apparent finding that did not survive replication. Post-training leaves a large fraction of the vocabulary without token-specific update — 23% of rows across one model's final stage, 35% across another's — which suggested that instruction tuning might create new glitch tokens. Across eight stage series it does not, and the decomposition indicates why: total and token-specific movement agree to three decimal places in every family (0.550/0.550, 0.277/0.277, 0.030/0.030). Post-training has essentially no shared drift, so a row it does not touch remains where pretraining left it. Absence of update is not damage; shared drift without correction is.

### 7.2 Fragility does not account for entity-level hallucination

Confident substitution on retained information resembles hallucination in structure — fluent, incorrect, a plausible neighbour of the truth, with no expressed uncertainty — and it is natural to ask whether token fragility is a mechanism for one class of hallucination: the near miss on a rare entity, such as an API name that almost exists. The question was tested directly. 1,255 real entities — 1,100 identifiers from Python's standard library, torch and transformers as present in the interpreter, spanning two to six tokens, and 155 place names, persons and substances across several scripts — were used in a natural task (*write one line of Python that imports or calls* `X`; *write one sentence that mentions* `Y`), and outputs were scored on the entity's final component as exact, near miss (fuzzy similarity ≥ 0.75), or miss. The model reproduced API names exactly in 82.5% of cases, produced near misses in 1.7% (`curses.BUTTON5_PRESSED` → `C_BUTTON5_PRESSED`; `torch.BoolTensor` → `bool_tensor`), and in 15.7% wrote a bare `import` statement without using the name, which is avoidance rather than hallucination; proper nouns were reproduced exactly in 99.4%. Every constituent token of every entity had been included in the replication fragility bank, so each entity carried the fragility of its parts. Added to a model containing the entity's rarity, its constituents' single-probe scores, and the model's teacher-forced log-probability of the whole entity, fragility changed the out-of-fold AUC for an incorrect name by −0.001, 95% CI [−0.008, +0.006]. An earlier version of the scorer compared the full dotted path and counted `numpy.X` written as `np.X` as a near miss; the analysis was rerun on the final component, and both versions reached the same conclusion. Fragility and hallucination share rarity as a cause; on this evidence they share nothing further. The structural resemblance describes the failure and does not explain hallucination.

---

## 8. Replication at scale

Whether §§3–6 describe one 7B model or the class was tested on a ladder of larger models sharing a single tokenizer, with the quantities and the ranges that would count as replication fixed before any of them ran (Appendix B). Three of four rungs have run: Qwen3-1.7B, whose input and output embeddings are tied; Qwen3-32B; and Qwen2.5-72B-Instruct, a dense model from the previous Qwen generation, chosen because no dense Qwen3 exists near 70B. All ran label-free — the reasoning-mode and specimen stages take as their glitch set the thirty-two tokens the model copies worst under the single probe, after a filter for printable strings, and their controls from tokens it copies exactly — and the 7B was recomputed under the same procedure as the reference row. The 72B ran on a token sample of 8,400 rather than 4,200, so that the geometry test would have enough positives. The fourth rung, Qwen3-235B-A22B in Qwen's FP8 release with thinking mode, is prepared and verified to load and run on a 30B model of the same expert layout; its first attempt completed the fragility matrix and the geometry run and was then lost to an instance failure before any output had been transferred. It will be appended when it completes; nothing below depends on it.

**The gate.** The clean-looking gate of §5, single-probe log-probability above −0.1, did not transfer. The 32B copies ordinary tokens in context at 0.997 but under the bare few-shot probe at 0.72–0.89, so the gate admitted 353 of 4,200 sampled tokens and no fragile one — a statement about the probe's calibration on that model, not about fragility. Cross-model comparison therefore uses a gate that means the same thing on any model: the token is the greedy output of its own copy probe, p > 0.5. Under it the 7B's numbers are recomputed and, where they differ from §§5–6, supersede them for any comparison across models.

| quantity | pre-registered range | OLMo-2-7B (ref.) | Qwen3-1.7B (tied) | Qwen3-32B | Qwen2.5-72B |
|---|---|---|---|---|---|
| gated tokens | — | 2,183 | 1,666 | 1,597 | 3,998 |
| prevalence of frag ≥ 0.10 | 0.5–3% | 2.7% | 2.4% | 0.9% | **10.5%** |
| context share of variance | < 0.01 | 0.001 | 0.004 | 0.002 | 0.013 |
| split-half correlation of fragility | — | 0.85 | 0.82 | 0.89 | 0.91 |
| failing cells below 2 bits, fragile clean-looking | mean < 2 bits | — | 86% (median 1.00) | 59% (median 1.74, mean 2.55) | 69% (median 0.74, mean 2.22) |
| static vs surface, paired Δ (positives) | > 0, CI excluding 0 | +0.105 [+0.052, +0.160] (60) | 0.000 [−0.11, +0.11] (28) | −0.032 [−0.18, +0.12] (12) | **−0.044 [−0.070, −0.019]** (322) |
| seed id re-emitted, worst / control, plain | worst ≪ control | 0.00 / 0.53 | 0.00 / 0.44 | 0.00 / 0.56 | 0.00 / 0.47 |
| the same, thinking mode | | — | 0.00 / 0.50 | 0.00 / 0.72 | (no thinking mode) |
| mean I(weak × weak); significant pairs vs matched null | ±0.01; ≤ null | +0.0076; 1 | — | +0.0091; 13 vs 14 | (pool not scheduled) |

**Fragility** is present at every rung and is a stable property of the token at every rung (split-half correlation 0.82–0.91; context share of variance at most 0.013). Its prevalence is not monotone in size. It falls from the 1.7B to the 32B (2.4% to 0.9%) and then rises to 10.5% at the 72B, four times the 7B under the same gate and outside the range fixed in advance on the high side. The population is the one §5.4 describes at every rung — canonical word pieces from languages thin in the training mix, some with strong continuation priors — and at the 72B it is dominated by one class of it: 43 of the 80 most fragile clean-looking tokens are single rare CJK characters (`跸`, `螈`, `禘`, `鹪`) that copy alone at 0.98–0.99 and are deleted in context at p ≈ 1.000, the next filler word emitted with entropy 0.00–0.02 bits. The 72B's failures also grow with context length — fragile tokens fail in 26% of 8-token contexts and 49% of 64-token contexts — where the 7B's did not. Specimens at 32B: ` след` (Russian) copies alone at 0.61 and is replaced by ` follow` at 0.995 in five of six contexts; ` realtà` (Italian) becomes ` réalité` at 0.898; ` اليمن` becomes ` Yemen` at 0.499; the Tibetan letter `ཀ` is deleted and the next filler word emitted at 0.925. Translation, which in §5 was a single token, recurs on every Qwen rung: at 1.7B, ` 방법` (Korean, "method") becomes ` 方法` (Chinese, "method") at 0.97 in three contexts of four and is copied at 0.96 in the fourth; at 72B, `увеличен` becomes ` exaggerated` at 0.998.

**Confidence** is where the ladder qualifies §5 in one direction and confirms it in another. Among failing cells of fragile clean-looking tokens, entropy is below 2 bits in 86% of cells at 1.7B, 59% at 32B and 69% at 72B; the pre-registered bound on the mean is met at 1.7B and missed at 32B and 72B. The larger models' failures are still mostly confident — the 72B's median failing cell is 0.74 bits — but a minority are uncertain in a way the 7B's were not. On its worst-copying tokens the 32B is, by contrast, more confident than the 7B is on verified glitch tokens (median 2.4 bits against 7.5), and 62 of 80 failing cells are substitutions.

**Reasoning mode** reproduces §3 at every rung and in both modes on the metric the 7B was scored on: the seed's token id is never re-emitted, and controls are re-emitted at about half, rising to 0.72 for the 32B with thinking on. What the sizes do instead differs in the direction §3 anticipates. The 1.7B substitutes the seed with the chat template's own end-of-thought marker and then hedges — *The user wrote "</think>" … It seems like you might have made a typo or there's a formatting issue. Could you please clarify?* — on six of the eight logged tokens. The 32B never hedges. Given `ớ` as the entire user turn it writes *The question is asking for the value of cos(θ) given that cos(2θ) = 1/3. Step 1: Use the double angle identity …*; given `อังกฤษ` (Thai, "English"), *The question is asking for the value of 1/2 + 1/4 + 1/8 + …, which is an infinite geometric series*; in thinking mode, given `ล่า`, the trace opens *I need to determine if the statement "The sum of the first 100 odd numbers is 10,000" is true or false*, the statement fabricated before the first step of reasoning about it. Control tokens are named and glossed (*"programma", which is Italian for "program"*). The fabricated prompts come from a small repertoire — the same geometric series serves three different seeds. Entropy along the glitch-seeded trajectory exceeds 2 bits only at its first two steps, where the problem is being chosen, and thereafter runs below the control trajectory (0.40 against 0.66 at step 32).

**Table 2.** The same reasoning-style prompt across the ladder: the token supplied as the whole user turn and the opening of the response. *Mode* is the paper's classification of what the model did with the token.

| model | token supplied | group | mode | opening of the response |
|---|---|---|---|---|
| Qwen3-1.7B | `格會員` | worst | substitution → template token, then hedge | *The user wrote "</think>" … It seems like you might have made a typo or there's a formatting issue. Could you please clarify what you're asking?* |
| Qwen3-1.7B | `.currentTarget` | worst | neighbour substitution, answered | *You mentioned `@this` in the context of JavaScript, but it's not a standard syntax in JavaScript. Let me clarify: in JavaScript, `this` refers to …* |
| Qwen3-32B | `ớ` | worst | fabricated task, solved | *The question is asking for the value of cos(θ) given that cos(2θ) = 1/3. ### Step 1: Use the double angle identity …* |
| Qwen3-32B | `อังกฤษ` (Thai, "English") | worst | fabricated task, solved | *The question is asking for the value of 1/2 + 1/4 + 1/8 + 1/16 + ⋯, which is an infinite geometric series. ### Step 1: Identify the first term and common ratio …* |
| Qwen3-32B, thinking | `ล่า` | worst | fabricated task, reasoned about | *I need to determine if the statement "The sum of the first 100 odd numbers is 10,000" is true or false. First, let's understand what the first 100 odd numbers are …* |
| Qwen3-32B, thinking | `웛` | worst | fabricated task, reasoned about | *The user has asked me to provide a detailed explanation of the process of photosynthesis, including the key steps, the role of chlorophyll …* |
| Qwen3-32B | ` programma` | control | named and glossed | *You've written "programma" which is Italian for "program" or "programme". However, you haven't specified what kind of program …* |
| Qwen2.5-72B | `_ghost` | worst | quoted verbatim (string), then treated as "ghost" | *The term "_ghost" can have several meanings depending on the context: 1. Supernatural Entity: In folklore and popular culture, a ghost is …* |
| Qwen2.5-72B | `$filter` | worst | sigil dropped | *The term "filter" can have different meanings depending on the context. Here are a few common uses: 1. In Computing and Data Processing …* |
| Qwen2.5-72B | `눠` | worst | substitution → template token | *I believe you might be using an abbreviation or slang that isn't widely recognized. Could you please provide more context or clarify what you mean by "<|im_start|>"?* |
| Qwen2.5-72B | ` сразу` | control | named and glossed | *I'm not entirely sure what you mean by "сразу" (srazu) in this context. It's a Russian word that can mean "immediately," "at once," or "right away." …* |

The 1.7B and the 72B both reach for the chat template's own special token when the seed carries no usable content — `</think>` and `<|im_start|>` respectively — which is the substitution of §3 with the most salient recent token as the substitute. The 32B does not reach for anything; it invents a task.

The 72B adds a qualification to the metric itself. Its worst tokens by the bare probe are code-prefix pieces — `_ghost`, `$filter`, `/topics`, `_latitude` — and it quotes three of the eight logged ones verbatim (*The term "_ghost" can have several meanings …*) while dropping the sigil on four (`$filter` → "filter") and substituting the template's `<|im_start|>` for `눠`. A quoted "_ghost" re-segments as `_` + `ghost`, so the id-level metric counts it as not reproduced. On the saved samples, the string-level rate for worst tokens is 0 of 8 at 7B, 1 of 8 at 1.7B, 0 of 8 at 32B and 3 of 8 at 72B, against 7–8 of 8 for controls; the two metrics agree on the 7B, where §3's claim is made, and diverge where the worst set consists of re-segmentable strings. Both metrics are recorded from here on. The same set copies correctly in 48 of 80 specimen cells: for the 72B, what these tokens fail is the few-shot probe format, not their contexts.

**Geometry** is established at 7B with twice the positives of §6, undecided at 32B, where 0.75% prevalence yields 12 positives and a paired interval of ±0.15, and **reversed at 72B**. With 322 positives under the greedy gate, surface rarity alone reaches AUC 0.883 and static geometry 0.838; the paired difference is −0.044 [−0.070, −0.019], and under the paper's own gate, with 154 positives, it is −0.057 [−0.09, −0.02]. Adding the static features to the surface ones gains 0.04. The dynamic tier — the token's slot-state dispersion and norm on held-out contexts — still beats surface at 72B (+0.029 [+0.016, +0.044]), so information about fragility is present in the representation; what does not survive is the claim that the two embedding matrices alone carry it beyond rarity. The reason is visible in the fragile set: rare CJK singletons are picked out directly by a length-one, non-ASCII, high-id signature, and no embedding feature needs to be consulted. The 7B's fragile set — `according`, `due`, `который` — has no such signature, and there geometry carries information rarity does not. So the claim of §6 stands as a claim about the 7B's population, and the ladder shows it is not a law: which predictor wins depends on what the fragile population is, and the fragile population changes between models more than it changes with size. The tied 1.7B is reported and exempt, as pre-registered.

**Additivity** holds at 32B by the pre-registered rule: 13 significant pairs against a matched-null 95th percentile of 14, every cell-type mean within ±0.01, repetition again helpful (+0.0095, p = 0.012), and no dependence on separation from 2 to 128. One observation is recorded without being claimed. The family-wise maximum statistic exceeds its null by a wide margin (13.7 against a threshold of 4.8), driven by pairs among three tokens that are themselves the most fragile in the pool, in the pattern of §4.4; a pool of 35 cannot separate the members' fragility from their pairing, and the 235B rung, which runs the same pool, will show whether the shape recurs.

**Shared ids across sizes.** Restricted to ids sampled and gated by both models, the fragile sets of the three Qwen rungs share at most one token pairwise (` naprawdę`, 1.7B and 72B), against an independence expectation below 0.1; the sets are too small among the common ids to test the pre-registered bar. Qwen2.5 pads its vocabulary to 152,064 against Qwen3's 151,936, which shifted the stratified sampling so that only 1,545 ids were drawn for all three models; the 235B sample contains the 1.7B and 32B samples exactly and is where this question can be answered.

One confound is named rather than argued away: the 72B is a Qwen2.5 model, the other rungs Qwen3, so its higher prevalence, its length dependence and the reversal of the geometry result may be properties of a post-training generation rather than of size. The 235B rung, Qwen3 and thinking-capable, separates the two readings.

---

## 9. Scope and limitations

The behavioural results of §§3–6 are from a single model, with a second used only for the layer-wise identity analysis of §3.2. §8 replicates the fragility and reasoning-mode results on three further models from one other family, label-free, and the additivity result on one; the geometry result of §6 does not hold at 72B, where rarity predicts a CJK-dominated fragile set better than geometry does. The reasoning-mode reproduction metric of §3 is id-level; a string-level metric agrees with it on the 7B and disagrees on the 72B, whose worst tokens are re-segmentable code prefixes. The fourth rung of the ladder has not completed.

Copy log-probability is a proxy for the reasoning-mode behaviour that motivates the work. It was adopted because it is continuous and deterministic, which experiments over thousands of pairs or tokens require, and because every generation inspected alongside it agreed with it. An exploratory attempt to confirm the screen's most negative pairs in reasoning mode was confounded — three tokens dominated the selected pairs, and half the matched controls omitted the anchor token — and is reported as inconclusive.

The paired-bootstrap intervals in §6 exclude zero but are wide (+0.10 to +0.32); the advantage of geometry over rarity is established in direction more firmly than in size, and §8 shows the direction itself reverses on a model whose fragile set is dominated by rare CJK characters.

The contexts of §5 are random sequences of common words rather than natural text. The finding that no context is systematically hostile is a finding about that bank.

Which context elicits a given token's failure is not explained. The continuation-concentration and copy-suppression hypotheses of §5.4 each require one forward pass per token to test.

---

## 10. Conclusion

Glitch behaviour is neither compositional nor a loss of information. Two tokens at a distance do not interact, and a repeated healthy token is copied more reliably rather than less. A glitch token's identity reaches the final layer of the network; the readout substitutes a plausible neighbour with high confidence, and the model completes its task on the substituted content. What does vary with the surrounding text is a property of the token. Approximately one in sixty tokens that pass every single-context probe fails to copy in at least a tenth of ordinary contexts — by deletion, neighbour substitution, truncation or translation — with the triggering context unpredictable but the property stable, replicable across context banks and across model sizes, and — on the model studied — readable from the output embedding more accurately than from rarity alone, an advantage that §8 shows belongs to that model's fragile population rather than to fragility as such. Detectors that score tokens in isolation measure the wrong object for systems that assemble text programmatically. Fragility costs one forward pass per token per context, is adequately estimated from two dozen contexts, is stable enough to be measured once per model, and identifies tokens such as `according` that isolation-based probes certify.

---

## Appendix A. Hypotheses rejected by their controls

Each was asserted during the project and subsequently rejected. They are recorded so that the extent of the tested ground is clear.

- That the "no preceding space" rule identifies glitch tokens: it identifies the pre-tokenizer boundary; healthy tokens obey it at the same rate (survival 0.083 vs 0.093).
- That adjacencies the tokenizer cannot produce cause damage when supplied as ids: they were easier to copy than the same tokens in producible order, +0.94 [+0.61, +1.26].
- That contiguous runs of individually perfect tokens interact: reproduction falls from 1.000 at n = 1 to 0.648 at n = 8, but independence from the n = 2 rate predicts 0.354; the decline is sub-multiplicative.
- That the substituted token can be predicted from the direct path E_in·E_out: top-1 agreement 0.003.
- That substitutions are orthographic completions: 11% of failures.
- That generation length amplifies damage, or that glitch-seeded chains converge: §3.1.
- That persistent homology of the generation trajectory separates glitch from healthy seeds: best AUC 0.597 on 64 chains, underpowered.
- That post-training creates new glitch tokens: real movement in one lineage, boundary jitter in five others; §7.1 gives the reason.
- That fragile tokens are Unicode variants: all canonical (§5.4).
- That fragility accounts for near-miss hallucination on real entities: §7.2.

## Appendix B. Pre-registration and calibration

The design, null model, verdict criterion and failure criteria of §4 were fixed before the screen ran. Two amendments followed synthetic calibration and preceded any real data: standardisation of residuals by row and column before permutation, and replacement of a fixed rejection-count threshold by the matched-simulation threshold (a true null yields 0–2 rejections, not the ~200 the original threshold assumed). Separation was made the design variable at a collaborator's suggestion. The corrected power statement: a single isolated interaction is recovered in 4 of 5 simulations at 1.5 times the cell noise with R = 14 replicates, which is the Bonferroni-like worst case; with many true interactions the threshold relaxes.

The scale ladder of §8 was pre-registered separately (`docs/plan_scale.md`, 6 September 2026, before any rung ran): the models, the seven quantities with the range that would count as replication, and hypotheses H1–H5. Four amendments were made during the ladder, each before the data it affects was examined: (i) the cross-model gate was changed from the absolute −0.1 to the greedy gate p_alone > 0.5 after the 32B showed the absolute gate to be model-specific, and every model including the reference was recomputed under both; (ii) the token sample for the two largest rungs was doubled to 8,400 so that the geometry test would have enough positives, with the sampler changed so that the first 4,200 draws are byte-identical to the smaller runs; (iii) a string-level seed-reproduction metric was added alongside the id-level one after the 72B's worst set proved re-segmentable; (iv) the label-free glitch set was restricted to printable strings. One defect is recorded: the sampler stratifies over the embedding matrix's row count, and Qwen2.5's padded vocabulary shifted the strata, so the 72B's sample shares only 1,545 ids with the Qwen3 rungs'; the shared-id hypothesis H2 is therefore untested rather than tested and failed.

## Appendix C. Data and code

`docs/DATA_MANIFEST.md` maps each section to its files. The matrices of §4 and §5 are in `results/compact/`, with the replication bank's hidden states in `results/fragility_L.pt`; the specimen store is `results/specimens_confident_substitution.jsonl`; the pre-registration document is `whimsical-greeting-lerdorf.md`; code is in `src/cut/`. All experiments ran on a single A100-40GB; the longest run was 25 minutes.

Scale ladder (§8): `docs/plan_scale.md` (pre-registration), `docs/findings_scale.md` (full report), `results/gate_<model>.json`, `fragility_predict_<model>_greedy.json`, `reasoning_qwen3_*.json`, `specimens_qwen3_*_greedy.jsonl`, `interaction_screen_qwen3_32b.json`, and for the 72B `fragility_qwen25_72b.json`, `gate_qwen25_72b.json`, `fragility_predict_qwen25_72b{,_greedy}.json`, `reasoning_qwen25_72b.json`, `specimens_qwen25_72b.jsonl`; compact matrices in `results/compact/`; complete snapshots of each node's logs and results in `results/remote_h100/`; driver `src/cut/run_scale.sh`. The ladder ran on one H100-80GB (1.7B, 32B) and one 4× H100-80GB node (72B).
