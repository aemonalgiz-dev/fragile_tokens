# Glitch Tokens Are Abandoned, Not Stillborn

**Draft — 2026-09-05, revision 2.** Revision 1 led with a two-population taxonomy.
A control run afterwards falsified it (§1.3) and the paper is restructured around
the mechanism, which has survived every control applied to it.

---

## Abstract

Glitch tokens are commonly assumed to be vocabulary rows that never received
gradient. We show the opposite: doomed rows receive *larger* total updates than
healthy rows — up to 5.2x — while the component of that update which is
token-specific collapses by step 32. The excess is shared drift, common to the
whole embedding table. Any diagnostic based on update magnitude therefore reports
the reverse of the truth. Decomposing updates into a shared-drift component and an
orthogonal token-specific residual makes the effect visible, and residual
suppression replicates across 6 models, 3 corpora, dense and MoE architectures.

Two practical consequences follow. Repairing embeddings *during* pretraining makes
glitch tokens significantly worse across 6 paired seeds, while post-hoc
neighbour-based repair helps; the deciding variable is where a row is placed, not
when. And a detector at AUC 0.987 has 3.0% precision at a usable threshold, so
acting on its ranking raises held-out perplexity from 15.9 to 183.4.

We also report that the field's two behavioural criteria — next-token entropy and
copy/repetition — score AUC 0.453 and 0.947 against the same labels, i.e. they do
not measure the same thing. We do not have a validated explanation for this and
say so.

---

## 1. Preliminaries

### 1.1 Specimens

Behaviourally verified glitch tokens in pythia-1.4b are overwhelmingly bare
sub-word fragments, and they fail in a characteristic way — the model emits the
*containing word* instead of reproducing the fragment:

| token | entropy | copy output |
|---|---|---|
| `ilarly` | 5.28 | ` similarly` |
| `icrosoft` | 6.32 | ` Microsoft` |
| `ortunately` | 4.23 | ` fortunately` |
| `pendicular` | 5.47 | ` perpendicular` |
| `emetery` | 5.18 | ` cemetery` |
| `bably` | 6.19 | ` probably` |
| `ecause` | — | ` because` |

Healthy controls of similar length copy cleanly (` originates` -> ` originates`,
` inhibit` -> ` inhibit`, `Chief` -> ` Chief`).

### 1.2 The two behavioural criteria disagree

Discriminating Magikarp's 36 verified tokens from 1,000 they tested and rejected:

| criterion | origin | AUC |
|---|---|---|
| copy / repetition | Magikarp (EMNLP'24), GlitchQuiz (USENIX'26) | **0.947** |
| next-token entropy | GlitchMiner (AAAI'26) | **0.453** |
| spelling | GlitchQuiz (USENIX'26) | 0.612 |

The copy figure is partly circular — Magikarp's labels come from a repetition
test. The independent criterion scores below chance. Whatever "glitch token"
denotes, these papers are not selecting the same set.

### 1.3 A retracted explanation

We initially attributed this to two populations with opposite entropy signatures:
*stillborn* tokens (zero token-specific gradient throughout) spiking in entropy,
and *abandoned* fragments carrying normal entropy. A control falsified it:

| group | n | entropy |
|---|---|---|
| whitespace, stillborn | 200 | **6.116** |
| whitespace, received gradient | 118 | **4.014** |
| non-whitespace, stillborn | 46 | 4.560 |
| non-whitespace, healthy | 300 | 4.733 |

The stillborn/entropy association holds **only within whitespace**. For
non-whitespace tokens it is absent and slightly reversed (AUC 0.436). Pythia's and
OLMo's `UNREACHABLE_MULTI_TOKEN` categories are 100% whitespace, which is what
produced the original result.

A surviving hypothesis, stated as such because it is fitted to the data that
falsified its predecessor and has not been independently tested: what matters is
whether the model holds a prior for the string from a **larger unit it knows**.
` researc` never appears alone but `research` does, so the model predicts
confidently (normal entropy) and fails only on reproduction; a nine-space
indent token has no containing unit, so entropy spikes. This accounts for all four cells above and
requires a pre-registered test on a held-out family.

The gradient-based split itself is measured and stands: 293 tokens receive zero
token-specific input update through step 64, versus 0.2% of healthy tokens. What
does not stand is the claim that it predicts an entropy phenotype.

## 2. Mechanism: abandonment, and why the obvious test inverts it

Split each row's update between checkpoints into the shared drift direction
(the mean update over all rows) and the orthogonal token-specific residual.

Glitch/healthy ratios, pythia-1.4b:

| step | total ‖ΔW‖ | shared | residual |
|---|---|---|---|
| 2 | 1.036 | 1.302 | 1.035 |
| 32 | 1.113 | 1.443 | **0.698** |
| 256 | **1.302** | 2.050 | **0.500** |

Doomed rows move *more* than healthy rows in total while learning less. Anyone
testing "was this row updated?" concludes these are the best-trained tokens in the
vocabulary. On the from-scratch rig (§4) the effect is starker: total ratio 5.236,
shared 13.839.

**Three named tokens at step 256.** The ratios above are population means; the
same inversion is visible token by token. `residual/total` is the fraction of a
row's motion that is token-specific:

| token | total ‖ΔW‖ | residual | fraction token-specific |
|---|---|---|---|
| `ecause` (glitch) | **0.0895** | 0.0232 | **0.259** |
| `icrosoft` (glitch) | **0.0914** | 0.0286 | **0.313** |
| ` originates` (healthy) | 0.0648 | 0.0416 | **0.642** |

Both glitch tokens moved *further* than the healthy token while acquiring less
than half as much token-specific structure.

**Replication.** Residual suppression in **6/6** models — pythia-410m/1.4b/6.9b
(Pile), OLMo-1B (Dolma), OLMo-2-1B/7B, OLMoE-1B-7B (MoE) — with ratios 0.146–0.736.
Masking holds in **5/5** models possessing a checkpoint at or before step 1000. The
two that do not show it (OLMo-1B, OLMoE) have no checkpoint before step 2000/10000,
and Pythia itself loses masking by step 1000; they are untestable in the relevant
window rather than counterexamples.

This resolves an open disagreement. The folk model (rows never received gradient)
is wrong. *Hub of Short Rows* (2608.29702) is right that the rows were updated —
and this identifies what those updates were: shared drift, not learning.

**Corollary (§5).** Shared drift may be partially *protective*, carrying doomed
rows off the centroid into a distinguishable region.

---

## 3. AUC is not a safe operating statistic here

Our late detector scores **AUC 0.987** against verified labels. Its **precision at
top-500 is 3.0%** — a 0.073% base rate makes those compatible. Acting on it:

| repair set | verified glitch | control | perplexity |
|---|---|---|---|
| top-500 | 5% → 5% | 70% → 60% | 15.92 → **87.34** |
| top-2000 | 5% → 10% | 70% → **50%** | 15.92 → **183.42** |
| top-2000 **+ behavioural guard** (717 rows) | 5% → **15%** | 70% → 70% | 15.92 → **15.92** |

The flagged set contains ` and`, ` of`, ` to`, ` is`, `,`, `.`. The features encode
"unusual geometry", and the hyper-frequent head is unusual in the opposite
direction from the starved tail; with 36 positives a linear model cannot separate
them. A two-stage filter (detector **and** behavioural screen) is required, and
even then repair mitigates rather than restores.

Every paper in this area reports AUC or F1; none reports precision at an operating
point.

---

## 4. Method: a rig with true frequency labels

Published work substitutes proxies for token frequency because real corpora are
unavailable. We train GPT-NeoX models (23M, untied embeddings) from scratch with a
deliberate tokenizer/corpus mismatch — BPE fitted on English+code+German, model
trained on English only — over 117M tokens. Every token's **exact** training count
is known; 712 have count zero. The manufactured glitch tokens are exactly the
expected kinds: German merges the English-only model never sees (`ischen`,
`über`, ` auf`, ` als`, ` wurde`), code punctuation (`()`, `('`, `):`), and
English fragments BPE can never emit because a longer merge always wins
(`owever`, ` includ`, ` charact`, ` describ`).

Intervention arms branch from a shared step-256 checkpoint so they are paired in
both weights and data order. 6 seeds.

**Caveat we discovered late and consider important:** manufactured zero-count
tokens are stillborn *by construction*. The rig silently samples one population,
which is why entropy worked there and fails on Pythia's fragment-heavy verified
set. Synthetic ground truth removed the failure mode that dominates in practice.

---

## 5. Intervention: repair late, and place rows carefully

Paired across 6 seeds, Δ vs baseline in doomed-token entropy (negative = repair):

| arm | Δ doomed | Δ healthy | Δ val loss |
|---|---|---|---|
| reinit @256 | **+0.183** | +0.050 | +0.0008 |
| freeze @256 | **+0.489** | +0.067 | **+0.0207** |
| mask @256 | **+0.235** | **+0.863** | +0.0008 |
| post-hoc centroid | −0.649 | +0.071 | +0.0077 |
| **post-hoc kNN** | **−1.245** | +0.001 | +0.0012 |

All CIs exclude zero. Every training-time intervention makes things **worse**.

We predicted that freezing repaired rows would prevent shared drift from
re-corrupting them; freeze is the worst arm. The surviving explanation is
positional: `reinit`/`freeze` park rows at the healthy **centroid**, the point of
maximum confusability, and `freeze` also forbids escape. kNN repair places rows
near *specific* neighbours. Late beats early because at step 14,000 you know where
the neighbours are; at step 256 the only available target is the centroid.

`mask` shrank the doomed-healthy gap by degrading *healthy* tokens (+0.863). The
gap metric is gameable; report both columns.

---

## 6. Negative results

Reported because each is well-powered against a validated instrument, and each
tests a hypothesis a reader would otherwise consider promising.

- **Compositional / "weld" glitch n-grams.** Violating the model's most confident
  continuations, matched on the inserted token's conditional probability:
  corr(commitment, damage) = **−0.003**, n=1200, commitment spanning 0.036–1.000.
  The welds themselves are real and easy to find: ` prerequ`->`isites` (p=0.994),
  ` pione`->`ers` (p=0.997), ` javax`->`.` (p=0.957), and a memorised citation
  template ` [****, ()](\`->`doibase` at p=1.000. Violating them does nothing.
  The same probe detects single-token glitches at 3.18 nats — a ~600× ratio.
  *Commitment and fragility are decoupled*, extending "Broken Tokens" (2506.19004).
- **Attractor behaviour.** Glitch tokens are not over-produced in open slots:
  top-50 enrichment **1.00×**, exactly their vocabulary share. The strongest slot
  attractors in the whole vocabulary are ordinary names that copy cleanly
  (` Julie`, ` Betty`, ` Patricia`, ` Jason`); no glitch token ranks near them. They are ~17% *less*
  context-sensitive than frequency-matched healthy tokens — inert, not explosive.
- **n-token repetition.** BPE builds repetition ladders (` ` has rungs at every
  count to 79, then 96/128/192/256/512), and copy success varies non-monotonically
  along them — base `-` copies at k=1..6, fails at 7, copies at 8, fails at 9-11.
  Multi-token repetitions copy at 6% vs 30% for
  single-token, but the effect vanishes under length matching (repetitive 12% vs
  non-repetitive 6% overall, gap ≈ 0 beyond 9 characters).
- **Hubness.** Glitch tokens are not hubs. AUC 0.26–0.40 in Pythia (backwards);
  the large OLMo ratios (179–405×) are a degenerate clique — 235 tokens holding
  68% of all k-occurrence mass.

---

## 7. What this paper does not claim

- **Not a better detector.** Head-to-head on the same task, Magikarp's three
  indicators beat our eight trajectory features: AUC 0.994 vs 0.983, recall@500
  75% vs 61%. Combining adds ~6pp.
- **We retract an earlier claim** that we found 13 glitch tokens the published
  method missed. All 13 rank 200–257 under Magikarp's own indicators and were
  tested and rejected by them; five with reproduction probability > 0.95. It is a
  verification-criteria disagreement, and belongs to §1 rather than to detection.
- **The early signal is a frequency detector.** `resid_i` reaches AUC 0.896 at
  step 2 and 0.921 at step 4 on pythia-6.9b — against labels produced
  independently for that exact model — where the standard geometric indicator sits
  at 0.451, below chance. But inside a frequency-matched pool it degrades to ~0.70,
  and anyone training a model already has exact counts. We present it as evidence
  that glitch-predictive information exists in the optimizer before it exists in
  the geometry, not as a tool.

## 8. Limitations

n=36 for Pythia's verified set. Two model families for the taxonomy. The
fragment/stillborn split uses a post-hoc string test that should be pre-registered.
One behavioural probe per phenotype. The rig is 23M parameters, and its glitch
fraction (4.35% zero-count) is 4–40× real models'.

## 9. What would strengthen it most

1. A third family's verified set for the taxonomy (GPT-2 and gpt-j-6b are
   labelled and have no checkpoints; OLMo-1.7-7B is labelled).
2. A pre-registered stillborn/abandoned criterion, ideally from tokenizer
   reachability rather than string shape.
3. Per-phenotype detection: entropy for stillborn, repetition for abandoned,
   reported separately — which no existing paper does.
