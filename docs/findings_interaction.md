# Non-contiguous token interactions: pre-registered screen

## The claim under test

Glitch-like behaviour is caused by a **non-contiguous subset** of a context —
members possibly far apart, possibly individually healthy:

```
[1, 231, 885, 9911, 1112, 231]         the REPETITION of 231 at distance
[1, 231, 885, 9911, 9999, 1922, 1013]  885 and 1013 CO-OCCURRING at distance
```

Every earlier experiment in this project tested *adjacent* pairs or *contiguous*
n-grams. This is the first test of the actual claim, and unlike the four claims
that died to post-hoc controls earlier, its null and failure criteria were fixed
before it ran, in a written plan kept outside this repository.

Model: `allenai/OLMo-2-1124-7B-Instruct`. Code: `src/cut/interaction_screen.py`,
`interaction_confirm.py`, `interaction_geometry.py`, `stats.py`
(`perm_null_additive`, `bh_fdr`, `interaction_test`), `calibrate_null.py`.

## Design (as run)

- **Pool**: 64 tokens sampled uniformly over the vocabulary and filtered to
  those the model reproduces perfectly alone (482/600 passed). Any interaction
  therefore cannot be inherited from a member.
- **Distance is the variable**: two long neutral filler sequences (common
  clean words); for each `d ∈ {2,4,8,16,32,64,128}` the carrier is
  `filler[:d+4]` with slots at positions 2 and 2+d. R = 14 replicates.
- **2×2 per ordered pair (a,b)**: `I = S11 − S10 − S01 + S00`, absent slot
  filled by a control token (three control pairs, two different controls so
  S00 is not itself a repetition). Diagonal a==b is the repetition case.
- **S** = teacher-forced copy logprob at the two slot positions. All 63,014
  cells scored; per-position vectors and slot hidden states (layers 16, 32)
  stored (`results/interaction_screen.pt`, 3.1 GB).
- **Null**: additive-model residuals, standardised per row/column by robust
  scale, permuted within replicate, BH-FDR q=0.05; family-wise max-statistic;
  verdict threshold from 20 no-interaction simulations matched to the observed
  heteroscedastic noise. Calibration (`calibrate_null.py`, R=14): 0 false
  rejections in 10/10 null runs after standardisation (5–18 before it).

## Results — clean pool

### Scorer sanity: passes
Positive-control glitch tokens dropped to −2.9 … −6.5 logprob against a
clean-pool single-cell baseline of −0.02. The screen would have seen an effect.

### Screen: no detectable interaction (pre-registered verdict)

```
BH-FDR q=0.05:  1 significant off-diagonal pair of 4032    matched-null 95th pct: 4
                0 significant diagonal (repetition) of 64
family-wise:    max|z| 29.06 vs threshold 28.78,  p = 0.035
power (single injected effect, 5 sims):  1.0×noise 0.2   1.5× 0.6   2.0× 1.0   3.0× 1.0
```

One rejection is what the matched null produces by itself (0–5). By the
criterion fixed in advance — more rejections than the matched null **and**
family-wise p < 0.05 — this is null.

**No distance dependence.** Off-diagonal mean I by distance:

| d | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| I off-diag | −0.0007 | +0.0036 | +0.0001 | +0.0021 | +0.0021 | +0.0026 | +0.0055 |
| I repetition | +0.0049 | +0.0032 | +0.0007 | +0.0043 | +0.0033 | +0.0113 | +0.0183 |

**Repetition at distance helps, robustly.** Diagonal mean I = +0.0066,
permutation p = 0.000 against the off-diagonal null — the pair copies *better*
than its parts, and the benefit grows with distance (+0.005 at d=2 → +0.018 at
d=128). That is induction: the second occurrence is easier because the first is
in context. Opposite sign to the hypothesis; small in absolute terms.

### The caveat: the clean pool sits at ceiling
Clean tokens copy at p ≈ 0.98; median cell noise is 0.005 logprob. There was
essentially no room for a combination to hurt. This pool was my recommendation
and it was the wrong one for detecting anything — a fair test needs tokens with
room to fail. See the mixed-pool control below.

### What the failures that do exist look like
Individually-clean tokens still fail to copy (slot logprob < −0.5) in **453 of
57,344 cells (0.8%)**. Those failures have structure, and it is not pairwise:

- **Concentrated in a few tokens.** Five tokens account for 47% of failing
  cells: `.Ver` (111 failures in slot i), `Sentence` (118 in slot j),
  ` Patriots` (91 in slot i), `callee`, `_cum`. Position-specific fragility.
- **Not reproducible across fillers.** Same-distance overlap of failing cells
  between the two filler sequences: 3, 0, 1, 0, 0, 0, 2. At d=4 one filler
  produced 68 failures and the other 1; at d=128, 158 vs 26.

So the failures factor as **(a fragile token) × (a hostile surrounding
context)**. The additive model absorbs token fragility as a main effect, which
is precisely why I is null. The "grouping" that breaks copying is a fragile
token plus its whole context — not two specific tokens at a distance.

### Confirmation (exploratory — nothing survived FDR)
Top-40 pairs by I vs matched-partner controls, chat-format "repeat then
explain", new filler, two distances each:

| | survivor | matched control |
|---|---|---|
| failure, strict match | 0.287 | 0.200 |
| **failure, case/punct-insensitive** | **0.100** | **0.087** |
| substitution | 0.050 | 0.031 |

Paired difference (primary): +0.013, 95% CI [−0.069, +0.094]. The strict-match
gap that looked suggestive was the model quoting the text back lowercased with
punctuation dropped (`.Ver` → `ver`) — normalisation, not substitution.

### Geometry
No FDR-significant pairs, so AUCs are undefined; Spearman with continuous I:
every static feature |ρ| < 0.07 (cos, direct-path cross terms, norm, id).
Activation-space interference at the mid layer correlates weakly (ρ ≈ +0.19)
and **decays cleanly with distance** — R_copy_i = 0.041 at d≤8, 0.005 at
16–32, 0.003 at d≥64 — but there is no damage for it to predict.

## Results — mixed pool (ceiling control)

Same design; pool = 32 **weak** tokens (single-token copy logprob in (−3, −0.3):
they copy sometimes, so a combination has room to push them over) + 32 clean.
`results/interaction_screen_mixed.*`, `interaction_confirm_mixed.json`,
`interaction_geometry_mixed.json`.

### Screen verdict: null — but underpowered, and that must be said

```
BH-FDR q=0.05:  1 significant off-diagonal of 4032   matched-null 95th pct: 14
                1 significant diagonal of 64
family-wise:    max|z| 13.97 vs threshold 11.06,  p = 0.005
power (single injected effect):  1.0× 0.2   1.5× 0.0   2.0× 0.0   3.0× 0.6
```

Weak tokens make the residual distribution far heavier-tailed (cell noise sd
0.013 vs 0.005; matched null now yields 3–16 spurious rejections). The
per-pair test cannot find an isolated effect below ~3× cell noise here, so
"1 rejection" is uninformative about *sparse* large effects. What the test
cannot resolve, the averages can.

### Co-occurrence is additive — the central negative result, and it is robust

Mean interaction by cell type, all distances pooled (n = cells, se over cells):

| a-kind × b-kind | n | mean S11 | **mean I** | se |
|---|---|---|---|---|
| clean × clean | 992 | −0.027 | +0.0011 | 0.0004 |
| clean × weak | 1024 | −0.146 | +0.0030 | 0.0016 |
| weak × clean | 1024 | −0.195 | +0.0033 | 0.0014 |
| **weak × weak** | 992 | **−0.313** | **+0.0076** | 0.0027 |

Weak tokens copy much worse (S11 falls 12×), but that is the **additive main
effect**. The interaction — damage beyond what each token does alone — is
slightly *positive* in every cell type, including weak×weak where there was
ample room to fail (mean +0.0076, ~3 se above zero). Two weak tokens at any
distance from 2 to 128 copy marginally *better* together than additivity
predicts, not worse. Across both pools, 8,064 ordered pairs, 14 replicates
each, there is no compositional damage from co-occurrence at a distance.

### Repetition: sign flips, and the negative side is context-specific

Clean pool: repetition **helps** (+0.0066, p=0.000, growing with distance —
induction). Mixed pool: diagonal mean −0.0130, p=0.000 against the off-diagonal
null, and the long band (d=64,128) flags 10 diagonal cells at fwer p=0.000.
That looks like archetype 1. The per-filler values say otherwise:

```
                        I(d=64)  f0/f1      I(d=128) f0/f1
'Tools'                 -0.006 / +0.001     -1.387 / -0.000
'Strategy'              +0.003 / +0.000     -0.784 / -0.000
'站'                     -0.003 / -0.006     -1.620 / -0.010
' HttpResponseMessage'  +0.002 / -0.029     -1.376 / +0.087
' Associated'           +0.002 / +0.009     -0.005 / -0.635
'razione'               +0.559 / -0.091     +1.096 / -0.027    <- positive
```

Almost every large negative sits in **one filler at d=128** with the other
filler at zero; `' Associated'` shows it in the *other* filler; `'razione'`
goes the opposite way. By kind, at d≥64: clean −0.018 (se 0.012), weak −0.053
(se 0.050) — neither distinguishable from zero. So: in a specific 132-token
context, several tokens fail to copy their **second** occurrence while copying
fine when they occur once. That is a genuine repetition-specific failure — the
2×2 controls for single-occurrence fragility — but it is a property of
(token, context), not of (token, token). It does not transfer to the other
filler. Same structure as the clean-pool fragility: **token × context**.

Exploratory lead only: the three repetition cells in the confirm set showed
~2× the 2nd→1st attention at layer 2 of three non-flagged tokens (0.046 vs
0.025, AUC 1.000). n = 3 vs 3; perfect separation on six points is a 1-in-20
event. Worth a real test with dozens of tokens; not evidence yet.

### Confirmation (exploratory — nothing survived FDR)

| | survivor (top-40 by I) | matched control |
|---|---|---|
| failure, strict | 0.863 | 0.588 |
| **failure, case/punct-insensitive** | **0.688** | **0.450** |
| substitution | 0.263 | 0.175 |

Paired difference +0.237, 95% CI [+0.100, +0.369] — an independent-data
effect on a new filler and a different task. Two reasons not to lean on it:
(1) pairs were selected on I without FDR, and the top-40 is three tokens' worth
of pairs (`/****…` in 15 of 40, two others in 9 and 8); (2) half the matched
controls swap the *anchor* out — `(a',b)` — so an always-failing anchor's main
effect leaks into the "pair" difference. Failure modes in the transcripts are
the ones seen all day: the unrenderable `/****…` token is **deleted** and the
text called "fragmented"; `carbohydr` is **completed** to "carbohydrate";
`estava` becomes "está nada". Deletion and lexical completion, not
pair-specific substitution.

### Geometry (mixed)
Nothing: every static and activation-space predictor |ρ| < 0.08 with I.
Interference again decays cleanly with distance (0.050 → 0.009 → 0.005) and
predicts nothing.

## What this experiment established

1. **Co-occurrence at a distance does not create glitch behaviour.** Additive
   in both pools, all cell types, d = 2…128. This is the cleanest negative in
   the project: pre-registered, calibrated null, scorer validated, and the
   averages are tight enough not to depend on the per-pair test's power.
2. **Repetition at a distance helps healthy tokens (induction) and is
   context-specifically harmful for some tokens in some long contexts.** The
   harmful cases do not reproduce across fillers; they are not a pair property.
3. **Nothing geometric predicts any of it** — static or dynamic, either pool.
4. **The recurring structure is token × context, not token × token.** A small
   set of tokens — some individually perfect — fail in a minority of surrounding
   contexts, and *which* contexts is not stable. That is the residue worth
   pursuing: **fragility rate** as a per-token property (fraction of random
   contexts in which a token fails to copy), which no single-context probe
   measures and which is where an agentic hazard would actually sit. It is
   N×C forward passes, not N², and the machinery here builds it directly.

Hardware note: the A100 is idle. Two hung SSH wrappers (from `setsid nohup …
&` inside ssh commands) were killed locally; no remote job was affected.
