# Idea 1: glitch tokens across pretraining (pythia-160m AND 1.4b)

Ground truth is BEHAVIOURAL, measured on the final model: the day-1-validated
copy probe run over all 50,304 tokens; glitch = worst 1% (n=504). We then ask how
well embedding geometry at checkpoint t predicts that final behaviour. Logistic
regression over 8 indicators, fit on half the vocabulary, scored on the held-out
half. 20 log-spaced checkpoints from step 0 to 143,000.

## Finding 1 -- final glitch status is predictable within the first few hundred steps

Combined-predictor AUC:

| step | 0 | 1 | 2 | 4 | 128 | 256 | 1000 | 143000 |
|---|---|---|---|---|---|---|---|---|
| AUC | .487 | .487 | **.873** | .885 | .845 | **.960** | .975 | .995 |

Chance at steps 0-1 (as it must be -- no information exists yet), then 0.873 after
**two optimizer steps**, and 0.95 by step 256 = **0.18% of training**.

No single indicator achieves this; the best solo is `resid_i` (0.842 at step 2).
The classic static detectors are useless this early: `unemb_norm`, `unemb_cnorm`,
`unemb_cos` sit at 0.506/0.506/0.512 -- pure chance -- until very late.
`unemb_norm` only reaches 0.995 at step 143,000, i.e. it works exactly when you no
longer need it.

## Finding 2 -- the early signal is gradient sparsity, not clairvoyance

`resid_i` carries everything early. Input-embedding gradients are SPARSE: only
tokens actually present in a batch receive a token-specific update. So this is
corpus-free frequency estimation read off the optimizer, not the model "knowing"
anything. Useful -- it needs no corpus counts and no behavioural testing -- but it
should not be oversold.

## Finding 3 -- glitch tokens are TWO populations, not one

Fraction of tokens with zero token-specific input update:

| step | all | glitch | healthy |
|---|---|---|---|
| 2  | 3.34% | **54.96%** | 2.82% |
| 32 | 0.63% | **52.58%** | 0.11% |
| 64 | 0.58% | **52.38%** | 0.06% |

- **Stillborn (~53%)**: literally never appear; zero token-specific gradient
  through step 64. Identifiable with near-certainty at step 2.
- **Abandoned (~47%)**: they DO appear and DO receive gradient, yet still end up
  behaviourally glitched. Invisible to any "was it updated" test.

(After step 128 the zero-fraction goes to 0.00% for every token -- almost certainly
numerical/optimizer-state leakage rather than real learning. Flagged, not relied on.)

## Finding 4 -- raw update magnitude HIDES abandonment; the shared/residual split reveals it

Decomposing each row's update into the global drift direction and the
token-specific residual (glitch:healthy ratios):

| step | total \|dW\| | shared | residual |
|---|---|---|---|
| 8   | 1.010 | 1.164 | 0.955 |
| 32  | 1.036 | 1.163 | **0.727** |
| 64  | 1.035 | 1.160 | **0.644** |
| 512 | 1.061 | 2.309 | 0.751 |
| 1000| 0.645 | 1.218 | 0.625 |

Raw magnitude shows **parity through step 512** -- glitch rows appear to be updated
as much as healthy ones, even slightly more. That parity is entirely global drift:
glitch rows are swept along the shared direction *more* than healthy rows
(ratio 1.16-2.31) while their token-specific learning falls behind from step ~8-32.

This resolves the open disagreement. The folk model ("glitch tokens are rows that
never received gradient") is wrong -- they receive large updates. "Hub of Short
Rows" (2608.29702) is right that the rows *were* updated, and this says what those
updates were: shared drift, not token-specific learning.

**Glitch tokens are mostly abandoned, not stillborn -- and for the ~47% that do
appear in the data, abandonment is the only story.**

## Caveat
Aggregate ratio separation at step 32 is NOT the same as per-token
discriminability; individual prediction needs step 256. Do not conflate the two.


---

# 1.4b replication (9x model size) -- all four findings hold

Combined-predictor AUC, pythia-1.4b:

| step | 0 | 1 | 2 | 128 | 256 | 1000 | 2000 | 143000 |
|---|---|---|---|---|---|---|---|---|
| AUC | .512 | .512 | **.832** | .840 | **.926** | **.960** | .968 | .946 |

Chance at 0-1, 0.832 after two steps, AUC>=0.90 at step 256 (0.18% of training,
identical to 160m), AUC>=0.95 at step 1000 (0.70%). Slightly later than 160m to
0.95 but the same shape. Again no single indicator gets there: best solo is
`unemb_cos` / `shared_o` reaching 0.90 only at step 512, and `unemb_norm` --
the standard static detector -- **never** reaches 0.90 at any checkpoint on 1.4b.

## Mechanism is size-invariant

Glitch:healthy ratios and zero-update fractions:

| step | 160m total | 160m shared | 160m resid | 1.4b total | 1.4b shared | 1.4b resid |
|---|---|---|---|---|---|---|
| 2   | 1.011 | 1.203 | 1.010 | 1.032 | 1.351 | 1.031 |
| 32  | 1.036 | 1.163 | **0.727** | 1.091 | 1.463 | **0.736** |
| 64  | 1.035 | 1.160 | 0.644 | 1.093 | 1.454 | 0.607 |
| 512 | 1.061 | 2.309 | 0.751 | 1.233 | 2.435 | 0.810 |

The residual ratio at step 32 is 0.727 vs 0.736 across a 9x size difference --
essentially the same number. Total magnitude shows parity (>=1.0) through step 512
in BOTH models while the residual has already collapsed. The masking effect is
not a small-model artifact.

Two populations replicate too -- zero token-specific input update at step 2:
54.96% of glitch vs 2.82% healthy (160m); **59.52% vs 2.78%** (1.4b).

## Bottom line
1. Final glitch status is predictable from embedding geometry at **0.18% of
   training**, on both model sizes, against behaviourally-defined ground truth.
2. Static detectors (norm, cos-to-mean) are at chance that early and are the
   *worst* choice for a training-time monitor -- on 1.4b `unemb_norm` never
   reaches AUC 0.90 at any checkpoint.
3. Glitch tokens split ~50/50 into stillborn (never appear) and abandoned
   (appear, get gradient, still degrade).
4. Raw update magnitude is actively misleading; the shared/residual split is
   what exposes abandonment, and it does so by step 32.
