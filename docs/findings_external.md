# External validation on Pythia, and an honest correction

Everything before this defined "glitch" with my own copy probe, so detector and
labels shared a source. Magikarp (Fishing for Magikarp, EMNLP 2024) publishes
per-token verification for EleutherAI/pythia-6.9b: 1,124 candidates behaviourally
tested, **36 verified under-trained**. All Pythia sizes share one tokenizer and
the same Pile data in the same order, so a token starved in 6.9b is starved in
1.4b -- valid, fully independent ground truth for the 1.4b trajectories.

## The correction I owe

My earlier headline ("AUC 0.95 at step 64") was scored against labels derived
from token frequency. That is closer to circular than I acknowledged: the early
signal IS a frequency detector, so predicting frequency-derived labels flatters
it. Against independent external labels the numbers are lower, and one of them
is much lower.

## Result 1 -- finding the 36 verified tokens in the whole vocabulary

| step | combined | `resid_i` (ours) | `unemb_cos` (standard) |
|---|---|---|---|
| 0 | .504 | .500 | .412 |
| **2** | .865 | **.911** | .412 |
| **8** | .874 | **.929** | .412 |
| 64 | .911 | .893 | .455 |
| 256 | .886 | .845 | .943 |
| 1000 | .915 | .839 | .965 |
| 143000 | .987 | .481 | .959 |

The two signals are **complementary in time, and cleanly so**. Our gradient-sparsity
signal hits AUC 0.91-0.93 at steps 2-8, where the standard static indicator sits at
**0.412 -- worse than chance**. Then they swap: `resid_i` decays to uselessness
(0.48) while `unemb_cos` becomes the better detector from step 256 on.

Early = "will this token be starved?" (a frequency question).
Late  = "is this token's representation broken?" (a geometry question).
Different questions, and only the first is answerable early.

## Result 2 -- the honest limitation

Scored *inside* Magikarp's own candidate pool (verified vs their false positives,
n=1,124) the task is much harder, because every token in that pool is already
rare:

| step | 2 | 32 | 256 | 512 | 1000 | peak |
|---|---|---|---|---|---|---|
| combined AUC | .583 | .728 | .762 | .829 | .872 | .891 (step 32k) |

Early AUC drops to ~0.70 and never reaches 0.90 at any checkpoint. **Within a
frequency-matched pool the early detector loses most of its power** -- which is
exactly what it should do if it is fundamentally a frequency detector. Reported
because it is the strongest objection to the method and it is real.

## Result 3 -- the n-gram-adjacent class is detectable at step 2

Magikarp labels 198 tokens `UNREACHABLE_MULTI_TOKEN`: they exist in the vocabulary
but the tokenizer can only ever emit them inside a longer sequence. That is the
closest labelled analogue in the literature to an n-gram glitch.

| step | 0 | 2 | 32 | 256 | 16000 |
|---|---|---|---|---|---|
| AUC | .455 | **.986** | **.999** | .991 | 1.000 |

Near-perfect from step 2. Partly trivial -- unreachable implies zero frequency, so
gradient sparsity catches them immediately -- but it does mean the sequence-
reachability class is fully identifiable at 0.001% of training.

## Result 4 -- additional glitch tokens Magikarp did not flag

Ranking `OK`-categorised tokens Magikarp never verified, using the LATE detector,
then checking them against our independent copy probe:

| score | copy-lp percentile | token |
|---|---|---|
| 49.8 | 1.7% | `'imonit'` |
| 49.7 | 1.0% | `'ÃÂÃÂÃÂÃÂ'` |
| 45.9 | 1.1% | `' {¶'` |
| 41.9 | 2.0% | `' $[]$'` |
| 40.1 | 1.4% | `' careg'` |
| 39.0 | 1.4% | `'medsc'` |
| 38.7 | 0.6% | `'1451450014514500'` |
| 37.6 | 1.5% | `'ICENSE'` |

**8 of 12 land in the worst 5% of the independent copy probe.** They are
recognisable glitch types: mojibake (`ÃÂ` = double-encoded UTF-8), truncated words
(`careg`, `medsc`, `ICENSE`), and a repeated numeric string.

The EARLY detector does NOT find these -- only 2/12, and its top candidates are
ordinary words (`'known'`, `' Please'`). Consistent with Result 1: early catches
starvation, not representational damage.

## Bottom line
- Early detection is externally validated (AUC 0.91-0.93 at steps 2-8) where the
  standard indicator is below chance -- but it is a frequency detector, and inside
  a frequency-matched pool it degrades to ~0.70.
- The two indicator families are complementary in time; neither works at both ends.
- The method finds real glitch tokens the published method missed, but only from
  the late checkpoint.
