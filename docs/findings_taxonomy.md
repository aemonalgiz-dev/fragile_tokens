# "Glitch token" is two phenomena, and the published criteria detect different ones

Set out to close a weakness -- every label in this project rested on one
behavioural probe (copy/repetition). Ran two additional published criteria that do
not involve copying. They disagree with each other, and the disagreement turns out
to be the finding.

## The criteria do not agree

On pythia-1.4b, discriminating Magikarp's 36 verified tokens from 1,000 they
tested and rejected:

| criterion | source | AUC |
|---|---|---|
| copy / repetition | Magikarp EMNLP'24, GlitchQuiz USENIX'26 | **0.947** |
| next-token entropy | GlitchMiner AAAI'26 | **0.453** (below chance) |
| spelling | GlitchQuiz USENIX'26 | 0.612 |

Note the copy number is partly circular -- Magikarp's labels come FROM a repetition
test, so a copy probe agreeing with them is self-confirmation. The genuinely
independent criterion actively disagrees.

Entropy also flips sign between settings: on our from-scratch rig, doomed tokens
had HIGHER entropy than healthy (6.58 vs 5.08); on Pythia's verified set they are
slightly LOWER. That is not noise -- it is two different populations.

## Two populations, opposite signatures

| group | n | mean entropy | % with zero token-specific update @step2 | @step64 |
|---|---|---|---|---|
| UNREACHABLE_MULTI_TOKEN | 198 | **6.159** | 99.5% | **99.5%** |
| Magikarp verified (26/36 are bare word-fragments) | 36 | **4.761** | 66.7% | **25.0%** |
| random healthy | 300-500 | 4.881 | 3.2% | 0.2% |

**STILLBORN.** Tokens the tokenizer can never emit. 99.5% receive zero
token-specific gradient at step 2 AND at step 64 -- they never appear, ever. With
no prior at all, entropy spikes (6.16 vs 4.88 healthy). Detected by
entropy-based methods; repetition may or may not fail.

**ABANDONED.** Bare word-fragments -- `'ecause'`, `'ccording'`, `'icrosoft'`,
`'ortunately'`. 66.7% start stillborn but only **25% remain so by step 64**: three
quarters of them DO begin receiving token-specific updates. They appear in the
data, get gradient, and are broken anyway. Because the model has a strong prior
from the containing word, entropy is normal (4.76 vs 4.88 healthy) -- it predicts
what follows `'ecause'` perfectly well. What it cannot do is REPRODUCE the
fragment; asked to copy `'ecause'` it emits `' because'`. Detected by repetition,
invisible to entropy.

## Why this matters

1. **It explains a disagreement nobody has flagged.** GlitchMiner maximises
   entropy to find glitch tokens; Magikarp and GlitchQuiz use repetition. Those
   are not two methods for one problem -- they are two methods for two different
   problems, and neither paper reports the other's population.

2. **It explains our own inconsistency.** Entropy was a fine endpoint on the
   from-scratch rig because manufactured zero-count tokens are stillborn by
   construction. It fails on Pythia's verified set because those are abandoned.
   Our rig's ground truth silently selected one population.

3. **It grounds the geometric taxonomy behaviourally.** The stillborn/abandoned
   split was previously a claim about gradient sparsity. It now has two distinct
   behavioural phenotypes with opposite entropy signatures, and each is what a
   different literature has been measuring.

4. **It reframes the single-criterion weakness.** The problem was never that our
   labels rest on one probe; it is that the field has no agreed definition and the
   criteria are not interchangeable. Any paper here should report both.

## Limits
n=36 for the verified set is small, and it is one model family. The
UNREACHABLE_MULTI group is definitionally stillborn, so that half is partly true
by construction -- the informative number is the abandoned group's drop from 66.7%
to 25.0% zero-update between step 2 and step 64, which is what shows they receive
real gradient and degrade anyway.

---

# Replication on OLMo-2 (442 verified labels vs Pythia's 36)

The taxonomy makes a quantitative prediction: entropy's discriminative power
should track how STILLBORN-HEAVY a verified set is, because entropy is blind to
the abandoned population. Two models, opposite mixes:

| | pythia-1.4b | OLMo-2-1B |
|---|---|---|
| verified labels | 36 | **442** |
| bare word-fragments (abandoned phenotype) | **72%** (26/36) | **29%** (127/442) |
| **entropy AUC vs healthy** | **0.453** | **0.646** |
| verified entropy | 4.761 | 6.086 |
| healthy entropy | 4.881 | 5.464 |
| zero token-specific update, verified | 66.7% @step2 | 52.2% @step300 |
| zero token-specific update, healthy | 3.2% | 2.8% |

Prediction confirmed in direction and magnitude: the fragment-heavy set (Pythia,
72%) defeats entropy entirely; the more stillborn-heavy set (OLMo-2, 29%) is
partially detectable by it. The stillborn signal is strong in both -- verified
tokens are ~19x more likely than healthy ones to have received zero
token-specific gradient.

## The abandoned population reflects the training corpus

Pythia's fragments are English prose: `'ecause'`, `'ccording'`, `'icrosoft'`,
`'ortunately'`, `'atever'`.

OLMo-2's are code identifiers: `'ERCHANTABILITY'` (licence boilerplate),
`'ictureBox'`, `'uspendLayout'`, `'adioButton'`, `'yclerView'`,
`'paredStatement'` -- Windows Forms and Java. OLMo-2's mix is code-heavier than
the Pile, and the abandoned tokens follow.

That is a further consistency check: abandoned tokens should be fragments of
strings the corpus contains in quantity but never in isolation, so their character
should change with the corpus. It does.

## Caveat
The OLMo-2 `UNREACHABLE_MULTI_TOKEN` group came back empty -- the label file is
from OLMo-2-1124-7B while the trajectory is OLMo-2-0425-1B, and category
assignments do not transfer across that pair even though the verified labels do
(cross-family AUCs were 0.96-0.98). So the OLMo-2 row is a two-group comparison,
not three.
