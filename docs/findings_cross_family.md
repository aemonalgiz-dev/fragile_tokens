# Cross-family verification: the mechanism holds; the geometry does not

Six models, three corpora (Pile / Dolma v1.5 / OLMo-2 mix), two tokenizer
families, 410M-7B, dense and MoE. Labels are Magikarp's published behavioural
verifications for each family -- and for pythia-6.9b they are labels for THAT
EXACT MODEL, removing the transfer assumption used everywhere else.

## Result 1 -- residual suppression replicates 6/6

Glitch/healthy ratio of the token-specific residual, at the point of deepest
suppression:

| model | corpus | verified | residual ratio | shared ratio |
|---|---|---|---|---|
| pythia-410m | Pile | 36 | **0.489** @256 | 1.70 |
| pythia-1.4b | Pile | 36 | **0.500** @256 | 2.05 |
| pythia-6.9b | Pile* | 36 | **0.736** @256 | 3.08 |
| OLMo-1B | Dolma | 438 | **0.146** @2000 | 2.21 |
| OLMo-2-1B | OLMo2 | 442 | **0.408** @30k | 1.47 |
| OLMo-2-7B | OLMo2 | 442 | **0.417** @5000 | 0.98 |
| OLMoE-1B-7B | OLMo/MoE | 98 | **0.363** @10k | 0.88 |

Every model. Different corpora, different tokenizers, dense and sparse. The
shared-drift ratio exceeds 1 early in every model that has early checkpoints.

## Result 2 -- the masking effect holds wherever it can be measured

The claim that makes the mechanism non-obvious: TOTAL update magnitude shows
parity or excess for doomed rows while the residual collapses, so anyone
measuring "was this row updated?" concludes the opposite of the truth.

| model | earliest ckpt | total ratio | residual ratio | masking? |
|---|---|---|---|---|
| pythia-410m | 2 | 1.04 - 1.30 | -> 0.489 | **yes** |
| pythia-1.4b | 2 | 1.04 - 1.30 | -> 0.500 | **yes** |
| pythia-6.9b | 2 | 1.04 - 1.14 | -> 0.736 | **yes** |
| OLMo-2-1B | 300 | **1.363** | 0.544 | **yes** |
| OLMo-2-7B | 1000 | 0.965 | 0.419 | **yes** (marginal) |
| OLMo-1B | 2000 | 0.218 | 0.146 | no |
| OLMoE | 10000 | 0.383 | 0.363 | no |

**5 of the 5 models with a checkpoint at or before step 1000 show masking; the 2
that lack early checkpoints do not.** That is the honest statement -- masking is
an EARLY phenomenon, and Pythia itself loses it by step 1000 (total ratio 0.929).
The two apparent failures are the two models that cannot be tested in the window
where the effect lives, not counterexamples.

## Result 3 -- early detection replicates on the exactly-labelled model

pythia-6.9b, against labels Magikarp produced for this very model:

| step | 0 | 1 | **2** | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| `resid_i` (ours) | .500 | .500 | **.896** | **.921** | .913 | .911 |
| `unemb_cos` (standard) | .451 | .451 | .451 | .451 | .451 | .453 |

Chance until an update exists, then 0.92 by step 4, while the standard geometric
indicator sits at 0.451 -- below chance -- because embedding geometry has not
differentiated yet. This is the one place our signal is not merely different but
strictly earlier than anything available.

OLMo-2-1B at its first post-init checkpoint (step 300): resid_i 0.959 vs
unemb_cos 0.846. Same ordering, larger sample (442 labels).

## Result 4 -- the hubness idea does NOT generalise

Hypothesis: glitch tokens become hubs (in many other tokens' k-NN lists), which
would geometrically explain why centroid-based repair backfires.

| model | hub(glitch) | hub(healthy) | ratio | AUC |
|---|---|---|---|---|
| pythia-410m | 11.3 | 10.0 | 1.13 | 0.388 |
| pythia-6.9b | 8.6 | 10.0 | 0.86 | 0.260 |
| OLMo-1B | 8.7 | 10.0 | 0.87 | 0.404 |
| OLMo-2-1B | 1112.8 | 5.1 | **217x** | 0.623 |
| OLMo-2-7B | 1456.9 | 3.6 | **405x** | 0.557 |
| OLMoE | 1328.3 | 7.4 | **179x** | 0.845 |

Wildly family-dependent, and in Pythia it runs BACKWARDS (AUC 0.26-0.40 -- glitch
tokens are mild anti-hubs). The extreme OLMo numbers are a degenerate cluster, not
a subtle geometry: in OLMo-2-1B, **235 tokens (0.23% of vocab) hold 68% of all
k-occurrence mass**, and 114 of the 442 glitch tokens sit in that clique. In
pythia-6.9b the comparable set holds 10% of mass and contains 1 of 36 glitch
tokens.

So this is a real difference in how model families treat dead vocabulary slots --
OLMo-2/OLMoE let untrained rows collapse into a mutual-neighbour clique, Pythia
does not -- but it is not a mechanism, and hubness is not a portable indicator.
Reported as a negative result.

## Bottom line
The central claim survives the strongest test available: residual suppression in
6/6 models across three corpora and two architectures, masking in 5/5 where it is
measurable, and early detection reproduced on a model whose labels were produced
independently by someone else. The geometric extension does not survive.
