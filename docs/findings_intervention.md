# Can we fix glitch tokens during training? No.

6 independent seeds (both model init and data order differ). Within a seed every
arm branches from that seed's own step-256 checkpoint, so arms are paired --
identical weights and identical data order up to the moment of intervention.
Doomed set = true corpus count < 100, known exactly because we own the corpus.

## Result

| arm | doomed entropy | healthy entropy | val loss |
|---|---|---|---|
| baseline | 6.627 (ref) | 5.090 (ref) | 3.6369 (ref) |
| reinit @256 | 6.810 **+0.183** | 5.140 +0.050 | +0.0008 |
| freeze @256 | 7.116 **+0.489** | 5.157 +0.067 | **+0.0207** |
| mask @256 | 6.863 **+0.235** | 5.952 **+0.863** | +0.0008 |
| posthoc centroid | 5.978 **-0.649** | 5.160 +0.071 | +0.0077 |
| **posthoc kNN** | 5.383 **-1.245** | 5.091 +0.001 | +0.0012 |

Paired 95% CIs on the per-seed difference, all excluding zero:

    reinit           +0.183 [+0.043, +0.309]
    freeze           +0.489 [+0.348, +0.620]
    mask             +0.235 [+0.091, +0.374]
    posthoc_centroid -0.649 [-1.166, -0.158]
    posthoc_knn      -1.245 [-1.381, -1.111]

**Every training-time intervention made glitch tokens significantly WORSE.**
Both post-hoc repairs helped. kNN repair is the clear winner: -1.245 nats of
damage removed, healthy tokens untouched (+0.001), val loss +0.0012.

## My causal explanation was wrong

Early repair failed on the local rig, and I proposed that shared drift -- measured
at 13.8x stronger for doomed rows -- re-corrupts the repaired embeddings over the
remaining steps. That predicted a specific outcome: freeze the repaired rows and
early repair should stick.

The freeze arm is the **worst** of all six, at +0.489, and it is the only arm with
a real val-loss cost (+0.0207, ~25x every other arm). Eliminating drift entirely
made the problem worse. The drift-corruption account is falsified.

## What actually explains all five arms

Where you put the row matters more than when, and the global centroid is the worst
possible place:

- `reinit` and `freeze` park doomed rows at the **healthy centroid**. That is the
  point of maximum confusability -- a row sitting on the mean is close to
  everything and distinguishable from nothing. `freeze` is worse than `reinit`
  because it also forbids escape.
- `posthoc_knn` places each row near **specific healthy neighbours**, which is
  distinctive. It wins by a wide margin.
- `posthoc_centroid` sits between them exactly as this predicts: it repairs
  (-0.649) but less than kNN, and it is the only repair that measurably degrades
  healthy tokens (+0.071) and val loss (+0.0077).

This also inverts the reading of the drift finding. Shared drift is not corrupting
doomed rows -- it may be partially rescuing them, by carrying them away from the
centroid into a distinguishable region. Freezing them on the mean removes that.

The reason late beats early is then straightforward: at step 14,000 you know where
the semantic neighbours are. At step 256 you do not, so the only available target
is the centroid, which is the one place you should not put them.

## `mask` degrades healthy tokens

Removing never-occurring tokens from the softmax raised healthy-token entropy by
+0.863 -- far more than it changed doomed tokens. Reported separately because the
doomed-minus-healthy "separation" metric would have scored this as an improvement.
That metric is gameable and should not be used alone.

## Practical conclusion
Do not try to repair glitch tokens during pretraining. Detect them early to decide
whether to **abort and re-tokenise**; repair them **after** training with
neighbour-based interpolation, which costs +0.0012 val loss and removes 1.245 nats
of damage.
