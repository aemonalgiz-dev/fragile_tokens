# Repairing a real model: AUC 0.987 is not a deployable detector

The synthetic rig said neighbour-based repair removes 1.245 nats at +0.0012 val
loss across 6 seeds. This is the same pipeline on pythia-1.4b, with glitch tokens
verified behaviourally on two models and perplexity on real held-out text.

## The headline warning

The late detector scores **AUC 0.987** against Magikarp's verified labels. Its
**precision at top-500 is 3.0%.**

That is not a contradiction -- it is what a 0.073% base rate does (36 positives in
49,572 tokens). AUC is a ranking statistic over all pairs; at any threshold you
would actually deploy, almost everything flagged is a false positive. Every paper
in this area, mine included until now, reports AUC/F1 and not precision at a usable
operating point.

## What that costs in practice

Repairing the detector's raw top-N destroys the model:

| repair set | verified glitch | our confirmed | control | perplexity |
|---|---|---|---|---|
| top-500 | 5% -> 5% | 20% -> 10% | 70% -> 60% | 15.92 -> **87.34 (+449%)** |
| top-2000 | 5% -> 10% | 20% -> 15% | 70% -> **50%** | 15.92 -> **183.42 (+1053%)** |
| top-2000 **+ behavioural guard** | 5% -> **15%** | 20% -> **25%** | 70% -> 70% | 15.92 -> **15.92 (+0.057%)** |

The reason is visible the moment you look at the flagged set. The top-500 contains:

    ' and', ' of', ' to', ' is', ' in', ' for', ' on', ' with', ' as', ' by',
    ' from', ',', '.', '-', ' ('

The most frequent tokens in the language. The features encode "unusual geometry",
and the hyper-frequent head is geometrically unusual too -- just in the opposite
direction from the starved tail. With 36 positives a linear model cannot separate
the two extremes, so it flags both. Overwriting the embedding of `' and'` with a
neighbour average is why perplexity went up 449%.

## The fix, and its honest size

Two-stage filter: repair only tokens that are BOTH detector-flagged AND
behaviourally impaired (copy-probe in the worst 5%). 2,000 flagged -> 717 repaired.

That is safe: controls unchanged, perplexity +0.057%. But the benefit is modest --
verified glitch tokens go 5% -> 15% repetition success, ours 20% -> 25%. Neither
approaches the 70% control level. **Repair helps; it does not restore.**

This is much weaker than the synthetic rig implied. On the rig the doomed set was
defined by TRUE corpus counts, so it contained no frequent tokens by construction
and repair was nearly free. On a real model you do not have counts, the detector
substitutes for them, and its errors are concentrated exactly where they hurt most.

## What to take from this
1. **Never deploy a glitch detector on AUC alone.** Report precision at the
   operating point. 0.987 AUC bought 3% precision here.
2. A behavioural screen is not optional -- it is the thing that makes repair safe.
3. Repair is a mitigation, not a cure: +10pp on a 65pp gap.
4. The synthetic-rig result overstated the case, because manufactured ground truth
   removed the exact failure mode that dominates in practice.
