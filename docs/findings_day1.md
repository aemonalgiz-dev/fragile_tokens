# Day 1: the seam hypothesis is falsified (pythia-1.4b, copy probe)

## Hypothesis tested
Compositional under-training: n-grams of individually-healthy tokens that the
model has *welded* into near-deterministic units should glitch when the weld is
violated -- damage from absence of VARIATION, not absence of data.

## What we built
`src/cut/` -- G1 entropy-collapse detector (context-invariant commitment sweep
over the full 50,304-token vocabulary across 10 diverse contexts), greedy chain
extraction, matched-improbability substitution, and a token-id-level copy probe
(teacher-forced logprob + greedy exact match). Nothing is ever decoded and
re-encoded, so retokenization can never confound the result.

## Result 1 -- welds are real, and G1 finds them cheaply
Top welds on pythia-1.4b: ` doesn`->`'t`, ` prerequ`->`isites` (p=.994),
` pione`->`ers` (p=.997), ` javax`->`.`, ` Bulgarian: `->`\n` (p=.994),
and a memorised citation template ` [****, ()](\`->`doibase` at **p=1.000**.
Median p(top1) at the seam = 0.807. Every constituent token is high-frequency
and passes single-token health screening. The phenomenon is exactly as
hypothesised, and O(|V|) to find -- no |V|^n search.

Welds also sharpen with scale: 0 tokens at H_max<1 in pythia-160m, 9 in
pythia-1.4b. (Two models only -- a lead, not a result.)

## Result 2 -- violating a weld does essentially nothing
Matched-improbability 2x2 (welded vs diffuse chain x intact vs substituted,
holding the inserted token's conditional probability constant at eps):

| eps   | welded drop | diffuse drop | interaction        |
|-------|-------------|--------------|--------------------|
| 1e-2  | 0.031       | -0.008       | +0.039 [-.01,+.09] |
| 1e-3  | 0.078       | -0.006       | +0.084 [+.01,+.17] |
| 1e-4  | 0.080       |  0.023       | +0.057 [-.02,+.13] |
| 1e-5  | 0.103       |  0.017       | +0.085 [+.02,+.17] |

Nominally significant at two bands -- but it does not survive the decisive test.

## Result 3 -- no dose-response on commitment (this is what kills it)
n=1200 chains, commitment spanning p(top1) = 0.036 to 1.000, eps=1e-5:

| quartile | p(top1)       | copy-logprob drop      | exact-match drop |
|----------|---------------|------------------------|------------------|
| Q1       | 0.036 - 0.275 | +0.009 [-.005,+.024]   | +2.0 pp          |
| Q2       | 0.275 - 0.599 | +0.006 [-.036,+.056]   | +1.0 pp          |
| Q3       | 0.599 - 0.870 | -0.005 [-.042,+.031]   | +4.0 pp          |
| Q4       | 0.870 - 1.000 | +0.005 [-.038,+.049]   | +1.7 pp          |

**corr(commitment, drop) = -0.003.  Q4-Q1 = -0.005, CI [-0.051,+0.043].**

Maximally violating a p=1.000 prediction is no worse than violating a p=0.036 one.

## Result 4 -- the instrument is not the problem (positive control)
Same copy probe, classic single-token glitch candidates (highest
cosine-to-mean unembedding, the untrained-row signature) vs healthy matched:

| | copy-logprob | exact-match |
|---|---|---|
| healthy          | -0.223 | 90.0% |
| glitch candidate | -3.399 |  0.0% |
| difference       | **+3.176** [+3.14,+3.21] | **-90 pp** |

The probe detects single-token glitches at 3.18 nats. Seam violations register
0.005 nats. **~600x ratio.** The null is a property of the phenomenon.

## What this means
**Commitment and fragility are decoupled.** The model's confidence structure has
no measurable relationship to its robustness structure. This extends "Broken
Tokens" (2506.19004) from "models tolerate non-canonical tokenizations" to the
much stronger "models tolerate maximal violation of their own most confident
predictions" -- and it is a real finding in its own right, cleanly measured.

The weld prior was wrong: welds are where the model is CONFIDENT, and confidence
turns out to be cheap to contradict. Whatever makes a single glitch token break
the model is not a property that composes along high-PMI seams.

## Live / dead
- DEAD: seam violation as a glitch generator (this model, this probe).
- STILL OPEN: G3 activation-manifold inversion -- searches for off-manifold
  states *directly* and assumes nothing about welds. Now the better route,
  precisely because the weld prior failed.
- CAVEAT: copy is a mechanical induction task. A probe requiring semantic USE of
  the sequence could differ. Worth one test before closing Idea 2 -- but note the
  probe had 3.18 nats of headroom and used none of it.
- UNTOUCHED: Idea 1 (training dynamics). Now the stronger bet.
