# Output-side test: are glitch tokens attractors? (pythia-1.4b)

Prompted by the ' 恒一' report (note.com/tank_ai). **Tokenizer claim verified
independently**: in o200k_base, `' 恒一'` is a single token (ID 198478) while
`'恒一'` is two; `' 太郎'`/`' 一郎'` are 2 tokens, `' 翔太'`/`' 大輔'`/`' 健二'` are 3;
and `' 恒一'` is the ONLY entry among all 200,019 vocabulary items containing that
name. The article's claim is correct. (The English auto-translation romanises the
kanji throughout, which makes it read as a Latin-script token -- it is not.)

That case is a SINGLE-token glitch (the author says so too), so it is not a
counterexample to the compositional null in findings_day1.md. But it exposed a
real gap: every existing detector, mine included, is **input-side** -- feed the
token in, see if the model copes. The reported 恒一 symptom is **output-side**: the
token is EMITTED where it does not belong, capturing a given-name slot.

## Design
- **SLOT** contexts (20): next token is an open-class choice -- a name, an
  identifier, a filename, a domain. Nothing grammatical constrains which one.
- **CLOSED** contexts (20): next token is grammatically/semantically forced.
- **BASE** contexts (20): broad natural text -> the model's own unigram estimate.
- attractor score `A(t) = log p_slot(t) - log p_base(t)`.
- Glitch set = worst 1% copy-logprob (behavioural). Compared against healthy
  tokens **matched on p_base** -- without that, rare tokens win by construction.

## Result: no attractor effect, and the sign is reversed

| | glitch | matched healthy | difference |
|---|---|---|---|
| over-production in OPEN SLOTS | +0.632 | +0.851 | **-0.219** [-0.290,-0.146] |
| over-production in CLOSED ctx | -1.383 | -1.582 | +0.199 [+0.114,+0.282] |

Glitch tokens are over-produced *less* than matched healthy tokens in open slots.

Top-50 intrusion, the direct analogue of the reported symptom:

| context | glitch share of top-50 | vocab share | enrichment |
|---|---|---|---|
| open slot | 1.0000% | 1.0019% | **1.00x** |
| closed | 0.0000% | 1.0019% | 0.00x |

Exactly their vocabulary share. No capture.

And the strongest slot attractors in the whole vocabulary are entirely
unremarkable: `' Julie'`, `' Betty'`, `' Patricia'`, `' Jason'`, `' Sally'` --
all copy cleanly. If a 恒一-style attractor existed in this model it would sit at
the top of that list. Nothing anomalous is there. The glitched tokens with the
highest A are word-continuation fragments (`'arently'`, `'orporated'`,
`'oubtedly'`) and replacement characters -- fragments, not slot captors.

## What DID show up: glitch tokens are contextually inert

Context sensitivity = how far a token's log-probability swings between an open
slot and a grammatically forced position:

- matched healthy: **2.433 nats** of swing
- behaviourally glitched: **2.015 nats**
- difference **-0.418**, 95% CI [-0.527, -0.302]

Glitch tokens respond ~17% less to context structure. That is coherent with an
under-trained representation -- a weakly-learned row is less modulated by what
surrounds it -- and it is the exact opposite of attractor behaviour. Being
under-trained makes a token contextually *flat*, not contextually explosive.

## Boundary of this result
Pythia-1.4b is a small base model with no instruction tuning. The 恒一 report
concerns a frontier post-trained system, and specifically its *auxiliary*
filename/title generator, which may be a separate distilled model. Attractor
dynamics created by post-training are not testable here. What this rules out is
that "behaviourally-glitched tokens are generally attractors" -- a property of
under-training itself. It does not rule out that a *particular* token became an
attractor through a *particular* training adjustment, which is what the article
actually claims.
