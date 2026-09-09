# Stage-relative abandonment: a falsified hypothesis, and what replaced it

## The hypothesis

The mechanism established elsewhere in this project is gradient sparsity: an
input-embedding row only receives a token-specific update on steps where its
token is in the batch. Post-training sees orders of magnitude fewer tokens than
pretraining, and instruction/preference data is far narrower than a web crawl.
So: a token could be adequately trained with respect to pretraining and
abandoned with respect to SFT/DPO/RLVR. "Glitch token" would be a property of a
model *and a training stage*, and auditing a base checkpoint would characterise
the wrong set for the deployed model.

Nobody in the glitch-token literature has measured glitch sets across stages,
and the prediction falls out of the mechanism rather than from fishing. It
looked like the strongest thing in the project.

## The sweep

Eight stage series, five independent lineages, all with matched vocabularies.
`src/cut/run_stages.sh`, sequential under an atomic lock.

| family | lineage | stages | vocab | tied |
|---|---|---|---|---|
| OLMo-2 1B / 7B | OLMo-2 | 4 (base→SFT→DPO→RLVR) | 100352 | no |
| OLMoE-1B-7B | OLMo/MoE | 3 | 50304 | no |
| Tulu-3-8B | Llama-3.1 | 3 (SFT→DPO→RLVR) | 128264 | no |
| Amber | LLaMA-1 | 3 (base→Chat→Safe) | 32000 | no |
| Qwen2.5-7B, Qwen3-8B | Qwen | 2 | 152k | no |
| Qwen2.5-1.5B | Qwen | 2 | 151936 | **yes** |

OLMo-2-13B failed on an HTTP 429 (unauthenticated rate limit); retriable, and it
would test scale within a lineage that already replicates, not independence.

Large vocabularies were subsampled to 40k rows for the behavioural probe;
coverage is always full-vocab.

## The result: it does not replicate

Turnover of the worst-1% copy-probe set, base → final post-trained stage, using
a rank-based threshold **within each stage** so the global formatting shift that
instruction tuning induces cancels:

| family | Jaccard vs base | newly glitched |
|---|---|---|
| olmo2_1b | 0.263 | 586 |
| olmo2_7b | 0.246 | 242 |
| qwen3_8b | 0.368 | 185 |
| qwen25_7b | 0.513 | 21 |
| tulu3_8b | 0.600 | 100 |
| amber | 0.608 | 78 |
| qwen25_1_5b (tied) | 0.651 | 100 |
| olmoe | 0.697 | 90 |

Turnover looks substantial everywhere. **It is almost entirely an artifact of
tokens jittering across the 1% cut.** The control: where did the newly-glitched
tokens sit in the base model's own ranking? The threshold is at rank ≈ probed/100.

| family | threshold rank | median base rank | ratio | fraction beyond 5× |
|---|---|---|---|---|
| olmo2_1b | 1004 | 6032 | **6.01×** | **0.548** |
| olmo2_7b | 400 | 1901 | **4.75×** | **0.479** |
| amber | 320 | 656 | 2.05× | 0.154 |
| tulu3_8b | 400 | 562 | 1.41× | 0.010 |
| olmoe | 504 | 660 | 1.31× | 0.000 |
| qwen3_8b | 400 | 520 | 1.30× | 0.038 |
| qwen25_7b | 772 | 900 | 1.17× | 0.000 |
| qwen25_1_5b (tied) | 400 | 487 | 1.22× | 0.000 |

Only OLMo-2 shows real movement, and 1B and 7B are the same lineage — not
independent replications. Five independent lineages fail the control.

The qualitative story fails even within OLMo-2. The 1B's newly-glitched tokens
are `'{\r\n'`, `'.Add'`, `'\tString'`, `'\tswitch'`, `'\tvirtual'`, `'*/\r\n'` —
which invited the reading "instruction tuning abandons CRLF and tab-indented
code". The 7B's are `'\'`, replacement characters, `.t`, `.S`, `.l`, `.L`. Same
lineage, different phenomenon. The interpretation was over-fitted to one model.

## Why it failed, mechanically

The reason is visible in the decomposition, and it should have been predicted
before the sweep. Glitch-row versus healthy-row update ratios per transition:

| family | transition | total | residual |
|---|---|---|---|
| amber | Amber→Chat | 0.550 | 0.550 |
| olmoe | SFT→Instruct | 0.209 | 0.209 |
| olmo2_7b | DPO→Instruct | 0.359 | 0.359 |
| tulu3_8b | DPO→RLVR | 0.385 | 0.385 |
| qwen25_7b | 7B→Instruct | 0.030 | 0.030 |
| qwen3_8b | Base→8B | 0.102 | 0.101 |

Total and residual are identical to three decimals everywhere: **post-training
embedding deltas are almost purely token-specific, with essentially no shared
drift.**

The pretraining mechanism is shared drift dragging a row while it receives no
token-specific correction — the row moves without being fixed. Post-training has
no shared drift to do the dragging. A row post-training never touches is
therefore *preserved at whatever the base model made it*, not damaged.

Zero token-specific update ≠ damage. The hypothesis conflated the two, and the
conflation is what made it wrong.

## What survives

**1. Coverage is real, large, and cheap to measure.** Rows receiving zero
token-specific update during a stage:

| family | transition | fraction |
|---|---|---|
| tulu3_8b | DPO → RLVR | **0.355** |
| olmo2_7b | DPO → Instruct | **0.231** |
| qwen25_7b | 7B → Instruct | 0.051 |
| olmoe | SFT → Instruct | 0.030 |
| olmo2_1b | SFT → DPO | 0.027 |

A third of the vocabulary can go completely untouched through an RLVR stage.
This is a novel measurement and costs two embedding matrices. It just does not
predict behavioural damage, per the above.

**2. A clean mechanistic contrast**, supported across eight families:
pretraining deltas are shared-drift-dominated, post-training deltas are
token-specific. This is why the pretraining glitch mechanism has no
post-training analogue, and it is a real finding in its own right — a negative
one with a reason attached.

**3. Corroboration of the tying result.** Qwen2.5-1.5B is the only tied series
in the sweep, and it is the only one where glitch rows move *more* than healthy
rows: total 4.852, residual 2.834, against below 1.0 for every untied family.
A tied row is a softmax negative on every step and receives dense gradient it
would not otherwise get. This independently reproduces the weight-tying scope
test run earlier on the HuggingFaceFW corpus ablations.

## Status

Hypothesis rejected. This is the fourth claim in this project falsified by a
control run after the claim was asserted — after hubness, the
stillborn/abandoned taxonomy, and BPE-unreachable adjacency. The controls are
doing their job; the honest read is that the project's positive results are
narrow, and the negative results with mechanisms attached are the more solid
contribution.
