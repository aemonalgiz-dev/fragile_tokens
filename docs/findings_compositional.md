# Compositional and reasoning-mode glitch behaviour

Motivated by a field report of two glitch tokens in `o200k` (ids 128188, 152383)
affecting GPT-5.3 reasoning models. Those weights are unavailable, so every
model claim below is measured on open models; the tokenizer claim is measured on
the actual `o200k_base` vocabulary, which is public.

Model unless stated: `allenai/OLMo-2-1124-7B-Instruct`, labels from Magikarp's
published verification for OLMo-2-7B.

---

## 1. The "you can't put a space before it" rule is the tokenizer, not the model

Both reported ids resolve in `o200k_base`:

| id | decodes to | codepoint |
|---|---|---|
| 128188 | `\U0004e50app` | U+4E50A (unassigned) + `app` |
| 152383 | `\U0007968app` | U+7968A (unassigned) + `app` |

Prefix survival, i.e. does the id still arrive as the final token:

| prefix | class | 128188 | 152383 |
|---|---|---|---|
| none / newline / tab | pre-tokenizer boundary | INTACT | INTACT |
| space, `"`, `'`, `(`, `[`, `{`, `.`, `,` | merging | **DROPPED** | **DROPPED** |
| doubled | boundary | INTACT | INTACT |

That reproduces the reported usage rule exactly, with no model involved. In
GPT-family BPE the pre-tokenizer splits on newline and tab, so the id survives;
space/quote/bracket merge rightward and the string re-encodes to something else.

**But it is not a glitch property.** In Pythia, a leading space re-segments
verified glitch tokens and healthy tokens at the same rate:

| prefix | glitch | healthy |
|---|---|---|
| space | 0.083 | 0.093 |
| newline | 1.000 | 0.992 |
| dquote | 1.000 | 0.943 |

So the usage rule is explained, and it is a red herring for glitchiness: ~90% of
all Pythia tokens behave that way.

**Consequence for prompt assembly:** a string that renders correctly can encode
to ids the model never sees adjacent, and this is invisible at the text layer.

---

## 2. The failure mode is confident misidentification, not degradation

`reasoning_drift.py`, 32 verified glitch vs 32 healthy tokens, same model, same
prompts, two regimes.

```
reproduced the seed token:   glitch 0.000    healthy 0.531
```

```
'.XtraLayout'     -> 'The token "layui" refers to a lightweight, modular JavaScript framework...'
' Hexatrigesimal' -> 'The token "Calculus" refers to a branch of mathematics...'
' woman'          -> 'The token "woman" is a noun in the English language...'   (correct)
```

The model names a *different* token and reasons fluently about that one. Entropy
while doing so is 0.85-1.30 — it is not uncertain. Task derailment was **1.000
in both arms**: it never stops working, it works on substituted content.

This is why entropy-based detectors (GlitchMiner) miss it, and it is the actual
agentic hazard: no uncertainty signal for a guardrail to catch.

### Falsified along the way

- **Entropy amplification with generation length.** Gap is +0.273 at step 8 and
  −0.288 at step 255. Wobbles around zero, never grows.
- **A shared attractor basin.** Chain alignment decays 0.0158 → 0.0001,
  identical to healthy (0.0173 → 0.0002). Different glitch seeds do not converge.
- **Representational collapse.** `forward_geom.py`: identity retrieval over a
  1124-candidate pool is 1.000 for glitch tokens at every layer through L24,
  equal to healthy. The information is present at the top of the network. The
  *readout* substitutes. This is confabulation, not information loss.

---

## 3. BPE-unreachable adjacency does not predict damage

If `encode(decode(a)+decode(b)) != [a,b]`, that adjacency essentially never
occurs in training but remains feedable as ids — the compositional analogue of
an unreachable token, computable from the tokenizer alone. 9.3% of random
healthy pairs have exactly one unreachable direction, giving a within-pair
control: identical tokens, identical individual health, only adjacency differs.

| condition | copy lp/tok | derail-ok | entropy |
|---|---|---|---|
| UNREACHABLE (a,b) | −3.783 | 1.000 | 1.105 |
| REACHABLE (b,a) | −4.724 | 1.000 | 1.118 |

Paired difference **+0.94** [+0.61, +1.26] — unreachable pairs are *easier*, the
opposite of the prediction; only 33.5% are worse. **Hypothesis falsified.**

---

## 4. Compositional failure is real, driven by rarity × length, not by structure

`ngram_identify.py`. Sequences built only from tokens the model reproduces
perfectly alone (n=1 accuracy is 1.000 by construction), so any failure at n>1
is not inherited from a member.

| n | rare tail | ordinary English |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 2 | 0.864 | 0.976 |
| 3 | 0.780 | 0.948 |
| 4 | 0.728 | 0.916 |
| 6 | 0.704 | 0.908 |
| 8 | **0.648** | **0.892** |

**35% of 8-token sequences of individually-perfect rare tokens are
misreproduced**; 11% for ordinary English. Failures look like the field report:

```
want ' haven numberWithInt see really another startures CrossAxisA'
got  ' havenventario see realmente otro rasgos Cruzado'      <- drifts into Spanish

want ' week aDecoder lesbisk His ReferentialAction defaultCenter'
got  ' week aDecoder lesbisk His 示 示 示 示 示 示 示 示'        <- CJK degeneration

want ' How strSql questions guy rootReducer RuntimeMethod'
got  ' How StraitSail questions Guy RootNodeMethod'           <- confident substitution
```

### Two qualifications that matter

**Not superadditive.** Per-adjacency failure estimated from n=2 is p=0.136;
independence predicts 0.864⁷ = 0.354 at n=8, observed is 0.648. The sequence is
*more* robust than independent compounding. There is no evidence of a special
compositional interaction — just more tokens, more chances.

**Not predictable from static features.** AUC for predicting which sequences
fail, within each n:

| predictor | AUC |
|---|---|
| unreachable adjacencies | 0.47–0.51 |
| whole sequence re-segments | 0.47–0.51 |
| character length | 0.42–0.75 (unstable) |

Tokenizer-only prediction is at chance. What does predict failure is token
rarity and sequence length — both known in advance, but not a structural
property of the grouping.

---

## 5. RETRACTED: "post-training creates new glitch tokens"

An earlier version of this section claimed that under-training is stage-relative
and that post-training abandons CRLF and tab-indented code tokens. **A
cross-family replication falsified it.** See `findings_stagewise.md` for the
full sweep; the summary is:

The claim rested on the worst-1% copy-probe set turning over 74% between base
and Instruct in OLMo-2-1B. The control that decides whether turnover means
anything is where the newly-glitched tokens sat in the BASE model's own ranking,
relative to the 1% cut. Tokens sitting just outside the cut crossing it is
noise; tokens from deep in the healthy region crossing it is the claim.

| family | lineage | median base rank / threshold | beyond 5× | verdict |
|---|---|---|---|---|
| olmo2_1b | OLMo-2 | 6.01× | 0.548 | real |
| olmo2_7b | OLMo-2 | 4.75× | 0.479 | real |
| amber | LLaMA-1 | 2.05× | 0.154 | mostly jitter |
| tulu3_8b | Llama-3.1 | 1.41× | 0.010 | jitter |
| olmoe | OLMo/MoE | 1.31× | 0.000 | jitter |
| qwen3_8b | Qwen3 | 1.30× | 0.038 | jitter |
| qwen25_7b | Qwen2.5 | 1.17× | 0.000 | jitter |

Five independent lineages say the turnover is boundary jitter. The CRLF/tab
interpretation does not even hold within OLMo-2: the 7B's movers are `'\'`,
replacement characters, `.t`, `.S`, `.l`.

**Why it was wrong, mechanically.** Post-training embedding deltas are almost
purely token-specific — total and residual ratios are identical to three decimal
places in every family (amber 0.550/0.550, olmoe 0.277/0.277, qwen25_7b
0.030/0.030). The pretraining mechanism is *shared drift dragging a row that
receives no token-specific correction*. Post-training has essentially no shared
drift, so a row it never touches is **preserved, not damaged**. Zero update ≠
damage; the original claim conflated them.

**What survives.** Coverage is real and large — 23% of rows receive zero
token-specific update across OLMo-2-7B's DPO→RLVR, 35.5% across Tulu-3's — it
simply has no behavioural consequence. And the tied control corroborates earlier
work: Qwen2.5-1.5B (tied) moves glitch rows 4.85×/2.83× versus below 1.0 for
every untied family, exactly as dense softmax-negative gradient predicts.

---

## 6. What this means for the original question

> is it predictable that certain groupings of tokens will result in glitch-like
> behaviour?

**Yes, but not from the structure of the grouping.** Every structural predictor
tested — BPE unreachability, re-segmentation, adjacency statistics — came in at
chance. The predictive quantities are token rarity and sequence length, and the
relationship is sub-multiplicative rather than interactive.

The finding with real teeth is §2 + §4 together: the failure is **silent
substitution at low entropy with no task derailment**. For agentic systems the
exposure is contexts dense in rare tokens — identifiers, paths, hashes, base64,
foreign script, tool output — where a 7B instruct model misreproduces roughly a
third of 8-token spans while continuing to complete the task normally.
