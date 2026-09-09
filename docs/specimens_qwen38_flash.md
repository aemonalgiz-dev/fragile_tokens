# Confident-substitution specimens

Every example in this file was produced by greedy decoding; `p` is the model's probability for the token it emitted, `H` the entropy in bits at that position, `p(target)` its probability for the correct token, `ret` the cosine between the last-layer slot state and the correct token's unembedding row (identity still pointing at its readout while the readout chose otherwise).

## Model

- **model**: `Qwen/Qwen3.8-Flash-Next-FP8`
- **commit**: `None`
- **dtype**: `torch.bfloat16`
- **num_layers**: `48`
- **hidden_size**: `2560`
- **vocab_size**: `248320`
- **tie_word_embeddings**: `False`
- **tokenizer_class**: `Qwen2Tokenizer`
- **architecture**: `?`
- **torch**: `2.14.0+cu130`
- **transformers**: `5.16.1`
- **python**: `3.10.12`
- **device**: `cuda:1`
- **decoding**: `greedy`
- **prompt_head**: `Repeat the text exactly.\nText: apple pie is good\nCopy: apple pie is good\nText: the quick brown fox\nCopy: the quick brown fox\nText:<context>\nCopy:<context-prefix>`
- **quantized**: `True`
- **fail_threshold_lp**: `-0.5`
- **n_contexts**: `24`
- **clean_gate_lp**: `-0.1`

## Fragile clean-looking tokens (pass the single probe; fail in some contexts)


### `' ilmaisia'` — id 219454

single-probe lp -0.082 · fragility 0.96 · mean lp -1.18 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' separ continue ex rel' [·] ' filename'` | 8 | `' gratuita'` (0.026) | **substitution** | 11.95 | 0.010 | -0.014 | `''` |
| worst | `' ret nature column between' [·] ' drug high plan'` | 64 | `' ilmaisia'` (0.136) | **correct** | 10.21 | 0.136 | -0.030 | `''` |
| worst | `' ver integ align android' [·] ' investig performance above'` | 64 | `' ilmaisia'` (0.264) | **correct** | 9.05 | 0.264 | -0.023 | `''` |
| best | `' on customers ball them' [·] ' fight js ear'` | 16 | `' ilmaisia'` (0.756) | **correct** | 3.50 | 0.756 | -0.025 | `''` |

### `' επίσης'` — id 194461

single-probe lp -0.072 · fragility 0.38 · mean lp -0.46 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' last down figure' [·] ' is distance los'` | 8 | `' также'` (0.743) | **substitution** | 2.74 | 0.094 | 0.017 | `''` |
| worst | `' conditions introdu want feature' [·] ' pan count energy'` | 32 | `' также'` (0.771) | **substitution** | 2.47 | 0.104 | 0.007 | `''` |
| worst | `' ver integ align android' [·] ' investig performance above'` | 64 | `' также'` (0.651) | **substitution** | 3.17 | 0.187 | -0.006 | `''` |
| best | `' ret nature column between' [·] ' drug high plan'` | 64 | `' επίσης'` (0.944) | **correct** | 0.73 | 0.944 | 0.017 | `''` |

### `' kokemuks'` — id 230321

single-probe lp -0.075 · fragility 0.33 · mean lp -0.40 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' last down figure' [·] ' is distance los'` | 8 | `' kokemuks'` (0.368) | **correct** | 6.56 | 0.368 | -0.013 | `''` |
| worst | `' separ continue ex rel' [·] ' filename'` | 8 | `' kokemuks'` (0.189) | **correct** | 6.01 | 0.189 | 0.013 | `''` |
| worst | `' ret nature column between' [·] ' drug high plan'` | 64 | `' kokemuks'` (0.538) | **correct** | 5.19 | 0.538 | -0.012 | `''` |
| best | `' met remove such trade' [·] ' or short send'` | 64 | `' kokemuks'` (0.979) | **correct** | 0.29 | 0.979 | -0.025 | `''` |

### `' унич'` — id 184098

single-probe lp -0.004 · fragility 0.33 · mean lp -0.62 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' ing' [·] ' contain job handle'` | 8 | `' унич'` (0.067) | **correct** | 11.13 | 0.067 | 0.002 | `''` |
| worst | `' taken' [·] ' longer visit own'` | 16 | `' unn'` (0.142) | **substitution** | 8.61 | 0.125 | -0.008 | `''` |
| worst | `' asked independ story pa' [·] ' fre further die'` | 32 | `' унич'` (0.079) | **correct** | 10.61 | 0.079 | -0.003 | `''` |
| best | `' met remove such trade' [·] ' or short send'` | 64 | `' унич'` (0.992) | **correct** | 0.15 | 0.992 | 0.013 | `''` |

### `' prostituerade'` — id 41308

single-probe lp -0.006 · fragility 0.21 · mean lp -0.30 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' met remove such trade' [·] ' or short send'` | 64 | `' prostitutes'` (0.603) | **substitution** | 2.10 | 0.196 | 0.026 | `''` |
| worst | `' ret nature column between' [·] ' drug high plan'` | 64 | `' prostitutes'` (0.543) | **substitution** | 2.94 | 0.200 | -0.014 | `''` |
| worst | `' ver integ align android' [·] ' investig performance above'` | 64 | `' prostituerade'` (0.871) | **correct** | 1.33 | 0.871 | -0.006 | `''` |
| best | `' separ continue ex rel' [·] ' filename'` | 8 | `' prostituerade'` (0.996) | **correct** | 0.06 | 0.996 | 0.006 | `''` |

### `' требованиями'` — id 231327

single-probe lp -0.024 · fragility 0.21 · mean lp -0.39 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' ing' [·] ' contain job handle'` | 8 | `'_requirements'` (0.293) | **substitution** | 5.39 | 0.037 | 0.025 | `''` |
| worst | `' conditions introdu want feature' [·] ' pan count energy'` | 32 | `'_requirements'` (0.242) | **substitution** | 5.52 | 0.242 | 0.022 | `''` |
| worst | `' layer remember official expl' [·] ' success dise render'` | 16 | `' требованиями'` (0.943) | **correct** | 0.81 | 0.943 | 0.026 | `''` |
| best | `' implement activity tree choose' [·] ' led comm future'` | 32 | `' требованиями'` (0.991) | **correct** | 0.15 | 0.991 | 0.026 | `''` |

### `' competências'` — id 245573

single-probe lp -0.015 · fragility 0.12 · mean lp -0.18 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' times cult comment addition' [·] ' structure edge bo'` | 16 | `' compétences'` (0.495) | **substitution** | 1.14 | 0.495 | 0.023 | `''` |
| worst | `' against compl settings' [·] ' poll trust case'` | 32 | `' competências'` (0.526) | **correct** | 1.10 | 0.526 | 0.064 | `''` |
| worst | `' together vot want' [·] ' obj news music'` | 16 | `' compétences'` (0.642) | **substitution** | 1.10 | 0.344 | 0.032 | `''` |
| best | `' updated import von lim' [·] ' able starting time'` | 64 | `' competências'` (0.993) | **correct** | 0.07 | 0.993 | 0.009 | `''` |

### `' ungkapnya'` — id 210392

single-probe lp -0.011 · fragility 0.12 · mean lp -0.18 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' ret nature column between' [·] ' drug high plan'` | 64 | `' ungkapnya'` (0.580) | **correct** | 5.98 | 0.580 | -0.009 | `''` |
| worst | `' times cult comment addition' [·] ' structure edge bo'` | 16 | `' ungkapnya'` (0.462) | **correct** | 7.15 | 0.462 | -0.019 | `''` |
| worst | `' together vot want' [·] ' obj news music'` | 16 | `' ungkapnya'` (0.539) | **correct** | 4.97 | 0.539 | -0.009 | `''` |
| best | `' simply los made later' [·] ' size above least'` | 8 | `' ungkapnya'` (0.961) | **correct** | 0.54 | 0.961 | 0.001 | `''` |

### `' следующего'` — id 208237

single-probe lp -0.015 · fragility 0.08 · mean lp -0.13 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' re der age former' [·] ' government pie positive'` | 8 | `' следующего'` (0.936) | **correct** | 0.86 | 0.936 | 0.061 | `''` |
| worst | `' layer remember official expl' [·] ' success dise render'` | 16 | `' '` (0.489) | **substitution** | 4.20 | 0.231 | 0.053 | `''` |
| worst | `' ver integ align android' [·] ' investig performance above'` | 64 | `' следующего'` (0.789) | **correct** | 2.55 | 0.789 | 0.038 | `''` |
| best | `' taken' [·] ' longer visit own'` | 16 | `' следующего'` (0.997) | **correct** | 0.05 | 0.997 | 0.069 | `''` |

## Verified glitch tokens (reference)


### `' szexf'` — id 214963  (single-probe lp -24.60, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' fully' [·] ' wrote date dise'` | `' wrote'` (0.991) | deletion | 0.14 | 0.000 | -0.066 |
| worst | `' re der age former' [·] ' government pie positive'` | `' government'` (0.958) | deletion | 0.54 | 0.000 | -0.036 |
| worst | `' simply los made later' [·] ' size above least'` | `' size'` (0.581) | deletion | 2.86 | 0.000 | -0.060 |
| best | `' conditions introdu want feature' [·] ' pan count energy'` | `' pan'` (0.034) | deletion | 13.35 | 0.000 | -0.044 |

### `' ForCanBeConverted'` — id 76549  (single-probe lp -19.15, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' on customers ball them' [·] ' fight js ear'` | `' fight'` (0.743) | deletion | 2.72 | 0.000 | -0.075 |
| worst | `' ret nature column between' [·] ' drug high plan'` | `' drug'` (0.805) | deletion | 1.73 | 0.000 | -0.072 |
| worst | `' re der age former' [·] ' government pie positive'` | `' government'` (0.856) | deletion | 1.83 | 0.000 | -0.055 |
| best | `' separ continue ex rel' [·] ' filename'` | `'<|im_start|>'` (0.061) | substitution | 12.44 | 0.000 | -0.062 |

### `'чнике'` — id 236885  (single-probe lp -14.59, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' last down figure' [·] ' is distance los'` | `'chnet'` (0.243) | substitution | 5.59 | 0.002 | -0.025 |
| worst | `' re der age former' [·] ' government pie positive'` | `'chine'` (0.236) | substitution | 4.98 | 0.004 | -0.019 |
| worst | `' separ continue ex rel' [·] ' filename'` | `'chine'` (0.519) | substitution | 4.05 | 0.004 | -0.017 |
| best | `' asked independ story pa' [·] ' fre further die'` | `'чнике'` (0.649) | correct | 3.58 | 0.649 | -0.016 |

### `'echslungs'` — id 221404  (single-probe lp -14.35, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' taken' [·] ' longer visit own'` | `' beispiels'` (0.021) | substitution | 12.46 | 0.000 | -0.060 |
| worst | `' ing' [·] ' contain job handle'` | `' onstage'` (0.015) | substitution | 12.30 | 0.000 | -0.066 |
| worst | `' fully' [·] ' wrote date dise'` | `' entsp'` (0.045) | substitution | 11.94 | 0.000 | -0.064 |
| best | `' ret nature column between' [·] ' drug high plan'` | `'stringLiteral'` (0.021) | substitution | 11.58 | 0.000 | -0.074 |

### `'полномо'` — id 166145  (single-probe lp -13.31, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' found product buf happened' [·] ' tour pro find'` | `'zenio'` (0.072) | substitution | 8.54 | 0.004 | -0.063 |
| worst | `' conditions introdu want feature' [·] ' pan count energy'` | `'zenio'` (0.119) | substitution | 8.22 | 0.004 | -0.025 |
| worst | `' last down figure' [·] ' is distance los'` | `'polation'` (0.051) | substitution | 9.26 | 0.006 | -0.037 |
| best | `' simply los made later' [·] ' size above least'` | `'zenio'` (0.036) | substitution | 10.55 | 0.023 | -0.044 |

### `'хотво'` — id 222763  (single-probe lp -13.19, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' implement activity tree choose' [·] ' led comm future'` | `' hentai'` (0.035) | substitution | 12.59 | 0.002 | -0.070 |
| worst | `' ver integ align android' [·] ' investig performance above'` | `' investigation'` (0.110) | substitution | 9.87 | 0.001 | -0.047 |
| worst | `' layer remember official expl' [·] ' success dise render'` | `'obot'` (0.043) | substitution | 11.73 | 0.002 | -0.066 |
| best | `' updated import von lim' [·] ' able starting time'` | `' hentai'` (0.038) | substitution | 11.56 | 0.029 | -0.067 |

### `'ejahteraan'` — id 198259  (single-probe lp -13.09, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' conditions introdu want feature' [·] ' pan count energy'` | `'weathermap'` (0.058) | substitution | 11.15 | 0.000 | -0.047 |
| worst | `' asked independ story pa' [·] ' fre further die'` | `'nehmung'` (0.018) | substitution | 11.91 | 0.000 | -0.049 |
| worst | `' simply los made later' [·] ' size above least'` | `'nerRadius'` (0.027) | substitution | 11.36 | 0.000 | -0.055 |
| best | `' against compl settings' [·] ' poll trust case'` | `'んばん'` (0.046) | substitution | 11.03 | 0.001 | -0.042 |

### `'átku'` — id 222109  (single-probe lp -12.31, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' against compl settings' [·] ' poll trust case'` | `'átku'` (0.954) | correct | 0.50 | 0.954 | -0.013 |
| worst | `' asked independ story pa' [·] ' fre further die'` | `'átku'` (0.949) | correct | 0.45 | 0.949 | -0.013 |
| worst | `' on customers ball them' [·] ' fight js ear'` | `'átku'` (0.972) | correct | 0.23 | 0.972 | -0.037 |
| best | `' np register range met' [·] ' sk them head'` | `'átku'` (0.999) | correct | 0.02 | 0.999 | -0.015 |

### `'�'` — id 74354  (single-probe lp -11.94, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' ing' [·] ' contain job handle'` | `' contain'` (0.966) | deletion | 0.28 | 0.000 | 0.094 |
| worst | `' told co it post' [·] ' establish anim rot'` | `' establish'` (0.942) | deletion | 0.38 | 0.000 | 0.086 |
| worst | `' updated import von lim' [·] ' able starting time'` | `' able'` (0.857) | deletion | 0.72 | 0.000 | 0.088 |
| best | `' ret nature column between' [·] ' drug high plan'` | `' drug'` (0.761) | deletion | 1.33 | 0.033 | 0.067 |

### `'_watch'` — id 56583  (single-probe lp -11.46, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' together vot want' [·] ' obj news music'` | `'_watch'` (1.000) | correct | 0.00 | 1.000 | 0.001 |
| worst | `' ing' [·] ' contain job handle'` | `'_watch'` (1.000) | correct | 0.00 | 1.000 | 0.009 |
| worst | `' last down figure' [·] ' is distance los'` | `'_watch'` (1.000) | correct | 0.00 | 1.000 | 0.017 |
| best | `' np register range met' [·] ' sk them head'` | `'_watch'` (1.000) | correct | 0.00 | 1.000 | 0.025 |

### `'/api'` — id 10195  (single-probe lp -11.36, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' last down figure' [·] ' is distance los'` | `'/api'` (1.000) | correct | 0.01 | 1.000 | 0.055 |
| worst | `' together vot want' [·] ' obj news music'` | `'/api'` (1.000) | correct | 0.00 | 1.000 | 0.012 |
| worst | `' simply los made later' [·] ' size above least'` | `'/api'` (1.000) | correct | 0.01 | 1.000 | 0.064 |
| best | `' friend etc strateg history' [·] ' ent tw against'` | `'/api'` (1.000) | correct | 0.00 | 1.000 | 0.023 |

### `'_flight'` — id 87262  (single-probe lp -11.26, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' last down figure' [·] ' is distance los'` | `'_flight'` (1.000) | correct | 0.01 | 1.000 | -0.012 |
| worst | `' fully' [·] ' wrote date dise'` | `'_flight'` (0.999) | correct | 0.01 | 0.999 | -0.006 |
| worst | `' layer remember official expl' [·] ' success dise render'` | `'_flight'` (1.000) | correct | 0.01 | 1.000 | 0.009 |
| best | `' updated import von lim' [·] ' able starting time'` | `'_flight'` (1.000) | correct | 0.00 | 1.000 | -0.002 |

### `'_inode'` — id 45185  (single-probe lp -11.20, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' offers fil push econ' [·] ' alt'` | `'_inode'` (0.999) | correct | 0.01 | 0.999 | -0.006 |
| worst | `' ing' [·] ' contain job handle'` | `'_inode'` (1.000) | correct | 0.01 | 1.000 | -0.010 |
| worst | `' layer remember official expl' [·] ' success dise render'` | `'_inode'` (1.000) | correct | 0.01 | 1.000 | -0.001 |
| best | `' friend etc strateg history' [·] ' ent tw against'` | `'_inode'` (1.000) | correct | 0.00 | 1.000 | 0.003 |

### `'_challenge'` — id 84391  (single-probe lp -11.07, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' simply los made later' [·] ' size above least'` | `'_challenge'` (0.998) | correct | 0.03 | 0.998 | -0.004 |
| worst | `' last down figure' [·] ' is distance los'` | `'_challenge'` (0.999) | correct | 0.02 | 0.999 | 0.009 |
| worst | `' separ continue ex rel' [·] ' filename'` | `'_challenge'` (1.000) | correct | 0.01 | 1.000 | 0.016 |
| best | `' told co it post' [·] ' establish anim rot'` | `'_challenge'` (1.000) | correct | 0.00 | 1.000 | -0.010 |

### `'elleicht'` — id 64122  (single-probe lp -11.07, fragility 0.38)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' simply los made later' [·] ' size above least'` | `'elleicht'` (0.438) | correct | 6.30 | 0.438 | -0.014 |
| worst | `' offers fil push econ' [·] ' alt'` | `'elleicht'` (0.286) | correct | 8.30 | 0.286 | -0.015 |
| worst | `' separ continue ex rel' [·] ' filename'` | `'elleicht'` (0.385) | correct | 6.15 | 0.385 | -0.034 |
| best | `' asked independ story pa' [·] ' fre further die'` | `'elleicht'` (0.970) | correct | 0.38 | 0.970 | -0.011 |

### `'{n'` — id 88216  (single-probe lp -11.05, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' taken' [·] ' longer visit own'` | `'{n'` (0.998) | correct | 0.03 | 0.998 | 0.008 |
| worst | `' on customers ball them' [·] ' fight js ear'` | `'{n'` (0.997) | correct | 0.05 | 0.997 | 0.013 |
| worst | `' offers fil push econ' [·] ' alt'` | `'{n'` (0.999) | correct | 0.02 | 0.999 | 0.016 |
| best | `' implement activity tree choose' [·] ' led comm future'` | `'{n'` (1.000) | correct | 0.00 | 1.000 | -0.007 |

### `'_outline'` — id 70499  (single-probe lp -11.02, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' last down figure' [·] ' is distance los'` | `'_outline'` (0.999) | correct | 0.01 | 0.999 | -0.007 |
| worst | `' simply los made later' [·] ' size above least'` | `'_outline'` (0.999) | correct | 0.01 | 0.999 | 0.015 |
| worst | `' asked independ story pa' [·] ' fre further die'` | `'_outline'` (1.000) | correct | 0.00 | 1.000 | -0.001 |
| best | `' friend etc strateg history' [·] ' ent tw against'` | `'_outline'` (1.000) | correct | 0.00 | 1.000 | 0.022 |

### `'/utils'` — id 21227  (single-probe lp -10.99, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' offers fil push econ' [·] ' alt'` | `'/utils'` (0.998) | correct | 0.02 | 0.998 | 0.022 |
| worst | `' separ continue ex rel' [·] ' filename'` | `'/utils'` (0.999) | correct | 0.01 | 0.999 | 0.022 |
| worst | `' on customers ball them' [·] ' fight js ear'` | `'/utils'` (0.999) | correct | 0.01 | 0.999 | 0.045 |
| best | `' friend etc strateg history' [·] ' ent tw against'` | `'/utils'` (1.000) | correct | 0.00 | 1.000 | 0.037 |

### `' โรงแรมบรรยากาศ'` — id 236333  (single-probe lp -10.76, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' implement activity tree choose' [·] ' led comm future'` | `' atof'` (0.018) | substitution | 11.60 | 0.000 | -0.037 |
| worst | `' ver integ align android' [·] ' investig performance above'` | `' vantag'` (0.032) | substitution | 11.53 | 0.000 | -0.011 |
| worst | `' taken' [·] ' longer visit own'` | `' atof'` (0.064) | substitution | 11.39 | 0.000 | -0.039 |
| best | `' against compl settings' [·] ' poll trust case'` | `' atof'` (0.029) | substitution | 11.65 | 0.001 | 0.022 |

### `'_THRESH'` — id 33674  (single-probe lp -10.69, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' offers fil push econ' [·] ' alt'` | `'_THRESH'` (0.997) | correct | 0.04 | 0.997 | 0.016 |
| worst | `' layer remember official expl' [·] ' success dise render'` | `'_THRESH'` (1.000) | correct | 0.01 | 0.999 | 0.006 |
| worst | `' updated import von lim' [·] ' able starting time'` | `'_THRESH'` (0.999) | correct | 0.02 | 0.999 | 0.004 |
| best | `' conditions introdu want feature' [·] ' pan count energy'` | `'_THRESH'` (1.000) | correct | 0.00 | 1.000 | 0.001 |
