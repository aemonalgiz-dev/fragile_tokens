# Confident-substitution specimens

Every example in this file was produced by greedy decoding; `p` is the model's probability for the token it emitted, `H` the entropy in bits at that position, `p(target)` its probability for the correct token, `ret` the cosine between the last-layer slot state and the correct token's unembedding row (identity still pointing at its readout while the readout chose otherwise).

## Model

- **model**: `Qwen/Qwen3-32B`
- **commit**: `9216db5781bf21249d130ec9da846c4624c16137`
- **dtype**: `torch.bfloat16`
- **num_layers**: `64`
- **hidden_size**: `5120`
- **vocab_size**: `151936`
- **tie_word_embeddings**: `False`
- **tokenizer_class**: `Qwen2Tokenizer`
- **architecture**: `Qwen3ForCausalLM`
- **torch**: `2.14.0+cu130`
- **transformers**: `5.16.1`
- **python**: `3.10.12`
- **device**: `cuda:0`
- **decoding**: `greedy`
- **prompt_head**: `Repeat the text exactly.\nText: apple pie is good\nCopy: apple pie is good\nText: the quick brown fox\nCopy: the quick brown fox\nText:<context>\nCopy:<context-prefix>`
- **quantized**: `False`
- **fail_threshold_lp**: `-0.5`
- **n_contexts**: `24`
- **clean_gate_lp**: `-0.1`

## Fragile clean-looking tokens (pass the single probe; fail in some contexts)


### `'镅'` — id 123357

single-probe lp -0.091 · fragility 0.17 · mean lp -0.27 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `' not'` (0.683) | **deletion** | 1.58 | 0.196 | 0.017 | `' not aff filename offset push context\nTe'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `' pick'` (0.597) | **deletion** | 1.97 | 0.300 | 0.054 | `' pick opportunity water long sol returns'` |
| worst | `' it driver when said' [·] ' account scale'` | 8 | `' account'` (0.634) | **deletion** | 1.38 | 0.319 | 0.015 | `' account scale\nText: 12'` |
| best | `' pages story clo sent' [·] ' took contact draw'` | 64 | `'镅'` (0.990) | **correct** | 0.11 | 0.990 | 0.001 | `'镅 took contact draw letter required are '` |

### `'🐨'` — id 150085

single-probe lp -0.035 · fragility 0.12 · mean lp -0.13 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' several port sw struct' [·] ' published echo mis'` | 64 | `' published'` (0.415) | **deletion** | 1.83 | 0.366 | 0.028 | `' published echo mis equal ass phone void'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `'🐨'` (0.566) | **correct** | 1.50 | 0.566 | 0.126 | `' pick opportunity water long sol returns'` |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `'🐨'` (0.574) | **correct** | 1.39 | 0.574 | 0.070 | `'🐨 not aff filename offset push context\n'` |
| best | `' it driver when said' [·] ' account scale'` | 8 | `'🐨'` (0.998) | **correct** | 0.04 | 0.998 | 0.060 | `'🐨 account scale\nText: 1'` |

### `'了半天'` — id 117498

single-probe lp -0.004 · fragility 0.08 · mean lp -0.07 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `'了半天'` (0.571) | **correct** | 1.61 | 0.571 | 0.003 | `'了半天 not aff filename offset push context'` |
| worst | `' several port sw struct' [·] ' published echo mis'` | 64 | `'了半天'` (0.620) | **correct** | 1.00 | 0.620 | -0.043 | `'了半天 published echo mis equal ass phone v'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `'了半天'` (0.867) | **correct** | 0.65 | 0.867 | -0.005 | `'了半天 pick opportunity water long sol retu'` |
| best | `' len' [·] ' zu std ensure'` | 8 | `'了半天'` (1.000) | **correct** | 0.00 | 1.000 | -0.054 | `'了半天 zu std ensure maint myself tf\n'` |

## Verified glitch tokens (reference)


### `'ớ'` — id 141628  (single-probe lp -20.98, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' rather prot consider obt' [·] ' not aff filename'` | `'\n'` (0.696) | substitution | 1.42 | 0.000 | 0.017 |
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.858) | deletion | 1.13 | 0.000 | -0.077 |
| worst | `' it driver when said' [·] ' account scale'` | `'<|im_end|>'` (0.969) | substitution | 0.38 | 0.000 | -0.097 |
| best | `' saying expected zu hash' [·] ' man specified looking'` | `'<|im_end|>'` (0.053) | substitution | 9.62 | 0.000 | -0.152 |

### `'เปอร์'` — id 135619  (single-probe lp -20.61, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' solution' [·] ' function come production'` | `'<|im_end|>'` (0.684) | substitution | 1.17 | 0.000 | -0.057 |
| worst | `' appear rep finally fl' [·] ' ro stage'` | `'<|im_end|>'` (0.852) | substitution | 0.94 | 0.000 | -0.111 |
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.713) | deletion | 1.52 | 0.000 | -0.106 |
| best | `' profession cent temper sal' [·] ' exception dans parser'` | `'<|im_end|>'` (0.460) | substitution | 3.70 | 0.000 | -0.088 |

### `'อังกฤษ'` — id 127887  (single-probe lp -20.44, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.926) | deletion | 0.67 | 0.000 | -0.093 |
| worst | `' att friends ke pretty' [·] ' establish near group'` | `' establish'` (0.906) | deletion | 0.88 | 0.000 | -0.077 |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.827) | deletion | 1.43 | 0.000 | -0.062 |
| best | `' profession cent temper sal' [·] ' exception dans parser'` | `'<|im_end|>'` (0.520) | substitution | 5.06 | 0.000 | -0.129 |

### `'ใช่'` — id 126984  (single-probe lp -20.35, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att friends ke pretty' [·] ' establish near group'` | `' establish'` (0.786) | deletion | 1.65 | 0.000 | 0.009 |
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.887) | deletion | 1.17 | 0.000 | -0.058 |
| worst | `' design paid internal center' [·] ' pick statement elements'` | `'<|im_end|>'` (0.369) | substitution | 3.38 | 0.000 | 0.182 |
| best | `' kill threat top sl' [·] ' under push since'` | `'upd'` (0.152) | substitution | 8.25 | 0.000 | -0.009 |

### `'แม้'` — id 126927  (single-probe lp -20.28, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att friends ke pretty' [·] ' establish near group'` | `' establish'` (0.907) | deletion | 0.85 | 0.000 | -0.008 |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | `' not'` (0.662) | deletion | 1.64 | 0.000 | -0.044 |
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.879) | deletion | 1.06 | 0.000 | -0.072 |
| best | `' kill threat top sl' [·] ' under push since'` | `'<|im_start|>'` (0.228) | substitution | 7.57 | 0.000 | -0.056 |

### `'ล่า'` — id 126892  (single-probe lp -20.28, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' solution' [·] ' function come production'` | `'<|im_end|>'` (0.553) | substitution | 1.49 | 0.000 | -0.054 |
| worst | `' appear rep finally fl' [·] ' ro stage'` | `'<|im_end|>'` (0.974) | substitution | 0.32 | 0.000 | -0.093 |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | `'\n'` (0.552) | substitution | 2.09 | 0.000 | 0.036 |
| best | `' saying expected zu hash' [·] ' man specified looking'` | `'<|im_start|>'` (0.413) | substitution | 5.38 | 0.000 | -0.172 |

### `'ทั'` — id 124963  (single-probe lp -20.14, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' rather prot consider obt' [·] ' not aff filename'` | `' l'` (0.961) | substitution | 0.40 | 0.000 | -0.085 |
| worst | `' it driver when said' [·] ' account scale'` | `'<|im_end|>'` (0.946) | substitution | 0.50 | 0.000 | -0.106 |
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.942) | deletion | 0.60 | 0.000 | -0.062 |
| best | `' profession cent temper sal' [·] ' exception dans parser'` | `'<|im_end|>'` (0.176) | substitution | 8.91 | 0.000 | -0.163 |

### `'ช่ว'` — id 124531  (single-probe lp -20.04, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.903) | deletion | 0.82 | 0.000 | -0.077 |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.769) | deletion | 1.59 | 0.000 | -0.036 |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | `'\n'` (0.700) | substitution | 1.30 | 0.000 | 0.016 |
| best | `' saying expected zu hash' [·] ' man specified looking'` | `'l'` (0.322) | substitution | 5.56 | 0.000 | -0.201 |

### `'<|fim_prefix|>'` — id 151659  (single-probe lp -20.00, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' solution' [·] ' function come production'` | `' function'` (0.660) | deletion | 1.20 | 0.000 | -0.116 |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | `' l'` (0.477) | substitution | 3.67 | 0.000 | -0.023 |
| worst | `' provide override milit cur' [·] ' complete built damage'` | `' l'` (0.271) | substitution | 6.69 | 0.000 | -0.078 |
| best | `' develop remain dep vers' [·] ' element hours though'` | `'<|im_end|>'` (0.150) | substitution | 8.72 | 0.000 | -0.083 |

### `' สิงหาคม'` — id 142447  (single-probe lp -19.99, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att friends ke pretty' [·] ' establish near group'` | `' establish'` (0.849) | deletion | 1.27 | 0.000 | -0.009 |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | `' not'` (0.339) | deletion | 2.43 | 0.000 | 0.078 |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.469) | deletion | 2.43 | 0.000 | -0.052 |
| best | `' saying expected zu hash' [·] ' man specified looking'` | `'<|im_start|>'` (0.296) | substitution | 5.71 | 0.000 | -0.090 |

### `'𬸪'` — id 123637  (single-probe lp -19.97, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' appear rep finally fl' [·] ' ro stage'` | `'<|im_end|>'` (0.879) | substitution | 0.76 | 0.000 | -0.004 |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | `'<|im_end|>'` (0.857) | substitution | 0.91 | 0.000 | 0.020 |
| worst | `' client met' [·] ' she len ha'` | `'<|im_end|>'` (0.685) | substitution | 1.32 | 0.000 | 0.119 |
| best | `' kill threat top sl' [·] ' under push since'` | `'<|im_start|>'` (0.549) | substitution | 3.15 | 0.000 | 0.032 |

### `'웛'` — id 149983  (single-probe lp -19.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' themselves line multiple things' [·] ' range car player'` | `'<|im_end|>'` (0.967) | substitution | 0.36 | 0.000 | 0.035 |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | `'<|im_end|>'` (0.963) | substitution | 0.45 | 0.000 | -0.042 |
| worst | `' it driver when said' [·] ' account scale'` | `'<|im_end|>'` (0.980) | substitution | 0.26 | 0.000 | -0.077 |
| best | `' len' [·] ' zu std ensure'` | `'n'` (0.249) | substitution | 7.10 | 0.000 | -0.153 |

### `'ล่าสุด'` — id 130460  (single-probe lp -19.62, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' sort foot social treatment' [·] ' person cont soon'` | `'<|im_end|>'` (0.563) | substitution | 2.20 | 0.000 | 0.041 |
| worst | `' solution' [·] ' function come production'` | `' function'` (0.676) | deletion | 1.72 | 0.000 | -0.042 |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.765) | deletion | 1.96 | 0.000 | -0.083 |
| best | `' kill threat top sl' [·] ' under push since'` | `'awn'` (0.133) | substitution | 8.38 | 0.000 | -0.015 |

### `'ยอดเยี่ยม'` — id 140332  (single-probe lp -19.62, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.706) | deletion | 1.61 | 0.000 | -0.112 |
| worst | `' solution' [·] ' function come production'` | `' function'` (0.617) | deletion | 1.86 | 0.000 | -0.099 |
| worst | `' appear rep finally fl' [·] ' ro stage'` | `'<|im_end|>'` (0.835) | substitution | 1.54 | 0.000 | -0.115 |
| best | `' saying expected zu hash' [·] ' man specified looking'` | `'<|im_start|>'` (0.296) | substitution | 5.73 | 0.000 | -0.149 |

### `'แก้ปัญหา'` — id 143828  (single-probe lp -19.42, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att friends ke pretty' [·] ' establish near group'` | `' establish'` (0.934) | deletion | 0.72 | 0.000 | 0.040 |
| worst | `' solution' [·] ' function come production'` | `' function'` (0.497) | deletion | 1.58 | 0.000 | -0.014 |
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.863) | deletion | 1.14 | 0.000 | -0.086 |
| best | `' develop remain dep vers' [·] ' element hours though'` | `'ailles'` (0.152) | substitution | 9.55 | 0.000 | -0.071 |

### `'อัพ'` — id 140665  (single-probe lp -19.39, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.647) | deletion | 1.76 | 0.000 | -0.092 |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | `' person'` (0.468) | deletion | 2.57 | 0.000 | 0.056 |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.845) | deletion | 1.64 | 0.000 | -0.005 |
| best | `' kill threat top sl' [·] ' under push since'` | `'\n\n'` (0.080) | substitution | 8.31 | 0.000 | -0.038 |

### `'ใหม่'` — id 126233  (single-probe lp -19.33, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.840) | deletion | 1.37 | 0.000 | -0.032 |
| worst | `' att friends ke pretty' [·] ' establish near group'` | `' establish'` (0.921) | deletion | 0.78 | 0.000 | -0.012 |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | `'<|im_end|>'` (0.409) | substitution | 2.58 | 0.000 | -0.002 |
| best | `' len' [·] ' zu std ensure'` | `'<|im_start|>'` (0.248) | substitution | 6.83 | 0.000 | -0.180 |

### `'ก็คือ'` — id 132973  (single-probe lp -19.29, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.775) | deletion | 1.21 | 0.000 | 0.001 |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | `' person'` (0.465) | deletion | 2.34 | 0.000 | 0.049 |
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.406) | deletion | 2.87 | 0.000 | -0.062 |
| best | `' saying expected zu hash' [·] ' man specified looking'` | `'<|im_start|>'` (0.197) | substitution | 7.44 | 0.000 | -0.163 |

### `'หน่อย'` — id 133229  (single-probe lp -19.29, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.539) | deletion | 2.47 | 0.000 | -0.054 |
| worst | `' veh calcul otherwise hard' [·] ' beh func tell'` | `'<|im_start|>'` (0.943) | substitution | 0.57 | 0.000 | -0.087 |
| worst | `' price late vi wom' [·] ' under available skin'` | `'<|im_start|>'` (0.922) | substitution | 0.80 | 0.000 | -0.173 |
| best | `' kill threat top sl' [·] ' under push since'` | `'<|im_start|>'` (0.450) | substitution | 4.70 | 0.000 | -0.001 |

### `'ค้น'` — id 133355  (single-probe lp -19.27, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.527) | deletion | 3.12 | 0.000 | -0.044 |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | `' not'` (0.743) | deletion | 1.69 | 0.000 | 0.094 |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | `'<|im_end|>'` (0.361) | substitution | 3.47 | 0.000 | 0.005 |
| best | `' profession cent temper sal' [·] ' exception dans parser'` | `'<|im_end|>'` (0.558) | substitution | 5.32 | 0.000 | -0.130 |
