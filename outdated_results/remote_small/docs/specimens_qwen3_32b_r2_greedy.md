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
- **clean_gate_lp**: `-0.693`

## Fragile clean-looking tokens (pass the single probe; fail in some contexts)


### `' след'` — id 92029

single-probe lp -0.487 · fragility 0.92 · mean lp -5.64 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `' follow'` (0.662) | **substitution** | 1.38 | 0.000 | 0.038 | `' follow person cont soon prote day intro'` |
| worst | `' att friends ke pretty' [·] ' establish near group'` | 32 | `' follow'` (0.947) | **substitution** | 0.37 | 0.000 | 0.016 | `' follow establish near group above walk '` |
| worst | `' themselves line multiple things' [·] ' range car player'` | 32 | `' foll'` (0.435) | **substitution** | 2.32 | 0.000 | 0.050 | `' foll range car player bar choose politi'` |
| best | `' solution' [·] ' function come production'` | 8 | `' след'` (0.988) | **correct** | 0.13 | 0.988 | 0.095 | `' след function come production font hand'` |

### `'��'` — id 92120

single-probe lp -0.513 · fragility 0.58 · mean lp -1.05 · canonical False · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' several port sw struct' [·] ' published echo mis'` | 64 | `' published'` (0.897) | **deletion** | 0.82 | 0.011 | -0.011 | `' published echo mis equal ass phone void'` |
| worst | `' develop remain dep vers' [·] ' element hours though'` | 16 | `' element'` (0.709) | **deletion** | 2.36 | 0.026 | 0.051 | `' element hours though performance establ'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `' key'` (0.503) | **substitution** | 2.67 | 0.064 | 0.069 | `' key pick opportunity water long sol ret'` |
| best | `' price late vi wom' [·] ' under available skin'` | 16 | `'��'` (0.981) | **correct** | 0.20 | 0.980 | 0.010 | `'�� under available skin sale del increas'` |

### `'וג'` — id 123945

single-probe lp -0.143 · fragility 0.54 · mean lp -0.84 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pages story clo sent' [·] ' took contact draw'` | 64 | `'tog'` (0.483) | **substitution** | 2.40 | 0.018 | -0.006 | `'tog took contact draw letter required ar'` |
| worst | `' it driver when said' [·] ' account scale'` | 8 | `'oğ'` (0.249) | **substitution** | 4.47 | 0.067 | 0.012 | `'oğ account scale\nText: 1'` |
| worst | `' themselves line multiple things' [·] ' range car player'` | 32 | `' range'` (0.471) | **deletion** | 4.77 | 0.099 | 0.055 | `' range car player bar choose political m'` |
| best | `' saying expected zu hash' [·] ' man specified looking'` | 32 | `'וג'` (0.955) | **correct** | 0.55 | 0.955 | 0.001 | `'וג man specified looking force max offer'` |

### `' realtà'` — id 134523

single-probe lp -0.281 · fragility 0.50 · mean lp -0.52 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' activ ne throw cam' [·] ' art dam ver'` | 8 | `' réalité'` (0.703) | **substitution** | 1.13 | 0.259 | 0.053 | `' réalité art dam ver\nText: '` |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `' réalité'` (0.571) | **substitution** | 1.31 | 0.392 | 0.042 | `' réalité not aff filename offset push co'` |
| worst | `' pages story clo sent' [·] ' took contact draw'` | 64 | `' réalité'` (0.519) | **substitution** | 1.23 | 0.458 | -0.018 | `' réalité took contact draw letter requir'` |
| best | `' kill threat top sl' [·] ' under push since'` | 32 | `' realtà'` (0.825) | **correct** | 0.78 | 0.825 | 0.034 | `' realtà under push since flow hours dim '` |

### `'자는'` — id 132343

single-probe lp -0.438 · fragility 0.50 · mean lp -1.05 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' themselves line multiple things' [·] ' range car player'` | 32 | `' range'` (0.392) | **deletion** | 5.06 | 0.007 | -0.007 | `' range car player bar choose political m'` |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `'<|im_start|>'` (0.704) | **substitution** | 2.44 | 0.040 | -0.034 | `'\n\nOkay, the user wants me'` |
| worst | `' profession cent temper sal' [·] ' exception dans parser'` | 64 | `'는'` (0.132) | **substitution** | 6.64 | 0.080 | -0.005 | `'는 exception dans parser look inf behind '` |
| best | `' len' [·] ' zu std ensure'` | 8 | `'자는'` (0.998) | **correct** | 0.03 | 0.998 | -0.042 | `'자는 zu std ensure maint myself tf\n'` |

### `'ർ'` — id 148979

single-probe lp -0.102 · fragility 0.50 · mean lp -1.06 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pages story clo sent' [·] ' took contact draw'` | 64 | `' took'` (0.851) | **deletion** | 1.09 | 0.029 | -0.008 | `' took contact draw letter required are p'` |
| worst | `' att friends ke pretty' [·] ' establish near group'` | 32 | `' establish'` (0.919) | **deletion** | 0.55 | 0.063 | 0.017 | `' establish near group above walk leave e'` |
| worst | `' profession cent temper sal' [·] ' exception dans parser'` | 64 | `' exception'` (0.914) | **deletion** | 0.58 | 0.071 | -0.019 | `' exception dans parser look inf behind h'` |
| best | `' veh calcul otherwise hard' [·] ' beh func tell'` | 8 | `'ർ'` (0.997) | **correct** | 0.04 | 0.997 | 0.012 | `'ർ beh func tell\nText: '` |

### `'�'` — id 242

single-probe lp -0.168 · fragility 0.50 · mean lp -1.24 · canonical False · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `' person'` (0.849) | **deletion** | 1.19 | 0.007 | 0.037 | `' person cont soon prote day introdu mach'` |
| worst | `' pages story clo sent' [·] ' took contact draw'` | 64 | `' took'` (0.832) | **deletion** | 1.10 | 0.009 | 0.021 | `' took contact draw letter required are p'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `' pick'` (0.918) | **deletion** | 1.01 | 0.019 | 0.061 | `' pick opportunity water long sol returns'` |
| best | `' client met' [·] ' she len ha'` | 64 | `'�'` (0.969) | **correct** | 0.36 | 0.969 | 0.061 | `'� she len ha possible ones shown most'` |

### `' והת'` — id 133258

single-probe lp -0.553 · fragility 0.46 · mean lp -0.60 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `' והת'` (0.137) | **correct** | 7.85 | 0.137 | 0.024 | `' והת person cont soon prote day introdu '` |
| worst | `' themselves line multiple things' [·] ' range car player'` | 32 | `' והת'` (0.295) | **correct** | 5.13 | 0.295 | 0.056 | `' והת range car player bar choose politic'` |
| worst | `' saying expected zu hash' [·] ' man specified looking'` | 32 | `' והת'` (0.329) | **correct** | 7.24 | 0.329 | 0.018 | `' והת man specified looking force max off'` |
| best | `' solution' [·] ' function come production'` | 8 | `' והת'` (0.892) | **correct** | 1.30 | 0.892 | 0.055 | `' והת function come production font handl'` |

### `' כסף'` — id 133359

single-probe lp -0.668 · fragility 0.42 · mean lp -0.45 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' several port sw struct' [·] ' published echo mis'` | 64 | `' כסף'` (0.422) | **correct** | 3.34 | 0.422 | -0.000 | `' כסף published echo mis equal ass phone '` |
| worst | `' price late vi wom' [·] ' under available skin'` | 16 | `' כסף'` (0.456) | **correct** | 2.63 | 0.456 | 0.080 | `' כסף under available skin sale del incre'` |
| worst | `' provide override milit cur' [·] ' complete built damage'` | 32 | `' כסף'` (0.494) | **correct** | 3.59 | 0.494 | 0.028 | `' כסף complete built damage idea seek lin'` |
| best | `' themselves line multiple things' [·] ' range car player'` | 32 | `' כסף'` (0.855) | **correct** | 1.27 | 0.855 | 0.071 | `' כסף range car player bar choose politic'` |

### `' להיות'` — id 128535

single-probe lp -0.514 · fragility 0.42 · mean lp -0.66 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' saying expected zu hash' [·] ' man specified looking'` | 32 | `' להיות'` (0.147) | **correct** | 8.45 | 0.147 | 0.027 | `' להיות man specified looking force max o'` |
| worst | `' appear rep finally fl' [·] ' ro stage'` | 16 | `'\n\n'` (0.169) | **substitution** | 7.63 | 0.140 | 0.013 | `"\n\nOkay, let's see. The"` |
| worst | `' kill threat top sl' [·] ' under push since'` | 32 | `' להיות'` (0.139) | **correct** | 8.78 | 0.139 | 0.015 | `' להיות under push since flow hours dim q'` |
| best | `' design paid internal center' [·] ' pick statement elements'` | 16 | `' להיות'` (0.905) | **correct** | 1.37 | 0.905 | 0.045 | `' להיות pick statement elements me hope e'` |

### `'ཀ'` — id 147474

single-probe lp -0.651 · fragility 0.38 · mean lp -0.67 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `' pick'` (0.884) | **deletion** | 1.10 | 0.041 | 0.091 | `' pick opportunity water long sol returns'` |
| worst | `' appear rep finally fl' [·] ' ro stage'` | 16 | `' ro'` (0.517) | **deletion** | 3.34 | 0.123 | -0.007 | `' ro stage\nText: flow well federal'` |
| worst | `' several port sw struct' [·] ' published echo mis'` | 64 | `' published'` (0.756) | **deletion** | 1.31 | 0.169 | 0.017 | `' published echo mis equal ass phone void'` |
| best | `' price late vi wom' [·] ' under available skin'` | 16 | `'ཀ'` (0.990) | **correct** | 0.16 | 0.990 | 0.065 | `'ཀ under available skin sale del increase'` |

### `'whereIn'` — id 85493

single-probe lp -0.459 · fragility 0.33 · mean lp -0.37 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pages story clo sent' [·] ' took contact draw'` | 64 | `'\n'` (0.395) | **substitution** | 3.06 | 0.240 | -0.047 | `'\nText: 1234'` |
| worst | `' develop remain dep vers' [·] ' element hours though'` | 16 | `'where'` (0.714) | **truncation** | 1.04 | 0.263 | 0.033 | `'whereIn element hours though performance'` |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `'\n'` (0.426) | **substitution** | 2.37 | 0.353 | 0.058 | `'\nText: sort foot social treatment\n'` |
| best | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `'whereIn'` (0.999) | **correct** | 0.01 | 0.999 | 0.019 | `'whereIn not aff filename offset push con'` |

### `'بدو'` — id 127961

single-probe lp -0.437 · fragility 0.33 · mean lp -0.65 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' provide override milit cur' [·] ' complete built damage'` | 32 | `'\n'` (0.105) | **substitution** | 9.11 | 0.016 | -0.010 | `'\nText: provide override milit cur\n'` |
| worst | `' activ ne throw cam' [·] ' art dam ver'` | 8 | `'bedo'` (0.601) | **substitution** | 4.78 | 0.032 | -0.017 | `'bedo art dam ver\nOkay, let'` |
| worst | `' price late vi wom' [·] ' under available skin'` | 16 | `'bedo'` (0.882) | **substitution** | 0.85 | 0.093 | -0.033 | `'bedo under available skin sale del incre'` |
| best | `' solution' [·] ' function come production'` | 8 | `'بدو'` (0.976) | **correct** | 0.36 | 0.976 | 0.032 | `'بدو function come production font handle'` |

### `'interopRequireDefault'` — id 35364

single-probe lp -0.265 · fragility 0.29 · mean lp -0.74 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `' person'` (0.593) | **deletion** | 2.62 | 0.006 | 0.075 | `' person cont soon prote day introdu mach'` |
| worst | `' themselves line multiple things' [·] ' range car player'` | 32 | `'\n'` (0.527) | **substitution** | 4.15 | 0.076 | 0.063 | `'\nText: complex full able themselves line'` |
| worst | `' kill threat top sl' [·] ' under push since'` | 32 | `'\n'` (0.226) | **substitution** | 8.02 | 0.094 | 0.015 | `'\nText: 1234'` |
| best | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `'interopRequireDefault'` (0.999) | **correct** | 0.02 | 0.999 | 0.075 | `'interopRequireDefault not aff filename o'` |

### `' прогн'` — id 141080

single-probe lp -0.680 · fragility 0.25 · mean lp -0.27 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' several port sw struct' [·] ' published echo mis'` | 64 | `' prog'` (0.613) | **substitution** | 1.91 | 0.199 | -0.024 | `' prog published echo mis equal ass phone'` |
| worst | `' len' [·] ' zu std ensure'` | 8 | `' прогн'` (0.386) | **correct** | 4.23 | 0.385 | -0.092 | `' прогн zu std ensure maint myself tf\n'` |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `' прогн'` (0.488) | **correct** | 2.41 | 0.488 | 0.047 | `' прогн not aff filename offset push cont'` |
| best | `' property content' [·] ' init except successful'` | 16 | `' прогн'` (0.991) | **correct** | 0.11 | 0.991 | -0.031 | `' прогн init except successful there nor '` |

### `'ঙ'` — id 148571

single-probe lp -0.293 · fragility 0.25 · mean lp -0.36 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' several port sw struct' [·] ' published echo mis'` | 64 | `' published'` (0.898) | **deletion** | 0.81 | 0.054 | 0.013 | `' published echo mis equal ass phone void'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `' pick'` (0.633) | **deletion** | 1.87 | 0.281 | 0.039 | `' pick opportunity water long sol returns'` |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `'ঙ'` (0.516) | **correct** | 1.75 | 0.516 | 0.060 | `'ঙ person cont soon prote day introdu mac'` |
| best | `' veh calcul otherwise hard' [·] ' beh func tell'` | 8 | `'ঙ'` (0.994) | **correct** | 0.09 | 0.994 | 0.038 | `'ঙ beh func tell\nText: '` |

### `' ofrece'` — id 88439

single-probe lp -0.524 · fragility 0.21 · mean lp -0.28 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' themselves line multiple things' [·] ' range car player'` | 32 | `' ofere'` (0.581) | **substitution** | 1.68 | 0.274 | 0.083 | `' oferece range car player bar choose pol'` |
| worst | `' finish child media foot' [·] ' especially home rece'` | 32 | `' ofere'` (0.521) | **substitution** | 1.54 | 0.316 | 0.056 | `' oferece especially home rece printf con'` |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `' ofrece'` (0.457) | **correct** | 1.65 | 0.457 | 0.064 | `' ofrece person cont soon prote day intro'` |
| best | `' len' [·] ' zu std ensure'` | 8 | `' ofrece'` (0.990) | **correct** | 0.10 | 0.990 | 0.048 | `' ofrece zu std ensure maint myself tf\n'` |

### `' программы'` — id 136059

single-probe lp -0.399 · fragility 0.21 · mean lp -0.45 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' several port sw struct' [·] ' published echo mis'` | 64 | `' programmes'` (0.385) | **substitution** | 2.56 | 0.002 | 0.008 | `' programmes published echo mis equal ass'` |
| worst | `' themselves line multiple things' [·] ' range car player'` | 32 | `' программы'` (0.522) | **correct** | 2.08 | 0.522 | 0.029 | `' программы range car player bar choose p'` |
| worst | `' it driver when said' [·] ' account scale'` | 8 | `' программы'` (0.549) | **correct** | 2.12 | 0.549 | 0.017 | `' программы account scale\nText: 1'` |
| best | `' client met' [·] ' she len ha'` | 64 | `' программы'` (0.992) | **correct** | 0.08 | 0.992 | 0.035 | `' программы she len ha possible ones show'` |

### `' טי'` — id 135345

single-probe lp -0.483 · fragility 0.17 · mean lp -0.24 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' provide override milit cur' [·] ' complete built damage'` | 32 | `' טי'` (0.288) | **correct** | 8.05 | 0.288 | -0.010 | `' טי complete built damage idea seek line'` |
| worst | `' price late vi wom' [·] ' under available skin'` | 16 | `' '` (0.501) | **substitution** | 2.50 | 0.391 | 0.038 | `'  טי under available skin sale del incre'` |
| worst | `' profession cent temper sal' [·] ' exception dans parser'` | 64 | `' טי'` (0.558) | **correct** | 2.75 | 0.558 | -0.037 | `' טי exception dans parser look inf behin'` |
| best | `' att friends ke pretty' [·] ' establish near group'` | 32 | `' טי'` (0.977) | **correct** | 0.35 | 0.977 | 0.032 | `' טי establish near group above walk leav'` |

### `'ℍ'` — id 149399

single-probe lp -0.657 · fragility 0.17 · mean lp -0.36 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `' pick'` (0.841) | **deletion** | 1.19 | 0.100 | 0.090 | `' pick opportunity water long sol returns'` |
| worst | `' veh calcul otherwise hard' [·] ' beh func tell'` | 8 | `' beh'` (0.518) | **deletion** | 1.64 | 0.429 | 0.065 | `' beh func tell\nText: 1'` |
| worst | `' themselves line multiple things' [·] ' range car player'` | 32 | `' range'` (0.482) | **deletion** | 1.65 | 0.452 | 0.098 | `' range car player bar choose political m'` |
| best | `' activ ne throw cam' [·] ' art dam ver'` | 8 | `'ℍ'` (0.993) | **correct** | 0.11 | 0.993 | 0.083 | `'ℍ art dam ver\nText: '` |

### `'𝓪'` — id 147910

single-probe lp -0.370 · fragility 0.17 · mean lp -0.28 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `' a'` (0.567) | **substitution** | 2.86 | 0.173 | 0.117 | `' a person cont soon prote day introdu ma'` |
| worst | `' themselves line multiple things' [·] ' range car player'` | 32 | `'𝓪'` (0.471) | **correct** | 2.41 | 0.471 | 0.084 | `'𝓪 range car player bar choose political '` |
| worst | `' it driver when said' [·] ' account scale'` | 8 | `'𝓪'` (0.611) | **correct** | 2.05 | 0.611 | 0.025 | `'𝓪 account scale\nText: 1'` |
| best | `' kill threat top sl' [·] ' under push since'` | 32 | `'𝓪'` (0.985) | **correct** | 0.23 | 0.985 | 0.028 | `'𝓪 under push since flow hours dim quite'` |

### `'👕'` — id 146416

single-probe lp -0.400 · fragility 0.17 · mean lp -0.25 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `' not'` (0.565) | **deletion** | 2.04 | 0.322 | 0.137 | `' not aff filename offset push context\nTe'` |
| worst | `' solution' [·] ' function come production'` | 8 | `'👕'` (0.526) | **correct** | 1.99 | 0.526 | 0.172 | `'👕 function come production font handle s'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `'👕'` (0.571) | **correct** | 2.33 | 0.571 | 0.148 | `'👕 pick opportunity water long sol return'` |
| best | `' finish child media foot' [·] ' especially home rece'` | 32 | `'👕'` (0.977) | **correct** | 0.29 | 0.977 | 0.137 | `'👕 especially home rece printf connect gr'` |

### `'ЛА'` — id 141115

single-probe lp -0.621 · fragility 0.17 · mean lp -0.28 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `' pick'` (0.714) | **deletion** | 1.73 | 0.150 | 0.074 | `' pick opportunity water long sol returns'` |
| worst | `' known ideas' [·] ' assert icon cert'` | 8 | `'ЛА'` (0.406) | **correct** | 2.40 | 0.406 | 0.046 | `'ЛА assert icon cert destroy pol\nText'` |
| worst | `' kill threat top sl' [·] ' under push since'` | 32 | `'ЛА'` (0.445) | **correct** | 2.70 | 0.445 | 0.019 | `'ЛА under push since flow hours dim quite'` |
| best | `' client met' [·] ' she len ha'` | 64 | `'ЛА'` (0.991) | **correct** | 0.12 | 0.991 | 0.079 | `'ЛА she len ha possible ones shown most'` |

### `'镅'` — id 123357

single-probe lp -0.091 · fragility 0.17 · mean lp -0.27 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `' not'` (0.683) | **deletion** | 1.58 | 0.196 | 0.017 | `' not aff filename offset push context\nTe'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `' pick'` (0.597) | **deletion** | 1.97 | 0.300 | 0.054 | `' pick opportunity water long sol returns'` |
| worst | `' it driver when said' [·] ' account scale'` | 8 | `' account'` (0.634) | **deletion** | 1.38 | 0.319 | 0.015 | `' account scale\nText: 12'` |
| best | `' pages story clo sent' [·] ' took contact draw'` | 64 | `'镅'` (0.990) | **correct** | 0.11 | 0.990 | 0.001 | `'镅 took contact draw letter required are '` |

### `'נע'` — id 134089

single-probe lp -0.668 · fragility 0.17 · mean lp -0.25 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' profession cent temper sal' [·] ' exception dans parser'` | 64 | `' exception'` (0.461) | **deletion** | 3.31 | 0.337 | -0.008 | `' exception dans parser look inf behind h'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `' pick'` (0.480) | **deletion** | 1.97 | 0.424 | 0.086 | `' pick opportunity water long sol returns'` |
| worst | `' pages story clo sent' [·] ' took contact draw'` | 64 | `' took'` (0.422) | **deletion** | 2.53 | 0.422 | 0.011 | `'נע took contact draw letter required are'` |
| best | `' len' [·] ' zu std ensure'` | 8 | `'נע'` (0.995) | **correct** | 0.09 | 0.995 | -0.035 | `'נע zu std ensure maint myself tf\n'` |

### `'🐨'` — id 150085

single-probe lp -0.035 · fragility 0.12 · mean lp -0.13 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' several port sw struct' [·] ' published echo mis'` | 64 | `' published'` (0.415) | **deletion** | 1.83 | 0.366 | 0.028 | `' published echo mis equal ass phone void'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `'🐨'` (0.566) | **correct** | 1.50 | 0.566 | 0.126 | `'🐨 pick opportunity water long sol return'` |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `'🐨'` (0.574) | **correct** | 1.39 | 0.574 | 0.070 | `'🐨 not aff filename offset push context\n'` |
| best | `' it driver when said' [·] ' account scale'` | 8 | `'🐨'` (0.998) | **correct** | 0.04 | 0.998 | 0.060 | `'🐨 account scale\nText: 1'` |

### `'וצ'` — id 123926

single-probe lp -0.317 · fragility 0.12 · mean lp -0.15 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' finish child media foot' [·] ' especially home rece'` | 32 | `'וצ'` (0.573) | **correct** | 3.94 | 0.573 | 0.044 | `'וצ especially home rece printf connect g'` |
| worst | `' known ideas' [·] ' assert icon cert'` | 8 | `'וצ'` (0.547) | **correct** | 3.74 | 0.547 | 0.026 | `'וצ assert icon cert destroy pol\nRepeat'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `'וצ'` (0.551) | **correct** | 2.63 | 0.551 | 0.065 | `'וצ pick opportunity water long sol retur'` |
| best | `' saying expected zu hash' [·] ' man specified looking'` | 32 | `'וצ'` (0.997) | **correct** | 0.05 | 0.997 | 0.015 | `'וצ man specified looking force max offer'` |

### `'ﺭ'` — id 144352

single-probe lp -0.421 · fragility 0.12 · mean lp -0.16 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `' not'` (0.527) | **deletion** | 1.49 | 0.411 | 0.072 | `' not aff filename offset push context\nTe'` |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `'ﺭ'` (0.592) | **correct** | 1.76 | 0.592 | 0.086 | `'ﺭ person cont soon prote day introdu mac'` |
| worst | `' att friends ke pretty' [·] ' establish near group'` | 32 | `'ﺭ'` (0.606) | **correct** | 1.22 | 0.606 | 0.082 | `'ﺭ establish near group above walk leave '` |
| best | `' provide override milit cur' [·] ' complete built damage'` | 32 | `'ﺭ'` (0.998) | **correct** | 0.03 | 0.998 | 0.056 | `'ﺭ complete built damage idea seek line p'` |

### `' ๆ'` — id 128630

single-probe lp -0.202 · fragility 0.12 · mean lp -0.21 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' provide override milit cur' [·] ' complete built damage'` | 32 | `' ๆ'` (0.157) | **correct** | 8.60 | 0.157 | -0.004 | `' ๆ complete built damage idea seek line '` |
| worst | `' appear rep finally fl' [·] ' ro stage'` | 16 | `' '` (0.414) | **substitution** | 2.04 | 0.267 | 0.002 | `'  ๆ ro stage\nText: flow'` |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `' ๆ'` (0.506) | **correct** | 3.75 | 0.506 | 0.030 | `' ๆ not aff filename offset push context\n'` |
| best | `' known ideas' [·] ' assert icon cert'` | 8 | `' ๆ'` (0.994) | **correct** | 0.07 | 0.994 | 0.043 | `' ๆ assert icon cert destroy pol\nText'` |

### `'stdexcept'` — id 94513

single-probe lp -0.386 · fragility 0.12 · mean lp -0.24 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `'\n'` (0.293) | **substitution** | 4.82 | 0.108 | 0.044 | `'\nperson cont soon prote day introdu mach'` |
| worst | `' appear rep finally fl' [·] ' ro stage'` | 16 | `'\n\n'` (0.354) | **substitution** | 3.58 | 0.157 | 0.059 | `'\n\nOkay, the user wants me to'` |
| worst | `' finish child media foot' [·] ' especially home rece'` | 32 | `'stdexcept'` (0.599) | **correct** | 3.59 | 0.599 | 0.023 | `'stdexcept especially home rece printf co'` |
| best | `' several port sw struct' [·] ' published echo mis'` | 64 | `'stdexcept'` (1.000) | **correct** | 0.00 | 1.000 | -0.000 | `'stdexcept published echo mis equal ass p'` |

### `'钐'` — id 122321

single-probe lp -0.683 · fragility 0.12 · mean lp -0.14 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `'钐'` (0.481) | **correct** | 2.82 | 0.481 | 0.043 | `'钐 not aff filename offset push context\n'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `'钐'` (0.502) | **correct** | 3.25 | 0.502 | 0.044 | `'钐 pick opportunity water long sol return'` |
| worst | `' pages story clo sent' [·] ' took contact draw'` | 64 | `'钐'` (0.620) | **correct** | 1.62 | 0.620 | -0.014 | `'钐 took contact draw letter required are '` |
| best | `' profession cent temper sal' [·] ' exception dans parser'` | 64 | `'钐'` (0.999) | **correct** | 0.01 | 0.999 | 0.009 | `'钐 exception dans parser look inf behind '` |

### `'𝕒'` — id 146898

single-probe lp -0.100 · fragility 0.12 · mean lp -0.27 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `' a'` (0.309) | **substitution** | 2.87 | 0.241 | 0.161 | `' a person cont soon prote day introdu ma'` |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `'𝕒'` (0.631) | **correct** | 2.37 | 0.631 | 0.076 | `'𝕒 not aff filename offset push context\n'` |
| worst | `' known ideas' [·] ' assert icon cert'` | 8 | `'𝕒'` (0.615) | **correct** | 1.90 | 0.615 | 0.067 | `'𝕒 assert icon cert destroy pol\nText'` |
| best | `' develop remain dep vers' [·] ' element hours though'` | 16 | `'𝕒'` (0.991) | **correct** | 0.13 | 0.991 | 0.081 | `'𝕒 element hours though performance estab'` |

### `'Ж'` — id 132042

single-probe lp -0.530 · fragility 0.12 · mean lp -0.16 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `' pick'` (0.524) | **deletion** | 1.66 | 0.408 | 0.109 | `' pick opportunity water long sol returns'` |
| worst | `' it driver when said' [·] ' account scale'` | 8 | `' account'` (0.522) | **deletion** | 1.23 | 0.460 | 0.095 | `' account scale\nText: instance it driver'` |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `'Ж'` (0.558) | **correct** | 1.60 | 0.558 | 0.116 | `'Ж person cont soon prote day introdu mac'` |
| best | `' len' [·] ' zu std ensure'` | 8 | `'Ж'` (0.999) | **correct** | 0.01 | 0.999 | 0.028 | `'Ж zu std ensure maint myself tf\n'` |

### `' абсол'` — id 142750

single-probe lp -0.465 · fragility 0.12 · mean lp -0.27 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' design paid internal center' [·] ' pick statement elements'` | 16 | `' absol'` (0.528) | **substitution** | 1.07 | 0.466 | 0.037 | `' absol pick statement elements me hope e'` |
| worst | `' pages story clo sent' [·] ' took contact draw'` | 64 | `' абсол'` (0.592) | **correct** | 0.99 | 0.592 | -0.072 | `' абсол took contact draw letter required'` |
| worst | `' solution' [·] ' function come production'` | 8 | `' абсол'` (0.590) | **correct** | 1.04 | 0.590 | 0.068 | `' абсол function come production font han'` |
| best | `' veh calcul otherwise hard' [·] ' beh func tell'` | 8 | `' абсол'` (0.967) | **correct** | 0.22 | 0.966 | -0.026 | `' абсол beh func tell\nText: '` |

### `'🗼'` — id 148272

single-probe lp -0.238 · fragility 0.12 · mean lp -0.43 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `' not'` (0.940) | **deletion** | 0.72 | 0.011 | 0.086 | `' not aff filename offset push context\n</'` |
| worst | `' pages story clo sent' [·] ' took contact draw'` | 64 | `' took'` (0.662) | **deletion** | 1.80 | 0.215 | -0.037 | `' took contact draw letter required are p'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `' pick'` (0.488) | **deletion** | 2.34 | 0.296 | 0.091 | `' pick opportunity water long sol returns'` |
| best | `' activ ne throw cam' [·] ' art dam ver'` | 8 | `'🗼'` (0.995) | **correct** | 0.08 | 0.995 | 0.126 | `'🗼 art dam ver\nText: '` |

### `' автор'` — id 132718

single-probe lp -0.619 · fragility 0.08 · mean lp -0.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' known ideas' [·] ' assert icon cert'` | 8 | `' автор'` (0.557) | **correct** | 1.65 | 0.557 | 0.063 | `' автор assert icon cert destroy pol\nText'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `' автор'` (0.583) | **correct** | 1.80 | 0.583 | 0.086 | `' автор pick opportunity water long sol r'` |
| worst | `' profession cent temper sal' [·] ' exception dans parser'` | 64 | `' автор'` (0.695) | **correct** | 1.16 | 0.695 | 0.059 | `' автор exception dans parser look inf be'` |
| best | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `' автор'` (0.983) | **correct** | 0.16 | 0.983 | 0.061 | `' автор not aff filename offset push cont'` |

### `'statusCode'` — id 44480

single-probe lp -0.519 · fragility 0.08 · mean lp -0.21 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' saying expected zu hash' [·] ' man specified looking'` | 32 | `'hashCode'` (0.651) | **substitution** | 1.64 | 0.061 | -0.015 | `'hashCode man specified looking force max'` |
| worst | `' veh calcul otherwise hard' [·] ' beh func tell'` | 8 | `'StatusCode'` (0.658) | **substitution** | 1.14 | 0.311 | 0.015 | `'StatusCode beh func tell\nText: '` |
| worst | `' solution' [·] ' function come production'` | 8 | `'statusCode'` (0.729) | **correct** | 1.21 | 0.729 | 0.046 | `'statusCode function come production font'` |
| best | `' activ ne throw cam' [·] ' art dam ver'` | 8 | `'statusCode'` (0.998) | **correct** | 0.03 | 0.998 | 0.028 | `'statusCode art dam ver\nText: '` |

### `'امي'` — id 131762

single-probe lp -0.668 · fragility 0.08 · mean lp -0.18 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' provide override milit cur' [·] ' complete built damage'` | 32 | `'امي'` (0.514) | **correct** | 2.27 | 0.514 | -0.018 | `'امي complete built damage idea seek line'` |
| worst | `' client met' [·] ' she len ha'` | 64 | `'امي'` (0.562) | **correct** | 3.80 | 0.562 | 0.046 | `'امي she len ha possible ones shown most'` |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `'امي'` (0.634) | **correct** | 1.80 | 0.634 | 0.058 | `'امي person cont soon prote day introdu m'` |
| best | `' len' [·] ' zu std ensure'` | 8 | `'امي'` (0.997) | **correct** | 0.05 | 0.997 | -0.050 | `'امي zu std ensure maint myself tf\n'` |

### `'ee'` — id 2127

single-probe lp -0.552 · fragility 0.08 · mean lp -0.13 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' profession cent temper sal' [·] ' exception dans parser'` | 64 | `'ée'` (0.474) | **substitution** | 2.71 | 0.198 | 0.012 | `'ée exception dans parser look inf behind'` |
| worst | `' kill threat top sl' [·] ' under push since'` | 32 | `'ee'` (0.370) | **correct** | 3.49 | 0.370 | 0.042 | `'ee under push since flow hours dim quite'` |
| worst | `' provide override milit cur' [·] ' complete built damage'` | 32 | `'ee'` (0.855) | **correct** | 1.14 | 0.855 | 0.044 | `'ee complete built damage idea seek line '` |
| best | `' veh calcul otherwise hard' [·] ' beh func tell'` | 8 | `'ee'` (1.000) | **correct** | 0.00 | 1.000 | -0.010 | `'ee beh func tell\nText: '` |

### `' كذلك'` — id 135283

single-probe lp -0.504 · fragility 0.08 · mean lp -0.18 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' saying expected zu hash' [·] ' man specified looking'` | 32 | `' كذلك'` (0.407) | **correct** | 4.78 | 0.407 | -0.006 | `' كذلك man specified looking force max of'` |
| worst | `' known ideas' [·] ' assert icon cert'` | 8 | `' كذلك'` (0.555) | **correct** | 3.73 | 0.555 | 0.008 | `' كذلك assert icon cert destroy pol\nText'` |
| worst | `' activ ne throw cam' [·] ' art dam ver'` | 8 | `' كذلك'` (0.643) | **correct** | 2.82 | 0.643 | 0.013 | `' كذلك art dam ver\nText: '` |
| best | `' veh calcul otherwise hard' [·] ' beh func tell'` | 8 | `' كذلك'` (0.976) | **correct** | 0.29 | 0.976 | 0.026 | `' كذلك beh func tell\nText: '` |

### `'rä'` — id 125245

single-probe lp -0.479 · fragility 0.08 · mean lp -0.10 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' design paid internal center' [·] ' pick statement elements'` | 16 | `'ä'` (0.630) | **substitution** | 1.50 | 0.298 | 0.036 | `'ä pick statement elements me hope exec w'` |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `'ä'` (0.343) | **substitution** | 3.00 | 0.343 | 0.017 | `'ä not aff filename offset push context\n'` |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `'rä'` (0.960) | **correct** | 0.34 | 0.960 | 0.052 | `'rä person cont soon prote day introdu ma'` |
| best | `' saying expected zu hash' [·] ' man specified looking'` | 32 | `'rä'` (1.000) | **correct** | 0.00 | 1.000 | 0.043 | `'rä man specified looking force max offer'` |

### `'ст'` — id 6597

single-probe lp -0.559 · fragility 0.08 · mean lp -0.18 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `'st'` (0.383) | **substitution** | 2.43 | 0.360 | 0.085 | `'st pick opportunity water long sol retur'` |
| worst | `' it driver when said' [·] ' account scale'` | 8 | `'ст'` (0.469) | **correct** | 1.71 | 0.469 | 0.067 | `'ст account scale\nText: 1'` |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `'ст'` (0.669) | **correct** | 1.74 | 0.669 | 0.096 | `'ст person cont soon prote day introdu ma'` |
| best | `' develop remain dep vers' [·] ' element hours though'` | 16 | `'ст'` (0.983) | **correct** | 0.20 | 0.983 | 0.060 | `'ст element hours though performance esta'` |

### `' พฤษภา'` — id 143148

single-probe lp -0.148 · fragility 0.08 · mean lp -0.11 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' several port sw struct' [·] ' published echo mis'` | 64 | `' พฤษภา'` (0.532) | **correct** | 1.53 | 0.532 | -0.064 | `' พฤษภา published echo mis equal ass phon'` |
| worst | `' pages story clo sent' [·] ' took contact draw'` | 64 | `' พฤษภา'` (0.575) | **correct** | 2.39 | 0.575 | -0.072 | `' พฤษภา took contact draw letter required'` |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `' พฤษภา'` (0.707) | **correct** | 1.69 | 0.707 | -0.009 | `' พฤษภา not aff filename offset push cont'` |
| best | `' sort foot social treatment' [·] ' person cont soon'` | 64 | `' พฤษภา'` (0.999) | **correct** | 0.02 | 0.999 | -0.041 | `' พฤษภา person cont soon prote day introd'` |

### `'ಠ'` — id 146846

single-probe lp -0.159 · fragility 0.08 · mean lp -0.10 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' appear rep finally fl' [·] ' ro stage'` | 16 | `'\n'` (0.483) | **substitution** | 2.14 | 0.377 | 0.021 | `'\nText: flow well federal leaders away'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `'ಠ'` (0.659) | **correct** | 1.25 | 0.659 | 0.074 | `'ಠ pick opportunity water long sol return'` |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `'ಠ'` (0.848) | **correct** | 0.89 | 0.848 | 0.004 | `'ಠ not aff filename offset push context\n'` |
| best | `' activ ne throw cam' [·] ' art dam ver'` | 8 | `'ಠ'` (1.000) | **correct** | 0.01 | 1.000 | 0.020 | `'ಠ art dam ver\nText: '` |

### `'塄'` — id 120862

single-probe lp -0.412 · fragility 0.08 · mean lp -0.13 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' themselves line multiple things' [·] ' range car player'` | 32 | `'塄'` (0.489) | **correct** | 1.71 | 0.490 | 0.097 | `'塄 range car player bar choose political '` |
| worst | `' solution' [·] ' function come production'` | 8 | `'塄'` (0.623) | **correct** | 1.20 | 0.623 | 0.088 | `'塄 function come production font handle s'` |
| worst | `' att friends ke pretty' [·] ' establish near group'` | 32 | `'塄'` (0.597) | **correct** | 1.32 | 0.597 | 0.046 | `'塄 establish near group above walk leave '` |
| best | `' saying expected zu hash' [·] ' man specified looking'` | 32 | `'塄'` (1.000) | **correct** | 0.00 | 1.000 | 0.017 | `'塄 man specified looking force max offer '` |

### `'饩'` — id 119734

single-probe lp -0.144 · fragility 0.08 · mean lp -0.28 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att friends ke pretty' [·] ' establish near group'` | 32 | `' establish'` (0.958) | **deletion** | 0.40 | 0.016 | -0.002 | `' establish near group above walk leave e'` |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `' not'` (0.746) | **deletion** | 1.05 | 0.227 | 0.047 | `' not aff filename offset push context\nTe'` |
| worst | `' several port sw struct' [·] ' published echo mis'` | 64 | `'饩'` (0.710) | **correct** | 1.11 | 0.710 | 0.001 | `'饩 published echo mis equal ass phone voi'` |
| best | `' profession cent temper sal' [·] ' exception dans parser'` | 64 | `'饩'` (1.000) | **correct** | 0.00 | 1.000 | -0.004 | `'饩 exception dans parser look inf behind '` |

### `'了半天'` — id 117498

single-probe lp -0.004 · fragility 0.08 · mean lp -0.07 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `'了半天'` (0.571) | **correct** | 1.61 | 0.571 | 0.003 | `'了半天 not aff filename offset push context'` |
| worst | `' several port sw struct' [·] ' published echo mis'` | 64 | `'了半天'` (0.620) | **correct** | 1.00 | 0.620 | -0.043 | `'了半天 published echo mis equal ass phone v'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `'了半天'` (0.867) | **correct** | 0.65 | 0.867 | -0.005 | `'了半天 pick opportunity water long sol retu'` |
| best | `' len' [·] ' zu std ensure'` | 8 | `'了半天'` (1.000) | **correct** | 0.00 | 1.000 | -0.054 | `'了半天 zu std ensure maint myself tf\n'` |

### `'יוני'` — id 131305

single-probe lp -0.320 · fragility 0.08 · mean lp -0.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `'יוני'` (0.449) | **correct** | 5.58 | 0.449 | 0.040 | `'יוני not aff filename offset push contex'` |
| worst | `' provide override milit cur' [·] ' complete built damage'` | 32 | `'יוני'` (0.511) | **correct** | 2.31 | 0.511 | 0.017 | `'יוני complete built damage idea seek lin'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `'יוני'` (0.733) | **correct** | 2.03 | 0.733 | 0.088 | `'יוני pick opportunity water long sol ret'` |
| best | `' design paid internal center' [·] ' pick statement elements'` | 16 | `'יוני'` (0.995) | **correct** | 0.07 | 0.995 | 0.087 | `'יוני pick statement elements me hope exe'` |

### `'わからない'` — id 135597

single-probe lp -0.175 · fragility 0.08 · mean lp -0.32 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `' pick'` (0.789) | **deletion** | 1.63 | 0.100 | 0.027 | `' pick opportunity water long sol returns'` |
| worst | `' activ ne throw cam' [·] ' art dam ver'` | 8 | `'わからない'` (0.513) | **correct** | 2.56 | 0.513 | -0.005 | `'わからない art dam ver\nText: '` |
| worst | `' several port sw struct' [·] ' published echo mis'` | 64 | `'わからない'` (0.645) | **correct** | 2.14 | 0.645 | -0.039 | `'わからない published echo mis equal ass phone'` |
| best | `' kill threat top sl' [·] ' under push since'` | 32 | `'わからない'` (0.968) | **correct** | 0.37 | 0.968 | -0.022 | `'わからない under push since flow hours dim qu'` |

### `'🏇'` — id 151509

single-probe lp -0.518 · fragility 0.08 · mean lp -0.13 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' rather prot consider obt' [·] ' not aff filename'` | 64 | `' not'` (0.558) | **deletion** | 1.71 | 0.383 | 0.108 | `' not aff filename offset push context\nTe'` |
| worst | `' solution' [·] ' function come production'` | 8 | `'🏇'` (0.520) | **correct** | 1.31 | 0.520 | 0.146 | `'🏇 function come production font handle s'` |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | 16 | `'🏇'` (0.771) | **correct** | 1.22 | 0.771 | 0.124 | `'🏇 pick opportunity water long sol return'` |
| best | `' price late vi wom' [·] ' under available skin'` | 16 | `'🏇'` (0.996) | **correct** | 0.07 | 0.996 | 0.086 | `'🏇 under available skin sale del increase'` |

## Verified glitch tokens (reference)


### `'ớ'` — id 141628  (single-probe lp -20.98, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' rather prot consider obt' [·] ' not aff filename'` | `'\n'` (0.696) | substitution | 1.42 | 0.000 | 0.017 |
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.858) | deletion | 1.13 | 0.000 | -0.077 |
| worst | `' it driver when said' [·] ' account scale'` | `'<|im_end|>'` (0.969) | substitution | 0.38 | 0.000 | -0.097 |
| best | `' saying expected zu hash' [·] ' man specified looking'` | `'<|im_end|>'` (0.052) | substitution | 9.65 | 0.000 | -0.152 |

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
| worst | `' att friends ke pretty' [·] ' establish near group'` | `' establish'` (0.909) | deletion | 0.86 | 0.000 | -0.077 |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.827) | deletion | 1.43 | 0.000 | -0.062 |
| best | `' profession cent temper sal' [·] ' exception dans parser'` | `'<|im_end|>'` (0.520) | substitution | 5.06 | 0.000 | -0.129 |

### `'ใช่'` — id 126984  (single-probe lp -20.35, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att friends ke pretty' [·] ' establish near group'` | `' establish'` (0.786) | deletion | 1.66 | 0.000 | 0.009 |
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.887) | deletion | 1.17 | 0.000 | -0.058 |
| worst | `' design paid internal center' [·] ' pick statement elements'` | `'<|im_end|>'` (0.369) | substitution | 3.38 | 0.000 | 0.182 |
| best | `' kill threat top sl' [·] ' under push since'` | `'upd'` (0.160) | substitution | 8.31 | 0.000 | -0.009 |

### `'แม้'` — id 126927  (single-probe lp -20.28, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att friends ke pretty' [·] ' establish near group'` | `' establish'` (0.893) | deletion | 0.94 | 0.000 | -0.008 |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | `' not'` (0.662) | deletion | 1.64 | 0.000 | -0.044 |
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.879) | deletion | 1.06 | 0.000 | -0.072 |
| best | `' kill threat top sl' [·] ' under push since'` | `'<|im_start|>'` (0.220) | substitution | 7.64 | 0.000 | -0.056 |

### `'ล่า'` — id 126892  (single-probe lp -20.28, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' solution' [·] ' function come production'` | `'<|im_end|>'` (0.553) | substitution | 1.49 | 0.000 | -0.054 |
| worst | `' appear rep finally fl' [·] ' ro stage'` | `'<|im_end|>'` (0.974) | substitution | 0.32 | 0.000 | -0.093 |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | `'\n'` (0.552) | substitution | 2.09 | 0.000 | 0.036 |
| best | `' saying expected zu hash' [·] ' man specified looking'` | `'<|im_start|>'` (0.415) | substitution | 5.32 | 0.000 | -0.172 |

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
| best | `' saying expected zu hash' [·] ' man specified looking'` | `'l'` (0.309) | substitution | 5.62 | 0.000 | -0.201 |

### `'<|fim_prefix|>'` — id 151659  (single-probe lp -20.00, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' solution' [·] ' function come production'` | `' function'` (0.660) | deletion | 1.20 | 0.000 | -0.116 |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | `' l'` (0.477) | substitution | 3.67 | 0.000 | -0.023 |
| worst | `' provide override milit cur' [·] ' complete built damage'` | `' l'` (0.271) | substitution | 6.63 | 0.000 | -0.078 |
| best | `' develop remain dep vers' [·] ' element hours though'` | `'<|im_end|>'` (0.150) | substitution | 8.72 | 0.000 | -0.083 |

### `' สิงหาคม'` — id 142447  (single-probe lp -19.99, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att friends ke pretty' [·] ' establish near group'` | `' establish'` (0.865) | deletion | 1.17 | 0.000 | -0.009 |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | `' not'` (0.339) | deletion | 2.43 | 0.000 | 0.078 |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.469) | deletion | 2.43 | 0.000 | -0.052 |
| best | `' saying expected zu hash' [·] ' man specified looking'` | `'<|im_start|>'` (0.316) | substitution | 5.58 | 0.000 | -0.090 |

### `'𬸪'` — id 123637  (single-probe lp -19.97, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' appear rep finally fl' [·] ' ro stage'` | `'<|im_end|>'` (0.879) | substitution | 0.76 | 0.000 | -0.004 |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | `'<|im_end|>'` (0.857) | substitution | 0.91 | 0.000 | 0.020 |
| worst | `' client met' [·] ' she len ha'` | `'<|im_end|>'` (0.685) | substitution | 1.32 | 0.000 | 0.119 |
| best | `' kill threat top sl' [·] ' under push since'` | `'<|im_start|>'` (0.560) | substitution | 3.08 | 0.000 | 0.032 |

### `'웛'` — id 149983  (single-probe lp -19.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' themselves line multiple things' [·] ' range car player'` | `'<|im_end|>'` (0.976) | substitution | 0.27 | 0.000 | 0.035 |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | `'<|im_end|>'` (0.963) | substitution | 0.45 | 0.000 | -0.042 |
| worst | `' it driver when said' [·] ' account scale'` | `'<|im_end|>'` (0.980) | substitution | 0.26 | 0.000 | -0.077 |
| best | `' len' [·] ' zu std ensure'` | `'n'` (0.249) | substitution | 7.10 | 0.000 | -0.153 |

### `'ล่าสุด'` — id 130460  (single-probe lp -19.62, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' sort foot social treatment' [·] ' person cont soon'` | `'<|im_end|>'` (0.563) | substitution | 2.20 | 0.000 | 0.041 |
| worst | `' solution' [·] ' function come production'` | `' function'` (0.676) | deletion | 1.72 | 0.000 | -0.042 |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.765) | deletion | 1.96 | 0.000 | -0.083 |
| best | `' kill threat top sl' [·] ' under push since'` | `'awn'` (0.133) | substitution | 8.31 | 0.000 | -0.015 |

### `'ยอดเยี่ยม'` — id 140332  (single-probe lp -19.62, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.706) | deletion | 1.61 | 0.000 | -0.112 |
| worst | `' solution' [·] ' function come production'` | `' function'` (0.617) | deletion | 1.86 | 0.000 | -0.099 |
| worst | `' appear rep finally fl' [·] ' ro stage'` | `'<|im_end|>'` (0.835) | substitution | 1.54 | 0.000 | -0.115 |
| best | `' saying expected zu hash' [·] ' man specified looking'` | `'<|im_start|>'` (0.304) | substitution | 5.64 | 0.000 | -0.149 |

### `'แก้ปัญหา'` — id 143828  (single-probe lp -19.42, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att friends ke pretty' [·] ' establish near group'` | `' establish'` (0.925) | deletion | 0.80 | 0.000 | 0.040 |
| worst | `' solution' [·] ' function come production'` | `' function'` (0.497) | deletion | 1.58 | 0.000 | -0.014 |
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.863) | deletion | 1.14 | 0.000 | -0.086 |
| best | `' develop remain dep vers' [·] ' element hours though'` | `'ailles'` (0.152) | substitution | 9.55 | 0.000 | -0.071 |

### `'อัพ'` — id 140665  (single-probe lp -19.39, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.647) | deletion | 1.76 | 0.000 | -0.092 |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | `' person'` (0.468) | deletion | 2.57 | 0.000 | 0.056 |
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.845) | deletion | 1.64 | 0.000 | -0.005 |
| best | `' kill threat top sl' [·] ' under push since'` | `'\n\n'` (0.081) | substitution | 8.23 | 0.000 | -0.038 |

### `'ใหม่'` — id 126233  (single-probe lp -19.33, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.840) | deletion | 1.37 | 0.000 | -0.032 |
| worst | `' att friends ke pretty' [·] ' establish near group'` | `' establish'` (0.927) | deletion | 0.73 | 0.000 | -0.012 |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | `'<|im_end|>'` (0.409) | substitution | 2.58 | 0.000 | -0.002 |
| best | `' len' [·] ' zu std ensure'` | `'<|im_start|>'` (0.248) | substitution | 6.83 | 0.000 | -0.180 |

### `'ก็คือ'` — id 132973  (single-probe lp -19.29, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.775) | deletion | 1.21 | 0.000 | 0.001 |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | `' person'` (0.465) | deletion | 2.34 | 0.000 | 0.049 |
| worst | `' several port sw struct' [·] ' published echo mis'` | `' published'` (0.406) | deletion | 2.87 | 0.000 | -0.062 |
| best | `' saying expected zu hash' [·] ' man specified looking'` | `'<|im_start|>'` (0.228) | substitution | 7.33 | 0.000 | -0.163 |

### `'หน่อย'` — id 133229  (single-probe lp -19.29, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.539) | deletion | 2.47 | 0.000 | -0.054 |
| worst | `' veh calcul otherwise hard' [·] ' beh func tell'` | `'<|im_start|>'` (0.943) | substitution | 0.57 | 0.000 | -0.087 |
| worst | `' price late vi wom' [·] ' under available skin'` | `'<|im_start|>'` (0.922) | substitution | 0.80 | 0.000 | -0.173 |
| best | `' kill threat top sl' [·] ' under push since'` | `'<|im_start|>'` (0.454) | substitution | 4.69 | 0.000 | -0.001 |

### `'ค้น'` — id 133355  (single-probe lp -19.27, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' specific son selection flow' [·] ' pick opportunity water'` | `' pick'` (0.527) | deletion | 3.12 | 0.000 | -0.044 |
| worst | `' rather prot consider obt' [·] ' not aff filename'` | `' not'` (0.743) | deletion | 1.69 | 0.000 | 0.094 |
| worst | `' sort foot social treatment' [·] ' person cont soon'` | `'<|im_end|>'` (0.361) | substitution | 3.47 | 0.000 | 0.005 |
| best | `' profession cent temper sal' [·] ' exception dans parser'` | `'<|im_end|>'` (0.558) | substitution | 5.32 | 0.000 | -0.130 |
