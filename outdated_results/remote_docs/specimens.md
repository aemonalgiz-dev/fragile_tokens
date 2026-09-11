# Confident-substitution specimens

Every example in this file was produced by greedy decoding; `p` is the model's probability for the token it emitted, `H` the entropy in bits at that position, `p(target)` its probability for the correct token, `ret` the cosine between the last-layer slot state and the correct token's unembedding row (identity still pointing at its readout while the readout chose otherwise).

## Model

- **model**: `allenai/OLMo-2-1124-7B-Instruct`
- **commit**: `470b1fba1ae01581f270116362ee4aa1b97f4c84`
- **dtype**: `torch.float16`
- **num_layers**: `32`
- **hidden_size**: `4096`
- **vocab_size**: `100352`
- **tie_word_embeddings**: `False`
- **tokenizer_class**: `TokenizersBackend`
- **architecture**: `Olmo2ForCausalLM`
- **torch**: `2.7.0`
- **transformers**: `5.16.1`
- **python**: `3.10.12`
- **device**: `cuda:0`
- **decoding**: `greedy`
- **prompt_head**: `Repeat the text exactly.\nText: apple pie is good\nCopy: apple pie is good\nText: the quick brown fox\nCopy: the quick brown fox\nText:<context>\nCopy:<context-prefix>`
- **fail_threshold_lp**: `-0.5`
- **n_contexts**: `24`

## Fragile clean-looking tokens (pass the single probe; fail in some contexts)


### `' according'` — id 4184

single-probe lp -0.001 · fragility 0.92 · mean lp -13.26 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' work kn she qual' [·] ' window qu drive'` | 16 | `' window'` (1.000) | **deletion** | 0.00 | 0.000 | -0.082 | `' window qu drive area road videos'` |
| worst | `' idx games ful parse' [·] ' begin states mar'` | 32 | `' begin'` (1.000) | **deletion** | 0.00 | 0.000 | -0.031 | `' begin states mar comm mean open att use'` |
| worst | `' everyone towards find' [·] ' track cover look'` | 8 | `' track'` (1.000) | **deletion** | 0.00 | 0.000 | -0.053 | `' track cover look laws'` |
| best | `' what rev hom draw' [·] ' body equ ensure'` | 32 | `' according'` (1.000) | **correct** | 0.00 | 1.000 | -0.079 | `' according body equ ensure mill section '` |

### `' который'` — id 98851

single-probe lp -0.017 · fragility 0.71 · mean lp -0.98 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' anything whether more redu' [·] ' was attr load'` | 16 | `' что'` (0.074) | **substitution** | 10.01 | 0.042 | -0.047 | `' что был attr load starting si cre'` |
| worst | `' large dif signific while' [·] ' money tre express'` | 64 | `' который'` (0.089) | **correct** | 9.24 | 0.089 | -0.096 | `' который money tre express st found cont'` |
| worst | `' env regular performance pos' [·] ' phot to evidence'` | 64 | `' который'` (0.154) | **correct** | 9.10 | 0.154 | -0.020 | `' который phot to evidence attr stuff po '` |
| best | `' tmp' [·] ' les related ep'` | 8 | `' который'` (0.936) | **correct** | 0.54 | 0.936 | 0.003 | `' который les related ep poss sec phone'` |

### `' única'` — id 89272

single-probe lp -0.024 · fragility 0.54 · mean lp -0.77 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' '` (0.259) | **substitution** | 4.37 | 0.053 | -0.084 | `'  ú nica inv bet deg int'` |
| worst | `' quality etc resources mut' [·] ' inst lot filter'` | 64 | `' ú'` (0.347) | **truncation** | 5.22 | 0.115 | -0.042 | `' únicamente inst lot filter belie decide'` |
| worst | `' grid logger conduct members' [·] ' cam title occ'` | 8 | `' ú'` (0.487) | **truncation** | 2.34 | 0.182 | -0.061 | `' únicam cam title occ'` |
| best | `' tmp' [·] ' les related ep'` | 8 | `' única'` (0.935) | **correct** | 0.70 | 0.935 | -0.012 | `' única les related ep poss sec phone'` |

### `' 查询'` — id 81628

single-probe lp -0.015 · fragility 0.46 · mean lp -0.95 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' QUERY'` (0.970) | **substitution** | 0.40 | 0.000 | -0.129 | `' QUERY look consider dest im as politica'` |
| worst | `' du generate career endl' [·] ' cross cool during'` | 32 | `' QUERY'` (0.785) | **substitution** | 1.98 | 0.085 | -0.091 | `' QUERY cross cool during massage especia'` |
| worst | `' work kn she qual' [·] ' window qu drive'` | 16 | `' QUERY'` (0.449) | **substitution** | 3.50 | 0.309 | -0.100 | `' QUERY window qu drive area road videos'` |
| best | `' tmp' [·] ' les related ep'` | 8 | `' 查询'` (0.877) | **correct** | 1.21 | 0.877 | 0.001 | `' 查询 les related ep poss sec phone'` |

### `' tienen'` — id 47539

single-probe lp -0.028 · fragility 0.46 · mean lp -0.62 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' potential loss has class' [·] ' edit birth gold'` | 64 | `' tienen'` (0.176) | **correct** | 7.20 | 0.176 | -0.086 | `' tienen edit birth gold adult wer super '` |
| worst | `' simply beg returned candid' [·] ' kind'` | 8 | `' tienen'` (0.219) | **correct** | 6.16 | 0.219 | -0.042 | `' tienen kind'` |
| worst | `' wom financial cout under' [·] ' false told super'` | 64 | `' tienen'` (0.222) | **correct** | 7.41 | 0.222 | -0.078 | `' tienen false told super des home safe p'` |
| best | `' car meet son probably' [·] ' machine contrib management'` | 16 | `' tienen'` (0.956) | **correct** | 0.48 | 0.956 | -0.081 | `' tienen machine contrib management'` |

### `' due'` — id 4245

single-probe lp -0.000 · fragility 0.42 · mean lp -7.51 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' large dif signific while' [·] ' money tre express'` | 64 | `' money'` (1.000) | **deletion** | 0.00 | 0.000 | -0.085 | `' money tre express st found context late'` |
| worst | `' idx games ful parse' [·] ' begin states mar'` | 32 | `' begin'` (1.000) | **deletion** | 0.00 | 0.000 | -0.065 | `' begin states mar comm mean open att use'` |
| worst | `' limited entry happy president' [·] ' comes website failed'` | 32 | `' comes'` (1.000) | **deletion** | 0.00 | 0.000 | -0.063 | `' comes website failed ps very all coll r'` |
| best | `' what rev hom draw' [·] ' body equ ensure'` | 32 | `' due'` (1.000) | **correct** | 0.00 | 1.000 | -0.093 | `' due body equ ensure mill section auf is'` |

### `' allerdings'` — id 86825

single-probe lp -0.092 · fragility 0.42 · mean lp -0.57 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' however'` (0.155) | **substitution** | 5.34 | 0.124 | -0.101 | `' however inv bet deg int part named ein'` |
| worst | `' wom financial cout under' [·] ' false told super'` | 64 | `' allerdings'` (0.168) | **correct** | 6.70 | 0.168 | -0.148 | `' allerdings false told super des home sa'` |
| worst | `' anything whether more redu' [·] ' was attr load'` | 16 | `' allerdings'` (0.310) | **correct** | 5.76 | 0.310 | -0.128 | `' allerdings was attr load starting si cr'` |
| best | `' what rev hom draw' [·] ' body equ ensure'` | 32 | `' allerdings'` (0.885) | **correct** | 1.17 | 0.885 | -0.134 | `' allerdings body equ ensure mill section'` |

### `' może'` — id 54621

single-probe lp -0.010 · fragility 0.42 · mean lp -0.51 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' char deter imp med' [·] ' road install aud'` | 16 | `' może'` (0.172) | **correct** | 7.82 | 0.172 | -0.034 | `' może road install aud text player carry'` |
| worst | `' large dif signific while' [·] ' money tre express'` | 64 | `' może'` (0.325) | **correct** | 4.95 | 0.325 | -0.045 | `' może money tre express st found context'` |
| worst | `' anything whether more redu' [·] ' was attr load'` | 16 | `' może'` (0.352) | **correct** | 6.60 | 0.352 | -0.026 | `' może was attr load starting si cre join'` |
| best | `' tmp' [·] ' les related ep'` | 8 | `' może'` (0.997) | **correct** | 0.05 | 0.997 | 0.006 | `' może les related ep poss sec phone'` |

### `' использов'` — id 89327

single-probe lp -0.007 · fragility 0.33 · mean lp -0.51 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' look'` (0.552) | **deletion** | 3.71 | 0.179 | -0.089 | `' look consider dest im as political func'` |
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' использов'` (0.241) | **correct** | 7.99 | 0.242 | -0.012 | `' использов inv bet deg int part named ei'` |
| worst | `' calls fac van' [·] ' lower takes padding'` | 32 | `' использов'` (0.345) | **correct** | 6.47 | 0.345 | -0.046 | `' использов lower takes padding stand equ'` |
| best | `' car meet son probably' [·] ' machine contrib management'` | 16 | `' использов'` (0.949) | **correct** | 0.66 | 0.949 | -0.071 | `' использов machine contrib management'` |

### `' strSQL'` — id 81253

single-probe lp -0.076 · fragility 0.29 · mean lp -0.48 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' SQL'` (0.682) | **substitution** | 2.77 | 0.046 | -0.133 | `' SQL look consider dest im as political '` |
| worst | `' everyone towards find' [·] ' track cover look'` | 8 | `' strSQL'` (0.214) | **correct** | 6.49 | 0.214 | -0.121 | `' strSQL track cover look laws'` |
| worst | `' anything whether more redu' [·] ' was attr load'` | 16 | `' SQL'` (0.244) | **substitution** | 5.32 | 0.233 | -0.089 | `' SQL was attr load starting si cre join'` |
| best | `' what rev hom draw' [·] ' body equ ensure'` | 32 | `' strSQL'` (0.965) | **correct** | 0.44 | 0.965 | -0.126 | `' strSQL body equ ensure mill section auf'` |

### `' setContentView'` — id 21793

single-probe lp -0.094 · fragility 0.25 · mean lp -0.38 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' setContentView'` (0.224) | **correct** | 7.32 | 0.224 | -0.047 | `' setContentView inv bet deg int part nam'` |
| worst | `' potential loss has class' [·] ' edit birth gold'` | 64 | `' setContentView'` (0.456) | **correct** | 6.49 | 0.456 | -0.060 | `' setContentView edit birth gold adult we'` |
| worst | `' quality etc resources mut' [·] ' inst lot filter'` | 64 | `' setContentView'` (0.525) | **correct** | 5.88 | 0.525 | -0.034 | `' setContentView inst lot filter belie de'` |
| best | `' tmp' [·] ' les related ep'` | 8 | `' setContentView'` (0.949) | **correct** | 0.78 | 0.949 | 0.013 | `' setContentView les related ep poss sec '` |

### `' mentre'` — id 93119

single-probe lp -0.002 · fragility 0.25 · mean lp -0.44 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' char deter imp med' [·] ' road install aud'` | 16 | `' serta'` (0.333) | **substitution** | 6.69 | 0.131 | -0.097 | `' serta road install aud text player carr'` |
| worst | `' quality etc resources mut' [·] ' inst lot filter'` | 64 | `' serta'` (0.307) | **substitution** | 7.20 | 0.136 | -0.030 | `' serta inst lot filter belie decided hom'` |
| worst | `' calls fac van' [·] ' lower takes padding'` | 32 | `'vanished'` (0.277) | **substitution** | 5.88 | 0.214 | -0.016 | `'vanished lower takes padding stand equip'` |
| best | `' grid logger conduct members' [·] ' cam title occ'` | 8 | `' mentre'` (0.998) | **correct** | 0.04 | 0.998 | -0.036 | `' mentre cam title occ'` |

### `' UIResponder'` — id 69921

single-probe lp -0.064 · fragility 0.25 · mean lp -0.36 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' RESP'` (0.264) | **substitution** | 6.70 | 0.140 | -0.139 | `' RESPONDerto look consider dest im as'` |
| worst | `' anything whether more redu' [·] ' was attr load'` | 16 | `' UIResponder'` (0.365) | **correct** | 5.26 | 0.365 | -0.118 | `' UIResponder was attr load starting si c'` |
| worst | `' everyone towards find' [·] ' track cover look'` | 8 | `' UIResponder'` (0.378) | **correct** | 5.47 | 0.378 | -0.112 | `' UIResponder track cover look laws'` |
| best | `' potential loss has class' [·] ' edit birth gold'` | 64 | `' UIResponder'` (0.976) | **correct** | 0.36 | 0.976 | -0.120 | `' UIResponder edit birth gold adult wer s'` |

### `' функци'` — id 55813

single-probe lp -0.017 · fragility 0.25 · mean lp -0.51 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' fonction'` (0.844) | **substitution** | 1.53 | 0.024 | -0.025 | `' fonction inv bet deg int part named ein'` |
| worst | `' du generate career endl' [·] ' cross cool during'` | 32 | `' fonction'` (0.270) | **substitution** | 2.81 | 0.177 | -0.058 | `' fonction cross cool during massage espe'` |
| worst | `' wom financial cout under' [·] ' false told super'` | 64 | `' fonction'` (0.487) | **substitution** | 2.19 | 0.340 | 0.014 | `' fonction false told super des home safe'` |
| best | `' setting letter see body' [·] ' teach'` | 8 | `' функци'` (0.972) | **correct** | 0.31 | 0.972 | 0.006 | `' функци teach'` |

### `' erklä'` — id 87207

single-probe lp -0.010 · fragility 0.25 · mean lp -0.42 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' '` (0.669) | **substitution** | 3.41 | 0.091 | -0.011 | `"  '?' inv bet deg int part named"` |
| worst | `' grid logger conduct members' [·] ' cam title occ'` | 8 | `' '` (0.623) | **substitution** | 2.18 | 0.248 | -0.006 | `'  ÜH cam title occ'` |
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' erklä'` (0.327) | **correct** | 5.58 | 0.327 | -0.067 | `' erklä look consider dest im as politica'` |
| best | `' tmp' [·] ' les related ep'` | 8 | `' erklä'` (0.998) | **correct** | 0.03 | 0.998 | 0.038 | `' erklä les related ep poss sec phone'` |

### `' WebDriverWait'` — id 64746

single-probe lp -0.013 · fragility 0.21 · mean lp -0.30 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' WAIT'` (0.681) | **substitution** | 2.36 | 0.180 | -0.089 | `" WAITING for element with id 'dest"` |
| worst | `' tmp' [·] ' les related ep'` | 8 | `' WAIT'` (0.478) | **substitution** | 2.39 | 0.419 | -0.059 | `' WAITING for related ep poss sec phone'` |
| worst | `' du generate career endl' [·] ' cross cool during'` | 32 | `' WebDriverWait'` (0.477) | **correct** | 2.39 | 0.477 | -0.097 | `' WebDriverWait cross cool during massage'` |
| best | `' grid logger conduct members' [·] ' cam title occ'` | 8 | `' WebDriverWait'` (0.980) | **correct** | 0.25 | 0.980 | -0.058 | `' WebDriverWait cam title occ'` |

### `' UIImagePickerController'` — id 78833

single-probe lp -0.007 · fragility 0.17 · mean lp -0.23 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' limited entry happy president' [·] ' comes website failed'` | 32 | `'UIImagePickerController'` (0.627) | **substitution** | 2.56 | 0.247 | -0.096 | `'UIImagePickerController comes website fa'` |
| worst | `' work kn she qual' [·] ' window qu drive'` | 16 | `' UIImagePickerController'` (0.542) | **correct** | 1.35 | 0.542 | -0.113 | `' UIImagePickerController window qu drive'` |
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' UIImagePickerController'` (0.574) | **correct** | 4.05 | 0.574 | -0.093 | `' UIImagePickerController inv bet deg int'` |
| best | `' everyone towards find' [·] ' track cover look'` | 8 | `' UIImagePickerController'` (0.986) | **correct** | 0.17 | 0.986 | -0.098 | `' UIImagePickerController track cover loo'` |

### `' enthus'` — id 16676

single-probe lp -0.039 · fragility 0.17 · mean lp -0.21 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' idx games ful parse' [·] ' begin states mar'` | 32 | `' enthusi'` (0.634) | **substitution** | 1.00 | 0.361 | -0.047 | `' enthusi begin states mar comm mean open'` |
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' enthusi'` (0.478) | **substitution** | 1.42 | 0.459 | -0.021 | `' enthusi inv bet deg int part named ein'` |
| worst | `' everyone towards find' [·] ' track cover look'` | 8 | `' enthus'` (0.516) | **correct** | 1.24 | 0.516 | -0.058 | `' enthus track cover look laws'` |
| best | `' limited entry happy president' [·] ' comes website failed'` | 32 | `' enthus'` (0.973) | **correct** | 0.20 | 0.973 | -0.056 | `' enthus comes website failed ps very all'` |

### `' tabBar'` — id 77881

single-probe lp -0.018 · fragility 0.17 · mean lp -0.28 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' TAB'` (0.614) | **substitution** | 2.13 | 0.226 | -0.131 | `' TABLE look consider dest im as politica'` |
| worst | `' limited entry happy president' [·] ' comes website failed'` | 32 | `' tabBar'` (0.330) | **correct** | 2.65 | 0.330 | -0.128 | `' tabBar comes website failed ps very all'` |
| worst | `' car meet son probably' [·] ' machine contrib management'` | 16 | `'TAB'` (0.403) | **substitution** | 2.14 | 0.390 | -0.128 | `'TABmachine contrib management'` |
| best | `' include hum label night' [·] ' js amount sen'` | 16 | `' tabBar'` (0.991) | **correct** | 0.12 | 0.991 | -0.118 | `' tabBar js amount sen anything ago requi'` |

### `' cuent'` — id 88338

single-probe lp -0.062 · fragility 0.17 · mean lp -0.30 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' cu'` (0.353) | **truncation** | 4.25 | 0.231 | -0.045 | `' cuente look consider dest im as politic'` |
| worst | `' calls fac van' [·] ' lower takes padding'` | 32 | `' cuent'` (0.360) | **correct** | 4.79 | 0.360 | -0.066 | `' cuent lower takes padding stand equipme'` |
| worst | `' anything whether more redu' [·] ' was attr load'` | 16 | `' cuenta'` (0.397) | **substitution** | 2.35 | 0.373 | -0.078 | `' cuenta was attr load starting si cre jo'` |
| best | `' wom financial cout under' [·] ' false told super'` | 64 | `' cuent'` (0.977) | **correct** | 0.23 | 0.977 | -0.025 | `' cuent false told super des home safe pr'` |

### `' страниц'` — id 62580

single-probe lp -0.021 · fragility 0.12 · mean lp -0.19 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' pá'` (0.423) | **substitution** | 2.72 | 0.209 | -0.050 | `' páage inv bet deg int part named'` |
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' страниц'` (0.485) | **correct** | 2.65 | 0.485 | -0.043 | `' страниц look consider dest im as politi'` |
| worst | `' work kn she qual' [·] ' window qu drive'` | 16 | `' страниц'` (0.540) | **correct** | 2.14 | 0.541 | -0.014 | `' страниц window qu drive area road video'` |
| best | `' calls fac van' [·] ' lower takes padding'` | 32 | `' страниц'` (0.993) | **correct** | 0.08 | 0.993 | -0.030 | `' страниц lower takes padding stand equip'` |

### `' gdzie'` — id 96591

single-probe lp -0.012 · fragility 0.12 · mean lp -0.25 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' gdzie'` (0.320) | **correct** | 4.86 | 0.320 | -0.044 | `' gdzie look consider dest im as politica'` |
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' gdzie'` (0.337) | **correct** | 6.47 | 0.336 | 0.029 | `' gdzie inv bet deg int part named ein'` |
| worst | `' large dif signific while' [·] ' money tre express'` | 64 | `' gdzie'` (0.497) | **correct** | 4.53 | 0.497 | -0.101 | `' gdzie money tre express st found contex'` |
| best | `' include hum label night' [·] ' js amount sen'` | 16 | `' gdzie'` (0.987) | **correct** | 0.19 | 0.987 | 0.005 | `' gdzie js amount sen anything ago requir'` |

### `' .'` — id 662

single-probe lp -0.089 · fragility 0.12 · mean lp -0.56 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' look'` (0.997) | **deletion** | 0.04 | 0.000 | -0.036 | `' look consider dest im as political func'` |
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' inv'` (0.922) | **deletion** | 0.56 | 0.050 | -0.018 | `' inv bet deg int part named ein undefine'` |
| worst | `' simply beg returned candid' [·] ' kind'` | 8 | `' .'` (0.579) | **correct** | 1.55 | 0.579 | -0.018 | `' . kind'` |
| best | `' du generate career endl' [·] ' cross cool during'` | 32 | `' .'` (1.000) | **correct** | 0.00 | 1.000 | -0.045 | `' . cross cool during massage especially '` |

### `' indeb'` — id 90897

single-probe lp -0.023 · fragility 0.12 · mean lp -0.20 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' grid logger conduct members' [·] ' cam title occ'` | 8 | `' inde'` (0.525) | **truncation** | 1.90 | 0.302 | -0.078 | `' indebted cam title occ'` |
| worst | `' anything whether more redu' [·] ' was attr load'` | 16 | `' indeb'` (0.503) | **correct** | 2.08 | 0.503 | -0.092 | `' indeb was attr load starting si cre joi'` |
| worst | `' simply beg returned candid' [·] ' kind'` | 8 | `' indeb'` (0.567) | **correct** | 1.71 | 0.567 | -0.056 | `' indeb kind'` |
| best | `' calls fac van' [·] ' lower takes padding'` | 32 | `' indeb'` (0.994) | **correct** | 0.07 | 0.994 | -0.069 | `' indeb lower takes padding stand equipme'` |

### `' 사용'` — id 41820

single-probe lp -0.010 · fragility 0.12 · mean lp -0.22 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' 사용'` (0.204) | **correct** | 8.13 | 0.204 | -0.117 | `' 사용 look consider dest im as political f'` |
| worst | `' anything whether more redu' [·] ' was attr load'` | 16 | `' 사용'` (0.482) | **correct** | 5.00 | 0.482 | -0.019 | `' 사용 was attr load starting si cre join'` |
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' 사용'` (0.579) | **correct** | 3.96 | 0.579 | -0.003 | `' 사용 inv bet deg int part named ein'` |
| best | `' setting letter see body' [·] ' teach'` | 8 | `' 사용'` (0.987) | **correct** | 0.20 | 0.987 | 0.034 | `' 사용 teach'` |

### `' memiliki'` — id 72236

single-probe lp -0.007 · fragility 0.12 · mean lp -0.23 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' wom financial cout under' [·] ' false told super'` | 64 | `' memiliki'` (0.243) | **correct** | 5.73 | 0.244 | -0.071 | `' memiliki false told super des home safe'` |
| worst | `' du generate career endl' [·] ' cross cool during'` | 32 | `' memiliki'` (0.450) | **correct** | 4.94 | 0.450 | -0.057 | `' memiliki cross cool during massage espe'` |
| worst | `' work kn she qual' [·] ' window qu drive'` | 16 | `' memiliki'` (0.520) | **correct** | 4.36 | 0.520 | -0.059 | `' memiliki window qu drive area road vide'` |
| best | `' everyone towards find' [·] ' track cover look'` | 8 | `' memiliki'` (0.994) | **correct** | 0.09 | 0.994 | -0.068 | `' memiliki track cover look laws'` |

### `' imprimir'` — id 93606

single-probe lp -0.029 · fragility 0.12 · mean lp -0.31 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' impres'` (0.531) | **substitution** | 2.84 | 0.227 | -0.101 | `' impres sion look consider dest im as'` |
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' imprimir'` (0.333) | **correct** | 3.57 | 0.333 | 0.012 | `' imprimir inv bet deg int part named ein'` |
| worst | `' live' [·] ' init agre filter'` | 8 | `' imprimir'` (0.450) | **correct** | 2.59 | 0.450 | 0.003 | `' imprimir init agre filter background pr'` |
| best | `' tmp' [·] ' les related ep'` | 8 | `' imprimir'` (0.990) | **correct** | 0.12 | 0.989 | 0.007 | `' imprimir les related ep poss sec phone'` |

### `' veya'` — id 88059

single-probe lp -0.082 · fragility 0.12 · mean lp -0.25 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' veya'` (0.461) | **correct** | 4.50 | 0.461 | -0.078 | `' veya look consider dest im as political'` |
| worst | `' anything whether more redu' [·] ' was attr load'` | 16 | `' veya'` (0.469) | **correct** | 6.27 | 0.470 | -0.070 | `' veya was attr load starting si cre join'` |
| worst | `' limited entry happy president' [·] ' comes website failed'` | 32 | `' veya'` (0.488) | **correct** | 4.90 | 0.488 | -0.065 | `' veya comes website failed ps very all c'` |
| best | `' tmp' [·] ' les related ep'` | 8 | `' veya'` (0.997) | **correct** | 0.05 | 0.997 | -0.007 | `' veya les related ep poss sec phone'` |

### `' weakSelf'` — id 74859

single-probe lp -0.065 · fragility 0.12 · mean lp -0.24 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' weakSelf'` (0.493) | **correct** | 5.59 | 0.493 | -0.092 | `' weakSelf inv bet deg int part named ein'` |
| worst | `' limited entry happy president' [·] ' comes website failed'` | 32 | `' weakSelf'` (0.573) | **correct** | 5.14 | 0.573 | -0.103 | `' weakSelf comes website failed ps very a'` |
| worst | `' quality etc resources mut' [·] ' inst lot filter'` | 64 | `' weakSelf'` (0.595) | **correct** | 4.12 | 0.595 | -0.066 | `' weakSelf inst lot filter belie decided '` |
| best | `' include hum label night' [·] ' js amount sen'` | 16 | `' weakSelf'` (0.962) | **correct** | 0.59 | 0.962 | -0.116 | `' weakSelf js amount sen anything ago req'` |

### `' selbst'` — id 35642

single-probe lp -0.009 · fragility 0.08 · mean lp -0.16 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' char deter imp med' [·] ' road install aud'` | 16 | `' selbst'` (0.564) | **correct** | 2.92 | 0.564 | -0.054 | `' selbst road install aud text player car'` |
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' selbst'` (0.597) | **correct** | 2.04 | 0.597 | -0.061 | `' selbst inv bet deg int part named ein'` |
| worst | `' work kn she qual' [·] ' window qu drive'` | 16 | `' selbst'` (0.677) | **correct** | 2.46 | 0.677 | -0.041 | `' selbst window qu drive area road videos'` |
| best | `' tmp' [·] ' les related ep'` | 8 | `' selbst'` (0.988) | **correct** | 0.14 | 0.988 | 0.026 | `' selbst les related ep poss sec phone'` |

### `' aValue'` — id 91011

single-probe lp -0.023 · fragility 0.08 · mean lp -0.21 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' aValue'` (0.156) | **correct** | 6.18 | 0.156 | -0.083 | `' aValue inv bet deg int part named ein'` |
| worst | `' anything whether more redu' [·] ' was attr load'` | 16 | `' a'` (0.460) | **truncation** | 3.61 | 0.314 | -0.116 | `' aValue was attr load starting si cre'` |
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' aValue'` (0.702) | **correct** | 2.08 | 0.702 | -0.146 | `' aValue look consider dest im as politic'` |
| best | `' calls fac van' [·] ' lower takes padding'` | 32 | `' aValue'` (0.993) | **correct** | 0.13 | 0.993 | -0.119 | `' aValue lower takes padding stand equipm'` |

### `' сохран'` — id 93781

single-probe lp -0.002 · fragility 0.08 · mean lp -0.29 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' сохран'` (0.154) | **correct** | 7.87 | 0.154 | -0.035 | `' сохран inv bet deg int part named ein'` |
| worst | `' anything whether more redu' [·] ' was attr load'` | 16 | `' сохран'` (0.596) | **correct** | 4.94 | 0.596 | -0.049 | `' сохран was attr load starting si cre jo'` |
| worst | `' grid logger conduct members' [·] ' cam title occ'` | 8 | `' сохран'` (0.624) | **correct** | 3.62 | 0.624 | -0.062 | `' сохран cam title occ'` |
| best | `' tmp' [·] ' les related ep'` | 8 | `' сохран'` (0.990) | **correct** | 0.17 | 0.990 | -0.001 | `' сохран les related ep poss sec phone'` |

### `' Own'` — id 21189

single-probe lp -0.004 · fragility 0.08 · mean lp -0.49 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' they option grand own' [·] ' manufact here recogn'` | 32 | `' manufact'` (0.996) | **deletion** | 0.05 | 0.000 | -0.077 | `' manufact here recogn parts happened fil'` |
| worst | `' live' [·] ' init agre filter'` | 8 | `' Own'` (0.577) | **correct** | 1.02 | 0.577 | -0.072 | `' Own init agre filter background prob al'` |
| worst | `' everyone towards find' [·] ' track cover look'` | 8 | `' Own'` (0.934) | **correct** | 0.42 | 0.934 | -0.084 | `' Own track cover look laws'` |
| best | `' du generate career endl' [·] ' cross cool during'` | 32 | `' Own'` (0.998) | **correct** | 0.02 | 0.998 | -0.102 | `' Own cross cool during massage especiall'` |

### `' tête'` — id 83067

single-probe lp -0.015 · fragility 0.08 · mean lp -0.21 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pen night server comple' [·] ' inv bet deg'` | 64 | `' té'` (0.583) | **substitution** | 1.75 | 0.337 | -0.043 | `' té inv bet deg int part named ein'` |
| worst | `' include hum label night' [·] ' js amount sen'` | 16 | `' tête'` (0.443) | **correct** | 2.94 | 0.443 | -0.056 | `' tête js amount sen anything ago require'` |
| worst | `' work kn she qual' [·] ' window qu drive'` | 16 | `' tête'` (0.627) | **correct** | 2.93 | 0.627 | -0.021 | `' tête window qu drive area road videos'` |
| best | `' calls fac van' [·] ' lower takes padding'` | 32 | `' tête'` (0.997) | **correct** | 0.05 | 0.997 | -0.052 | `' tête lower takes padding stand equipmen'` |

### `' Gö'` — id 83246

single-probe lp -0.006 · fragility 0.08 · mean lp -0.11 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' tmp' [·] ' les related ep'` | 8 | `' Gö'` (0.446) | **correct** | 2.21 | 0.446 | -0.036 | `' Gö les related ep poss sec phone'` |
| worst | `' calls fac van' [·] ' lower takes padding'` | 32 | `' gö'` (0.510) | **substitution** | 1.11 | 0.480 | -0.043 | `' gö lower takes padding stand equipment '` |
| worst | `' grid logger conduct members' [·] ' cam title occ'` | 8 | `' Gö'` (0.745) | **correct** | 1.54 | 0.745 | -0.049 | `' Gö cam title occ'` |
| best | `' wom financial cout under' [·] ' false told super'` | 64 | `' Gö'` (0.999) | **correct** | 0.01 | 0.999 | -0.072 | `' Gö false told super des home safe pract'` |

### `' Ness'` — id 88602

single-probe lp -0.013 · fragility 0.08 · mean lp -0.12 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' include hum label night' [·] ' js amount sen'` | 16 | `' ness'` (0.656) | **substitution** | 0.94 | 0.343 | -0.063 | `' ness js amount sen anything ago require'` |
| worst | `' work kn she qual' [·] ' window qu drive'` | 16 | `' Ness'` (0.589) | **correct** | 0.98 | 0.589 | -0.038 | `' Ness window qu drive area road videos'` |
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' Ness'` (0.629) | **correct** | 0.96 | 0.629 | -0.069 | `' Ness look consider dest im as political'` |
| best | `' potential loss has class' [·] ' edit birth gold'` | 64 | `' Ness'` (0.999) | **correct** | 0.01 | 0.999 | -0.083 | `' Ness edit birth gold adult wer super po'` |

### `' mouseX'` — id 81990

single-probe lp -0.002 · fragility 0.08 · mean lp -0.10 · canonical True · Magikarp `OK`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' anything whether more redu' [·] ' was attr load'` | 16 | `' MOUSE'` (0.557) | **substitution** | 1.97 | 0.343 | -0.083 | `' MOUSEX was attr load starting si cre'` |
| worst | `' business comment began influ' [·] ' look consider dest'` | 16 | `' mouseX'` (0.563) | **correct** | 3.33 | 0.563 | -0.105 | `' mouseX look consider dest im as politic'` |
| worst | `' limited entry happy president' [·] ' comes website failed'` | 32 | `' mouseX'` (0.878) | **correct** | 0.90 | 0.878 | -0.081 | `' mouseX comes website failed ps very all'` |
| best | `' large dif signific while' [·] ' money tre express'` | 64 | `' mouseX'` (0.997) | **correct** | 0.04 | 0.997 | -0.063 | `' mouseX money tre express st found conte'` |

## Verified glitch tokens (reference)


### `' PodsDummy'` — id 71390  (single-probe lp -9.41, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | `' look'` (0.990) | deletion | 0.13 | 0.000 | -0.153 |
| worst | `' calls fac van' [·] ' lower takes padding'` | `' According'` (0.103) | substitution | 9.47 | 0.000 | -0.149 |
| worst | `' char deter imp med' [·] ' road install aud'` | `' th'` (0.037) | substitution | 8.84 | 0.000 | -0.179 |
| best | `' tmp' [·] ' les related ep'` | `'contador'` (0.069) | substitution | 11.24 | 0.000 | -0.121 |

### `'<|fim_prefix|>'` — id 100258  (single-probe lp -24.22, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | `' look'` (1.000) | deletion | 0.00 | 0.000 | -0.119 |
| worst | `' what rev hom draw' [·] ' body equ ensure'` | `' The'` (0.626) | substitution | 1.38 | 0.000 | -0.239 |
| worst | `' quality etc resources mut' [·] ' inst lot filter'` | `' inst'` (0.971) | deletion | 0.26 | 0.000 | -0.183 |
| best | `' live' [·] ' init agre filter'` | `' '` (0.309) | substitution | 6.94 | 0.000 | -0.120 |

### `'<|im_end|>'` — id 100265  (single-probe lp -24.58, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | `' look'` (0.991) | deletion | 0.12 | 0.000 | -0.141 |
| worst | `' tmp' [·] ' les related ep'` | `'$'` (0.985) | substitution | 0.20 | 0.000 | -0.164 |
| worst | `' large dif signific while' [·] ' money tre express'` | `'$'` (0.657) | substitution | 1.69 | 0.000 | -0.209 |
| best | `' char deter imp med' [·] ' road install aud'` | `'$.'` (0.223) | substitution | 7.30 | 0.000 | -0.197 |

### `' ForCanBeConvertedToForeach'` — id 80371  (single-probe lp -8.19, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | `' 示'` (0.308) | substitution | 7.49 | 0.000 | -0.193 |
| worst | `' pen night server comple' [·] ' inv bet deg'` | `'ccione'` (0.034) | substitution | 10.87 | 0.000 | -0.147 |
| worst | `' wom financial cout under' [·] ' false told super'` | `' incididunt'` (0.030) | substitution | 10.95 | 0.000 | -0.140 |
| best | `' setting letter see body' [·] ' teach'` | `' 示'` (0.089) | substitution | 10.95 | 0.000 | -0.147 |

### `'useRal'` — id 89471  (single-probe lp -23.64, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' business comment began influ' [·] ' look consider dest'` | `' look'` (0.997) | deletion | 0.03 | 0.000 | -0.164 |
| worst | `' env regular performance pos' [·] ' phot to evidence'` | `'.'` (0.992) | substitution | 0.09 | 0.000 | -0.172 |
| worst | `' large dif signific while' [·] ' money tre express'` | `' money'` (0.790) | deletion | 1.03 | 0.000 | -0.221 |
| best | `' live' [·] ' init agre filter'` | `' ver'` (0.143) | substitution | 7.81 | 0.000 | -0.117 |

### `')paren'` — id 51345  (single-probe lp -16.13, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' they option grand own' [·] ' manufact here recogn'` | `'(paren'` (0.964) | substitution | 0.43 | 0.002 | -0.097 |
| worst | `' idx games ful parse' [·] ' begin states mar'` | `'(paren'` (0.863) | substitution | 1.41 | 0.002 | -0.129 |
| worst | `' business comment began influ' [·] ' look consider dest'` | `'(paren'` (0.465) | substitution | 5.39 | 0.002 | -0.180 |
| best | `' tmp' [·] ' les related ep'` | `')'` (0.297) | truncation | 4.20 | 0.060 | -0.116 |

### `'.SuppressLint'` — id 88791  (single-probe lp -12.56, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' wom financial cout under' [·] ' false told super'` | `'SuppressLint'` (0.915) | substitution | 1.44 | 0.000 | -0.161 |
| worst | `' car meet son probably' [·] ' machine contrib management'` | `'SuppressLint'` (0.737) | substitution | 3.83 | 0.000 | -0.153 |
| worst | `' potential loss has class' [·] ' edit birth gold'` | `'SuppressLint'` (0.849) | substitution | 2.43 | 0.000 | -0.152 |
| best | `' simply beg returned candid' [·] ' kind'` | `'SuppressLint'` (0.342) | substitution | 9.03 | 0.001 | -0.117 |

### `'addPreferredGap'` — id 35375  (single-probe lp -8.92, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' pen night server comple' [·] ' inv bet deg'` | `' Bentley'` (0.080) | substitution | 9.99 | 0.000 | -0.126 |
| worst | `' simply beg returned candid' [·] ' kind'` | `' 示'` (0.038) | substitution | 11.89 | 0.000 | -0.129 |
| worst | `' du generate career endl' [·] ' cross cool during'` | `'⠀⠀'` (0.128) | substitution | 10.50 | 0.000 | -0.167 |
| best | `' what rev hom draw' [·] ' body equ ensure'` | `'⠀⠀'` (0.091) | substitution | 11.60 | 0.002 | -0.191 |

### `'<|extra_id_0|>'` — id 100256  (single-probe lp -23.31, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' what rev hom draw' [·] ' body equ ensure'` | `' the'` (0.999) | substitution | 0.01 | 0.000 | -0.234 |
| worst | `' everyone towards find' [·] ' track cover look'` | `' my'` (0.998) | substitution | 0.03 | 0.000 | -0.229 |
| worst | `' idx games ful parse' [·] ' begin states mar'` | `' the'` (0.989) | substitution | 0.10 | 0.000 | 0.016 |
| best | `' calls fac van' [·] ' lower takes padding'` | `' many'` (0.154) | substitution | 7.01 | 0.000 | -0.185 |

### `'.ContentAlignment'` — id 33694  (single-probe lp -7.05, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' they option grand own' [·] ' manufact here recogn'` | `' .'` (0.862) | truncation | 2.11 | 0.000 | -0.100 |
| worst | `' pen night server comple' [·] ' inv bet deg'` | `'TION'` (0.155) | substitution | 10.74 | 0.000 | -0.148 |
| worst | `' simply beg returned candid' [·] ' kind'` | `' 示'` (0.062) | substitution | 11.66 | 0.001 | -0.149 |
| best | `' what rev hom draw' [·] ' body equ ensure'` | `'_ATTACHMENT'` (0.018) | substitution | 12.23 | 0.017 | -0.195 |

### `'_typeDefinition'` — id 67705  (single-probe lp -9.48, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' anything whether more redu' [·] ' was attr load'` | `'ccione'` (0.163) | substitution | 7.49 | 0.000 | -0.174 |
| worst | `' large dif signific while' [·] ' money tre express'` | `' their'` (0.429) | substitution | 7.38 | 0.000 | -0.213 |
| worst | `' work kn she qual' [·] ' window qu drive'` | `' departamento'` (0.058) | substitution | 10.76 | 0.000 | -0.175 |
| best | `' tmp' [·] ' les related ep'` | `'olicitud'` (0.038) | substitution | 10.39 | 0.001 | -0.140 |

### `':)])'` — id 96904  (single-probe lp -15.78, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' car meet son probably' [·] ' machine contrib management'` | `'))'` (0.205) | substitution | 5.58 | 0.000 | -0.166 |
| worst | `' setting letter see body' [·] ' teach'` | `')]'` (0.302) | substitution | 3.77 | 0.000 | -0.135 |
| worst | `' limited entry happy president' [·] ' comes website failed'` | `' ))'` (0.226) | substitution | 4.93 | 0.000 | -0.112 |
| best | `' du generate career endl' [·] ' cross cool during'` | `')]'` (0.097) | substitution | 8.84 | 0.002 | -0.138 |

### `':UIButtonType'` — id 87361  (single-probe lp -9.04, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' car meet son probably' [·] ' machine contrib management'` | `'-Type'` (0.076) | substitution | 9.87 | 0.000 | -0.163 |
| worst | `' large dif signific while' [·] ' money tre express'` | `' TypeScript'` (0.090) | substitution | 8.07 | 0.000 | -0.185 |
| worst | `' grid logger conduct members' [·] ' cam title occ'` | `'-Type'` (0.390) | substitution | 7.02 | 0.000 | -0.143 |
| best | `' du generate career endl' [·] ' cross cool during'` | `'DbType'` (0.062) | substitution | 10.16 | 0.002 | -0.185 |

### `'utedString'` — id 40630  (single-probe lp -8.75, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' setting letter see body' [·] ' teach'` | `'String'` (0.614) | substitution | 4.12 | 0.009 | -0.181 |
| worst | `' everyone towards find' [·] ' track cover look'` | `'String'` (0.280) | substitution | 7.31 | 0.012 | -0.167 |
| worst | `' idx games ful parse' [·] ' begin states mar'` | `'String'` (0.823) | substitution | 2.06 | 0.014 | -0.131 |
| best | `' tmp' [·] ' les related ep'` | `'utedString'` (0.513) | correct | 5.65 | 0.513 | -0.127 |

### `'okableCall'` — id 77666  (single-probe lp -6.49, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' calls fac van' [·] ' lower takes padding'` | `'ẩu'` (0.245) | substitution | 8.88 | 0.001 | -0.161 |
| worst | `' simply beg returned candid' [·] ' kind'` | `'имв'` (0.040) | substitution | 11.22 | 0.001 | -0.145 |
| worst | `' car meet son probably' [·] ' machine contrib management'` | `'Xã'` (0.067) | substitution | 10.83 | 0.001 | -0.194 |
| best | `' env regular performance pos' [·] ' phot to evidence'` | `'ẩu'` (0.033) | substitution | 11.55 | 0.004 | -0.128 |

### `' dequeueReusableCellWithIdentifier'` — id 73905  (single-probe lp -3.35, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' potential loss has class' [·] ' edit birth gold'` | `' Deer'` (0.054) | substitution | 10.05 | 0.007 | -0.143 |
| worst | `' business comment began influ' [·] ' look consider dest'` | `'IQUE'` (0.147) | substitution | 8.74 | 0.017 | -0.150 |
| worst | `' pen night server comple' [·] ' inv bet deg'` | `' Decompiled'` (0.087) | substitution | 9.55 | 0.020 | -0.099 |
| best | `' include hum label night' [·] ' js amount sen'` | `' dequeueReusableCellWithIdentifier'` (0.203) | correct | 8.48 | 0.203 | -0.149 |

### `'ketøy'` — id 81885  (single-probe lp -7.72, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' live' [·] ' init agre filter'` | `'contador'` (0.981) | substitution | 0.29 | 0.000 | -0.107 |
| worst | `' large dif signific while' [·] ' money tre express'` | `'contador'` (0.588) | substitution | 4.71 | 0.000 | -0.150 |
| worst | `' setting letter see body' [·] ' teach'` | `'ýt'` (0.340) | substitution | 7.87 | 0.001 | -0.144 |
| best | `' pen night server comple' [·] ' inv bet deg'` | `'tableFuture'` (0.051) | substitution | 9.86 | 0.008 | -0.097 |

### `'$fdata'` — id 87941  (single-probe lp -19.44, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' anything whether more redu' [·] ' was attr load'` | `'ccione'` (0.177) | substitution | 7.97 | 0.000 | -0.191 |
| worst | `' quality etc resources mut' [·] ' inst lot filter'` | `'ted'` (0.358) | substitution | 7.01 | 0.000 | -0.173 |
| worst | `' simply beg returned candid' [·] ' kind'` | `'data'` (0.168) | substitution | 8.02 | 0.000 | -0.161 |
| best | `' setting letter see body' [·] ' teach'` | `'ftime'` (0.065) | substitution | 9.72 | 0.000 | -0.192 |

### `'(MigrationBuilder'` — id 54421  (single-probe lp -10.37, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' grid logger conduct members' [·] ' cam title occ'` | `';'` (0.077) | substitution | 10.09 | 0.000 | -0.170 |
| worst | `' pen night server comple' [·] ' inv bet deg'` | `'ază'` (0.064) | substitution | 10.48 | 0.000 | -0.159 |
| worst | `' char deter imp med' [·] ' road install aud'` | `' 示'` (0.043) | substitution | 11.30 | 0.000 | -0.157 |
| best | `' wom financial cout under' [·] ' false told super'` | `'（'` (0.027) | substitution | 11.50 | 0.002 | -0.169 |

### `'.AdapterView'` — id 71559  (single-probe lp -6.41, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' anything whether more redu' [·] ' was attr load'` | `' AdapterView'` (0.531) | substitution | 5.06 | 0.002 | -0.114 |
| worst | `' simply beg returned candid' [·] ' kind'` | `' AdapterView'` (0.715) | substitution | 2.62 | 0.002 | -0.096 |
| worst | `' potential loss has class' [·] ' edit birth gold'` | `' AdapterView'` (0.809) | substitution | 2.30 | 0.004 | -0.134 |
| best | `' live' [·] ' init agre filter'` | `'.AdapterView'` (0.143) | correct | 10.30 | 0.143 | -0.106 |

## Reasoning-mode substitutions (chat template, 256-token greedy chains)

model `allenai/OLMo-2-1124-7B-Instruct`; seed reproduced: glitch 0.000, healthy 0.531

| group | seed token | response (first 200 chars) |
|---|---|---|
| glitch | `'.XtraLayout'` | ' The token "layui" refers to a lightweight, modular, and flexible JavaScript framework primarily used for building web interfaces. It is designed to simplify the development process and enhance the us' |
| glitch | `'")){\r\n'` | " The token you've provided, `){`, is a closing curly brace (`}`) in programming syntax. In many programming languages, curly braces are used to denote the beginning and end of a block of code. This bl" |
| glitch | `' Hexatrigesimal'` | ' The token "Calculus" refers to a branch of mathematics that deals with the study of change, motion, and accumulation. It involves the concepts of limits, derivatives, integrals, and infinite series. ' |
| glitch | `'雅黑'` | ' The token "턱" (ttuk) in Korean Hangul script represents the consonant sound "t" or "d" depending on the context, and it is combined with the vowel "u" (ㅜ) to form the syllable "ttuk" or "tuk," which ' |
| glitch | `' didn'` | ' The token "don" is a short form or abbreviation of "do not." It is commonly used in English as a command or directive to tell someone not to do something. Here\'s a step-by-step explanation:\n\n1. **Ide' |
| glitch | `'.AddModelError'` | ' The token `ErrorDictionary` is not a standard token in programming languages like C#, Java, Python, or JavaScript. However, it might be a misunderstanding or a typo. The term "ErrorDictionary" doesn\'' |
| glitch | `'.JpaRepository'` | ' The token "IpaRepository" does not directly correspond to a widely recognized term or acronym in English. However, breaking down the components might help us understand its possible meaning:\n\n1. **I*' |
| glitch | `'.ContextCompat'` | ' The token `ContextCompat` refers to a static utility class in the Android SDK that provides a way to access the current theme context for drawing operations. This class is part of the `android.conten' |
| healthy | `' woman'` | ' The token "woman" is a noun in the English language. Here\'s a step-by-step explanation:\n\n1. **Token Identification**: The token "woman" is a written or printed symbol that represents a concept, idea,' |
| healthy | `' life'` | ' The token "life" is a concept that has been extensively explored across various disciplines, including philosophy, biology, psychology, and linguistics. Here\'s a step-by-step explanation of what "lif' |
| healthy | `' rumpe'` | ' The token "r rumpe" does not immediately correspond to a widely recognized term or abbreviation in English or any other language that I am aware of up to my last knowledge update in early 2023. Howev' |
| healthy | `' start'` | ' The token "start" in various contexts can have different meanings, but in a general computing or programming context, it often refers to the beginning of an operation or process.\n\n**Step-by-step expl' |
| healthy | `' actually'` | ' The token "actually" is an adverb in the English language. Adverbs modify verbs, adjectives, or other adverbs, often providing additional information about manner, place, time, frequency, degree, lev' |
| healthy | `' way'` | ' The token "way" can be interpreted in several contexts, depending on the field or discipline it is used in. Here are a few possibilities:\n\n1. **Mathematics and Logic**: In mathematics, "way" could re' |
| healthy | `',\r\n'` | ' The token you\'ve provided, `",` is a quotation mark or double quote character in ASCII. Here\'s a step-by-step explanation:\n\n1. **Character Identification**: The token is a double quote character (")"' |
| healthy | `'.onOptionsItemSelected'` | ' The token "OptionsItemSelected" is not a standalone token with a universally recognized meaning in the context of programming languages or computer science. Instead, it appears to be a combination of' |

## Hallucination-bridge near-misses (259 of 1255 entities)

| entity | type | model produced | frag_max | familiarity |
|---|---|---|---|---|
| `ssl.ALERT_DESCRIPTION_PROTOCOL_VERSION` | code | `ALERT_DESCRIPTION_PROTOCOL_VERSION` | 1.00 | -7.54 |
| `Nukuʻalofa` | noun | `Nuku'alofa` | 0.96 | -2.42 |
| `asyncio.Transport` | code | `asyncio.transport` | 0.92 | -8.44 |
| `numpy.ravel_multi_index` | code | `np.ravel_multi_index` | 0.88 | -6.56 |
| `transformers.DepthEstimationPipeline` | code | `DepthEstimationPipeline` | 0.83 | -8.30 |
| `transformers.IMAGE_PROCESSOR_MAPPING` | code | `IMAGE_PROCESSOR_MAPPING` | 0.75 | -7.89 |
| `socket.TIPC_MEDIUM_IMPORTANCE` | code | `sock.TIPC_MEDIUM_IMPORTANCE` | 0.75 | -8.92 |
| `socket.TIPC_LOW_IMPORTANCE` | code | `sock.TIPC_LOW_IMPORTANCE` | 0.75 | -7.74 |
| `curses.BUTTON5_PRESSED` | code | `curses.C_BUTTON5_PRESSED` | 0.71 | -7.23 |
| `transformers.PeVideoVideoProcessor` | code | `PeVideoVideoProcessor` | 0.71 | -8.05 |
| `curses.BUTTON4_DOUBLE_CLICKED` | code | `BUTTON4_DOUBLE_CLICKED` | 0.71 | -8.45 |
| `curses.BUTTON1_PRESSED` | code | `BUTTON1_PRESSED` | 0.71 | -5.67 |
| `curses.BUTTON3_PRESSED` | code | `curses.CTRL_PRESSED` | 0.71 | -6.10 |
| `curses.BUTTON5_DOUBLE_CLICKED` | code | `BUTTON5_DOUBLE_CLICKED` | 0.71 | -8.56 |
| `ast.Assert` | code | `Assert` | 0.67 | -13.13 |
| `logging.WARNING` | code | `logging.warning` | 0.67 | -10.07 |
| `ssl.OP_NO_COMPRESSION` | code | `OP_NO_COMPRESSION` | 0.62 | -5.57 |
| `ssl.OP_CIPHER_SERVER_PREFERENCE` | code | `OP_CIPHER_SERVER_PREFERENCE` | 0.62 | -5.49 |
| `ast.Assign` | code | `Assign` | 0.58 | -13.29 |
| `ssl.SO_TYPE` | code | `SO_TYPE` | 0.58 | -10.49 |
| `socket.SOCK_RDM` | code | `sock.SOCK_RDM` | 0.58 | -8.81 |
| `socket.SOCK_CLOEXEC` | code | `SOCK_CLOEXEC` | 0.58 | -5.48 |
| `socket.SO_SNDLOWAT` | code | `sock.SO_SNDLOWAT` | 0.58 | -6.07 |
| `abc.abstractclassmethod` | code | `abstractclassmethod` | 0.54 | -10.63 |
| `typing.SupportsBytes` | code | `SupportsBytes` | 0.54 | -10.33 |
| `typing.SupportsFloat` | code | `SupportsFloat` | 0.54 | -9.09 |
| `transformers.MaskGenerationPipeline` | code | `MaskGenerationPipeline` | 0.50 | -10.42 |
| `reprlib.recursive_repr` | code | `recursive_repr` | 0.50 | -5.60 |
| `transformers.MaskFormerImageProcessor` | code | `MaskFormerImageProcessor` | 0.50 | -9.52 |
| `tarfile.REGTYPE` | code | `tf.REGTYPE` | 0.46 | -8.50 |
| `os.MutableMapping` | code | `MutableMapping` | 0.42 | -9.61 |
| `typing.MutableMapping` | code | `MutableMapping` | 0.42 | -7.68 |
| `ast.MatchClass` | code | `MatchClass` | 0.38 | -13.51 |
| `numpy.random.exponential` | code | `nprandom.exponential` | 0.38 | -6.12 |
| `torch.FloatTensor` | code | `FloatTensor` | 0.33 | -7.15 |
| `ast.Expression` | code | `Expression` | 0.33 | -13.06 |
| `xml.etree.ElementTree.VERSION` | code | `xml.etree.ElementTree` | 0.29 | -3.89 |
| `torch.nn.InstanceNorm2d` | code | `InstanceNorm2d` | 0.29 | -3.59 |
| `ast.Delete` | code | `Delete` | 0.21 | -13.43 |
| `torch.DispatchKey` | code | `torch.dispatch_key` | 0.21 | -12.29 |
| `socket.MSG_CONFIRM` | code | `sock.MSG_CONFIRM` | 0.21 | -9.13 |
| `unittest.SkipTest` | code | `ut.skipTest` | 0.21 | -6.88 |
| `socket.AF_RDS` | code | `sock.AF_RDS` | 0.21 | -9.15 |
| `random.NV_MAGICCONST` | code | `NV_MAGICCONST` | 0.21 | -8.58 |
| `random.SG_MAGICCONST` | code | `SG_MAGICCONST` | 0.21 | -7.69 |
| `torch.nn.ConvTranspose1d` | code | `ConvTranspose1d` | 0.21 | -4.78 |
| `socket.AF_QIPCRTR` | code | `sock.AF_QIPCRTR` | 0.21 | -10.15 |
| `fractions.math` | code | `fractions` | 0.17 | -7.73 |
| `torch.index_put` | code | `index_put` | 0.17 | -8.68 |
| `decimal.ExtendedContext` | code | `ExtendedContext` | 0.17 | -8.27 |
| `socket.IP_MULTICAST_LOOP` | code | `sock.IP_MULTICAST_LOOP` | 0.17 | -6.90 |
| `uuid.RESERVED_FUTURE` | code | `uuidRESERVED_FUTURE` | 0.17 | -6.88 |
| `numpy.SHIFT_UNDERFLOW` | code | `SHIFT_UNDERFLOW` | 0.17 | -10.21 |
| `numpy.FPE_UNDERFLOW` | code | `FPE_UNDERFLOW` | 0.17 | -9.01 |
| `socket.AI_NUMERICSERV` | code | `AI_NUMERICSERV` | 0.17 | -5.37 |
| `numpy.linalg.linalg` | code | `numpy.linalg` | 0.12 | -7.51 |
| `sched.namedtuple` | code | `namedtuple` | 0.12 | -10.74 |
| `ast.ExtSlice` | code | `ExtSlice` | 0.12 | -10.96 |
| `ast.TypeIgnore` | code | `TypeIgnore` | 0.12 | -11.89 |
| `curses.color_content` | code | `color_content` | 0.12 | -8.42 |
| `curses.pair_content` | code | `pair_content` | 0.12 | -9.20 |
| `numpy.nanpercentile` | code | `nanpercentile` | 0.12 | -8.41 |
| `numpy.linalg.lstsq` | code | `numpy.linalg` | 0.12 | -4.94 |
| `numpy.linalg.pinv` | code | `numpy.linalg` | 0.12 | -5.50 |
| `socket.MSG_ERRQUEUE` | code | `sock.MSG_ERRQUEUE` | 0.12 | -10.21 |
| `asyncio.AbstractServer` | code | `AbstractServer` | 0.12 | -7.47 |
| `os.O_RSYNC` | code | `os.osync` | 0.12 | -8.01 |
| `socket.CAN_ERR_FLAG` | code | `sock.CAN_ERR_FLAG` | 0.12 | -8.65 |
| `sqlite3.SQLITE_ATTACH` | code | `sqllite3.SQLITE_ATTACH` | 0.12 | -6.48 |
| `torch.nn.TransformerEncoder` | code | `TransformerEncoder` | 0.12 | -4.28 |
| `asyncio.AbstractEventLoop` | code | `AbstractEventLoop` | 0.12 | -4.39 |
| `torch.nn.TransformerEncoderLayer` | code | `TransformerEncoderLayer` | 0.12 | -3.79 |
| `torch.ConcreteModuleTypeBuilder` | code | `ConcreteModuleTypeBuilder` | 0.12 | -9.78 |
| `socket.CAN_EFF_FLAG` | code | `sock.CAN_EFF_FLAG` | 0.12 | -7.63 |
| `graphlib.TopologicalSorter` | code | `TopologicalSorter` | 0.12 | -5.68 |
| `ast.copy_location` | code | `copy_location` | 0.08 | -9.50 |
| `numpy.True_` | code | `np.True_` | 0.08 | -9.65 |
| `numpy.ndenumerate` | code | `ndenumerate` | 0.08 | -7.65 |
| `numpy.ScalarType` | code | `ScalarType` | 0.08 | -6.39 |
| `ast.Ellipsis` | code | `Ellipsis` | 0.08 | -5.90 |
| `torch.distributions.Independent` | code | `torch.distributions` | 0.08 | -5.30 |
| `torch.distributions.Geometric` | code | `torch.distributions` | 0.08 | -5.56 |
| `sqlite3.threadsafety` | code | `threadsafety` | 0.08 | -7.00 |
| `tokenize.StopTokenizing` | code | `StopTokenizing` | 0.08 | -7.98 |
| `tarfile.RECORDSIZE` | code | `tf.RECORDSIZE` | 0.08 | -6.92 |
| `pickle.SHORT_BINBYTES` | code | `pl.ShortBinBytes` | 0.08 | -8.48 |
| `transformers.DFineForObjectDetection` | code | `DFineForObjectDetection` | 0.08 | -11.06 |
| `tarfile.PAX_NAME_FIELDS` | code | `PAX_NAME_FIELDS` | 0.08 | -8.85 |
| `torch.distributions.wishart` | code | `torch.distributions` | 0.08 | -5.25 |
| `torch.distributions.multinomial` | code | `torch.distributions` | 0.08 | -4.91 |
| `tarfile.OutsideDestinationError` | code | `OutsideDestinationError` | 0.08 | -8.81 |
| `torch.distributions.Weibull` | code | `torch.distributions` | 0.08 | -5.05 |
| `torch.distributions.dirichlet` | code | `torch.distributions` | 0.08 | -4.53 |
| `numpy.histogram` | code | `np.histogram` | 0.04 | -11.87 |
| `numpy.byte` | code | `np.byte` | 0.04 | -10.61 |
| `sched.scheduler` | code | `scheduler` | 0.04 | -11.77 |
| `unittest.main` | code | `unittest` | 0.04 | -8.69 |
| `numpy.transpose` | code | `transpose` | 0.04 | -10.66 |
| `numpy.maximum` | code | `np.maximum` | 0.04 | -12.80 |
| `numpy.partition` | code | `partition` | 0.04 | -10.54 |
| `numpy.append` | code | `np.append` | 0.04 | -10.88 |
| `numpy.angle` | code | `np.angle` | 0.04 | -11.79 |
| `numpy.concatenate` | code | `concatenate` | 0.04 | -12.36 |
| `numpy.argmax` | code | `np.argmax` | 0.04 | -10.88 |
| `numpy.real` | code | `np.real` | 0.04 | -10.83 |
| `copy.deepcopy` | code | `deepcopy` | 0.04 | -8.30 |
| `unittest.skip` | code | `unittest` | 0.04 | -8.08 |
| `datetime.date` | code | `datetime` | 0.04 | -6.47 |
| `numpy.unpackbits` | code | `unpackbits` | 0.04 | -8.40 |
| `operator.imod` | code | `operator` | 0.04 | -11.85 |
| `ssl.warnings` | code | `warnings` | 0.04 | -8.95 |
| `torch.detach_` | code | `torchoverlay.detach_` | 0.04 | -9.23 |
| `unittest.installHandler` | code | `installHandler` | 0.04 | -10.25 |
| `numpy.nanmedian` | code | `np.nanmedian` | 0.04 | -10.45 |
| `numpy.iscomplex` | code | `np.iscomplex` | 0.04 | -8.03 |
| `numpy.row_stack` | code | `np.row_stack` | 0.04 | -8.50 |
| `abc.abstractproperty` | code | `abstractproperty` | 0.04 | -10.95 |
| `numpy.random.choice` | code | `numpy.random` | 0.04 | -6.77 |
| `numpy.around` | code | `np.around` | 0.04 | -8.10 |
| `logging.raiseExceptions` | code | `raiseExceptions` | 0.04 | -6.88 |
| `numpy.array_equal` | code | `array_equal` | 0.04 | -8.40 |
| `glob.itertools` | code | `itertools` | 0.04 | -9.48 |
| `numpy.random.random` | code | `numpy.random` | 0.04 | -7.22 |
| `numpy.broadcast_to` | code | `broadcast_to` | 0.04 | -8.65 |
| `torch.BoolTensor` | code | `torch.bool_tensor` | 0.04 | -6.01 |
| `numpy.deprecate` | code | `deprecate` | 0.04 | -6.59 |
| `numpy.random.geometric` | code | `nprandom.geometric` | 0.04 | -6.92 |
| `asyncio.BufferedProtocol` | code | `BufferedProtocol` | 0.04 | -8.23 |
| `dataclasses.asdict` | code | `dataclasses` | 0.04 | -5.20 |
| `decimal.DivisionImpossible` | code | `DivisionImpossible` | 0.04 | -8.55 |
| `numpy.add_newdoc` | code | `add_newdoc` | 0.04 | -7.08 |
| `numpy.random.standard_gamma` | code | `nprandom.standard_gamma` | 0.04 | -8.12 |
| `numpy.SHIFT_OVERFLOW` | code | `SHIFT_OVERFLOW` | 0.04 | -10.72 |
| `pickle.EXT2` | code | `pickle_ext2` | 0.04 | -8.78 |
| `numpy.compare_chararrays` | code | `compare_chararrays` | 0.04 | -10.20 |
| `http.client.errno` | code | `http.client` | 0.04 | -8.06 |
| `urllib.parse.quote` | code | `urllib.parse` | 0.04 | -4.73 |
| `numpy.random.wald` | code | `nprandom.wald` | 0.04 | -6.99 |
| `numpy.bitwise_and` | code | `np.bitwise_and` | 0.04 | -6.46 |
| `numpy.arcsinh` | code | `np.arcsinh` | 0.04 | -6.27 |
| `numpy.isposinf` | code | `np.isposinf` | 0.04 | -7.09 |
| `http.client.BAD_REQUEST` | code | `httpc.BAD_REQUEST` | 0.04 | -8.02 |
| `numpy.setdiff1d` | code | `setdiff1d` | 0.04 | -5.98 |
| `numpy.random.ranf` | code | `nprandom.ranf` | 0.04 | -5.96 |
| `torch.BFloat16Tensor` | code | `bfloat16_tensor` | 0.04 | -5.08 |
| `torch.nn.PixelUnshuffle` | code | `PixelUnshuffle` | 0.04 | -6.15 |
| `torch.nn.functional.threshold_` | code | `torch.nn.functional` | 0.04 | -5.68 |
| `zipfile.compressor_names` | code | `compressor_names` | 0.04 | -7.35 |
| `torch.nn.functional.batch_norm` | code | `torch.nn.functional` | 0.04 | -5.14 |
| `torch.channels_last_3d` | code | `channels_last_3d` | 0.04 | -6.54 |
| `torch.nn.functional.celu_` | code | `torch.nn.functional` | 0.04 | -6.78 |
| `socket.IPPROTO_ESP` | code | `sock.IPPROTO_ESP` | 0.04 | -5.22 |
| `torch.nn.functional.selu_` | code | `torch.nn.functional` | 0.04 | -5.52 |
| `ssl.SSLWantReadError` | code | `SSLWantReadError` | 0.04 | -5.54 |
| `socket.BTPROTO_HCI` | code | `sock.BTPROTO_HCI` | 0.04 | -7.41 |
| `socket.VMADDR_PORT_ANY` | code | `sock.VMADDR_PORT_ANY` | 0.04 | -8.30 |
| `numpy.tracemalloc_domain` | code | `np.tracemalloc_domain` | 0.04 | -8.42 |
| `socket.CAN_EFF_MASK` | code | `sock.CAN_EFF_MASK` | 0.04 | -7.94 |
| `torch.conv_transpose1d` | code | `conv_transpose1d` | 0.04 | -6.09 |
| `torch.nn.ReflectionPad3d` | code | `ReflectionPad3d` | 0.04 | -4.88 |
| `torch.threshold` | code | `threshold` | 0.00 | -11.01 |
| `calendar.week` | code | `calendar` | 0.00 | -10.47 |
| `ast.Return` | code | `Return` | 0.00 | -13.54 |
| `statistics.mode` | code | `statistics` | 0.00 | -11.52 |
| `re.Scanner` | code | `Scanner` | 0.00 | -10.98 |
| `statistics.math` | code | `statistics` | 0.00 | -11.66 |
| `ast.Attribute` | code | `Attribute` | 0.00 | -13.02 |
| `torch.square` | code | `torc.square` | 0.00 | -10.30 |
| `cmd.string` | code | `string` | 0.00 | -11.40 |
| `dis.collections` | code | `collections` | 0.00 | -14.01 |
| `ast.keyword` | code | `keyword` | 0.00 | -12.35 |
| `torch.ScriptClass` | code | `ScriptClass` | 0.00 | -10.19 |
| `torch.detach_copy` | code | `torchopt.detach_copy` | 0.00 | -9.91 |
| `ssl.match_hostname` | code | `match_hostname` | 0.00 | -7.25 |
| `torch.OptionalType` | code | `OptionalType` | 0.00 | -8.58 |
| `pickle.FunctionType` | code | `FunctionType` | 0.00 | -8.40 |
| `torch.CharTensor` | code | `CharTensor` | 0.00 | -7.32 |
| `tokenize.Name` | code | `tokenize` | 0.00 | -9.32 |
| `operator.isub` | code | `operator` | 0.00 | -8.18 |
| `torch.polygamma` | code | `polygamma` | 0.00 | -9.37 |
| `typing.abstractmethod` | code | `abstractmethod` | 0.00 | -8.81 |
| `gzip.decompress` | code | `decompress` | 0.00 | -6.61 |
| `array.typecodes` | code | `typecodes` | 0.00 | -11.63 |
| `socket.IP_OPTIONS` | code | `sock.IP_OPTIONS` | 0.00 | -11.29 |
| `torch.as_tensor` | code | `as_tensor` | 0.00 | -6.32 |
| `typing.AsyncIterator` | code | `AsyncIterator` | 0.00 | -7.91 |
| `logging.currentframe` | code | `currentframe` | 0.00 | -10.25 |
| `ast.NameConstant` | code | `NameConstant` | 0.00 | -8.72 |
| `torch.alpha_dropout` | code | `alpha_dropout` | 0.00 | -11.07 |
| `torch.complex128` | code | `complex128` | 0.00 | -7.15 |
| `torch.JITException` | code | `JITException` | 0.00 | -8.56 |
| `torch.native_channel_shuffle` | code | `native_channel_shuffle` | 0.00 | -11.16 |
| `torch.bitwise_or` | code | `bitwise_or` | 0.00 | -7.22 |
| `asyncio.TimeoutError` | code | `TimeoutError` | 0.00 | -5.31 |
| `sys.getswitchinterval` | code | `s.getswitchinterval` | 0.00 | -5.12 |
| `torch.nn.functional.interpolate` | code | `torch.nn.functional` | 0.00 | -5.56 |
| `torch.nn.functional.embedding` | code | `torch.nn.functional` | 0.00 | -5.96 |
| `torch.isposinf` | code | `torchoverflow.isposinf` | 0.00 | -7.94 |
| `torch.rnn_relu_cell` | code | `rnn_relu_cell` | 0.00 | -9.79 |
| `torch.nn.functional.fold` | code | `torch.nn.functional` | 0.00 | -7.76 |
| `sqlite3.complete_statement` | code | `sqld3.complete_statement` | 0.00 | -6.32 |
| `inspect.ismethoddescriptor` | code | `ismethoddescriptor` | 0.00 | -7.68 |
| `torch.UnionType` | code | `UnionType` | 0.00 | -7.22 |
| `statistics.hypot` | code | `statistics` | 0.00 | -7.22 |
| `torch.native_layer_norm` | code | `native_layer_norm` | 0.00 | -9.17 |
| `torch.PyObjectType` | code | `PyObjectType` | 0.00 | -8.37 |
| `torch.DeviceObjType` | code | `DeviceObjType` | 0.00 | -9.55 |
| `ssl.DefaultVerifyPaths` | code | `DefaultVerifyPaths` | 0.00 | -7.79 |
| `torch.optim.AdamW` | code | `torch.optim` | 0.00 | -5.53 |
| `ast.AsyncFunctionDef` | code | `AsyncFunctionDef` | 0.00 | -7.80 |
| `torch.optim.Rprop` | code | `torch.optim` | 0.00 | -9.89 |
| `ssl.TLSVersion` | code | `TLSVersion` | 0.00 | -5.92 |
| `contextlib.deque` | code | `contextlib` | 0.00 | -6.92 |
| `asyncio.StreamReaderProtocol` | code | `StreamReaderProtocol` | 0.00 | -5.77 |
| `torch.quasirandom` | code | `quasirandom` | 0.00 | -6.58 |
| `torch.SymIntType` | code | `SymIntType` | 0.00 | -9.07 |
| `torch.logcumsumexp` | code | `logcumsumexp` | 0.00 | -7.45 |
| `inspect.getclasstree` | code | `getclasstree` | 0.00 | -6.43 |
| `torch.LiteScriptModule` | code | `torch.litescript` | 0.00 | -7.92 |
| `csv.QUOTE_MINIMAL` | code | `csv_QUOTE_MINIMAL` | 0.00 | -6.47 |
| `torch.QInt8Storage` | code | `QInt8Storage` | 0.00 | -8.25 |
| `bisect.insort_right` | code | `insort_right` | 0.00 | -3.68 |
| `socket.EAI_BADFLAGS` | code | `sock.EAI_BADFLAGS` | 0.00 | -6.61 |
| `shutil.SpecialFileError` | code | `SpecialFileError` | 0.00 | -7.64 |
| `torch.bitwise_right_shift` | code | `torc.nn.bitwise_right_shift` | 0.00 | -7.06 |
| `contextlib.asynccontextmanager` | code | `asynccontextmanager` | 0.00 | -4.78 |
| `torch.unify_type_list` | code | `unify_type_list` | 0.00 | -8.29 |
| `zipfile.structEndArchive` | code | `structEndArchive` | 0.00 | -10.00 |
| `torch.sym_constrain_range` | code | `sym_constrain_range` | 0.00 | -10.87 |
| `difflib.context_diff` | code | `context_diff` | 0.00 | -5.93 |
| `transformers.get_constant_schedule` | code | `get_constant_schedule` | 0.00 | -9.39 |
| `torch.cosine_embedding_loss` | code | `cosine_embedding_loss` | 0.00 | -7.99 |
| `ssl.cert_time_to_seconds` | code | `cert_time_to_seconds` | 0.00 | -9.86 |
| `transformers.image_processing_base` | code | `image_processing_base` | 0.00 | -8.52 |
| `logging.setLogRecordFactory` | code | `setLogRecordFactory` | 0.00 | -7.78 |
| `zipfile.structCentralDir` | code | `structCentralDir` | 0.00 | -8.95 |
| `torch.ComplexDoubleStorage` | code | `ComplexDoubleStorage` | 0.00 | -6.35 |
| `pickle.PyStringMap` | code | `PyStringMap` | 0.00 | -7.09 |
| `ssl.HAS_SSLv2` | code | `HAS_SSLv2` | 0.00 | -6.61 |
| `torch.import_ir_module_from_buffer` | code | `import_ir_module_from_buffer` | 0.00 | -8.05 |
| `transformers.is_neptune_available` | code | `is_neptune_available` | 0.00 | -9.88 |
| `torch.quantized_gru_cell` | code | `torch.quantized` | 0.00 | -6.91 |
| `asyncio.DefaultEventLoopPolicy` | code | `DefaultEventLoopPolicy` | 0.00 | -4.61 |
| `transformers.is_clearml_available` | code | `is_clearml_available` | 0.00 | -9.60 |
| `weakref.getweakrefcount` | code | `getweakrefcount` | 0.00 | -4.25 |
| `xml.etree.ElementTree.dump` | code | `xml.etree.ElementTree` | 0.00 | -4.00 |
| `transformers.image_processing_bit_fast` | code | `image_processing_bit_fast` | 0.00 | -9.66 |
| `sys.get_int_max_str_digits` | code | `_.get_int_max_str_digits` | 0.00 | -6.67 |
| `torch.nn.functional.rms_norm` | code | `torch.nn.functional` | 0.00 | -7.02 |
| `transformers.models.bit.configuration_bit` | code | `transformers.models.bitconfiguration_bit` | 0.00 | -8.60 |
| `torch.DisableTorchFunction` | code | `DisableTorchFunction` | 0.00 | -8.32 |
| `transformers.image_processing_auto_fast` | code | `image_processing_auto_fast` | 0.00 | -8.95 |
| `socket.CAN_J1939` | code | `sCAN_J1939` | 0.00 | -7.21 |
| `xml.etree.ElementTree.parse` | code | `xml.etree.ElementTree` | 0.00 | -2.97 |
| `transformers.models.auto.auto_factory` | code | `transformers.models.auto` | 0.00 | -7.93 |
| `transformers.ClvpFeatureExtractor` | code | `ClvpFeatureExtractor` | 0.00 | -8.04 |
| `transformers.models.depth_anything` | code | `models.depth_anything` | 0.00 | -8.53 |
| `heapq.heappushpop` | code | `heappushpop` | 0.00 | -3.56 |
| `transformers.glue_compute_metrics` | code | `glue_compute_metrics` | 0.00 | -8.70 |
