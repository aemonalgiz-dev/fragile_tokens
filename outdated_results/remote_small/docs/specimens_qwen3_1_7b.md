# Confident-substitution specimens

Every example in this file was produced by greedy decoding; `p` is the model's probability for the token it emitted, `H` the entropy in bits at that position, `p(target)` its probability for the correct token, `ret` the cosine between the last-layer slot state and the correct token's unembedding row (identity still pointing at its readout while the readout chose otherwise).

## Model

- **model**: `Qwen/Qwen3-1.7B`
- **commit**: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- **dtype**: `torch.bfloat16`
- **num_layers**: `28`
- **hidden_size**: `2048`
- **vocab_size**: `151936`
- **tie_word_embeddings**: `True`
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

## Fragile clean-looking tokens (pass the single probe; fail in some contexts)


### `' 방법'` — id 130864

single-probe lp -0.045 · fragility 0.75 · mean lp -1.24 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | 16 | `' 方法'` (0.969) | **substitution** | 0.21 | 0.029 | 0.194 | `' 方法 air email serial weap plan commit sp'` |
| worst | `' reach ps things drug' [·] ' associated season government'` | 32 | `' 方法'` (0.969) | **substitution** | 0.21 | 0.029 | 0.119 | `' 方法 associated season government search '` |
| worst | `' module times chang doesn' [·] ' sale index syn'` | 8 | `' 方法'` (0.931) | **substitution** | 0.37 | 0.067 | 0.112 | `' 方法 sale index syn\nText: the'` |
| best | `' autom different contin selection' [·] ' how sem link'` | 64 | `' 방법'` (0.957) | **correct** | 0.26 | 0.957 | 0.128 | `' 방법 how sem link scient statement um car'` |

### `' �'` — id 58230

single-probe lp -0.055 · fragility 0.67 · mean lp -3.57 · canonical False · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' park col our server' [·] ' into fine son'` | 32 | `' into'` (1.000) | **deletion** | 0.00 | 0.000 | -0.057 | `' into fine son path cross we milit virtu'` |
| worst | `' child vs' [·] ' long my opp'` | 8 | `' long'` (1.000) | **deletion** | 0.01 | 0.000 | 0.001 | `' long my opp we maybe\nText:'` |
| worst | `' live mat abs cool' [·] ' wom tem foreign'` | 16 | `' wom'` (0.990) | **deletion** | 0.10 | 0.000 | -0.011 | `' wom tem foreign point ou fun contrib st'` |
| best | `' nil tried ens recent' [·] ' takes implements es'` | 16 | `' �'` (1.000) | **correct** | 0.00 | 1.000 | -0.045 | `' � takes implements es operator empty ge'` |

### `' או'` — id 128255

single-probe lp -0.073 · fragility 0.58 · mean lp -0.97 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' msg' [·] ' loaded win far'` | 16 | `' ou'` (0.731) | **substitution** | 1.62 | 0.087 | 0.110 | `' ou loaded win far une profession achie '` |
| worst | `' child vs' [·] ' long my opp'` | 8 | `' long'` (0.625) | **deletion** | 1.60 | 0.109 | 0.172 | `' long my opp we maybe\nText:'` |
| worst | `' abs dem log min' [·] ' particular line child'` | 64 | `' hoặc'` (0.361) | **substitution** | 2.80 | 0.133 | 0.093 | `' hoặc particular line child thread watch'` |
| best | `' nil tried ens recent' [·] ' takes implements es'` | 16 | `' או'` (0.977) | **correct** | 0.27 | 0.977 | 0.106 | `' או takes implements es operator empty g'` |

### `' Marshal'` — id 35667

single-probe lp -0.074 · fragility 0.46 · mean lp -2.11 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' very queue check patients' [·] ' extrem education friend'` | 16 | `'Marshal'` (0.997) | **substitution** | 0.03 | 0.002 | -0.008 | `'Marshal extrem education friend\nText: th'` |
| worst | `' security flag room rem' [·] ' respons attack suggest'` | 64 | `'Marshal'` (0.997) | **substitution** | 0.03 | 0.002 | -0.003 | `'Marshal respons attack suggest porn host'` |
| worst | `' abs dem log min' [·] ' particular line child'` | 64 | `'Marshal'` (0.997) | **substitution** | 0.03 | 0.003 | 0.016 | `'Marshal particular line child thread wat'` |
| best | `' lib btn' [·] ' policy another expl'` | 8 | `' Marshal'` (0.999) | **correct** | 0.02 | 0.999 | 0.061 | `' Marshal policy another expl int stateme'` |

### `' לכל'` — id 128710

single-probe lp -0.025 · fragility 0.33 · mean lp -0.41 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' ide arr exper bul' [·] ' among write evid'` | 32 | `' whilst'` (0.284) | **substitution** | 5.54 | 0.195 | 0.038 | `' whilst among write evid became iss thei'` |
| worst | `' autom different contin selection' [·] ' how sem link'` | 64 | `' whilst'` (0.515) | **substitution** | 2.97 | 0.243 | 0.074 | `' whilst how sem link scient statement um'` |
| worst | `' abs dem log min' [·] ' particular line child'` | 64 | `' לכל'` (0.387) | **correct** | 4.25 | 0.387 | 0.133 | `' לכל particular line child thread watch '` |
| best | `' fem reports rep probably' [·] ' sup place wr'` | 32 | `' לכל'` (0.971) | **correct** | 0.35 | 0.971 | 0.116 | `' לכל sup place wr mill blue that extends'` |

### `'ो'` — id 54575

single-probe lp -0.040 · fragility 0.33 · mean lp -0.67 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' accept repe incre' [·] ' fire prem qual'` | 8 | `'ó'` (0.443) | **substitution** | 1.89 | 0.005 | 0.076 | `'ó fire prem qual specific\nText:'` |
| worst | `' child vs' [·] ' long my opp'` | 8 | `' long'` (0.384) | **deletion** | 2.45 | 0.125 | 0.100 | `' long my opp we maybe\nText:'` |
| worst | `' security flag room rem' [·] ' respons attack suggest'` | 64 | `'ó'` (0.482) | **substitution** | 2.47 | 0.122 | 0.029 | `'ó respons attack suggest porn host red s'` |
| best | `' lib btn' [·] ' policy another expl'` | 8 | `'ो'` (0.999) | **correct** | 0.02 | 0.999 | 0.085 | `'ो policy another expl int statement\nText'` |

### `' Dịch'` — id 134586

single-probe lp -0.001 · fragility 0.33 · mean lp -0.75 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' live mat abs cool' [·] ' wom tem foreign'` | 16 | `'\n'` (0.840) | **substitution** | 0.85 | 0.054 | 0.047 | `'\nText: the quick brown fox jumps'` |
| worst | `' abs dem log min' [·] ' particular line child'` | 64 | `'\n'` (0.626) | **substitution** | 1.32 | 0.109 | 0.082 | `'\nDịch: alert between item military'` |
| worst | `' autom different contin selection' [·] ' how sem link'` | 64 | `'\n'` (0.846) | **substitution** | 0.80 | 0.048 | 0.096 | `'\nDịch: off mother raise mass'` |
| best | `' nil tried ens recent' [·] ' takes implements es'` | 16 | `' Dịch'` (0.999) | **correct** | 0.01 | 0.999 | 0.067 | `' Dịch takes implements es operator empty'` |

### `' �'` — id 94491

single-probe lp -0.000 · fragility 0.21 · mean lp -0.26 · canonical False · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' park col our server' [·] ' into fine son'` | 32 | `' into'` (0.913) | **deletion** | 0.44 | 0.085 | 0.086 | `' into fine son path cross we milit virtu'` |
| worst | `' child vs' [·] ' long my opp'` | 8 | `' long'` (0.797) | **deletion** | 0.74 | 0.202 | 0.133 | `' long my opp we maybe\nText:'` |
| worst | `' msg' [·] ' loaded win far'` | 16 | `' loaded'` (0.560) | **deletion** | 1.05 | 0.436 | 0.110 | `' � loaded win far une profession achie g'` |
| best | `' lib btn' [·] ' policy another expl'` | 8 | `' �'` (1.000) | **correct** | 0.00 | 1.000 | 0.139 | `' � policy another expl int statement\nTex'` |

### `' אית'` — id 139430

single-probe lp -0.022 · fragility 0.21 · mean lp -0.32 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' accept repe incre' [·] ' fire prem qual'` | 8 | `' neat'` (0.409) | **substitution** | 4.41 | 0.055 | 0.075 | `' neat fire prem qual specific\nText:'` |
| worst | `' msg' [·] ' loaded win far'` | 16 | `' אית'` (0.400) | **correct** | 4.12 | 0.400 | 0.052 | `' אית loaded win far une profession achie'` |
| worst | `' child vs' [·] ' long my opp'` | 8 | `' אית'` (0.448) | **correct** | 2.24 | 0.448 | 0.095 | `' אית long my opp we maybe\nText'` |
| best | `' extrem solid moving to' [·] ' eval'` | 8 | `' אית'` (0.990) | **correct** | 0.14 | 0.990 | 0.103 | `' אית eval\nText: the quick brown'` |

### `' ש'` — id 87798

single-probe lp -0.013 · fragility 0.21 · mean lp -0.35 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' ide arr exper bul' [·] ' among write evid'` | 32 | `' sh'` (0.805) | **substitution** | 0.94 | 0.123 | 0.108 | `' sh among write evid became iss their na'` |
| worst | `' child vs' [·] ' long my opp'` | 8 | `' sh'` (0.650) | **substitution** | 1.30 | 0.271 | 0.218 | `' sh long my opp we maybe\nText'` |
| worst | `' msg' [·] ' loaded win far'` | 16 | `' sh'` (0.558) | **substitution** | 1.44 | 0.338 | 0.197 | `' sh loaded win far une profession achie '` |
| best | `' reach ps things drug' [·] ' associated season government'` | 32 | `' ש'` (0.993) | **correct** | 0.07 | 0.992 | 0.224 | `' ש associated season government search n'` |

### `' PropertyChanged'` — id 43189

single-probe lp -0.026 · fragility 0.17 · mean lp -0.26 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' autom different contin selection' [·] ' how sem link'` | 64 | `'PropertyChanged'` (0.852) | **substitution** | 0.60 | 0.148 | 0.068 | `'PropertyChanged how sem link scient stat'` |
| worst | `' api court dec trust' [·] ' rock print van'` | 64 | `'PropertyChanged'` (0.679) | **substitution** | 0.91 | 0.321 | 0.060 | `'PropertyChanged rock print van find\nText'` |
| worst | `' security flag room rem' [·] ' respons attack suggest'` | 64 | `'PropertyChanged'` (0.593) | **substitution** | 0.98 | 0.407 | 0.034 | `'PropertyChanged respons attack suggest p'` |
| best | `' extrem solid moving to' [·] ' eval'` | 8 | `' PropertyChanged'` (0.999) | **correct** | 0.01 | 0.999 | 0.095 | `' PropertyChanged eval\nText: the quick br'` |

### `' naprawdę'` — id 133732

single-probe lp -0.050 · fragility 0.12 · mean lp -0.21 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' child vs' [·] ' long my opp'` | 8 | `' really'` (0.632) | **substitution** | 1.60 | 0.299 | 0.077 | `' really long my opp we maybe\nText'` |
| worst | `' msg' [·] ' loaded win far'` | 16 | `' naprawdę'` (0.598) | **correct** | 1.42 | 0.598 | 0.070 | `' naprawdę loaded win far une profession '` |
| worst | `' fra ca little bel' [·] ' param money text'` | 64 | `' naprawdę'` (0.584) | **correct** | 3.12 | 0.584 | 0.031 | `' naprawdę param money text copyright del'` |
| best | `' security flag room rem' [·] ' respons attack suggest'` | 64 | `' naprawdę'` (0.981) | **correct** | 0.22 | 0.981 | 0.055 | `' naprawdę respons attack suggest porn ho'` |

### `' وغيرها'` — id 137052

single-probe lp -0.019 · fragility 0.12 · mean lp -0.19 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' lib btn' [·] ' policy another expl'` | 8 | `'另有'` (0.386) | **substitution** | 3.20 | 0.340 | 0.106 | `'另有政策另述另一解释說明\n'` |
| worst | `' module times chang doesn' [·] ' sale index syn'` | 8 | `' وغيرها'` (0.514) | **correct** | 5.16 | 0.514 | 0.044 | `' وغيرها sale index syn\nText: the'` |
| worst | `' msg' [·] ' loaded win far'` | 16 | `' وغيرها'` (0.601) | **correct** | 2.26 | 0.601 | 0.078 | `' وغيرها loaded win far une profession ac'` |
| best | `' autom different contin selection' [·] ' how sem link'` | 64 | `' وغيرها'` (0.983) | **correct** | 0.22 | 0.983 | 0.062 | `' وغيرها how sem link scient statement um'` |

### `' getResource'` — id 87794

single-probe lp -0.000 · fragility 0.12 · mean lp -0.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' ide arr exper bul' [·] ' among write evid'` | 32 | `'getResource'` (0.678) | **substitution** | 0.92 | 0.320 | 0.029 | `'getResource among write evid became iss '` |
| worst | `' abs dem log min' [·] ' particular line child'` | 64 | `'getResource'` (0.495) | **substitution** | 1.07 | 0.495 | 0.051 | `'getResource particular line child thread'` |
| worst | `' api court dec trust' [·] ' rock print van'` | 64 | `' getResource'` (0.561) | **correct** | 1.01 | 0.561 | 0.090 | `' getResource rock print van find\nText:'` |
| best | `' module times chang doesn' [·] ' sale index syn'` | 8 | `' getResource'` (1.000) | **correct** | 0.00 | 1.000 | 0.070 | `' getResource sale index syn\nText: the'` |

### `' пот'` — id 80703

single-probe lp -0.024 · fragility 0.12 · mean lp -0.31 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' lib btn' [·] ' policy another expl'` | 8 | `' pot'` (0.958) | **substitution** | 0.26 | 0.042 | 0.195 | `' pot policy another expl int statement\nT'` |
| worst | `' child vs' [·] ' long my opp'` | 8 | `' pot'` (0.749) | **substitution** | 0.88 | 0.243 | 0.169 | `' pot long my opp we maybe\nText'` |
| worst | `' msg' [·] ' loaded win far'` | 16 | `' pot'` (0.772) | **substitution** | 0.84 | 0.221 | 0.221 | `' pot loaded win far une profession achie'` |
| best | `' fra ca little bel' [·] ' param money text'` | 64 | `' пот'` (0.998) | **correct** | 0.02 | 0.998 | 0.224 | `' пот param money text copyright deliver '` |

### `' �'` — id 17284

single-probe lp -0.047 · fragility 0.12 · mean lp -0.29 · canonical False · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' child vs' [·] ' long my opp'` | 8 | `' long'` (0.979) | **deletion** | 0.15 | 0.020 | 0.011 | `' long my opp we maybe\nText:'` |
| worst | `' live mat abs cool' [·] ' wom tem foreign'` | 16 | `' wom'` (0.721) | **deletion** | 1.01 | 0.265 | 0.019 | `' wom tem foreign point ou fun contrib st'` |
| worst | `' security flag room rem' [·] ' respons attack suggest'` | 64 | `' respons'` (0.531) | **deletion** | 1.01 | 0.469 | 0.023 | `' respons attack suggest porn host red st'` |
| best | `' nil tried ens recent' [·] ' takes implements es'` | 16 | `' �'` (1.000) | **correct** | 0.00 | 1.000 | 0.034 | `' � takes implements es operator empty ge'` |

### `' NotFound'` — id 23420

single-probe lp -0.021 · fragility 0.12 · mean lp -0.23 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' fem reports rep probably' [·] ' sup place wr'` | 32 | `'NotFound'` (0.940) | **substitution** | 0.33 | 0.060 | 0.093 | `'NotFound sup place wr mill blue that ext'` |
| worst | `' api court dec trust' [·] ' rock print van'` | 64 | `' NotFound'` (0.562) | **correct** | 1.00 | 0.562 | 0.095 | `' NotFound rock print van find\nText:'` |
| worst | `' security flag room rem' [·] ' respons attack suggest'` | 64 | `' NotFound'` (0.593) | **correct** | 0.98 | 0.593 | 0.043 | `' NotFound respons attack suggest porn ho'` |
| best | `' lib btn' [·] ' policy another expl'` | 8 | `' NotFound'` (1.000) | **correct** | 0.01 | 1.000 | 0.140 | `' NotFound policy another expl int statem'` |

### `'�'` — id 99

single-probe lp -0.029 · fragility 0.12 · mean lp -0.22 · canonical False · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' child vs' [·] ' long my opp'` | 8 | `' long'` (0.957) | **deletion** | 0.35 | 0.033 | -0.020 | `' long my opp we maybe\nText:'` |
| worst | `' accept repe incre' [·] ' fire prem qual'` | 8 | `'�'` (0.386) | **correct** | 3.91 | 0.386 | 0.005 | `'� fire prem qual specific\nText:'` |
| worst | `' module times chang doesn' [·] ' sale index syn'` | 8 | `'�'` (0.540) | **correct** | 1.93 | 0.540 | -0.018 | `'� sale index syn\nText: the'` |
| best | `' park col our server' [·] ' into fine son'` | 32 | `'�'` (1.000) | **correct** | 0.00 | 1.000 | -0.023 | `'� into fine son path cross we milit'` |

### `' DIRECT'` — id 20230

single-probe lp -0.007 · fragility 0.08 · mean lp -0.57 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' api court dec trust' [·] ' rock print van'` | 64 | `'DIRECT'` (1.000) | **substitution** | 0.00 | 0.000 | 0.061 | `'DIRECT rock print van find\nText:'` |
| worst | `' security flag room rem' [·] ' respons attack suggest'` | 64 | `'DIRECT'` (0.986) | **substitution** | 0.11 | 0.014 | 0.022 | `'DIRECT respons attack suggest porn host '` |
| worst | `' autom different contin selection' [·] ' how sem link'` | 64 | `' DIRECT'` (0.679) | **correct** | 0.91 | 0.679 | 0.039 | `' DIRECT how sem link scient statement um'` |
| best | `' lib btn' [·] ' policy another expl'` | 8 | `' DIRECT'` (1.000) | **correct** | 0.00 | 1.000 | 0.085 | `' DIRECT policy another expl int statemen'` |

### `' dataTable'` — id 62766

single-probe lp -0.028 · fragility 0.08 · mean lp -0.09 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' autom different contin selection' [·] ' how sem link'` | 64 | `'dataTable'` (0.562) | **substitution** | 1.00 | 0.437 | 0.045 | `'dataTable how sem link scient statement '` |
| worst | `' api court dec trust' [·] ' rock print van'` | 64 | `'dataTable'` (0.562) | **substitution** | 0.99 | 0.438 | 0.084 | `'dataTable rock print van find\nText:'` |
| worst | `' abs dem log min' [·] ' particular line child'` | 64 | `' dataTable'` (0.915) | **correct** | 0.42 | 0.915 | 0.075 | `' dataTable particular line child thread '` |
| best | `' module times chang doesn' [·] ' sale index syn'` | 8 | `' dataTable'` (0.999) | **correct** | 0.01 | 0.999 | 0.094 | `' dataTable sale index syn\nText: the'` |

### `' سريع'` — id 140675

single-probe lp -0.062 · fragility 0.08 · mean lp -0.10 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' display active width ir' [·] ' et foreach dict'` | 32 | `' سريع'` (0.572) | **correct** | 4.23 | 0.572 | 0.091 | `' سريع et foreach dict heard that leading'` |
| worst | `' live mat abs cool' [·] ' wom tem foreign'` | 16 | `' سريع'` (0.579) | **correct** | 4.04 | 0.579 | 0.008 | `' سريع wom tem foreign point ou fun contr'` |
| worst | `' fa let sever getting' [·] ' goal oper import'` | 8 | `' سريع'` (0.826) | **correct** | 1.85 | 0.826 | 0.022 | `' سريع goal oper import\nText: the'` |
| best | `' das material events customer' [·] ' air email serial'` | 16 | `' سريع'` (0.994) | **correct** | 0.10 | 0.994 | 0.119 | `' سريع air email serial weap plan commit '` |

### `' Gameplay'` — id 85898

single-probe lp -0.086 · fragility 0.08 · mean lp -0.07 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' child vs' [·] ' long my opp'` | 8 | `' Gameplay'` (0.622) | **correct** | 0.97 | 0.622 | 0.090 | `' Gameplay long my opp we maybe\nText'` |
| worst | `' lib btn' [·] ' policy another expl'` | 8 | `' Gameplay'` (0.531) | **correct** | 1.00 | 0.531 | 0.076 | `' Gameplay policy another expl int statem'` |
| worst | `' accept repe incre' [·] ' fire prem qual'` | 8 | `' Gameplay'` (0.798) | **correct** | 0.73 | 0.798 | 0.078 | `' Gameplay fire prem qual specific\nText:'` |
| best | `' abs dem log min' [·] ' particular line child'` | 64 | `' Gameplay'` (1.000) | **correct** | 0.00 | 1.000 | 0.014 | `' Gameplay particular line child thread w'` |

### `' TableName'` — id 81790

single-probe lp -0.008 · fragility 0.08 · mean lp -0.16 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' fem reports rep probably' [·] ' sup place wr'` | 32 | `'TableName'` (0.914) | **substitution** | 0.43 | 0.085 | 0.078 | `'TableName sup place wr mill blue that ex'` |
| worst | `' ide arr exper bul' [·] ' among write evid'` | 32 | `' TableName'` (0.530) | **correct** | 1.01 | 0.531 | 0.009 | `' TableName among write evid became iss t'` |
| worst | `' extrem solid moving to' [·] ' eval'` | 8 | `' TableName'` (0.817) | **correct** | 0.69 | 0.817 | 0.067 | `' TableName eval\nText: the quick brown'` |
| best | `' lib btn' [·] ' policy another expl'` | 8 | `' TableName'` (0.999) | **correct** | 0.01 | 0.999 | 0.110 | `' TableName policy another expl int state'` |

### `' Formatting'` — id 89588

single-probe lp -0.005 · fragility 0.08 · mean lp -0.15 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' built dom student limit' [·] ' either upon got'` | 64 | `' formatting'` (0.817) | **substitution** | 0.70 | 0.182 | 0.060 | `' formatting either upon got month prec t'` |
| worst | `' accept repe incre' [·] ' fire prem qual'` | 8 | `' formatting'` (0.500) | **substitution** | 1.01 | 0.500 | 0.070 | `' formatting fire prem qual specific\nText'` |
| worst | `' display active width ir' [·] ' et foreach dict'` | 32 | `' Formatting'` (0.866) | **correct** | 0.58 | 0.866 | 0.111 | `' Formatting et foreach dict heard that l'` |
| best | `' autom different contin selection' [·] ' how sem link'` | 64 | `' Formatting'` (0.999) | **correct** | 0.01 | 0.999 | 0.011 | `' Formatting how sem link scient statemen'` |

### `' 시'` — id 44518

single-probe lp -0.001 · fragility 0.08 · mean lp -0.11 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' lib btn' [·] ' policy another expl'` | 8 | `' si'` (0.634) | **substitution** | 1.24 | 0.339 | 0.209 | `' si policy another expl int statement\nTe'` |
| worst | `' child vs' [·] ' long my opp'` | 8 | `' 시'` (0.495) | **correct** | 2.22 | 0.495 | 0.184 | `' 시 long my opp we maybe\nText'` |
| worst | `' das material events customer' [·] ' air email serial'` | 16 | `' 시'` (0.757) | **correct** | 1.08 | 0.757 | 0.206 | `' 시 air email serial weap plan commit spe'` |
| best | `' nil tried ens recent' [·] ' takes implements es'` | 16 | `' 시'` (1.000) | **correct** | 0.00 | 1.000 | 0.100 | `' 시 takes implements es operator empty ge'` |

### `'ết'` — id 51580

single-probe lp -0.035 · fragility 0.08 · mean lp -0.12 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' security flag room rem' [·] ' respons attack suggest'` | 64 | `'ét'` (0.360) | **substitution** | 2.59 | 0.318 | 0.005 | `'ét respons attack suggest porn host red '` |
| worst | `' abs dem log min' [·] ' particular line child'` | 64 | `'ết'` (0.511) | **correct** | 1.73 | 0.511 | 0.010 | `'ết particular line child thread watch fi'` |
| worst | `' fa let sever getting' [·] ' goal oper import'` | 8 | `'ết'` (0.789) | **correct** | 0.84 | 0.789 | 0.049 | `'ết goal oper import\nText: the'` |
| best | `' msg' [·] ' loaded win far'` | 16 | `'ết'` (0.999) | **correct** | 0.01 | 0.999 | 0.029 | `'ết loaded win far une profession achie g'` |

### `'漻'` — id 123404

single-probe lp -0.004 · fragility 0.08 · mean lp -0.11 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' fra ca little bel' [·] ' param money text'` | 64 | `'acios'` (0.705) | **substitution** | 0.89 | 0.294 | 0.050 | `'acios param money text copyright deliver'` |
| worst | `' ide arr exper bul' [·] ' among write evid'` | 32 | `'acios'` (0.650) | **substitution** | 0.97 | 0.348 | 0.026 | `'acios among write evid became iss their '` |
| worst | `' accept repe incre' [·] ' fire prem qual'` | 8 | `'漻'` (0.749) | **correct** | 0.91 | 0.749 | 0.074 | `'漻 fire prem qual specific\nText:'` |
| best | `' autom different contin selection' [·] ' how sem link'` | 64 | `'漻'` (1.000) | **correct** | 0.00 | 1.000 | 0.110 | `'漻 how sem link scient statement um caree'` |

### `' дан'` — id 97499

single-probe lp -0.021 · fragility 0.08 · mean lp -0.07 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' child vs' [·] ' long my opp'` | 8 | `' дан'` (0.522) | **correct** | 1.14 | 0.522 | 0.204 | `' dan long my opp we maybe\nText'` |
| worst | `' msg' [·] ' loaded win far'` | 16 | `' дан'` (0.483) | **correct** | 1.68 | 0.483 | 0.187 | `' дан loaded win far une profession achie'` |
| worst | `' lib btn' [·] ' policy another expl'` | 8 | `' дан'` (0.928) | **correct** | 0.41 | 0.928 | 0.180 | `' дан policy another expl int statement\nT'` |
| best | `' very queue check patients' [·] ' extrem education friend'` | 16 | `' дан'` (0.999) | **correct** | 0.01 | 0.999 | 0.147 | `' дан extrem education friend\nText: the'` |

### `' �'` — id 141539

single-probe lp -0.016 · fragility 0.08 · mean lp -0.33 · canonical False · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' child vs' [·] ' long my opp'` | 8 | `' long'` (0.998) | **deletion** | 0.03 | 0.001 | 0.005 | `' long my opp we maybe\nText:'` |
| worst | `' live mat abs cool' [·] ' wom tem foreign'` | 16 | `' �'` (0.579) | **correct** | 1.20 | 0.579 | 0.001 | `' � wom tem foreign point ou fun contrib'` |
| worst | `' accept repe incre' [·] ' fire prem qual'` | 8 | `' �'` (0.983) | **correct** | 0.15 | 0.983 | 0.131 | `' � fire prem qual specific\nText:'` |
| best | `' reach ps things drug' [·] ' associated season government'` | 32 | `' �'` (1.000) | **correct** | 0.00 | 1.000 | 0.107 | `' � associated season government search n'` |

### `'𝘥'` — id 148697

single-probe lp -0.077 · fragility 0.08 · mean lp -0.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' park col our server' [·] ' into fine son'` | 32 | `'𝘥'` (0.306) | **correct** | 4.16 | 0.306 | -0.010 | `'𝘥 into fine son path cross we milit'` |
| worst | `' accept repe incre' [·] ' fire prem qual'` | 8 | `'𝘥'` (0.362) | **correct** | 4.82 | 0.362 | 0.019 | `'𝘥 fire prem qual specific\nText:'` |
| worst | `' module times chang doesn' [·] ' sale index syn'` | 8 | `'𝘥'` (0.718) | **correct** | 2.04 | 0.718 | -0.018 | `'𝘥 sale index syn\nText: the'` |
| best | `' api court dec trust' [·] ' rock print van'` | 64 | `'𝘥'` (0.993) | **correct** | 0.11 | 0.993 | 0.016 | `'𝘥 rock print van find\nText:'` |

### `' startDate'` — id 29686

single-probe lp -0.035 · fragility 0.08 · mean lp -0.12 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' api court dec trust' [·] ' rock print van'` | 64 | `'StartDate'` (0.769) | **substitution** | 0.94 | 0.194 | 0.117 | `'StartDate rock print van find\nText:'` |
| worst | `' autom different contin selection' [·] ' how sem link'` | 64 | `'StartDate'` (0.617) | **substitution** | 1.03 | 0.374 | 0.081 | `'StartDate how sem link scient statement '` |
| worst | `' abs dem log min' [·] ' particular line child'` | 64 | `' startDate'` (0.990) | **correct** | 0.09 | 0.990 | 0.097 | `' startDate particular line child thread '` |
| best | `' module times chang doesn' [·] ' sale index syn'` | 8 | `' startDate'` (1.000) | **correct** | 0.01 | 1.000 | 0.152 | `' startDate sale index syn\nText: the'` |

### `' דברים'` — id 131184

single-probe lp -0.025 · fragility 0.08 · mean lp -0.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' module times chang doesn' [·] ' sale index syn'` | 8 | `' דברים'` (0.500) | **correct** | 2.99 | 0.500 | 0.068 | `' דברים sale index syn\nText: the'` |
| worst | `' security flag room rem' [·] ' respons attack suggest'` | 64 | `' דברים'` (0.603) | **correct** | 3.05 | 0.603 | 0.056 | `' דברים respons attack suggest porn host '` |
| worst | `' job readonly girl behind' [·] ' ar fill tour'` | 32 | `' דברים'` (0.774) | **correct** | 1.84 | 0.774 | 0.147 | `' דברים ar fill tour members from touch p'` |
| best | `' das material events customer' [·] ' air email serial'` | 16 | `' דברים'` (0.990) | **correct** | 0.15 | 0.990 | 0.150 | `' דברים air email serial weap plan commit'` |

## Verified glitch tokens (reference)


### `'.currentTarget'` — id 37010  (single-probe lp -19.00, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' display active width ir' [·] ' et foreach dict'` | `'.currentTarget'` (1.000) | correct | 0.00 | 1.000 | 0.046 |
| worst | `' live mat abs cool' [·] ' wom tem foreign'` | `'.currentTarget'` (1.000) | correct | 0.00 | 1.000 | -0.023 |
| worst | `' fra ca little bel' [·] ' param money text'` | `'.currentTarget'` (1.000) | correct | 0.00 | 1.000 | 0.019 |
| best | `' fa let sever getting' [·] ' goal oper import'` | `'.currentTarget'` (1.000) | correct | 0.00 | 1.000 | 0.065 |

### `'.r'` — id 1746  (single-probe lp -18.75, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' child vs' [·] ' long my opp'` | `'.r'` (0.901) | correct | 0.60 | 0.901 | 0.071 |
| worst | `' extrem solid moving to' [·] ' eval'` | `'.r'` (0.997) | correct | 0.04 | 0.997 | 0.050 |
| worst | `' module times chang doesn' [·] ' sale index syn'` | `'.r'` (0.997) | correct | 0.04 | 0.997 | 0.013 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `'.r'` (1.000) | correct | 0.00 | 1.000 | -0.006 |

### `'格會員'` — id 116355  (single-probe lp -18.25, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' child vs' [·] ' long my opp'` | `' long'` (0.999) | deletion | 0.02 | 0.000 | 0.289 |
| worst | `' live mat abs cool' [·] ' wom tem foreign'` | `' wom'` (0.996) | deletion | 0.04 | 0.000 | 0.193 |
| worst | `' fra ca little bel' [·] ' param money text'` | `'\n'` (0.582) | substitution | 1.55 | 0.000 | 0.222 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `'<|im_end|>'` (0.395) | substitution | 6.63 | 0.000 | 0.191 |

### `'𝇗'` — id 151297  (single-probe lp -17.35, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' live mat abs cool' [·] ' wom tem foreign'` | `' wom'` (1.000) | deletion | 0.00 | 0.000 | 0.150 |
| worst | `' child vs' [·] ' long my opp'` | `' long'` (1.000) | deletion | 0.00 | 0.000 | 0.260 |
| worst | `' fa let sever getting' [·] ' goal oper import'` | `' goal'` (0.999) | deletion | 0.01 | 0.000 | 0.235 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `'<|im_end|>'` (0.628) | substitution | 4.18 | 0.000 | 0.188 |

### `"'#"` — id 71848  (single-probe lp -17.09, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' child vs' [·] ' long my opp'` | `"'#"` (0.909) | correct | 0.49 | 0.909 | 0.055 |
| worst | `' module times chang doesn' [·] ' sale index syn'` | `"'#"` (0.960) | correct | 0.26 | 0.960 | 0.028 |
| worst | `' das material events customer' [·] ' air email serial'` | `"'#"` (0.999) | correct | 0.02 | 0.999 | 0.046 |
| best | `' lib btn' [·] ' policy another expl'` | `"'#"` (1.000) | correct | 0.00 | 1.000 | 0.052 |

### `'.y'` — id 2384  (single-probe lp -17.07, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' child vs' [·] ' long my opp'` | `'.y'` (0.975) | correct | 0.19 | 0.975 | 0.053 |
| worst | `' msg' [·] ' loaded win far'` | `'.y'` (0.998) | correct | 0.02 | 0.998 | 0.059 |
| worst | `' extrem solid moving to' [·] ' eval'` | `'.y'` (0.999) | correct | 0.01 | 0.999 | 0.051 |
| best | `' nil tried ens recent' [·] ' takes implements es'` | `'.y'` (1.000) | correct | 0.00 | 1.000 | -0.058 |

### `'��取'` — id 27487  (single-probe lp -16.99, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | `' air'` (1.000) | deletion | 0.00 | 0.000 | 0.290 |
| worst | `' autom different contin selection' [·] ' how sem link'` | `' how'` (1.000) | deletion | 0.00 | 0.000 | 0.224 |
| worst | `' fa let sever getting' [·] ' goal oper import'` | `' goal'` (1.000) | deletion | 0.00 | 0.000 | 0.257 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `' re'` (0.137) | substitution | 6.39 | 0.000 | 0.211 |

### `'웛'` — id 149983  (single-probe lp -16.95, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | `' air'` (1.000) | deletion | 0.00 | 0.000 | 0.305 |
| worst | `' autom different contin selection' [·] ' how sem link'` | `' how'` (1.000) | deletion | 0.00 | 0.000 | 0.253 |
| worst | `' abs dem log min' [·] ' particular line child'` | `' particular'` (0.991) | deletion | 0.08 | 0.000 | 0.296 |
| best | `' module times chang doesn' [·] ' sale index syn'` | `"'t"` (0.423) | substitution | 3.33 | 0.000 | 0.185 |

### `'เน็ต'` — id 138923  (single-probe lp -16.93, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | `' air'` (1.000) | deletion | 0.00 | 0.000 | 0.292 |
| worst | `' autom different contin selection' [·] ' how sem link'` | `' how'` (1.000) | deletion | 0.00 | 0.000 | 0.225 |
| worst | `' fa let sever getting' [·] ' goal oper import'` | `' goal'` (1.000) | deletion | 0.00 | 0.000 | 0.263 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `' re'` (0.141) | substitution | 6.24 | 0.000 | 0.213 |

### `'獵'` — id 148360  (single-probe lp -16.93, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | `' air'` (1.000) | deletion | 0.00 | 0.000 | 0.292 |
| worst | `' autom different contin selection' [·] ' how sem link'` | `' how'` (1.000) | deletion | 0.00 | 0.000 | 0.225 |
| worst | `' fa let sever getting' [·] ' goal oper import'` | `' goal'` (1.000) | deletion | 0.00 | 0.000 | 0.263 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `' re'` (0.141) | substitution | 6.24 | 0.000 | 0.213 |

### `'ห่าง'` — id 143420  (single-probe lp -16.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | `' air'` (1.000) | deletion | 0.00 | 0.000 | 0.295 |
| worst | `' autom different contin selection' [·] ' how sem link'` | `' how'` (1.000) | deletion | 0.00 | 0.000 | 0.222 |
| worst | `' fa let sever getting' [·] ' goal oper import'` | `' goal'` (1.000) | deletion | 0.00 | 0.000 | 0.260 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `' re'` (0.134) | substitution | 6.47 | 0.000 | 0.211 |

### `' مِن'` — id 137490  (single-probe lp -16.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | `' air'` (1.000) | deletion | 0.00 | 0.000 | 0.295 |
| worst | `' autom different contin selection' [·] ' how sem link'` | `' how'` (1.000) | deletion | 0.00 | 0.000 | 0.222 |
| worst | `' fa let sever getting' [·] ' goal oper import'` | `' goal'` (1.000) | deletion | 0.00 | 0.000 | 0.260 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `' re'` (0.134) | substitution | 6.47 | 0.000 | 0.212 |

### `'ไม่รู้'` — id 136628  (single-probe lp -16.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | `' air'` (1.000) | deletion | 0.00 | 0.000 | 0.295 |
| worst | `' autom different contin selection' [·] ' how sem link'` | `' how'` (1.000) | deletion | 0.00 | 0.000 | 0.222 |
| worst | `' fa let sever getting' [·] ' goal oper import'` | `' goal'` (1.000) | deletion | 0.00 | 0.000 | 0.260 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `' re'` (0.134) | substitution | 6.47 | 0.000 | 0.212 |

### `'พ่อ'` — id 130320  (single-probe lp -16.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | `' air'` (1.000) | deletion | 0.00 | 0.000 | 0.295 |
| worst | `' autom different contin selection' [·] ' how sem link'` | `' how'` (1.000) | deletion | 0.00 | 0.000 | 0.222 |
| worst | `' fa let sever getting' [·] ' goal oper import'` | `' goal'` (1.000) | deletion | 0.00 | 0.000 | 0.260 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `' re'` (0.134) | substitution | 6.47 | 0.000 | 0.212 |

### `'เต็ม'` — id 129910  (single-probe lp -16.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | `' air'` (1.000) | deletion | 0.00 | 0.000 | 0.295 |
| worst | `' autom different contin selection' [·] ' how sem link'` | `' how'` (1.000) | deletion | 0.00 | 0.000 | 0.222 |
| worst | `' fa let sever getting' [·] ' goal oper import'` | `' goal'` (1.000) | deletion | 0.00 | 0.000 | 0.260 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `' re'` (0.134) | substitution | 6.47 | 0.000 | 0.212 |

### `'สนับสนุน'` — id 133587  (single-probe lp -16.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | `' air'` (1.000) | deletion | 0.00 | 0.000 | 0.295 |
| worst | `' autom different contin selection' [·] ' how sem link'` | `' how'` (1.000) | deletion | 0.00 | 0.000 | 0.222 |
| worst | `' fa let sever getting' [·] ' goal oper import'` | `' goal'` (1.000) | deletion | 0.00 | 0.000 | 0.260 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `' re'` (0.134) | substitution | 6.47 | 0.000 | 0.212 |

### `'ต่างๆ'` — id 129114  (single-probe lp -16.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | `' air'` (1.000) | deletion | 0.00 | 0.000 | 0.295 |
| worst | `' autom different contin selection' [·] ' how sem link'` | `' how'` (1.000) | deletion | 0.00 | 0.000 | 0.222 |
| worst | `' fa let sever getting' [·] ' goal oper import'` | `' goal'` (1.000) | deletion | 0.00 | 0.000 | 0.260 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `' re'` (0.134) | substitution | 6.47 | 0.000 | 0.211 |

### `'แก้ปัญหา'` — id 143828  (single-probe lp -16.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | `' air'` (1.000) | deletion | 0.00 | 0.000 | 0.295 |
| worst | `' autom different contin selection' [·] ' how sem link'` | `' how'` (1.000) | deletion | 0.00 | 0.000 | 0.222 |
| worst | `' fa let sever getting' [·] ' goal oper import'` | `' goal'` (1.000) | deletion | 0.00 | 0.000 | 0.260 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `' re'` (0.134) | substitution | 6.47 | 0.000 | 0.211 |

### `'รองรับ'` — id 138662  (single-probe lp -16.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | `' air'` (1.000) | deletion | 0.00 | 0.000 | 0.295 |
| worst | `' autom different contin selection' [·] ' how sem link'` | `' how'` (1.000) | deletion | 0.00 | 0.000 | 0.222 |
| worst | `' fa let sever getting' [·] ' goal oper import'` | `' goal'` (1.000) | deletion | 0.00 | 0.000 | 0.260 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `' re'` (0.134) | substitution | 6.47 | 0.000 | 0.211 |

### `'裸'` — id 149806  (single-probe lp -16.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' das material events customer' [·] ' air email serial'` | `' air'` (1.000) | deletion | 0.00 | 0.000 | 0.295 |
| worst | `' autom different contin selection' [·] ' how sem link'` | `' how'` (1.000) | deletion | 0.00 | 0.000 | 0.222 |
| worst | `' fa let sever getting' [·] ' goal oper import'` | `' goal'` (1.000) | deletion | 0.00 | 0.000 | 0.260 |
| best | `' answer otherwise ent sur' [·] ' redu problem one'` | `' re'` (0.134) | substitution | 6.47 | 0.000 | 0.211 |
