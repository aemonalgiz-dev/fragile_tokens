# Confident-substitution specimens

Every example in this file was produced by greedy decoding; `p` is the model's probability for the token it emitted, `H` the entropy in bits at that position, `p(target)` its probability for the correct token, `ret` the cosine between the last-layer slot state and the correct token's unembedding row (identity still pointing at its readout while the readout chose otherwise).

## Model

- **model**: `Qwen/Qwen3-235B-A22B-FP8`
- **commit**: `39eb2b067ea6b8e3e1dd97d3cd0c7ffeaf3e1a35`
- **dtype**: `torch.bfloat16`
- **num_layers**: `94`
- **hidden_size**: `4096`
- **vocab_size**: `151936`
- **tie_word_embeddings**: `False`
- **tokenizer_class**: `Qwen2Tokenizer`
- **architecture**: `Qwen3MoeForCausalLM`
- **torch**: `2.14.0+cu130`
- **transformers**: `5.16.1`
- **python**: `3.10.12`
- **device**: `cuda:0`
- **decoding**: `greedy`
- **prompt_head**: `Repeat the text exactly.\nText: apple pie is good\nCopy: apple pie is good\nText: the quick brown fox\nCopy: the quick brown fox\nText:<context>\nCopy:<context-prefix>`
- **quantized**: `True`
- **fail_threshold_lp**: `-0.5`
- **n_contexts**: `24`
- **clean_gate_lp**: `-0.1`

## Fragile clean-looking tokens (pass the single probe; fail in some contexts)


### `'ཧ'` — id 150460

single-probe lp -0.021 · fragility 1.00 · mean lp -8.84 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.994) | **deletion** | 0.06 | 0.000 | 0.067 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.988) | **deletion** | 0.13 | 0.000 | 0.080 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.995) | **deletion** | 0.06 | 0.000 | 0.064 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'\n\n'` (0.536) | **substitution** | 1.89 | 0.024 | 0.108 | `''` |

### `'ঙ'` — id 148571

single-probe lp -0.003 · fragility 1.00 · mean lp -3.53 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.935) | **deletion** | 0.57 | 0.015 | 0.036 | `''` |
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.947) | **deletion** | 0.42 | 0.004 | 0.021 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.983) | **deletion** | 0.14 | 0.001 | 0.028 | `''` |
| best | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'ঙ'` (0.657) | **correct** | 1.62 | 0.657 | 0.065 | `''` |

### `'🚆'` — id 148722

single-probe lp -0.009 · fragility 1.00 · mean lp -13.55 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.994) | **deletion** | 0.06 | 0.000 | 0.031 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.988) | **deletion** | 0.13 | 0.000 | 0.023 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.522) | **substitution** | 2.05 | 0.000 | 0.019 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'\n'` (0.345) | **substitution** | 3.13 | 0.032 | 0.048 | `''` |

### `'ལ'` — id 148576

single-probe lp -0.010 · fragility 1.00 · mean lp -7.56 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.981) | **deletion** | 0.19 | 0.000 | 0.095 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.994) | **deletion** | 0.08 | 0.000 | 0.084 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.463) | **substitution** | 2.34 | 0.000 | 0.088 | `''` |
| best | `' den namespace items pas' [·] ' person cont sound'` | 64 | `'ལ'` (0.610) | **correct** | 2.64 | 0.610 | 0.105 | `''` |

### `'ම'` — id 147514

single-probe lp -0.020 · fragility 1.00 · mean lp -9.64 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.991) | **deletion** | 0.11 | 0.000 | 0.042 | `''` |
| worst | `' las touch recently hard' [·] ' beh func eff'` | 8 | `'\n'` (0.271) | **substitution** | 3.47 | 0.000 | 0.013 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.974) | **deletion** | 0.27 | 0.000 | 0.047 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'\n\n'` (0.327) | **substitution** | 4.37 | 0.094 | 0.086 | `''` |

### `' לתת'` — id 133211

single-probe lp -0.022 · fragility 1.00 · mean lp -6.45 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `'<|im_end|>'` (0.484) | **substitution** | 3.32 | 0.000 | -0.036 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.533) | **deletion** | 3.23 | 0.000 | -0.020 | `''` |
| worst | `' fund port sw struct' [·] ' term echo rev'` | 64 | `'<|im_end|>'` (0.600) | **substitution** | 3.44 | 0.000 | -0.044 | `''` |
| best | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' לתת'` (0.181) | **correct** | 7.36 | 0.181 | 0.008 | `''` |

### `' והת'` — id 133258

single-probe lp -0.002 · fragility 1.00 · mean lp -3.97 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `"'t"` (0.408) | **substitution** | 4.80 | 0.000 | -0.007 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `'leftrightarrow'` (0.073) | **substitution** | 8.86 | 0.000 | 0.002 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `'战略合作'` (0.109) | **substitution** | 7.55 | 0.005 | 0.013 | `''` |
| best | `' it staff when said' [·] ' account vers'` | 8 | `' והת'` (0.487) | **correct** | 5.20 | 0.487 | 0.028 | `''` |

### `'렇'` — id 124934

single-probe lp -0.057 · fragility 1.00 · mean lp -5.10 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.983) | **deletion** | 0.20 | 0.000 | 0.007 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.632) | **deletion** | 1.95 | 0.000 | 0.020 | `''` |
| worst | `' fund port sw struct' [·] ' term echo rev'` | 64 | `' term'` (0.722) | **deletion** | 2.30 | 0.001 | -0.015 | `''` |
| best | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'\n'` (0.304) | **substitution** | 3.90 | 0.237 | 0.031 | `''` |

### `'𝕒'` — id 146898

single-probe lp -0.018 · fragility 1.00 · mean lp -6.16 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (1.000) | **deletion** | 0.00 | 0.000 | 0.041 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.997) | **deletion** | 0.05 | 0.000 | 0.076 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.991) | **deletion** | 0.10 | 0.000 | 0.044 | `''` |
| best | `' opport cent positive quality' [·] ' weight dispatch ideas'` | 64 | `' weight'` (0.582) | **deletion** | 2.61 | 0.089 | 0.038 | `''` |

### `'펙'` — id 144609

single-probe lp -0.041 · fragility 1.00 · mean lp -5.76 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.992) | **deletion** | 0.11 | 0.000 | 0.024 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.995) | **deletion** | 0.06 | 0.000 | 0.015 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `' ax'` (0.629) | **deletion** | 2.22 | 0.000 | 0.044 | `''` |
| best | `' it staff when said' [·] ' account vers'` | 8 | `'펙'` (0.472) | **correct** | 3.44 | 0.472 | 0.018 | `''` |

### `'윽'` — id 145360

single-probe lp -0.097 · fragility 1.00 · mean lp -8.09 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.997) | **deletion** | 0.03 | 0.000 | -0.007 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.984) | **deletion** | 0.18 | 0.000 | 0.008 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.322) | **substitution** | 2.68 | 0.000 | 0.025 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'unction'` (0.389) | **substitution** | 2.84 | 0.303 | 0.030 | `''` |

### `'僔'` — id 123365

single-probe lp -0.010 · fragility 1.00 · mean lp -6.46 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (1.000) | **deletion** | 0.00 | 0.000 | 0.039 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.982) | **deletion** | 0.22 | 0.000 | 0.043 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.256) | **substitution** | 3.53 | 0.000 | 0.050 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'僔'` (0.350) | **correct** | 4.53 | 0.350 | 0.066 | `''` |

### `'بدو'` — id 127961

single-probe lp -0.014 · fragility 1.00 · mean lp -7.06 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.396) | **substitution** | 2.61 | 0.000 | 0.026 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `'\n'` (0.429) | **substitution** | 2.42 | 0.000 | 0.042 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.939) | **deletion** | 0.55 | 0.000 | 0.061 | `''` |
| best | `' den namespace items pas' [·] ' person cont sound'` | 64 | `'dojo'` (0.734) | **substitution** | 1.63 | 0.164 | 0.022 | `''` |

### `'شهور'` — id 143044

single-probe lp -0.081 · fragility 1.00 · mean lp -4.29 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'<|im_end|>'` (0.339) | **substitution** | 2.41 | 0.000 | 0.040 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `'</think>'` (0.397) | **substitution** | 2.68 | 0.000 | 0.014 | `''` |
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'\n'` (0.332) | **substitution** | 2.51 | 0.000 | 0.072 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'شهور'` (0.424) | **correct** | 2.78 | 0.424 | 0.016 | `''` |

### `'احتجاج'` — id 142689

single-probe lp -0.040 · fragility 1.00 · mean lp -9.97 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'\n'` (0.501) | **substitution** | 1.71 | 0.000 | 0.044 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.457) | **substitution** | 2.09 | 0.000 | 0.047 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.402) | **deletion** | 2.65 | 0.000 | -0.010 | `''` |
| best | `' origin child below namespace' [·] ' lock home rece'` | 32 | `' lock'` (0.339) | **deletion** | 2.60 | 0.046 | 0.019 | `''` |

### `'חוץ'` — id 129918

single-probe lp -0.010 · fragility 1.00 · mean lp -16.11 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.978) | **deletion** | 0.20 | 0.000 | -0.013 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.962) | **deletion** | 0.38 | 0.000 | -0.019 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.953) | **deletion** | 0.39 | 0.000 | 0.001 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'<|im_end|>'` (0.421) | **substitution** | 3.50 | 0.000 | 0.001 | `''` |

### `'גות'` — id 132867

single-probe lp -0.066 · fragility 1.00 · mean lp -10.13 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.946) | **deletion** | 0.55 | 0.000 | 0.030 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.922) | **deletion** | 0.55 | 0.000 | 0.027 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.975) | **deletion** | 0.28 | 0.000 | 0.029 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'\n\n'` (0.301) | **substitution** | 4.81 | 0.004 | 0.035 | `''` |

### `'镘'` — id 121776

single-probe lp -0.059 · fragility 1.00 · mean lp -4.59 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.983) | **deletion** | 0.15 | 0.000 | 0.026 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.994) | **deletion** | 0.06 | 0.000 | 0.037 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.631) | **substitution** | 1.56 | 0.000 | 0.044 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'镘'` (0.787) | **correct** | 1.60 | 0.787 | 0.015 | `''` |

### `'嗐'` — id 121211

single-probe lp -0.018 · fragility 0.96 · mean lp -6.57 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.977) | **deletion** | 0.22 | 0.000 | 0.063 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.960) | **deletion** | 0.38 | 0.000 | 0.060 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.564) | **substitution** | 2.14 | 0.000 | 0.052 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'嗐'` (0.830) | **correct** | 1.28 | 0.829 | 0.073 | `''` |

### `'בחר'` — id 129930

single-probe lp -0.059 · fragility 0.96 · mean lp -5.64 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'\n'` (0.411) | **substitution** | 2.30 | 0.000 | 0.038 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.581) | **substitution** | 2.46 | 0.000 | 0.020 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.821) | **deletion** | 1.58 | 0.000 | 0.034 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'בחר'` (0.844) | **correct** | 1.59 | 0.844 | 0.058 | `''` |

### `'работать'` — id 132991

single-probe lp -0.056 · fragility 0.96 · mean lp -4.76 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'<|im_end|>'` (0.345) | **substitution** | 2.37 | 0.000 | 0.052 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.975) | **deletion** | 0.26 | 0.000 | 0.070 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `'<|im_end|>'` (0.641) | **substitution** | 2.00 | 0.003 | 0.033 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'работать'` (0.599) | **correct** | 2.00 | 0.599 | 0.006 | `''` |

### `'سبة'` — id 126014

single-probe lp -0.017 · fragility 0.96 · mean lp -6.73 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.420) | **substitution** | 1.99 | 0.000 | 0.034 | `''` |
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'\n'` (0.431) | **substitution** | 2.45 | 0.000 | 0.042 | `''` |
| worst | `' origin child below namespace' [·] ' lock home rece'` | 32 | `'</think>'` (0.408) | **substitution** | 2.69 | 0.000 | -0.010 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'سبة'` (0.999) | **correct** | 0.01 | 0.999 | 0.008 | `''` |

### `'ம'` — id 146150

single-probe lp -0.038 · fragility 0.96 · mean lp -4.95 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.575) | **substitution** | 1.66 | 0.000 | 0.052 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `' under'` (0.980) | **deletion** | 0.21 | 0.000 | 0.072 | `''` |
| worst | `' fund port sw struct' [·] ' term echo rev'` | 64 | `' term'` (0.991) | **deletion** | 0.11 | 0.000 | 0.018 | `''` |
| best | `' pot content' [·] ' init abs comfort'` | 16 | `' init'` (0.527) | **deletion** | 1.51 | 0.410 | 0.066 | `''` |

### `'앉'` — id 144125

single-probe lp -0.009 · fragility 0.96 · mean lp -7.86 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.983) | **deletion** | 0.17 | 0.000 | 0.005 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.950) | **deletion** | 0.44 | 0.000 | 0.011 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.917) | **deletion** | 0.60 | 0.000 | 0.030 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'앉'` (0.984) | **correct** | 0.19 | 0.984 | 0.019 | `''` |

### `'נסה'` — id 132576

single-probe lp -0.000 · fragility 0.96 · mean lp -10.79 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.807) | **deletion** | 1.33 | 0.000 | 0.020 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.358) | **substitution** | 2.02 | 0.000 | 0.021 | `''` |
| worst | `' song released top sl' [·] ' under sus since'` | 32 | `'\n'` (0.428) | **substitution** | 2.84 | 0.000 | -0.011 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'נסה'` (0.693) | **correct** | 2.52 | 0.693 | 0.039 | `''` |

### `' באו'` — id 132355

single-probe lp -0.003 · fragility 0.96 · mean lp -5.94 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.718) | **deletion** | 1.52 | 0.000 | -0.017 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.545) | **deletion** | 2.03 | 0.000 | -0.022 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.654) | **deletion** | 2.20 | 0.000 | 0.017 | `''` |
| best | `' den namespace items pas' [·] ' person cont sound'` | 64 | `' באו'` (0.973) | **correct** | 0.27 | 0.973 | 0.004 | `''` |

### `' כולו'` — id 143460

single-probe lp -0.018 · fragility 0.96 · mean lp -8.27 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.936) | **deletion** | 0.53 | 0.000 | 0.020 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.973) | **deletion** | 0.26 | 0.000 | 0.019 | `''` |
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'\n'` (0.673) | **substitution** | 1.62 | 0.000 | 0.042 | `''` |
| best | `' pot content' [·] ' init abs comfort'` | 16 | `' כולו'` (0.781) | **correct** | 1.98 | 0.781 | 0.040 | `''` |

### `' рассмат'` — id 142145

single-probe lp -0.044 · fragility 0.96 · mean lp -8.81 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.746) | **deletion** | 1.56 | 0.000 | -0.025 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `'<|im_end|>'` (0.848) | **substitution** | 1.27 | 0.000 | -0.011 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.453) | **substitution** | 2.14 | 0.000 | -0.011 | `''` |
| best | `' den namespace items pas' [·] ' person cont sound'` | 64 | `' рассмат'` (0.909) | **correct** | 1.00 | 0.909 | 0.001 | `''` |

### `'שוק'` — id 126710

single-probe lp -0.002 · fragility 0.96 · mean lp -9.07 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.989) | **deletion** | 0.11 | 0.000 | 0.020 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.420) | **substitution** | 2.03 | 0.000 | 0.005 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.881) | **deletion** | 0.69 | 0.000 | -0.002 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'שוק'` (0.701) | **correct** | 2.11 | 0.701 | 0.031 | `''` |

### `'댁'` — id 145249

single-probe lp -0.004 · fragility 0.96 · mean lp -4.42 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.996) | **deletion** | 0.05 | 0.000 | 0.021 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.950) | **deletion** | 0.53 | 0.000 | 0.070 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `' ax'` (0.354) | **deletion** | 3.47 | 0.000 | 0.049 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'댁'` (0.994) | **correct** | 0.09 | 0.994 | 0.026 | `''` |

### `'дачи'` — id 134630

single-probe lp -0.050 · fragility 0.96 · mean lp -2.97 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.812) | **deletion** | 1.51 | 0.002 | 0.009 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.402) | **substitution** | 2.50 | 0.000 | 0.024 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.792) | **deletion** | 1.58 | 0.006 | 0.034 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'дачи'` (0.937) | **correct** | 0.68 | 0.937 | -0.012 | `''` |

### `' כשה'` — id 132284

single-probe lp -0.006 · fragility 0.96 · mean lp -11.13 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' fund port sw struct' [·] ' term echo rev'` | 64 | `' term'` (0.783) | **deletion** | 1.34 | 0.000 | -0.046 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `'</think>'` (0.659) | **substitution** | 1.82 | 0.000 | -0.012 | `''` |
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'</think>'` (0.767) | **substitution** | 1.31 | 0.000 | -0.014 | `''` |
| best | `' den namespace items pas' [·] ' person cont sound'` | 64 | `' כשה'` (0.465) | **correct** | 1.91 | 0.465 | 0.002 | `''` |

### `'𝙰'` — id 151576

single-probe lp -0.051 · fragility 0.96 · mean lp -3.82 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.01 | 0.000 | 0.052 | `''` |
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.978) | **deletion** | 0.21 | 0.000 | 0.058 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.940) | **deletion** | 0.51 | 0.001 | 0.069 | `''` |
| best | `' fund port sw struct' [·] ' term echo rev'` | 64 | `'𝙰'` (0.914) | **correct** | 0.63 | 0.914 | 0.047 | `''` |

### `'בי'` — id 124268

single-probe lp -0.048 · fragility 0.96 · mean lp -6.36 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.940) | **deletion** | 0.70 | 0.000 | 0.044 | `''` |
| worst | `' printf ne throw sy' [·] ' art human ver'` | 8 | `' bidi'` (0.457) | **substitution** | 3.00 | 0.000 | 0.036 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `' under'` (0.957) | **deletion** | 0.37 | 0.000 | 0.043 | `''` |
| best | `' origin child below namespace' [·] ' lock home rece'` | 32 | `'בי'` (0.775) | **correct** | 1.73 | 0.775 | 0.052 | `''` |

### `'ត'` — id 146568

single-probe lp -0.025 · fragility 0.96 · mean lp -5.02 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.997) | **deletion** | 0.04 | 0.000 | 0.040 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.948) | **deletion** | 0.53 | 0.000 | 0.069 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `' under'` (0.939) | **deletion** | 0.69 | 0.001 | 0.045 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'ត'` (0.417) | **correct** | 2.73 | 0.417 | 0.066 | `''` |

### `'镅'` — id 123357

single-probe lp -0.031 · fragility 0.92 · mean lp -3.90 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.998) | **deletion** | 0.03 | 0.000 | 0.026 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.967) | **deletion** | 0.30 | 0.000 | 0.057 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.979) | **deletion** | 0.22 | 0.001 | 0.054 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'镅'` (0.967) | **correct** | 0.34 | 0.967 | 0.037 | `''` |

### `'狴'` — id 122676

single-probe lp -0.001 · fragility 0.92 · mean lp -3.86 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.01 | 0.000 | 0.021 | `''` |
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.973) | **deletion** | 0.21 | 0.000 | 0.043 | `''` |
| worst | `' song released top sl' [·] ' under sus since'` | 32 | `' under'` (0.988) | **deletion** | 0.14 | 0.001 | 0.082 | `''` |
| best | `' origin child below namespace' [·] ' lock home rece'` | 32 | `'狴'` (0.987) | **correct** | 0.13 | 0.987 | 0.063 | `''` |

### `'ﺭ'` — id 144352

single-probe lp -0.078 · fragility 0.92 · mean lp -3.38 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.01 | 0.000 | 0.054 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.960) | **deletion** | 0.42 | 0.000 | 0.067 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.970) | **deletion** | 0.30 | 0.001 | 0.073 | `''` |
| best | `' bad consist' [·] ' assert signific cert'` | 8 | `'ﺭ'` (0.910) | **correct** | 1.01 | 0.911 | 0.085 | `''` |

### `'骀'` — id 120007

single-probe lp -0.010 · fragility 0.92 · mean lp -5.15 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.505) | **substitution** | 1.77 | 0.000 | 0.043 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.890) | **deletion** | 0.78 | 0.000 | 0.038 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.782) | **deletion** | 1.11 | 0.000 | 0.017 | `''` |
| best | `' word override script cur' [·] ' environment mis reach'` | 32 | `'骀'` (0.929) | **correct** | 0.62 | 0.929 | 0.032 | `''` |

### `'🤜'` — id 147964

single-probe lp -0.062 · fragility 0.92 · mean lp -4.72 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.01 | 0.000 | 0.001 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.989) | **deletion** | 0.11 | 0.000 | 0.035 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.943) | **deletion** | 0.43 | 0.000 | -0.010 | `''` |
| best | `' las touch recently hard' [·] ' beh func eff'` | 8 | `'🤜'` (0.978) | **correct** | 0.21 | 0.978 | 0.036 | `''` |

### `'𝗦'` — id 147178

single-probe lp -0.093 · fragility 0.92 · mean lp -3.84 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.992) | **deletion** | 0.10 | 0.000 | 0.081 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.974) | **deletion** | 0.25 | 0.000 | 0.066 | `''` |
| worst | `' pot content' [·] ' init abs comfort'` | 16 | `' init'` (0.980) | **deletion** | 0.16 | 0.000 | 0.115 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'𝗦'` (0.866) | **correct** | 0.95 | 0.866 | 0.121 | `''` |

### `'崚'` — id 120680

single-probe lp -0.028 · fragility 0.92 · mean lp -4.99 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.948) | **deletion** | 0.33 | 0.000 | 0.029 | `''` |
| worst | `' pot content' [·] ' init abs comfort'` | 16 | `' init'` (0.887) | **deletion** | 0.76 | 0.000 | 0.039 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.990) | **deletion** | 0.12 | 0.000 | 0.055 | `''` |
| best | `' las touch recently hard' [·] ' beh func eff'` | 8 | `'崚'` (0.967) | **correct** | 0.33 | 0.967 | -0.003 | `''` |

### `'暅'` — id 123185

single-probe lp -0.021 · fragility 0.92 · mean lp -5.21 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.02 | 0.000 | 0.013 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.902) | **deletion** | 0.57 | 0.000 | 0.038 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.989) | **deletion** | 0.12 | 0.000 | 0.046 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'暅'` (0.819) | **correct** | 1.72 | 0.819 | 0.003 | `''` |

### `'интер'` — id 142796

single-probe lp -0.018 · fragility 0.92 · mean lp -5.43 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' client met' [·] ' she len rais'` | 64 | `'inter'` (0.997) | **substitution** | 0.04 | 0.000 | -0.008 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `'inter'` (1.000) | **substitution** | 0.00 | 0.000 | -0.048 | `''` |
| worst | `' len' [·] ' mobile std drop'` | 8 | `'inter'` (0.997) | **substitution** | 0.04 | 0.000 | -0.011 | `''` |
| best | `' origin child below namespace' [·] ' lock home rece'` | 32 | `'интер'` (0.774) | **correct** | 1.17 | 0.774 | 0.023 | `''` |

### `'عراق'` — id 139185

single-probe lp -0.018 · fragility 0.92 · mean lp -5.75 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.407) | **substitution** | 1.96 | 0.000 | 0.044 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.821) | **deletion** | 1.26 | 0.000 | 0.025 | `''` |
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'\n'` (0.582) | **substitution** | 1.74 | 0.000 | 0.055 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'عراق'` (0.793) | **correct** | 1.66 | 0.793 | 0.040 | `''` |

### `'ことができます'` — id 127997

single-probe lp -0.063 · fragility 0.92 · mean lp -8.39 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.369) | **substitution** | 2.28 | 0.000 | 0.031 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.919) | **deletion** | 0.54 | 0.000 | 0.024 | `''` |
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'\n'` (0.679) | **substitution** | 1.46 | 0.000 | 0.038 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'ことができます'` (0.891) | **correct** | 0.86 | 0.891 | 0.002 | `''` |

### `' להעביר'` — id 140069

single-probe lp -0.074 · fragility 0.92 · mean lp -3.77 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'<|im_end|>'` (0.418) | **substitution** | 2.59 | 0.000 | -0.014 | `''` |
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'<|im_end|>'` (0.247) | **substitution** | 5.09 | 0.001 | -0.001 | `''` |
| worst | `' it staff when said' [·] ' account vers'` | 8 | `'<|im_end|>'` (0.499) | **substitution** | 4.03 | 0.007 | 0.012 | `''` |
| best | `' origin child below namespace' [·] ' lock home rece'` | 32 | `' להעביר'` (0.916) | **correct** | 0.97 | 0.916 | -0.015 | `''` |

### `'ἤ'` — id 148499

single-probe lp -0.013 · fragility 0.92 · mean lp -3.83 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.975) | **deletion** | 0.24 | 0.000 | 0.005 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.981) | **deletion** | 0.22 | 0.000 | 0.032 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.434) | **substitution** | 2.68 | 0.001 | -0.011 | `''` |
| best | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'ἤ'` (0.912) | **correct** | 0.90 | 0.912 | 0.058 | `''` |

### `'栟'` — id 120334

single-probe lp -0.001 · fragility 0.92 · mean lp -5.06 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.02 | 0.000 | 0.044 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `' ax'` (0.882) | **deletion** | 0.77 | 0.000 | 0.059 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `' under'` (0.995) | **deletion** | 0.07 | 0.000 | 0.083 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'栟'` (0.946) | **correct** | 0.44 | 0.946 | 0.056 | `''` |

### `'عا'` — id 125072

single-probe lp -0.052 · fragility 0.88 · mean lp -3.97 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' stuff' [·] ' function come tab'` | 8 | `'acia'` (0.464) | **substitution** | 3.96 | 0.000 | 0.017 | `''` |
| worst | `' las touch recently hard' [·] ' beh func eff'` | 8 | `'cea'` (0.240) | **substitution** | 5.21 | 0.001 | 0.002 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.873) | **deletion** | 1.10 | 0.003 | 0.016 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'عا'` (0.999) | **correct** | 0.02 | 0.999 | 0.008 | `''` |

### `'иона'` — id 126079

single-probe lp -0.020 · fragility 0.88 · mean lp -2.57 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.903) | **deletion** | 0.61 | 0.000 | 0.055 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.629) | **substitution** | 1.76 | 0.000 | 0.034 | `''` |
| worst | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'iona'` (0.927) | **substitution** | 0.53 | 0.009 | 0.030 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'иона'` (0.649) | **correct** | 0.97 | 0.649 | 0.016 | `''` |

### `'عظيم'` — id 135723

single-probe lp -0.003 · fragility 0.88 · mean lp -4.82 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.342) | **substitution** | 2.33 | 0.000 | 0.077 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `'\n'` (0.348) | **substitution** | 2.37 | 0.000 | 0.028 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.938) | **deletion** | 0.48 | 0.000 | 0.022 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'عظيم'` (0.997) | **correct** | 0.04 | 0.996 | 0.007 | `''` |

### `'跏'` — id 120922

single-probe lp -0.004 · fragility 0.88 · mean lp -6.48 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.986) | **deletion** | 0.12 | 0.000 | 0.037 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.980) | **deletion** | 0.21 | 0.000 | 0.043 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.875) | **deletion** | 1.00 | 0.000 | 0.024 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'跏'` (0.956) | **correct** | 0.39 | 0.956 | 0.018 | `''` |

### `'趔'` — id 121105

single-probe lp -0.028 · fragility 0.88 · mean lp -5.66 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `'</think>'` (0.299) | **substitution** | 2.69 | 0.000 | 0.032 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.993) | **deletion** | 0.07 | 0.000 | 0.030 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.885) | **deletion** | 0.86 | 0.000 | 0.021 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'趔'` (0.955) | **correct** | 0.37 | 0.955 | 0.046 | `''` |

### `'点了点头'` — id 112385

single-probe lp -0.001 · fragility 0.88 · mean lp -3.63 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'<|im_end|>'` (0.558) | **substitution** | 2.10 | 0.000 | 0.001 | `''` |
| worst | `' song released top sl' [·] ' under sus since'` | 32 | `' under'` (0.745) | **deletion** | 1.57 | 0.000 | 0.033 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.615) | **deletion** | 1.43 | 0.001 | 0.022 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'点了点头'` (0.945) | **correct** | 0.37 | 0.945 | 0.004 | `''` |

### `'使え'` — id 140708

single-probe lp -0.061 · fragility 0.88 · mean lp -3.45 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.332) | **substitution** | 2.98 | 0.000 | 0.016 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' reusable'` (0.432) | **substitution** | 2.99 | 0.002 | 0.029 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' usability'` (0.590) | **substitution** | 2.84 | 0.003 | 0.014 | `''` |
| best | `' pot content' [·] ' init abs comfort'` | 16 | `'使え'` (0.986) | **correct** | 0.17 | 0.986 | 0.010 | `''` |

### `'ܠ'` — id 146725

single-probe lp -0.030 · fragility 0.88 · mean lp -6.26 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.991) | **deletion** | 0.12 | 0.000 | 0.030 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.878) | **deletion** | 0.76 | 0.000 | 0.015 | `''` |
| worst | `' pot content' [·] ' init abs comfort'` | 16 | `' init'` (0.992) | **deletion** | 0.09 | 0.000 | 0.032 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'ܠ'` (0.946) | **correct** | 0.53 | 0.946 | 0.104 | `''` |

### `'읽'` — id 135124

single-probe lp -0.086 · fragility 0.88 · mean lp -2.65 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.997) | **deletion** | 0.03 | 0.000 | 0.040 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.995) | **deletion** | 0.06 | 0.000 | 0.006 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `' unread'` (0.501) | **substitution** | 1.92 | 0.005 | 0.032 | `''` |
| best | `' pot content' [·] ' init abs comfort'` | 16 | `'읽'` (0.890) | **correct** | 0.84 | 0.890 | 0.017 | `''` |

### `'ཀ'` — id 147474

single-probe lp -0.010 · fragility 0.88 · mean lp -3.57 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.993) | **deletion** | 0.06 | 0.000 | 0.072 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.965) | **deletion** | 0.30 | 0.000 | 0.104 | `''` |
| worst | `' fund port sw struct' [·] ' term echo rev'` | 64 | `' term'` (0.994) | **deletion** | 0.08 | 0.002 | 0.083 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'ཀ'` (0.928) | **correct** | 0.76 | 0.928 | 0.075 | `''` |

### `' אליו'` — id 134332

single-probe lp -0.031 · fragility 0.88 · mean lp -8.30 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' opport cent positive quality' [·] ' weight dispatch ideas'` | 64 | `' weight'` (0.463) | **deletion** | 2.30 | 0.000 | 0.008 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'</think>'` (0.427) | **substitution** | 2.19 | 0.000 | 0.030 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `'</think>'` (0.522) | **substitution** | 2.86 | 0.000 | -0.027 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `' אליו'` (0.744) | **correct** | 2.43 | 0.744 | 0.044 | `''` |

### `' всег'` — id 130355

single-probe lp -0.016 · fragility 0.88 · mean lp -4.43 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.818) | **deletion** | 1.22 | 0.000 | 0.034 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'<|im_end|>'` (0.725) | **substitution** | 1.67 | 0.000 | -0.001 | `''` |
| worst | `' thems line sem things' [·] ' range car player'` | 32 | `'<|im_end|>'` (0.802) | **substitution** | 1.61 | 0.000 | 0.032 | `''` |
| best | `' printf ne throw sy' [·] ' art human ver'` | 8 | `' всег'` (0.930) | **correct** | 0.65 | 0.930 | -0.000 | `''` |

### `'藟'` — id 123668

single-probe lp -0.003 · fragility 0.88 · mean lp -3.58 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.983) | **deletion** | 0.17 | 0.000 | 0.090 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.998) | **deletion** | 0.02 | 0.000 | 0.060 | `''` |
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.986) | **deletion** | 0.14 | 0.000 | 0.044 | `''` |
| best | `' thems line sem things' [·] ' range car player'` | 32 | `'藟'` (0.896) | **correct** | 0.72 | 0.896 | 0.122 | `''` |

### `'ഏ'` — id 150443

single-probe lp -0.008 · fragility 0.88 · mean lp -4.39 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (1.000) | **deletion** | 0.00 | 0.000 | 0.027 | `''` |
| worst | `' pot content' [·] ' init abs comfort'` | 16 | `' init'` (0.933) | **deletion** | 0.52 | 0.000 | 0.053 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.998) | **deletion** | 0.03 | 0.000 | 0.098 | `''` |
| best | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'ഏ'` (0.966) | **correct** | 0.40 | 0.966 | 0.102 | `''` |

### `'ར'` — id 147504

single-probe lp -0.001 · fragility 0.88 · mean lp -3.64 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.987) | **deletion** | 0.14 | 0.000 | 0.094 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.979) | **deletion** | 0.19 | 0.001 | 0.072 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.932) | **deletion** | 0.64 | 0.001 | 0.109 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'ར'` (0.937) | **correct** | 0.57 | 0.937 | 0.106 | `''` |

### `'⧫'` — id 149062

single-probe lp -0.006 · fragility 0.88 · mean lp -3.50 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pot content' [·] ' init abs comfort'` | 16 | `' init'` (0.875) | **deletion** | 0.91 | 0.001 | 0.037 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.01 | 0.000 | 0.001 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.993) | **deletion** | 0.07 | 0.000 | 0.014 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'⧫'` (0.971) | **correct** | 0.33 | 0.971 | 0.034 | `''` |

### `'بيض'` — id 134070

single-probe lp -0.003 · fragility 0.88 · mean lp -6.13 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.980) | **deletion** | 0.19 | 0.000 | 0.031 | `''` |
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'\n'` (0.473) | **substitution** | 1.86 | 0.000 | 0.071 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.575) | **deletion** | 2.00 | 0.000 | 0.032 | `''` |
| best | `' origin child below namespace' [·] ' lock home rece'` | 32 | `'بيض'` (0.901) | **correct** | 0.92 | 0.901 | 0.006 | `''` |

### `'杷'` — id 119843

single-probe lp -0.036 · fragility 0.88 · mean lp -4.28 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.983) | **deletion** | 0.16 | 0.000 | 0.053 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.993) | **deletion** | 0.09 | 0.000 | 0.016 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.792) | **deletion** | 1.16 | 0.000 | 0.036 | `''` |
| best | `' pot content' [·] ' init abs comfort'` | 16 | `'杷'` (0.954) | **correct** | 0.38 | 0.954 | 0.025 | `''` |

### `' MetroFramework'` — id 51355

single-probe lp -0.027 · fragility 0.88 · mean lp -6.97 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `' under'` (0.395) | **deletion** | 2.61 | 0.000 | 0.056 | `''` |
| worst | `' stuff' [·] ' function come tab'` | 8 | `'<|im_start|>'` (0.695) | **substitution** | 1.72 | 0.000 | 0.064 | `''` |
| worst | `' pot content' [·] ' init abs comfort'` | 16 | `'<|im_start|>'` (0.688) | **substitution** | 1.79 | 0.000 | 0.073 | `''` |
| best | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' MetroFramework'` (0.734) | **correct** | 1.67 | 0.734 | 0.063 | `''` |

### `' האוויר'` — id 141248

single-probe lp -0.013 · fragility 0.83 · mean lp -7.58 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.997) | **deletion** | 0.04 | 0.000 | 0.043 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'<|im_end|>'` (0.387) | **substitution** | 2.15 | 0.000 | 0.067 | `''` |
| worst | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'<|im_end|>'` (0.410) | **substitution** | 2.32 | 0.000 | 0.058 | `''` |
| best | `' opport cent positive quality' [·] ' weight dispatch ideas'` | 64 | `' האוויר'` (0.870) | **correct** | 1.02 | 0.870 | 0.059 | `''` |

### `'馃'` — id 120751

single-probe lp -0.033 · fragility 0.83 · mean lp -3.04 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.994) | **deletion** | 0.06 | 0.000 | 0.058 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.01 | 0.000 | 0.013 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.933) | **deletion** | 0.63 | 0.006 | 0.085 | `''` |
| best | `' thems line sem things' [·] ' range car player'` | 32 | `'馃'` (0.477) | **correct** | 1.77 | 0.477 | 0.078 | `''` |

### `'졌'` — id 128036

single-probe lp -0.017 · fragility 0.83 · mean lp -3.35 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.823) | **deletion** | 1.36 | 0.000 | 0.020 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `' under'` (0.815) | **deletion** | 1.47 | 0.002 | 0.026 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.447) | **substitution** | 2.56 | 0.001 | -0.037 | `''` |
| best | `' las touch recently hard' [·] ' beh func eff'` | 8 | `'졌'` (0.921) | **correct** | 0.63 | 0.921 | -0.013 | `''` |

### `'當您'` — id 114322

single-probe lp -0.003 · fragility 0.83 · mean lp -5.79 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.315) | **substitution** | 2.26 | 0.000 | -0.013 | `''` |
| worst | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'\n\n'` (0.317) | **substitution** | 2.29 | 0.000 | 0.013 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.988) | **deletion** | 0.14 | 0.000 | 0.001 | `''` |
| best | `' fund port sw struct' [·] ' term echo rev'` | 64 | `'當您'` (0.762) | **correct** | 1.62 | 0.763 | 0.001 | `''` |

### `'媸'` — id 121333

single-probe lp -0.003 · fragility 0.83 · mean lp -3.83 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.982) | **deletion** | 0.15 | 0.000 | 0.045 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.961) | **deletion** | 0.30 | 0.000 | 0.062 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.927) | **deletion** | 0.61 | 0.001 | 0.068 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'媸'` (0.980) | **correct** | 0.18 | 0.980 | 0.064 | `''` |

### `'ܥ'` — id 147447

single-probe lp -0.004 · fragility 0.83 · mean lp -4.32 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.970) | **deletion** | 0.22 | 0.000 | 0.070 | `''` |
| worst | `' pot content' [·] ' init abs comfort'` | 16 | `' init'` (0.972) | **deletion** | 0.24 | 0.000 | 0.071 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `' under'` (0.995) | **deletion** | 0.08 | 0.000 | 0.078 | `''` |
| best | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'ܥ'` (0.936) | **correct** | 0.88 | 0.936 | 0.091 | `''` |

### `'荽'` — id 120316

single-probe lp -0.014 · fragility 0.83 · mean lp -4.31 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.996) | **deletion** | 0.05 | 0.000 | 0.049 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `' ax'` (0.414) | **deletion** | 2.36 | 0.000 | 0.007 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.998) | **deletion** | 0.02 | 0.000 | 0.019 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'荽'` (0.979) | **correct** | 0.21 | 0.979 | 0.044 | `''` |

### `'ർ'` — id 148979

single-probe lp -0.018 · fragility 0.83 · mean lp -3.00 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.01 | 0.000 | 0.054 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.976) | **deletion** | 0.26 | 0.000 | 0.049 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.986) | **deletion** | 0.17 | 0.002 | 0.037 | `''` |
| best | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'ർ'` (0.993) | **correct** | 0.09 | 0.993 | 0.071 | `''` |

### `'柙'` — id 120061

single-probe lp -0.000 · fragility 0.83 · mean lp -3.77 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.853) | **deletion** | 0.81 | 0.000 | 0.051 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.953) | **deletion** | 0.44 | 0.008 | 0.038 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.995) | **deletion** | 0.05 | 0.000 | 0.030 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'柙'` (0.536) | **correct** | 1.46 | 0.536 | 0.064 | `''` |

### `'怄'` — id 119759

single-probe lp -0.005 · fragility 0.79 · mean lp -3.02 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.995) | **deletion** | 0.05 | 0.000 | 0.013 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.991) | **deletion** | 0.09 | 0.001 | 0.057 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.936) | **deletion** | 0.53 | 0.001 | 0.073 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'怄'` (0.963) | **correct** | 0.31 | 0.963 | 0.061 | `''` |

### `'螗'` — id 123543

single-probe lp -0.041 · fragility 0.79 · mean lp -2.40 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.974) | **deletion** | 0.25 | 0.001 | 0.072 | `''` |
| worst | `' opport cent positive quality' [·] ' weight dispatch ideas'` | 64 | `' weight'` (0.287) | **deletion** | 2.55 | 0.006 | 0.076 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.959) | **deletion** | 0.33 | 0.000 | 0.045 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'螗'` (0.982) | **correct** | 0.19 | 0.982 | 0.056 | `''` |

### `'త'` — id 147368

single-probe lp -0.003 · fragility 0.79 · mean lp -4.55 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.964) | **deletion** | 0.40 | 0.000 | 0.038 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.984) | **deletion** | 0.17 | 0.000 | 0.020 | `''` |
| worst | `' client met' [·] ' she len rais'` | 64 | `' she'` (0.340) | **deletion** | 5.54 | 0.001 | 0.010 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'త'` (0.989) | **correct** | 0.12 | 0.989 | 0.059 | `''` |

## Verified glitch tokens (reference)


### `'larınd'` — id 129956  (single-probe lp -21.32, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | `'\n\n'` (0.637) | substitution | 1.63 | 0.000 | -0.057 |
| worst | `' margin prot consider cult' [·] ' not million provided'` | `' not'` (0.951) | deletion | 0.46 | 0.000 | -0.020 |
| worst | `' att attempt ke quite' [·] ' tor near group'` | `'\n'` (0.553) | substitution | 1.49 | 0.000 | 0.017 |
| best | `' pot content' [·] ' init abs comfort'` | `'larınd'` (0.673) | correct | 2.89 | 0.673 | 0.003 |

### `'吏'` — id 151148  (single-probe lp -20.77, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' bad consist' [·] ' assert signific cert'` | `' assert'` (0.984) | deletion | 0.17 | 0.000 | 0.219 |
| worst | `' att attempt ke quite' [·] ' tor near group'` | `'<|im_end|>'` (0.404) | substitution | 2.20 | 0.000 | 0.018 |
| worst | `' thems line sem things' [·] ' range car player'` | `' range'` (0.866) | deletion | 0.75 | 0.000 | 0.129 |
| best | `' word override script cur' [·] ' environment mis reach'` | `'</think>'` (0.517) | substitution | 2.58 | 0.000 | -0.047 |

### `'𝄹'` — id 151290  (single-probe lp -20.66, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.976) | deletion | 0.19 | 0.000 | 0.054 |
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (1.000) | deletion | 0.01 | 0.000 | 0.079 |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | `' doing'` (0.981) | deletion | 0.16 | 0.000 | 0.068 |
| best | `' client met' [·] ' she len rais'` | `' she'` (0.396) | deletion | 3.46 | 0.000 | 0.081 |

### `'พื้นที่'` — id 126795  (single-probe lp -20.49, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' bad consist' [·] ' assert signific cert'` | `' assert'` (0.918) | deletion | 0.70 | 0.000 | 0.178 |
| worst | `' margin prot consider cult' [·] ' not million provided'` | `'\n\n'` (0.802) | substitution | 1.05 | 0.000 | 0.196 |
| worst | `' word override script cur' [·] ' environment mis reach'` | `' environment'` (0.921) | deletion | 0.58 | 0.000 | 0.069 |
| best | `' para course pat wom' [·] ' under available carry'` | `'\n\n'` (0.259) | substitution | 5.33 | 0.000 | -0.027 |

### `'ได้ง่าย'` — id 140750  (single-probe lp -20.31, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (0.915) | deletion | 0.57 | 0.000 | 0.159 |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | `' doing'` (0.921) | deletion | 0.48 | 0.000 | 0.119 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.926) | deletion | 0.52 | 0.000 | 0.136 |
| best | `' client met' [·] ' she len rais'` | `' she'` (0.358) | deletion | 3.17 | 0.000 | 0.037 |

### `'กระเป๋า'` — id 136695  (single-probe lp -20.31, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (1.000) | deletion | 0.01 | 0.000 | 0.137 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.953) | deletion | 0.35 | 0.000 | 0.112 |
| worst | `' margin prot consider cult' [·] ' not million provided'` | `' not'` (0.951) | deletion | 0.39 | 0.000 | 0.202 |
| best | `' den namespace items pas' [·] ' person cont sound'` | `'<|im_end|>'` (0.484) | substitution | 2.08 | 0.000 | 0.138 |

### `'燐'` — id 149824  (single-probe lp -20.30, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' bad consist' [·] ' assert signific cert'` | `' assert'` (0.989) | deletion | 0.13 | 0.000 | 0.179 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.935) | deletion | 0.50 | 0.000 | 0.059 |
| worst | `' pot content' [·] ' init abs comfort'` | `' init'` (0.346) | deletion | 2.20 | 0.000 | 0.208 |
| best | `' den namespace items pas' [·] ' person cont sound'` | `'<|im_end|>'` (0.738) | substitution | 1.63 | 0.000 | 0.087 |

### `'สนับสนุ'` — id 133586  (single-probe lp -20.25, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' specific isn parents owner' [·] ' doing gold water'` | `' doing'` (0.939) | deletion | 0.43 | 0.000 | 0.090 |
| worst | `' videos means leave didn' [·] ' ax global frame'` | `'\n\n'` (0.300) | substitution | 2.38 | 0.000 | 0.211 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.838) | deletion | 1.06 | 0.000 | 0.160 |
| best | `' client met' [·] ' she len rais'` | `'<|im_end|>'` (0.521) | substitution | 2.60 | 0.000 | 0.174 |

### `'มากขึ้น'` — id 131240  (single-probe lp -20.20, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' bad consist' [·] ' assert signific cert'` | `'\n\n'` (0.441) | substitution | 1.67 | 0.000 | 0.155 |
| worst | `' word override script cur' [·] ' environment mis reach'` | `' environment'` (0.596) | deletion | 1.28 | 0.000 | 0.192 |
| worst | `' pot content' [·] ' init abs comfort'` | `' init'` (0.973) | deletion | 0.24 | 0.000 | -0.040 |
| best | `' den namespace items pas' [·] ' person cont sound'` | `'<|im_end|>'` (0.390) | substitution | 3.15 | 0.000 | 0.215 |

### `'��取'` — id 27487  (single-probe lp -20.19, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' pot content' [·] ' init abs comfort'` | `' init'` (0.971) | deletion | 0.27 | 0.000 | 0.195 |
| worst | `' bad consist' [·] ' assert signific cert'` | `' assert'` (0.941) | deletion | 0.56 | 0.000 | 0.189 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.959) | deletion | 0.34 | 0.000 | 0.161 |
| best | `' client met' [·] ' she len rais'` | `'<|im_start|>'` (0.320) | substitution | 2.80 | 0.000 | 0.042 |

### `'ארגוני'` — id 137670  (single-probe lp -20.19, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.980) | deletion | 0.17 | 0.000 | 0.016 |
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (0.763) | deletion | 1.21 | 0.000 | 0.037 |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | `' doing'` (0.852) | deletion | 0.91 | 0.000 | 0.016 |
| best | `' len' [·] ' mobile std drop'` | `'</think>'` (0.588) | substitution | 2.30 | 0.000 | 0.011 |

### `'הבנה'` — id 143454  (single-probe lp -20.11, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (0.999) | deletion | 0.02 | 0.000 | 0.025 |
| worst | `' videos means leave didn' [·] ' ax global frame'` | `'\n'` (0.631) | substitution | 1.41 | 0.000 | 0.038 |
| worst | `' origin child below namespace' [·] ' lock home rece'` | `'\n'` (0.683) | substitution | 1.39 | 0.000 | 0.036 |
| best | `' client met' [·] ' she len rais'` | `'</think>'` (0.603) | substitution | 2.03 | 0.000 | 0.031 |

### `' สิงหาคม'` — id 142447  (single-probe lp -20.09, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (0.969) | deletion | 0.24 | 0.000 | 0.165 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.960) | deletion | 0.31 | 0.000 | 0.106 |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | `' doing'` (0.964) | deletion | 0.28 | 0.000 | 0.110 |
| best | `' den namespace items pas' [·] ' person cont sound'` | `'</think>'` (0.386) | substitution | 2.33 | 0.000 | 0.068 |

### `'ความเป็น'` — id 132966  (single-probe lp -20.06, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' pot content' [·] ' init abs comfort'` | `' abs'` (0.929) | substitution | 0.54 | 0.000 | 0.127 |
| worst | `' word override script cur' [·] ' environment mis reach'` | `' environment'` (0.722) | deletion | 1.52 | 0.000 | -0.068 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.571) | deletion | 1.88 | 0.000 | 0.067 |
| best | `' client met' [·] ' she len rais'` | `' and'` (0.171) | substitution | 5.17 | 0.000 | -0.043 |

### `'แม้'` — id 126927  (single-probe lp -19.77, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' pot content' [·] ' init abs comfort'` | `' init'` (0.358) | deletion | 3.44 | 0.000 | 0.014 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.938) | deletion | 0.46 | 0.000 | 0.193 |
| worst | `' client met' [·] ' she len rais'` | `'?;\n\n'` (0.525) | substitution | 1.52 | 0.000 | 0.120 |
| best | `' printf ne throw sy' [·] ' art human ver'` | `'\n\n'` (0.248) | substitution | 6.59 | 0.000 | 0.189 |

### `'量'` — id 147225  (single-probe lp -19.77, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (0.995) | deletion | 0.05 | 0.000 | 0.179 |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | `' doing'` (0.938) | deletion | 0.47 | 0.000 | 0.085 |
| worst | `' para course pat wom' [·] ' under available carry'` | `' under'` (0.991) | deletion | 0.10 | 0.000 | 0.116 |
| best | `' den namespace items pas' [·] ' person cont sound'` | `'</think>'` (0.405) | substitution | 2.35 | 0.000 | 0.108 |

### `'กิจกรรม'` — id 130551  (single-probe lp -19.65, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' bad consist' [·] ' assert signific cert'` | `' assert'` (0.984) | deletion | 0.17 | 0.000 | 0.250 |
| worst | `' word override script cur' [·] ' environment mis reach'` | `' environment'` (0.530) | deletion | 2.23 | 0.000 | -0.043 |
| worst | `' song released top sl' [·] ' under sus since'` | `' under'` (0.991) | deletion | 0.10 | 0.000 | 0.205 |
| best | `' fund port sw struct' [·] ' term echo rev'` | `'<|im_end|>'` (0.570) | substitution | 2.87 | 0.000 | -0.010 |

### `'มักจะ'` — id 139176  (single-probe lp -19.65, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (0.995) | deletion | 0.06 | 0.000 | 0.176 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.932) | deletion | 0.47 | 0.000 | 0.143 |
| worst | `' para course pat wom' [·] ' under available carry'` | `' under'` (0.995) | deletion | 0.06 | 0.000 | 0.191 |
| best | `' client met' [·] ' she len rais'` | `'<|im_end|>'` (0.267) | substitution | 3.28 | 0.000 | 0.072 |

### `'แก้ปัญหา'` — id 143828  (single-probe lp -19.51, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.977) | deletion | 0.20 | 0.000 | 0.091 |
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (0.999) | deletion | 0.01 | 0.000 | 0.138 |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | `' doing'` (0.977) | deletion | 0.22 | 0.000 | 0.102 |
| best | `' den namespace items pas' [·] ' person cont sound'` | `' person'` (0.368) | deletion | 2.40 | 0.000 | 0.116 |

### `'น้ำหนัก'` — id 135989  (single-probe lp -19.47, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | `' environment'` (0.894) | deletion | 0.74 | 0.000 | 0.160 |
| worst | `' pot content' [·] ' init abs comfort'` | `' abs'` (0.925) | substitution | 0.53 | 0.000 | 0.212 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.623) | deletion | 1.55 | 0.000 | 0.143 |
| best | `' den namespace items pas' [·] ' person cont sound'` | `'<|im_end|>'` (0.885) | substitution | 0.81 | 0.000 | 0.210 |
