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
- **clean_gate_lp**: `-0.693`

## Fragile clean-looking tokens (pass the single probe; fail in some contexts)


### `'ཧ'` — id 150460

single-probe lp -0.021 · fragility 1.00 · mean lp -8.84 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.994) | **deletion** | 0.06 | 0.000 | 0.067 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.979) | **deletion** | 0.21 | 0.000 | 0.080 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.995) | **deletion** | 0.06 | 0.000 | 0.064 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'\n\n'` (0.547) | **substitution** | 1.70 | 0.024 | 0.108 | `''` |

### `'ଓ'` — id 149343

single-probe lp -0.170 · fragility 1.00 · mean lp -8.31 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.994) | **deletion** | 0.06 | 0.000 | 0.054 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.991) | **deletion** | 0.08 | 0.000 | 0.032 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.982) | **deletion** | 0.19 | 0.000 | 0.073 | `''` |
| best | `' develop raw dep elements' [·] ' element anything press'` | 16 | `'<|im_end|>'` (0.272) | **substitution** | 3.77 | 0.165 | 0.043 | `''` |

### `'�'` — id 120

single-probe lp -0.453 · fragility 1.00 · mean lp -8.66 · canonical False · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.02 | 0.000 | -0.007 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.987) | **deletion** | 0.12 | 0.000 | 0.006 | `''` |
| worst | `' origin child below namespace' [·] ' lock home rece'` | 32 | `' lock'` (0.979) | **deletion** | 0.23 | 0.000 | -0.003 | `''` |
| best | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'\n'` (0.454) | **substitution** | 3.14 | 0.276 | 0.009 | `''` |

### `'ঙ'` — id 148571

single-probe lp -0.003 · fragility 1.00 · mean lp -3.53 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.955) | **deletion** | 0.42 | 0.008 | 0.036 | `''` |
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.947) | **deletion** | 0.42 | 0.004 | 0.021 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.983) | **deletion** | 0.14 | 0.001 | 0.028 | `''` |
| best | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'ঙ'` (0.689) | **correct** | 1.53 | 0.689 | 0.065 | `''` |

### `'🚆'` — id 148722

single-probe lp -0.009 · fragility 1.00 · mean lp -13.55 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.987) | **deletion** | 0.11 | 0.000 | 0.031 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.969) | **deletion** | 0.28 | 0.000 | 0.023 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.441) | **substitution** | 2.21 | 0.000 | 0.019 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'\n'` (0.355) | **substitution** | 3.19 | 0.042 | 0.048 | `''` |

### `'𝓪'` — id 147910

single-probe lp -0.120 · fragility 1.00 · mean lp -5.89 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.970) | **deletion** | 0.24 | 0.000 | 0.040 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.996) | **deletion** | 0.05 | 0.000 | 0.023 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.995) | **deletion** | 0.06 | 0.000 | 0.062 | `''` |
| best | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'\n'` (0.309) | **substitution** | 3.68 | 0.114 | 0.066 | `''` |

### `'💱'` — id 147977

single-probe lp -0.180 · fragility 1.00 · mean lp -9.11 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.01 | 0.000 | 0.027 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `' ax'` (0.651) | **deletion** | 1.91 | 0.000 | 0.022 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.998) | **deletion** | 0.03 | 0.000 | 0.050 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'💱'` (0.441) | **correct** | 2.44 | 0.441 | 0.064 | `''` |

### `'༎'` — id 147152

single-probe lp -0.267 · fragility 1.00 · mean lp -7.16 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.01 | 0.000 | -0.004 | `''` |
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.991) | **deletion** | 0.10 | 0.000 | -0.026 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.978) | **deletion** | 0.23 | 0.000 | 0.029 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `' mobile'` (0.759) | **deletion** | 1.72 | 0.116 | 0.039 | `''` |

### `'ལ'` — id 148576

single-probe lp -0.010 · fragility 1.00 · mean lp -7.56 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.986) | **deletion** | 0.15 | 0.000 | 0.095 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.993) | **deletion** | 0.09 | 0.000 | 0.084 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.524) | **substitution** | 2.19 | 0.000 | 0.088 | `''` |
| best | `' den namespace items pas' [·] ' person cont sound'` | 64 | `'ལ'` (0.393) | **correct** | 3.22 | 0.393 | 0.105 | `''` |

### `'ම'` — id 147514

single-probe lp -0.020 · fragility 1.00 · mean lp -9.64 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.996) | **deletion** | 0.06 | 0.000 | 0.042 | `''` |
| worst | `' las touch recently hard' [·] ' beh func eff'` | 8 | `'\n'` (0.279) | **substitution** | 3.46 | 0.000 | 0.013 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.972) | **deletion** | 0.28 | 0.000 | 0.047 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `' mobile'` (0.246) | **deletion** | 4.23 | 0.149 | 0.086 | `''` |

### `'摛'` — id 123142

single-probe lp -0.538 · fragility 1.00 · mean lp -9.41 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (1.000) | **deletion** | 0.00 | 0.000 | 0.022 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.982) | **deletion** | 0.17 | 0.000 | 0.028 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.364) | **substitution** | 2.02 | 0.000 | 0.036 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `' mobile'` (0.415) | **deletion** | 2.64 | 0.039 | 0.043 | `''` |

### `'搒'` — id 123144

single-probe lp -0.325 · fragility 1.00 · mean lp -8.76 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.360) | **substitution** | 3.08 | 0.000 | 0.050 | `''` |
| worst | `' it staff when said' [·] ' account vers'` | 8 | `'<|im_end|>'` (0.633) | **substitution** | 2.51 | 0.000 | 0.048 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.989) | **deletion** | 0.10 | 0.000 | 0.032 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'\n\n'` (0.551) | **substitution** | 2.47 | 0.002 | 0.074 | `''` |

### `'قرأ'` — id 133615

single-probe lp -0.192 · fragility 1.00 · mean lp -8.76 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.581) | **deletion** | 2.29 | 0.000 | 0.011 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.436) | **substitution** | 2.06 | 0.000 | 0.037 | `''` |
| worst | `' fund port sw struct' [·] ' term echo rev'` | 64 | `'<|im_end|>'` (0.643) | **substitution** | 2.55 | 0.000 | -0.016 | `''` |
| best | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' readable'` (0.430) | **substitution** | 3.14 | 0.066 | 0.002 | `''` |

### `' לתת'` — id 133211

single-probe lp -0.022 · fragility 1.00 · mean lp -6.45 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `'<|im_end|>'` (0.467) | **substitution** | 3.27 | 0.000 | -0.036 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.470) | **deletion** | 3.53 | 0.000 | -0.020 | `''` |
| worst | `' fund port sw struct' [·] ' term echo rev'` | 64 | `'<|im_end|>'` (0.478) | **substitution** | 3.82 | 0.000 | -0.044 | `''` |
| best | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' לתת'` (0.181) | **correct** | 7.36 | 0.181 | 0.008 | `''` |

### `' והת'` — id 133258

single-probe lp -0.002 · fragility 1.00 · mean lp -3.97 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `"'t"` (0.452) | **substitution** | 4.70 | 0.000 | -0.007 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `'战略合作'` (0.088) | **substitution** | 9.06 | 0.001 | 0.002 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `'战略合作'` (0.109) | **substitution** | 7.55 | 0.005 | 0.013 | `''` |
| best | `' it staff when said' [·] ' account vers'` | 8 | `' והת'` (0.288) | **correct** | 6.89 | 0.288 | 0.028 | `''` |

### `'צבא'` — id 132235

single-probe lp -0.111 · fragility 1.00 · mean lp -12.57 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' song released top sl' [·] ' under sus since'` | 32 | `' under'` (0.999) | **deletion** | 0.02 | 0.000 | 0.045 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.874) | **deletion** | 0.89 | 0.000 | 0.030 | `''` |
| worst | `' las touch recently hard' [·] ' beh func eff'` | 8 | `'\n'` (0.618) | **substitution** | 1.68 | 0.000 | -0.022 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'<|im_end|>'` (0.249) | **substitution** | 3.67 | 0.104 | 0.030 | `''` |

### `'それぞ'` — id 132282

single-probe lp -0.134 · fragility 1.00 · mean lp -10.48 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (1.000) | **deletion** | 0.00 | 0.000 | -0.029 | `''` |
| worst | `' opport cent positive quality' [·] ' weight dispatch ideas'` | 64 | `'\n'` (0.442) | **substitution** | 2.85 | 0.000 | 0.000 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.885) | **deletion** | 0.91 | 0.000 | -0.010 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'</think>'` (0.213) | **substitution** | 5.90 | 0.107 | -0.050 | `''` |

### `'렇'` — id 124934

single-probe lp -0.057 · fragility 1.00 · mean lp -5.10 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.977) | **deletion** | 0.26 | 0.000 | 0.007 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.749) | **deletion** | 1.52 | 0.000 | 0.020 | `''` |
| worst | `' fund port sw struct' [·] ' term echo rev'` | 64 | `' term'` (0.828) | **deletion** | 1.50 | 0.000 | -0.015 | `''` |
| best | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'렇'` (0.358) | **correct** | 3.80 | 0.358 | 0.031 | `''` |

### `'팍'` — id 144974

single-probe lp -0.233 · fragility 1.00 · mean lp -9.52 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.974) | **deletion** | 0.23 | 0.000 | 0.028 | `''` |
| worst | `' fund port sw struct' [·] ' term echo rev'` | 64 | `' term'` (0.902) | **deletion** | 0.66 | 0.000 | 0.002 | `''` |
| worst | `' den namespace items pas' [·] ' person cont sound'` | 64 | `' person'` (0.939) | **deletion** | 0.61 | 0.000 | -0.003 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'\n\n'` (0.424) | **substitution** | 2.36 | 0.107 | 0.028 | `''` |

### `'ᄀ'` — id 146312

single-probe lp -0.500 · fragility 1.00 · mean lp -8.37 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.997) | **deletion** | 0.04 | 0.000 | 0.025 | `''` |
| worst | `' pot content' [·] ' init abs comfort'` | 16 | `' init'` (0.989) | **deletion** | 0.12 | 0.000 | 0.060 | `''` |
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.994) | **deletion** | 0.06 | 0.000 | 0.031 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `' mobile'` (0.699) | **deletion** | 1.78 | 0.095 | 0.089 | `''` |

### `'ביר'` — id 125810

single-probe lp -0.442 · fragility 1.00 · mean lp -12.59 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.990) | **deletion** | 0.13 | 0.000 | 0.007 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (1.000) | **deletion** | 0.01 | 0.000 | -0.003 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.927) | **deletion** | 0.65 | 0.000 | 0.003 | `''` |
| best | `' origin child below namespace' [·] ' lock home rece'` | 32 | `'\n\n'` (0.274) | **substitution** | 4.94 | 0.020 | 0.015 | `''` |

### `'𝕒'` — id 146898

single-probe lp -0.018 · fragility 1.00 · mean lp -6.16 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (1.000) | **deletion** | 0.01 | 0.000 | 0.041 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.997) | **deletion** | 0.05 | 0.000 | 0.076 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.988) | **deletion** | 0.14 | 0.000 | 0.044 | `''` |
| best | `' opport cent positive quality' [·] ' weight dispatch ideas'` | 64 | `' weight'` (0.373) | **deletion** | 3.25 | 0.176 | 0.038 | `''` |

### `'⇗'` — id 146975

single-probe lp -0.387 · fragility 1.00 · mean lp -5.47 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.01 | 0.000 | 0.023 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.991) | **deletion** | 0.10 | 0.000 | 0.047 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `' ax'` (0.790) | **deletion** | 1.30 | 0.000 | 0.023 | `''` |
| best | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'⇗'` (0.564) | **correct** | 2.68 | 0.563 | 0.070 | `''` |

### `'מוצא'` — id 135695

single-probe lp -0.542 · fragility 1.00 · mean lp -15.92 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.995) | **deletion** | 0.06 | 0.000 | -0.005 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `' under'` (0.984) | **deletion** | 0.17 | 0.000 | 0.021 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.949) | **deletion** | 0.34 | 0.000 | -0.044 | `''` |
| best | `' opport cent positive quality' [·] ' weight dispatch ideas'` | 64 | `' weight'` (0.552) | **deletion** | 2.10 | 0.000 | 0.009 | `''` |

### `'赇'` — id 120686

single-probe lp -0.251 · fragility 1.00 · mean lp -6.69 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.905) | **deletion** | 0.64 | 0.000 | 0.052 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.992) | **deletion** | 0.08 | 0.000 | 0.026 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.960) | **deletion** | 0.38 | 0.000 | 0.026 | `''` |
| best | `' origin child below namespace' [·] ' lock home rece'` | 32 | `'\n\n'` (0.316) | **substitution** | 2.71 | 0.023 | 0.017 | `''` |

### `'חשב'` — id 126112

single-probe lp -0.205 · fragility 1.00 · mean lp -16.56 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.490) | **deletion** | 2.05 | 0.000 | -0.015 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.720) | **deletion** | 1.35 | 0.000 | 0.010 | `''` |
| worst | `' pot content' [·] ' init abs comfort'` | 16 | `'\n'` (0.739) | **substitution** | 1.44 | 0.000 | 0.023 | `''` |
| best | `' origin child below namespace' [·] ' lock home rece'` | 32 | `'<|im_end|>'` (0.386) | **substitution** | 2.69 | 0.001 | 0.020 | `''` |

### `'쩔'` — id 144515

single-probe lp -0.176 · fragility 1.00 · mean lp -6.48 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.907) | **deletion** | 0.69 | 0.000 | 0.053 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.854) | **deletion** | 0.92 | 0.000 | 0.061 | `''` |
| worst | `' opport cent positive quality' [·] ' weight dispatch ideas'` | 64 | `'\n\n'` (0.380) | **substitution** | 2.40 | 0.000 | 0.060 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'쩔'` (0.542) | **correct** | 3.23 | 0.542 | 0.068 | `''` |

### `'ﻧ'` — id 144524

single-probe lp -0.393 · fragility 1.00 · mean lp -5.93 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.990) | **deletion** | 0.12 | 0.000 | 0.072 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `' under'` (0.922) | **deletion** | 0.79 | 0.000 | 0.082 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.980) | **deletion** | 0.28 | 0.000 | 0.099 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'ﻧ'` (0.450) | **correct** | 4.08 | 0.450 | 0.114 | `''` |

### `'펙'` — id 144609

single-probe lp -0.041 · fragility 1.00 · mean lp -5.76 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.988) | **deletion** | 0.15 | 0.000 | 0.024 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.995) | **deletion** | 0.06 | 0.000 | 0.015 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `' ax'` (0.631) | **deletion** | 1.99 | 0.000 | 0.044 | `''` |
| best | `' it staff when said' [·] ' account vers'` | 8 | `'펙'` (0.503) | **correct** | 3.16 | 0.504 | 0.018 | `''` |

### `'윽'` — id 145360

single-probe lp -0.097 · fragility 1.00 · mean lp -8.09 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.998) | **deletion** | 0.03 | 0.000 | -0.007 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.985) | **deletion** | 0.16 | 0.000 | 0.008 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.339) | **substitution** | 2.74 | 0.000 | 0.025 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'unction'` (0.347) | **substitution** | 2.74 | 0.347 | 0.030 | `''` |

### `'퐁'` — id 145478

single-probe lp -0.126 · fragility 1.00 · mean lp -5.13 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.965) | **deletion** | 0.30 | 0.000 | 0.036 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `' under'` (0.997) | **deletion** | 0.04 | 0.000 | 0.062 | `''` |
| worst | `' song released top sl' [·] ' under sus since'` | 32 | `' under'` (0.999) | **deletion** | 0.02 | 0.000 | 0.047 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'퐁'` (0.558) | **correct** | 2.52 | 0.558 | 0.035 | `''` |

### `'끈'` — id 144474

single-probe lp -0.236 · fragility 1.00 · mean lp -8.02 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.995) | **deletion** | 0.06 | 0.000 | 0.018 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.972) | **deletion** | 0.26 | 0.000 | 0.029 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.520) | **substitution** | 1.54 | 0.000 | 0.022 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'끈'` (0.621) | **correct** | 2.11 | 0.621 | 0.006 | `''` |

### `'ﻢ'` — id 144847

single-probe lp -0.659 · fragility 1.00 · mean lp -6.74 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (1.000) | **deletion** | 0.01 | 0.000 | 0.054 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `' under'` (0.993) | **deletion** | 0.09 | 0.000 | 0.084 | `''` |
| worst | `' pot content' [·] ' init abs comfort'` | 16 | `' init'` (0.986) | **deletion** | 0.13 | 0.000 | 0.086 | `''` |
| best | `' song released top sl' [·] ' under sus since'` | 32 | `' under'` (0.858) | **deletion** | 1.12 | 0.062 | 0.083 | `''` |

### `'랫'` — id 144151

single-probe lp -0.230 · fragility 1.00 · mean lp -8.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pot content' [·] ' init abs comfort'` | 16 | `' init'` (0.960) | **deletion** | 0.27 | 0.000 | 0.050 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.955) | **deletion** | 0.34 | 0.000 | 0.056 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.954) | **deletion** | 0.36 | 0.000 | 0.050 | `''` |
| best | `' song released top sl' [·] ' under sus since'` | 32 | `' under'` (0.596) | **deletion** | 2.08 | 0.151 | 0.010 | `''` |

### `'僔'` — id 123365

single-probe lp -0.010 · fragility 1.00 · mean lp -6.46 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (1.000) | **deletion** | 0.00 | 0.000 | 0.039 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.986) | **deletion** | 0.18 | 0.000 | 0.043 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `' ax'` (0.318) | **deletion** | 3.14 | 0.000 | 0.050 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'僔'` (0.409) | **correct** | 4.40 | 0.409 | 0.066 | `''` |

### `'יצור'` — id 127251

single-probe lp -0.274 · fragility 1.00 · mean lp -13.33 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.980) | **deletion** | 0.20 | 0.000 | -0.000 | `''` |
| worst | `' las touch recently hard' [·] ' beh func eff'` | 8 | `'<|im_end|>'` (0.623) | **substitution** | 1.87 | 0.000 | -0.018 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.394) | **substitution** | 2.39 | 0.000 | 0.014 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'<|im_end|>'` (0.309) | **substitution** | 6.17 | 0.001 | 0.010 | `''` |

### `'مواف'` — id 143326

single-probe lp -0.422 · fragility 1.00 · mean lp -7.90 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.646) | **deletion** | 1.87 | 0.000 | 0.035 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.280) | **substitution** | 2.66 | 0.000 | 0.045 | `''` |
| worst | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'</think>'` (0.833) | **substitution** | 0.98 | 0.000 | 0.061 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'</think>'` (0.775) | **substitution** | 1.76 | 0.017 | 0.025 | `''` |

### `'コンテン'` — id 143618

single-probe lp -0.364 · fragility 1.00 · mean lp -7.83 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.476) | **substitution** | 1.85 | 0.000 | 0.074 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.381) | **deletion** | 2.41 | 0.000 | 0.041 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.854) | **deletion** | 0.87 | 0.000 | 0.068 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'\n'` (0.610) | **substitution** | 2.11 | 0.027 | 0.073 | `''` |

### `'ﻳ'` — id 144383

single-probe lp -0.256 · fragility 1.00 · mean lp -9.05 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.981) | **deletion** | 0.19 | 0.000 | 0.065 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.997) | **deletion** | 0.04 | 0.000 | 0.054 | `''` |
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.982) | **deletion** | 0.16 | 0.000 | 0.034 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'\n\n'` (0.412) | **substitution** | 3.64 | 0.195 | 0.103 | `''` |

### `'بدو'` — id 127961

single-probe lp -0.014 · fragility 1.00 · mean lp -7.06 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.423) | **substitution** | 2.31 | 0.000 | 0.026 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `'\n'` (0.460) | **substitution** | 2.05 | 0.000 | 0.042 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.932) | **deletion** | 0.61 | 0.000 | 0.061 | `''` |
| best | `' den namespace items pas' [·] ' person cont sound'` | 64 | `'dojo'` (0.465) | **substitution** | 2.21 | 0.362 | 0.022 | `''` |

### `'تها'` — id 128412

single-probe lp -0.167 · fragility 1.00 · mean lp -8.46 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `'</think>'` (0.431) | **substitution** | 3.10 | 0.000 | -0.030 | `''` |
| worst | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'</think>'` (0.709) | **substitution** | 2.58 | 0.000 | -0.028 | `''` |
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'</think>'` (0.256) | **substitution** | 4.25 | 0.000 | -0.030 | `''` |
| best | `' unit rep anyone fl' [·] ' ro tf'` | 16 | `'تها'` (0.475) | **correct** | 4.51 | 0.475 | -0.025 | `''` |

### `'شهور'` — id 143044

single-probe lp -0.081 · fragility 1.00 · mean lp -4.29 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'</think>'` (0.367) | **substitution** | 2.42 | 0.000 | 0.040 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `'</think>'` (0.401) | **substitution** | 2.57 | 0.000 | 0.014 | `''` |
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'\n'` (0.321) | **substitution** | 2.49 | 0.000 | 0.072 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'شهور'` (0.576) | **correct** | 2.73 | 0.576 | 0.016 | `''` |

### `' พฤษภา'` — id 143148

single-probe lp -0.218 · fragility 1.00 · mean lp -4.91 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.988) | **deletion** | 0.13 | 0.000 | 0.084 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'</think>'` (0.233) | **substitution** | 3.01 | 0.000 | 0.100 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.823) | **deletion** | 1.34 | 0.000 | 0.030 | `''` |
| best | `' pot content' [·] ' init abs comfort'` | 16 | `' init'` (0.514) | **deletion** | 2.32 | 0.243 | 0.064 | `''` |

### `'профессиона'` — id 143807

single-probe lp -0.560 · fragility 1.00 · mean lp -6.25 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'<|im_end|>'` (0.330) | **substitution** | 2.30 | 0.000 | 0.081 | `''` |
| worst | `' opport cent positive quality' [·] ' weight dispatch ideas'` | 64 | `'<|im_start|>'` (0.594) | **substitution** | 2.08 | 0.000 | 0.053 | `''` |
| worst | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'<|im_start|>'` (0.523) | **substitution** | 1.44 | 0.000 | 0.077 | `''` |
| best | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.726) | **deletion** | 1.70 | 0.098 | 0.090 | `''` |

### `'阗'` — id 121283

single-probe lp -0.360 · fragility 1.00 · mean lp -6.55 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.946) | **deletion** | 0.42 | 0.000 | 0.017 | `''` |
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.947) | **deletion** | 0.43 | 0.000 | 0.025 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.333) | **deletion** | 2.70 | 0.000 | 0.012 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'</think>'` (0.734) | **substitution** | 1.69 | 0.060 | 0.033 | `''` |

### `'احتجاج'` — id 142689

single-probe lp -0.040 · fragility 1.00 · mean lp -9.97 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'\n'` (0.472) | **substitution** | 1.72 | 0.000 | 0.044 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.542) | **substitution** | 1.90 | 0.000 | 0.047 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.477) | **deletion** | 2.50 | 0.000 | -0.010 | `''` |
| best | `' origin child below namespace' [·] ' lock home rece'` | 32 | `'<|im_end|>'` (0.315) | **substitution** | 2.90 | 0.131 | 0.019 | `''` |

### `'חוץ'` — id 129918

single-probe lp -0.010 · fragility 1.00 · mean lp -16.11 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.998) | **deletion** | 0.02 | 0.000 | -0.013 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.990) | **deletion** | 0.12 | 0.000 | -0.019 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.966) | **deletion** | 0.30 | 0.000 | 0.001 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'<|im_end|>'` (0.469) | **substitution** | 3.54 | 0.000 | 0.001 | `''` |

### `'عالج'` — id 141039

single-probe lp -0.608 · fragility 1.00 · mean lp -9.91 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.956) | **deletion** | 0.38 | 0.000 | 0.004 | `''` |
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'<|im_end|>'` (0.398) | **substitution** | 1.97 | 0.000 | 0.035 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.415) | **substitution** | 2.45 | 0.000 | 0.040 | `''` |
| best | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.427) | **deletion** | 2.57 | 0.007 | -0.008 | `''` |

### `'عطاء'` — id 141069

single-probe lp -0.237 · fragility 1.00 · mean lp -8.06 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.960) | **deletion** | 0.28 | 0.000 | 0.058 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.293) | **deletion** | 2.95 | 0.000 | 0.051 | `''` |
| worst | `' printf ne throw sy' [·] ' art human ver'` | 8 | `'<|im_end|>'` (0.483) | **substitution** | 1.60 | 0.000 | 0.044 | `''` |
| best | `' pot content' [·] ' init abs comfort'` | 16 | `'عطاء'` (0.878) | **correct** | 1.08 | 0.878 | 0.040 | `''` |

### `'оличество'` — id 98614

single-probe lp -0.150 · fragility 1.00 · mean lp -5.15 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'<|im_start|>'` (0.391) | **substitution** | 2.26 | 0.000 | 0.036 | `''` |
| worst | `' opport cent positive quality' [·] ' weight dispatch ideas'` | 64 | `'\n\n'` (0.290) | **substitution** | 2.66 | 0.000 | 0.025 | `''` |
| worst | `' fund port sw struct' [·] ' term echo rev'` | 64 | `'<|im_start|>'` (0.463) | **substitution** | 2.30 | 0.000 | -0.030 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'<|im_start|>'` (0.551) | **substitution** | 1.57 | 0.379 | 0.009 | `''` |

### `'جزاء'` — id 139947

single-probe lp -0.603 · fragility 1.00 · mean lp -9.98 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.507) | **substitution** | 1.77 | 0.000 | 0.029 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.925) | **deletion** | 0.59 | 0.000 | 0.019 | `''` |
| worst | `' song released top sl' [·] ' under sus since'` | 32 | `' under'` (0.948) | **deletion** | 0.47 | 0.000 | 0.028 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'</think>'` (0.175) | **substitution** | 6.01 | 0.002 | -0.010 | `''` |

### `'شغل'` — id 138391

single-probe lp -0.644 · fragility 1.00 · mean lp -7.39 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.375) | **substitution** | 2.32 | 0.000 | 0.001 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.547) | **deletion** | 2.43 | 0.000 | 0.020 | `''` |
| worst | `' song released top sl' [·] ' under sus since'` | 32 | `' under'` (0.981) | **deletion** | 0.20 | 0.000 | 0.018 | `''` |
| best | `' develop raw dep elements' [·] ' element anything press'` | 16 | `'شغل'` (0.549) | **correct** | 2.91 | 0.549 | -0.004 | `''` |

### `'شت'` — id 129678

single-probe lp -0.110 · fragility 1.00 · mean lp -5.00 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' len' [·] ' mobile std drop'` | 8 | `'cht'` (0.912) | **substitution** | 0.59 | 0.000 | -0.021 | `''` |
| worst | `' song released top sl' [·] ' under sus since'` | 32 | `'s'` (0.851) | **substitution** | 1.00 | 0.000 | -0.005 | `''` |
| worst | `' opport cent positive quality' [·] ' weight dispatch ideas'` | 64 | `'\n\n'` (0.308) | **substitution** | 2.88 | 0.000 | 0.061 | `''` |
| best | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'شت'` (0.613) | **correct** | 1.10 | 0.613 | 0.024 | `''` |

### `'طحن'` — id 138763

single-probe lp -0.281 · fragility 1.00 · mean lp -7.28 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.943) | **deletion** | 0.46 | 0.000 | 0.056 | `''` |
| worst | `' song released top sl' [·] ' under sus since'` | 32 | `' under'` (0.965) | **deletion** | 0.30 | 0.000 | 0.052 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.955) | **deletion** | 0.38 | 0.000 | 0.098 | `''` |
| best | `' develop raw dep elements' [·] ' element anything press'` | 16 | `'\n'` (0.662) | **substitution** | 1.43 | 0.023 | 0.081 | `''` |

### `'שאר'` — id 129588

single-probe lp -0.268 · fragility 1.00 · mean lp -14.15 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.977) | **deletion** | 0.20 | 0.000 | 0.004 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.900) | **deletion** | 0.79 | 0.000 | -0.003 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.575) | **deletion** | 2.22 | 0.000 | 0.013 | `''` |
| best | `' origin child below namespace' [·] ' lock home rece'` | 32 | `'\n\n'` (0.754) | **substitution** | 1.97 | 0.000 | -0.003 | `''` |

### `'גות'` — id 132867

single-probe lp -0.066 · fragility 1.00 · mean lp -10.13 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.939) | **deletion** | 0.61 | 0.000 | 0.030 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.992) | **deletion** | 0.08 | 0.000 | 0.027 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.932) | **deletion** | 0.66 | 0.000 | 0.029 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'\n\n'` (0.352) | **substitution** | 4.93 | 0.004 | 0.035 | `''` |

### `'镘'` — id 121776

single-probe lp -0.059 · fragility 1.00 · mean lp -4.59 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.984) | **deletion** | 0.13 | 0.000 | 0.026 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.995) | **deletion** | 0.05 | 0.000 | 0.037 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.583) | **substitution** | 1.44 | 0.000 | 0.044 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'镘'` (0.775) | **correct** | 1.73 | 0.775 | 0.015 | `''` |

### `'ὂ'` — id 149620

single-probe lp -0.687 · fragility 1.00 · mean lp -7.08 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' pot content' [·] ' init abs comfort'` | 16 | `' init'` (0.924) | **deletion** | 0.50 | 0.000 | 0.069 | `''` |
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.957) | **deletion** | 0.29 | 0.000 | 0.031 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.974) | **deletion** | 0.22 | 0.000 | 0.022 | `''` |
| best | `' den namespace items pas' [·] ' person cont sound'` | 64 | `'ὂ'` (0.469) | **correct** | 3.34 | 0.469 | -0.000 | `''` |

### `'ጽ'` — id 151466

single-probe lp -0.544 · fragility 1.00 · mean lp -9.74 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.02 | 0.000 | 0.011 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.402) | **substitution** | 2.20 | 0.000 | 0.022 | `''` |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | 16 | `' doing'` (0.944) | **deletion** | 0.33 | 0.000 | 0.031 | `''` |
| best | `' den namespace items pas' [·] ' person cont sound'` | 64 | `' person'` (0.581) | **deletion** | 3.09 | 0.114 | 0.046 | `''` |

### `'حرية'` — id 136716

single-probe lp -0.540 · fragility 0.96 · mean lp -6.53 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.478) | **substitution** | 1.90 | 0.000 | 0.065 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.789) | **deletion** | 1.43 | 0.001 | 0.043 | `''` |
| worst | `' las touch recently hard' [·] ' beh func eff'` | 8 | `'<|im_start|>'` (0.338) | **substitution** | 2.45 | 0.000 | -0.003 | `''` |
| best | `' origin child below namespace' [·] ' lock home rece'` | 32 | `'حرية'` (0.751) | **correct** | 2.60 | 0.751 | 0.046 | `''` |

### `'גה'` — id 131251

single-probe lp -0.483 · fragility 0.96 · mean lp -7.39 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.936) | **deletion** | 0.63 | 0.000 | 0.032 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.856) | **deletion** | 1.04 | 0.000 | 0.030 | `''` |
| worst | `' pot content' [·] ' init abs comfort'` | 16 | `'\n'` (0.483) | **substitution** | 2.09 | 0.000 | 0.032 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'גה'` (0.728) | **correct** | 2.14 | 0.728 | -0.011 | `''` |

### `'יוני'` — id 131305

single-probe lp -0.227 · fragility 0.96 · mean lp -9.54 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.01 | 0.000 | 0.045 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.843) | **deletion** | 1.03 | 0.000 | 0.052 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'</think>'` (0.682) | **substitution** | 1.56 | 0.000 | 0.032 | `''` |
| best | `' den namespace items pas' [·] ' person cont sound'` | 64 | `'יוני'` (0.898) | **correct** | 0.96 | 0.898 | 0.010 | `''` |

### `'嗐'` — id 121211

single-probe lp -0.018 · fragility 0.96 · mean lp -6.57 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.978) | **deletion** | 0.21 | 0.000 | 0.063 | `''` |
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.933) | **deletion** | 0.52 | 0.000 | 0.060 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.500) | **substitution** | 2.36 | 0.000 | 0.052 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'嗐'` (0.843) | **correct** | 1.19 | 0.843 | 0.073 | `''` |

### `'蹒'` — id 121878

single-probe lp -0.416 · fragility 0.96 · mean lp -6.07 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.996) | **deletion** | 0.04 | 0.000 | 0.031 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.977) | **deletion** | 0.20 | 0.000 | 0.018 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.437) | **substitution** | 2.39 | 0.000 | 0.003 | `''` |
| best | `' design remain notice early' [·] ' doing additional methods'` | 16 | `'蹒'` (0.658) | **correct** | 1.47 | 0.658 | 0.013 | `''` |

### `'בחר'` — id 129930

single-probe lp -0.059 · fragility 0.96 · mean lp -5.64 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'\n'` (0.414) | **substitution** | 2.35 | 0.000 | 0.038 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.552) | **substitution** | 2.02 | 0.000 | 0.020 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.662) | **deletion** | 2.58 | 0.000 | 0.034 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'בחר'` (0.875) | **correct** | 1.32 | 0.875 | 0.058 | `''` |

### `'אבל'` — id 140623

single-probe lp -0.578 · fragility 0.96 · mean lp -8.21 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' song released top sl' [·] ' under sus since'` | 32 | `' under'` (0.999) | **deletion** | 0.02 | 0.000 | 0.023 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.376) | **substitution** | 2.11 | 0.000 | 0.043 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.951) | **deletion** | 0.42 | 0.000 | 0.033 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'אבל'` (0.544) | **correct** | 3.46 | 0.544 | 0.059 | `''` |

### `'работать'` — id 132991

single-probe lp -0.056 · fragility 0.96 · mean lp -4.76 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'<|im_end|>'` (0.371) | **substitution** | 2.35 | 0.000 | 0.052 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.947) | **deletion** | 0.49 | 0.000 | 0.070 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `'<|im_end|>'` (0.641) | **substitution** | 2.00 | 0.003 | 0.033 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'работать'` (0.694) | **correct** | 1.53 | 0.694 | 0.006 | `''` |

### `'فاع'` — id 126119

single-probe lp -0.178 · fragility 0.96 · mean lp -4.63 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.861) | **deletion** | 1.08 | 0.000 | 0.028 | `''` |
| worst | `' unit rep anyone fl' [·] ' ro tf'` | 16 | `'<|im_end|>'` (0.385) | **substitution** | 2.37 | 0.000 | 0.012 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `' under'` (0.998) | **deletion** | 0.03 | 0.000 | 0.041 | `''` |
| best | `' origin child below namespace' [·] ' lock home rece'` | 32 | `'فاع'` (0.749) | **correct** | 1.41 | 0.749 | 0.028 | `''` |

### `'سبة'` — id 126014

single-probe lp -0.017 · fragility 0.96 · mean lp -6.73 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'</think>'` (0.402) | **substitution** | 2.12 | 0.000 | 0.034 | `''` |
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'\n'` (0.397) | **substitution** | 2.67 | 0.000 | 0.042 | `''` |
| worst | `' origin child below namespace' [·] ' lock home rece'` | 32 | `'</think>'` (0.352) | **substitution** | 2.63 | 0.000 | -0.010 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'سبة'` (1.000) | **correct** | 0.01 | 0.999 | 0.008 | `''` |

### `'ம'` — id 146150

single-probe lp -0.038 · fragility 0.96 · mean lp -4.95 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.592) | **substitution** | 1.43 | 0.000 | 0.052 | `''` |
| worst | `' para course pat wom' [·] ' under available carry'` | 16 | `' under'` (0.980) | **deletion** | 0.21 | 0.000 | 0.072 | `''` |
| worst | `' fund port sw struct' [·] ' term echo rev'` | 64 | `' term'` (0.994) | **deletion** | 0.08 | 0.000 | 0.018 | `''` |
| best | `' pot content' [·] ' init abs comfort'` | 16 | `' init'` (0.527) | **deletion** | 1.51 | 0.410 | 0.066 | `''` |

### `'앉'` — id 144125

single-probe lp -0.009 · fragility 0.96 · mean lp -7.86 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.987) | **deletion** | 0.14 | 0.000 | 0.005 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.966) | **deletion** | 0.32 | 0.000 | 0.011 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.921) | **deletion** | 0.59 | 0.000 | 0.030 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'앉'` (0.980) | **correct** | 0.23 | 0.980 | 0.019 | `''` |

### `'늦'` — id 144211

single-probe lp -0.544 · fragility 0.96 · mean lp -7.24 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.999) | **deletion** | 0.01 | 0.000 | 0.032 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.265) | **substitution** | 3.00 | 0.000 | 0.017 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.990) | **deletion** | 0.11 | 0.000 | 0.026 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'늦'` (0.600) | **correct** | 2.04 | 0.600 | 0.057 | `''` |

### `'נסה'` — id 132576

single-probe lp -0.000 · fragility 0.96 · mean lp -10.79 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.839) | **deletion** | 1.17 | 0.000 | 0.020 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n'` (0.393) | **substitution** | 1.88 | 0.000 | 0.021 | `''` |
| worst | `' song released top sl' [·] ' under sus since'` | 32 | `'\n'` (0.330) | **substitution** | 2.97 | 0.000 | -0.011 | `''` |
| best | `' stuff' [·] ' function come tab'` | 8 | `'נסה'` (0.726) | **correct** | 2.33 | 0.726 | 0.039 | `''` |

### `' באו'` — id 132355

single-probe lp -0.003 · fragility 0.96 · mean lp -5.94 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.471) | **deletion** | 2.18 | 0.000 | -0.017 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.555) | **deletion** | 2.16 | 0.000 | -0.022 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.504) | **deletion** | 2.63 | 0.001 | 0.017 | `''` |
| best | `' den namespace items pas' [·] ' person cont sound'` | 64 | `' באו'` (0.840) | **correct** | 0.98 | 0.840 | 0.004 | `''` |

### `' כולו'` — id 143460

single-probe lp -0.018 · fragility 0.96 · mean lp -8.27 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' design remain notice early' [·] ' doing additional methods'` | 16 | `' doing'` (0.936) | **deletion** | 0.53 | 0.000 | 0.020 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.988) | **deletion** | 0.14 | 0.000 | 0.019 | `''` |
| worst | `' parts major mobile veh' [·] ' man fn enough'` | 32 | `'\n'` (0.549) | **substitution** | 2.03 | 0.000 | 0.042 | `''` |
| best | `' pot content' [·] ' init abs comfort'` | 16 | `' כולו'` (0.781) | **correct** | 1.98 | 0.781 | 0.040 | `''` |

### `' рассмат'` — id 142145

single-probe lp -0.044 · fragility 0.96 · mean lp -8.81 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.706) | **deletion** | 1.68 | 0.000 | -0.025 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `'<|im_end|>'` (0.832) | **substitution** | 1.31 | 0.000 | -0.011 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.463) | **substitution** | 2.08 | 0.000 | -0.011 | `''` |
| best | `' den namespace items pas' [·] ' person cont sound'` | 64 | `' рассмат'` (0.833) | **correct** | 1.42 | 0.833 | 0.001 | `''` |

### `'שוק'` — id 126710

single-probe lp -0.002 · fragility 0.96 · mean lp -9.07 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.979) | **deletion** | 0.19 | 0.000 | 0.020 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.454) | **substitution** | 1.68 | 0.000 | 0.005 | `''` |
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (0.974) | **deletion** | 0.21 | 0.000 | -0.002 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'שוק'` (0.782) | **correct** | 1.72 | 0.783 | 0.031 | `''` |

### `'댁'` — id 145249

single-probe lp -0.004 · fragility 0.96 · mean lp -4.42 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.994) | **deletion** | 0.08 | 0.000 | 0.021 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.954) | **deletion** | 0.49 | 0.000 | 0.070 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `' ax'` (0.412) | **deletion** | 3.32 | 0.000 | 0.049 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'댁'` (0.996) | **correct** | 0.06 | 0.996 | 0.026 | `''` |

### `'𝗠'` — id 147995

single-probe lp -0.139 · fragility 0.96 · mean lp -5.18 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | 32 | `' tor'` (1.000) | **deletion** | 0.00 | 0.000 | 0.068 | `''` |
| worst | `' word override script cur' [·] ' environment mis reach'` | 32 | `' environment'` (0.989) | **deletion** | 0.13 | 0.000 | 0.072 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `' ax'` (0.896) | **deletion** | 0.77 | 0.000 | 0.051 | `''` |
| best | `' len' [·] ' mobile std drop'` | 8 | `'𝗠'` (0.856) | **correct** | 1.21 | 0.856 | 0.116 | `''` |

### `'дачи'` — id 134630

single-probe lp -0.050 · fragility 0.96 · mean lp -2.97 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' margin prot consider cult' [·] ' not million provided'` | 64 | `' not'` (0.841) | **deletion** | 1.25 | 0.001 | 0.009 | `''` |
| worst | `' videos means leave didn' [·] ' ax global frame'` | 64 | `'\n\n'` (0.257) | **substitution** | 2.90 | 0.000 | 0.024 | `''` |
| worst | `' bad consist' [·] ' assert signific cert'` | 8 | `' assert'` (0.858) | **deletion** | 1.17 | 0.004 | 0.034 | `''` |
| best | `' client met' [·] ' she len rais'` | 64 | `'дачи'` (0.871) | **correct** | 1.38 | 0.871 | -0.012 | `''` |

## Verified glitch tokens (reference)


### `'larınd'` — id 129956  (single-probe lp -21.32, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' videos means leave didn' [·] ' ax global frame'` | `'\n\n'` (0.500) | substitution | 2.19 | 0.000 | -0.057 |
| worst | `' margin prot consider cult' [·] ' not million provided'` | `' not'` (0.933) | deletion | 0.57 | 0.000 | -0.020 |
| worst | `' att attempt ke quite' [·] ' tor near group'` | `'\n'` (0.545) | substitution | 1.58 | 0.000 | 0.017 |
| best | `' pot content' [·] ' init abs comfort'` | `'larınd'` (0.673) | correct | 2.89 | 0.673 | 0.003 |

### `'吏'` — id 151148  (single-probe lp -20.77, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' bad consist' [·] ' assert signific cert'` | `' assert'` (0.982) | deletion | 0.19 | 0.000 | 0.219 |
| worst | `' att attempt ke quite' [·] ' tor near group'` | `'<|im_end|>'` (0.400) | substitution | 2.11 | 0.000 | 0.018 |
| worst | `' thems line sem things' [·] ' range car player'` | `' range'` (0.962) | deletion | 0.30 | 0.000 | 0.129 |
| best | `' word override script cur' [·] ' environment mis reach'` | `'</think>'` (0.618) | substitution | 2.22 | 0.000 | -0.047 |

### `'𝄹'` — id 151290  (single-probe lp -20.66, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.976) | deletion | 0.19 | 0.000 | 0.054 |
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (1.000) | deletion | 0.01 | 0.000 | 0.079 |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | `' doing'` (0.981) | deletion | 0.16 | 0.000 | 0.068 |
| best | `' client met' [·] ' she len rais'` | `' she'` (0.547) | deletion | 2.78 | 0.000 | 0.081 |

### `'พื้นที่'` — id 126795  (single-probe lp -20.49, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' bad consist' [·] ' assert signific cert'` | `' assert'` (0.927) | deletion | 0.64 | 0.000 | 0.178 |
| worst | `' margin prot consider cult' [·] ' not million provided'` | `'\n\n'` (0.778) | substitution | 1.13 | 0.000 | 0.196 |
| worst | `' word override script cur' [·] ' environment mis reach'` | `' environment'` (0.844) | deletion | 1.13 | 0.000 | 0.069 |
| best | `' para course pat wom' [·] ' under available carry'` | `'\n\n'` (0.259) | substitution | 5.33 | 0.000 | -0.027 |

### `'ได้ง่าย'` — id 140750  (single-probe lp -20.31, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (0.982) | deletion | 0.17 | 0.000 | 0.159 |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | `' doing'` (0.921) | deletion | 0.48 | 0.000 | 0.119 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.926) | deletion | 0.52 | 0.000 | 0.136 |
| best | `' client met' [·] ' she len rais'` | `'<|im_end|>'` (0.435) | substitution | 3.00 | 0.000 | 0.037 |

### `'กระเป๋า'` — id 136695  (single-probe lp -20.31, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (1.000) | deletion | 0.01 | 0.000 | 0.137 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.953) | deletion | 0.35 | 0.000 | 0.112 |
| worst | `' margin prot consider cult' [·] ' not million provided'` | `' not'` (0.948) | deletion | 0.42 | 0.000 | 0.202 |
| best | `' den namespace items pas' [·] ' person cont sound'` | `'</think>'` (0.592) | substitution | 2.23 | 0.000 | 0.138 |

### `'燐'` — id 149824  (single-probe lp -20.30, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' bad consist' [·] ' assert signific cert'` | `' assert'` (0.989) | deletion | 0.13 | 0.000 | 0.179 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.935) | deletion | 0.50 | 0.000 | 0.059 |
| worst | `' pot content' [·] ' init abs comfort'` | `' init'` (0.346) | deletion | 2.20 | 0.000 | 0.208 |
| best | `' den namespace items pas' [·] ' person cont sound'` | `'<|im_end|>'` (0.895) | substitution | 0.88 | 0.000 | 0.087 |

### `'สนับสนุ'` — id 133586  (single-probe lp -20.25, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' specific isn parents owner' [·] ' doing gold water'` | `' doing'` (0.939) | deletion | 0.43 | 0.000 | 0.090 |
| worst | `' videos means leave didn' [·] ' ax global frame'` | `'\n\n'` (0.314) | substitution | 2.30 | 0.000 | 0.211 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.838) | deletion | 1.06 | 0.000 | 0.160 |
| best | `' client met' [·] ' she len rais'` | `'<|im_end|>'` (0.697) | substitution | 2.06 | 0.000 | 0.174 |

### `'มากขึ้น'` — id 131240  (single-probe lp -20.20, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' bad consist' [·] ' assert signific cert'` | `'  \n\n'` (0.473) | substitution | 1.63 | 0.000 | 0.155 |
| worst | `' word override script cur' [·] ' environment mis reach'` | `' environment'` (0.608) | deletion | 1.16 | 0.000 | 0.192 |
| worst | `' pot content' [·] ' init abs comfort'` | `' init'` (0.973) | deletion | 0.24 | 0.000 | -0.040 |
| best | `' den namespace items pas' [·] ' person cont sound'` | `'<|im_end|>'` (0.418) | substitution | 3.15 | 0.000 | 0.215 |

### `'��取'` — id 27487  (single-probe lp -20.19, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' pot content' [·] ' init abs comfort'` | `' init'` (0.971) | deletion | 0.27 | 0.000 | 0.195 |
| worst | `' bad consist' [·] ' assert signific cert'` | `' assert'` (0.932) | deletion | 0.63 | 0.000 | 0.189 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.959) | deletion | 0.34 | 0.000 | 0.161 |
| best | `' client met' [·] ' she len rais'` | `'<|im_end|>'` (0.298) | substitution | 2.74 | 0.000 | 0.042 |

### `'ארגוני'` — id 137670  (single-probe lp -20.19, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.980) | deletion | 0.17 | 0.000 | 0.016 |
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (0.994) | deletion | 0.06 | 0.000 | 0.037 |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | `' doing'` (0.852) | deletion | 0.91 | 0.000 | 0.016 |
| best | `' len' [·] ' mobile std drop'` | `'</think>'` (0.661) | substitution | 2.00 | 0.000 | 0.011 |

### `'הבנה'` — id 143454  (single-probe lp -20.11, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (0.999) | deletion | 0.02 | 0.000 | 0.025 |
| worst | `' videos means leave didn' [·] ' ax global frame'` | `'\n'` (0.605) | substitution | 1.60 | 0.000 | 0.038 |
| worst | `' origin child below namespace' [·] ' lock home rece'` | `'\n'` (0.681) | substitution | 1.40 | 0.000 | 0.036 |
| best | `' client met' [·] ' she len rais'` | `'</think>'` (0.590) | substitution | 2.12 | 0.000 | 0.031 |

### `' สิงหาคม'` — id 142447  (single-probe lp -20.09, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (0.978) | deletion | 0.19 | 0.000 | 0.165 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.960) | deletion | 0.31 | 0.000 | 0.106 |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | `' doing'` (0.964) | deletion | 0.28 | 0.000 | 0.110 |
| best | `' den namespace items pas' [·] ' person cont sound'` | `' person'` (0.404) | deletion | 2.17 | 0.000 | 0.068 |

### `'ความเป็น'` — id 132966  (single-probe lp -20.06, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' pot content' [·] ' init abs comfort'` | `' abs'` (0.929) | substitution | 0.54 | 0.000 | 0.127 |
| worst | `' word override script cur' [·] ' environment mis reach'` | `' environment'` (0.690) | deletion | 1.67 | 0.000 | -0.068 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.571) | deletion | 1.88 | 0.000 | 0.067 |
| best | `' client met' [·] ' she len rais'` | `'\n\n'` (0.219) | substitution | 5.27 | 0.000 | -0.043 |

### `'แม้'` — id 126927  (single-probe lp -19.77, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' pot content' [·] ' init abs comfort'` | `' init'` (0.358) | deletion | 3.44 | 0.000 | 0.014 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.938) | deletion | 0.46 | 0.000 | 0.193 |
| worst | `' client met' [·] ' she len rais'` | `'?;\n\n'` (0.516) | substitution | 1.66 | 0.000 | 0.120 |
| best | `' printf ne throw sy' [·] ' art human ver'` | `'\n\n'` (0.372) | substitution | 5.73 | 0.000 | 0.189 |

### `'量'` — id 147225  (single-probe lp -19.77, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (0.998) | deletion | 0.03 | 0.000 | 0.179 |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | `' doing'` (0.938) | deletion | 0.47 | 0.000 | 0.085 |
| worst | `' para course pat wom' [·] ' under available carry'` | `' under'` (0.991) | deletion | 0.10 | 0.000 | 0.116 |
| best | `' den namespace items pas' [·] ' person cont sound'` | `' person'` (0.439) | deletion | 2.42 | 0.000 | 0.108 |

### `'กิจกรรม'` — id 130551  (single-probe lp -19.65, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' bad consist' [·] ' assert signific cert'` | `' assert'` (0.983) | deletion | 0.18 | 0.000 | 0.250 |
| worst | `' word override script cur' [·] ' environment mis reach'` | `' environment'` (0.405) | deletion | 2.43 | 0.000 | -0.043 |
| worst | `' song released top sl' [·] ' under sus since'` | `' under'` (0.992) | deletion | 0.09 | 0.000 | 0.205 |
| best | `' fund port sw struct' [·] ' term echo rev'` | `'<|im_end|>'` (0.249) | substitution | 4.44 | 0.000 | -0.010 |

### `'มักจะ'` — id 139176  (single-probe lp -19.65, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (0.960) | deletion | 0.31 | 0.000 | 0.176 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.932) | deletion | 0.47 | 0.000 | 0.143 |
| worst | `' para course pat wom' [·] ' under available carry'` | `' under'` (0.995) | deletion | 0.06 | 0.000 | 0.191 |
| best | `' client met' [·] ' she len rais'` | `' she'` (0.239) | deletion | 3.12 | 0.000 | 0.072 |

### `'แก้ปัญหา'` — id 143828  (single-probe lp -19.51, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.977) | deletion | 0.20 | 0.000 | 0.091 |
| worst | `' att attempt ke quite' [·] ' tor near group'` | `' tor'` (0.974) | deletion | 0.21 | 0.000 | 0.138 |
| worst | `' specific isn parents owner' [·] ' doing gold water'` | `' doing'` (0.977) | deletion | 0.22 | 0.000 | 0.102 |
| best | `' den namespace items pas' [·] ' person cont sound'` | `' person'` (0.465) | deletion | 2.43 | 0.000 | 0.116 |

### `'น้ำหนัก'` — id 135989  (single-probe lp -19.47, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' word override script cur' [·] ' environment mis reach'` | `' environment'` (0.850) | deletion | 0.96 | 0.000 | 0.160 |
| worst | `' pot content' [·] ' init abs comfort'` | `' abs'` (0.925) | substitution | 0.53 | 0.000 | 0.212 |
| worst | `' design remain notice early' [·] ' doing additional methods'` | `' doing'` (0.623) | deletion | 1.55 | 0.000 | 0.143 |
| best | `' den namespace items pas' [·] ' person cont sound'` | `'<|im_end|>'` (0.856) | substitution | 0.97 | 0.000 | 0.210 |
