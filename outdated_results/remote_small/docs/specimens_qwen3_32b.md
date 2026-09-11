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

## Fragile clean-looking tokens (pass the single probe; fail in some contexts)


## Verified glitch tokens (reference)


### `'ớ'` — id 141628  (single-probe lp -20.98, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.809) | substitution | 0.81 | 0.000 | 0.006 |
| worst | `' cal them' [·] ' someone damage educ'` | `'<|im_end|>'` (0.434) | substitution | 2.09 | 0.000 | 0.044 |
| worst | `' requ ang to concern' [·] ' professional'` | `' professional'` (0.555) | deletion | 2.35 | 0.000 | -0.075 |
| best | `' orig val sales std' [·] ' throws uses js'` | `'<|im_end|>'` (0.224) | substitution | 7.38 | 0.000 | -0.006 |

### `'อังกฤษ'` — id 127887  (single-probe lp -20.44, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' favor times chang single' [·] ' blue index layout'` | `' blue'` (0.915) | deletion | 0.65 | 0.000 | 0.073 |
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.628) | substitution | 1.45 | 0.000 | -0.045 |
| worst | `' random chat incre' [·] ' across edge levels'` | `'<|im_end|>'` (0.438) | substitution | 2.42 | 0.000 | 0.171 |
| best | `' lib didn' [·] ' bed another column'` | `'<tool_call>'` (0.349) | substitution | 6.33 | 0.000 | -0.072 |

### `'ใช่'` — id 126984  (single-probe lp -20.35, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' requ ang to concern' [·] ' professional'` | `' professional'` (0.355) | deletion | 2.69 | 0.000 | -0.012 |
| worst | `' ar ref det ren' [·] ' pos click both'` | `'<|im_end|>'` (0.611) | substitution | 2.14 | 0.000 | -0.045 |
| worst | `' don hot being there' [·] ' loaded night der'` | `' loaded'` (0.451) | deletion | 2.68 | 0.000 | -0.146 |
| best | `' orig val sales std' [·] ' throws uses js'` | `'<|im_end|>'` (0.067) | substitution | 10.12 | 0.000 | -0.019 |

### `'ล่า'` — id 126892  (single-probe lp -20.28, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.863) | substitution | 0.62 | 0.000 | -0.041 |
| worst | `' cal them' [·] ' someone damage educ'` | `'<|im_end|>'` (0.445) | substitution | 2.51 | 0.000 | 0.047 |
| worst | `' lib didn' [·] ' bed another column'` | `'<|im_start|>'` (0.870) | substitution | 1.16 | 0.000 | -0.094 |
| best | `' between item slow pet' [·] ' mit dans photo'` | `'<|im_end|>'` (0.408) | substitution | 4.83 | 0.000 | -0.168 |

### `' สิงหาคม'` — id 142447  (single-probe lp -19.99, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' ar ref det ren' [·] ' pos click both'` | `'<|im_start|>'` (0.885) | substitution | 1.05 | 0.000 | -0.092 |
| worst | `' random chat incre' [·] ' across edge levels'` | `'\n'` (0.292) | substitution | 4.14 | 0.000 | 0.101 |
| worst | `' don hot being there' [·] ' loaded night der'` | `'<|im_end|>'` (0.454) | substitution | 2.37 | 0.000 | -0.136 |
| best | `' nil gives compl money' [·] ' fem specified es'` | `'<tool_call>'` (0.099) | substitution | 7.53 | 0.000 | -0.059 |

### `'𬸪'` — id 123637  (single-probe lp -19.97, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' cal them' [·] ' someone damage educ'` | `'<|im_end|>'` (0.557) | substitution | 1.86 | 0.000 | 0.019 |
| worst | `' don hot being there' [·] ' loaded night der'` | `'<|im_end|>'` (0.451) | substitution | 1.76 | 0.000 | -0.057 |
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.863) | substitution | 0.80 | 0.000 | -0.002 |
| best | `' manufact ter ide arr' [·] ' greater catch deb'` | `'<|im_end|>'` (0.398) | substitution | 2.62 | 0.000 | -0.032 |

### `'웛'` — id 149983  (single-probe lp -19.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' don hot being there' [·] ' loaded night der'` | `'<|im_end|>'` (0.961) | substitution | 0.31 | 0.000 | -0.081 |
| worst | `' cal them' [·] ' someone damage educ'` | `'<|im_end|>'` (0.677) | substitution | 1.58 | 0.000 | 0.010 |
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.986) | substitution | 0.15 | 0.000 | -0.053 |
| best | `' variable sense except bul' [·] ' wom tem agreed'` | `'<|im_end|>'` (0.583) | substitution | 3.93 | 0.000 | -0.165 |

### `'แก้ปัญหา'` — id 143828  (single-probe lp -19.42, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' don hot being there' [·] ' loaded night der'` | `' loaded'` (0.522) | deletion | 2.20 | 0.000 | -0.154 |
| worst | `' cal them' [·] ' someone damage educ'` | `'<|im_end|>'` (0.443) | substitution | 2.49 | 0.000 | 0.043 |
| worst | `' requ ang to concern' [·] ' professional'` | `'<|im_end|>'` (0.657) | substitution | 1.87 | 0.000 | -0.020 |
| best | `' between item slow pet' [·] ' mit dans photo'` | `'apl'` (0.231) | substitution | 7.90 | 0.000 | -0.137 |

### `'อัพ'` — id 140665  (single-probe lp -19.39, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `' tried'` (0.725) | deletion | 1.46 | 0.000 | -0.028 |
| worst | `' don hot being there' [·] ' loaded night der'` | `'\n'` (0.376) | substitution | 2.43 | 0.000 | -0.155 |
| worst | `' walk directly ent sur' [·] ' major press one'` | `'orge'` (0.430) | substitution | 4.20 | 0.000 | -0.082 |
| best | `' that extends deter ev' [·] ' pract camp pie'` | `'\n'` (0.046) | substitution | 10.17 | 0.000 | -0.197 |

### `'ค้น'` — id 133355  (single-probe lp -19.27, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.665) | substitution | 1.37 | 0.000 | -0.082 |
| worst | `' cal them' [·] ' someone damage educ'` | `' someone'` (0.333) | deletion | 2.69 | 0.000 | 0.084 |
| worst | `' random chat incre' [·] ' across edge levels'` | `'\n\n'` (0.393) | substitution | 2.66 | 0.000 | 0.132 |
| best | `' measure below invol chance' [·] ' enter email weap'` | `'<|im_end|>'` (0.285) | substitution | 6.57 | 0.000 | -0.087 |

### `'กระเป๋า'` — id 136695  (single-probe lp -19.25, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.736) | substitution | 1.04 | 0.000 | -0.020 |
| worst | `' cal them' [·] ' someone damage educ'` | `' someone'` (0.586) | deletion | 1.96 | 0.000 | 0.103 |
| worst | `' requ ang to concern' [·] ' professional'` | `' professional'` (0.464) | deletion | 2.08 | 0.000 | -0.001 |
| best | `' nil gives compl money' [·] ' fem specified es'` | `'<tool_call>'` (0.328) | substitution | 6.39 | 0.000 | -0.075 |

### `' แต่ก็'` — id 139599  (single-probe lp -19.16, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' cal them' [·] ' someone damage educ'` | `' someone'` (0.692) | deletion | 1.55 | 0.000 | 0.021 |
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.803) | substitution | 0.98 | 0.000 | -0.022 |
| worst | `' requ ang to concern' [·] ' professional'` | `' professional'` (0.367) | deletion | 2.33 | 0.000 | -0.045 |
| best | `' redu raise mus order' [·] ' rights energ send'` | `'<|im_end|>'` (0.736) | substitution | 2.84 | 0.000 | -0.111 |

### `'พ่อ'` — id 130320  (single-probe lp -18.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `' tried'` (0.483) | deletion | 1.28 | 0.000 | -0.041 |
| worst | `' requ ang to concern' [·] ' professional'` | `' professional'` (0.469) | deletion | 2.40 | 0.000 | -0.043 |
| worst | `' random chat incre' [·] ' across edge levels'` | `'\n\n'` (0.353) | substitution | 2.36 | 0.000 | 0.118 |
| best | `' lib didn' [·] ' bed another column'` | `'<tool_call>'` (0.064) | substitution | 9.40 | 0.000 | -0.119 |

### `'𝇗'` — id 151297  (single-probe lp -18.86, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' favor times chang single' [·] ' blue index layout'` | `' mon'` (0.981) | substitution | 0.23 | 0.000 | 0.038 |
| worst | `' that extends deter ev' [·] ' pract camp pie'` | `' mon'` (0.574) | substitution | 1.52 | 0.000 | -0.114 |
| worst | `' don hot being there' [·] ' loaded night der'` | `' mon'` (0.805) | substitution | 1.43 | 0.000 | -0.093 |
| best | `' measure below invol chance' [·] ' enter email weap'` | `' mon'` (0.404) | substitution | 3.89 | 0.000 | 0.098 |

### `'บุรี'` — id 133153  (single-probe lp -18.84, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `' tried'` (0.578) | deletion | 2.52 | 0.000 | -0.006 |
| worst | `' even' [·] ' necessary es imp'` | `'<|im_start|>'` (0.363) | substitution | 4.08 | 0.000 | 0.031 |
| worst | `' requ ang to concern' [·] ' professional'` | `'<|im_start|>'` (0.431) | substitution | 3.02 | 0.000 | -0.047 |
| best | `' nil gives compl money' [·] ' fem specified es'` | `'<tool_call>'` (0.263) | substitution | 6.65 | 0.000 | -0.018 |

### `'สมบู'` — id 138059  (single-probe lp -18.68, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.752) | substitution | 1.04 | 0.000 | -0.053 |
| worst | `' even' [·] ' necessary es imp'` | `'<|im_end|>'` (0.819) | substitution | 1.35 | 0.000 | 0.025 |
| worst | `' requ ang to concern' [·] ' professional'` | `' professional'` (0.761) | deletion | 1.57 | 0.000 | -0.040 |
| best | `' fa let btn comput' [·] ' inj oper import'` | `'<|im_end|>'` (0.222) | substitution | 7.21 | 0.000 | -0.172 |

### `'ราคาถูก'` — id 138415  (single-probe lp -18.65, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `' tried'` (0.841) | deletion | 0.75 | 0.000 | -0.003 |
| worst | `' cal them' [·] ' someone damage educ'` | `' someone'` (0.367) | deletion | 2.74 | 0.000 | 0.048 |
| worst | `' random chat incre' [·] ' across edge levels'` | `'\n\n'` (0.379) | substitution | 3.26 | 0.000 | 0.255 |
| best | `' nil gives compl money' [·] ' fem specified es'` | `'<|im_end|>'` (0.371) | substitution | 5.89 | 0.000 | -0.100 |

### `'เต็ม'` — id 129910  (single-probe lp -18.29, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.554) | substitution | 1.14 | 0.000 | -0.091 |
| worst | `' cal them' [·] ' someone damage educ'` | `'<|im_end|>'` (0.338) | substitution | 2.65 | 0.000 | 0.063 |
| worst | `' don hot being there' [·] ' loaded night der'` | `'<|im_end|>'` (0.771) | substitution | 1.50 | 0.000 | -0.163 |
| best | `' fa let btn comput' [·] ' inj oper import'` | `'<|im_start|>'` (0.270) | substitution | 6.50 | 0.000 | -0.162 |

### `'แข็'` — id 127932  (single-probe lp -18.20, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `' tried'` (0.538) | deletion | 1.47 | 0.000 | -0.053 |
| worst | `' don hot being there' [·] ' loaded night der'` | `'<|im_start|>'` (0.406) | substitution | 2.36 | 0.000 | -0.155 |
| worst | `' random chat incre' [·] ' across edge levels'` | `'\n\n'` (0.192) | substitution | 4.34 | 0.000 | 0.215 |
| best | `' orig val sales std' [·] ' throws uses js'` | `'\n'` (0.143) | substitution | 7.99 | 0.000 | -0.012 |

### `'บอกว่า'` — id 133219  (single-probe lp -18.18, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' cal them' [·] ' someone damage educ'` | `'\n'` (0.269) | substitution | 2.97 | 0.000 | 0.017 |
| worst | `' random chat incre' [·] ' across edge levels'` | `'\n'` (0.170) | substitution | 6.53 | 0.000 | 0.238 |
| worst | `' house' [·] ' tried win far'` | `' tried'` (0.676) | deletion | 2.74 | 0.000 | 0.012 |
| best | `' between item slow pet' [·] ' mit dans photo'` | `'<|im_end|>'` (0.245) | substitution | 7.79 | 0.000 | -0.110 |
