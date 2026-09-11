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

single-probe lp -0.487 · fragility 0.83 · mean lp -5.03 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' between item slow pet' [·] ' mit dans photo'` | 64 | `' follow'` (0.995) | **substitution** | 0.06 | 0.000 | 0.004 | `' follow mit dans photo cell object stud '` |
| worst | `' child objects' [·] ' long my pur'` | 8 | `' follow'` (0.952) | **substitution** | 0.45 | 0.000 | 0.051 | `' follow long my pur we tour\nOkay'` |
| worst | `' les rev into browser' [·] ' path arch we'` | 32 | `' follow'` (0.979) | **substitution** | 0.20 | 0.000 | 0.015 | `' follow path arch we prevent anything ap'` |
| best | `' der saved ren political' [·] ' exc police'` | 64 | `' след'` (0.999) | **correct** | 0.02 | 0.999 | 0.023 | `' след exc police\nText: 1'` |

### `'��'` — id 92120

single-probe lp -0.513 · fragility 0.79 · mean lp -1.88 · canonical False · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' don hot being there' [·] ' loaded night der'` | 64 | `' loaded'` (0.512) | **deletion** | 2.01 | 0.005 | -0.004 | `' loaded night der learn any setup phot r'` |
| worst | `' told text happened owner' [·] ' out calls upon'` | 64 | `'キー'` (0.277) | **substitution** | 2.93 | 0.011 | 0.011 | `'キー out calls upon recomm kn link hon'` |
| worst | `' manufact ter ide arr' [·] ' greater catch deb'` | 32 | `' greater'` (0.890) | **deletion** | 0.98 | 0.035 | 0.026 | `' greater catch deb move cause aud iss th'` |
| best | `' fa let btn comput' [·] ' inj oper import'` | 8 | `'��'` (0.955) | **correct** | 0.47 | 0.955 | -0.012 | `'�� inj oper import\nText: '` |

### `' והת'` — id 133258

single-probe lp -0.553 · fragility 0.54 · mean lp -0.64 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' random chat incre' [·] ' across edge levels'` | 8 | `'\n\n'` (0.188) | **substitution** | 7.27 | 0.156 | 0.051 | `"\n\nOkay, let's see. The"` |
| worst | `' requ ang to concern' [·] ' professional'` | 8 | `' והת'` (0.250) | **correct** | 5.70 | 0.250 | 0.053 | `' והת professional\nText: 12'` |
| worst | `' cal them' [·] ' someone damage educ'` | 32 | `' והת'` (0.275) | **correct** | 6.74 | 0.275 | 0.065 | `' והת someone damage educ things plus sol'` |
| best | `' favor times chang single' [·] ' blue index layout'` | 8 | `' והת'` (0.869) | **correct** | 1.70 | 0.869 | 0.054 | `' והת blue index layout\nText: '` |

### `'ཀ'` — id 147474

single-probe lp -0.651 · fragility 0.42 · mean lp -0.62 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' manufact ter ide arr' [·] ' greater catch deb'` | 32 | `' greater'` (0.925) | **deletion** | 0.54 | 0.059 | 0.072 | `' greater catch deb move cause aud iss th'` |
| worst | `' random chat incre' [·] ' across edge levels'` | 8 | `'\n'` (0.428) | **substitution** | 3.88 | 0.139 | 0.090 | `"\nOkay, let's see. The"` |
| worst | `' even' [·] ' necessary es imp'` | 32 | `'\n'` (0.461) | **substitution** | 3.05 | 0.192 | 0.073 | `'\nCopy: necessary\nCopy: es'` |
| best | `' nil gives compl money' [·] ' fem specified es'` | 16 | `'ཀ'` (0.985) | **correct** | 0.23 | 0.985 | 0.068 | `'ཀ fem specified es argument agre liter c'` |

### `'자는'` — id 132343

single-probe lp -0.438 · fragility 0.42 · mean lp -0.52 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' random chat incre' [·] ' across edge levels'` | 8 | `'\n'` (0.184) | **substitution** | 7.93 | 0.017 | -0.033 | `'\nText: random chat incre\nCopy'` |
| worst | `' redu raise mus order' [·] ' rights energ send'` | 64 | `'<|im_start|>'` (0.440) | **substitution** | 3.29 | 0.302 | -0.023 | `'\n\nOkay, the user wants me'` |
| worst | `' don hot being there' [·] ' loaded night der'` | 64 | `'자는'` (0.422) | **correct** | 4.55 | 0.422 | -0.037 | `'자는 loaded night der learn any setup phot'` |
| best | `' favor times chang single' [·] ' blue index layout'` | 8 | `'자는'` (0.990) | **correct** | 0.14 | 0.990 | -0.011 | `'자는 blue index layout\nText: '` |

### `' realtà'` — id 134523

single-probe lp -0.281 · fragility 0.38 · mean lp -0.65 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' even' [·] ' necessary es imp'` | 32 | `' réalité'` (0.898) | **substitution** | 0.64 | 0.074 | 0.052 | `' réalité necessary es imp cast long prev'` |
| worst | `' that extends deter ev' [·] ' pract camp pie'` | 32 | `' réalité'` (0.867) | **substitution** | 0.67 | 0.117 | -0.011 | `' réalité pract camp pie calcul state pre'` |
| worst | `' walk directly ent sur' [·] ' major press one'` | 16 | `' réalité'` (0.829) | **substitution** | 0.72 | 0.163 | 0.056 | `' réalité major press one ey arch control'` |
| best | `' lib didn' [·] ' bed another column'` | 8 | `' realtà'` (0.937) | **correct** | 0.45 | 0.937 | 0.037 | `' realtà bed another column int soon\nText'` |

### `'بدو'` — id 127961

single-probe lp -0.437 · fragility 0.33 · mean lp -0.52 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' ar ref det ren' [·] ' pos click both'` | 64 | `'بدو'` (0.201) | **correct** | 7.95 | 0.201 | -0.028 | `'بدو pos click both die isn empty matter'` |
| worst | `' lib didn' [·] ' bed another column'` | 8 | `'bedo'` (0.555) | **substitution** | 3.19 | 0.217 | 0.001 | `'bedo bed another column int soon\nText'` |
| worst | `' orig val sales std' [·] ' throws uses js'` | 32 | `'بدو'` (0.259) | **correct** | 6.79 | 0.259 | -0.041 | `'بدو throws uses js invest cut determin m'` |
| best | `' child objects' [·] ' long my pur'` | 8 | `'بدو'` (0.971) | **correct** | 0.31 | 0.971 | 0.053 | `'بدو long my pur we tour\nText'` |

### `' прогн'` — id 141080

single-probe lp -0.680 · fragility 0.21 · mean lp -0.36 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' don hot being there' [·] ' loaded night der'` | 64 | `' prog'` (0.366) | **substitution** | 3.48 | 0.209 | -0.022 | `' progne loaded night der learn any setup'` |
| worst | `' random chat incre' [·] ' across edge levels'` | 8 | `' prog'` (0.294) | **substitution** | 4.68 | 0.276 | 0.055 | `' prog across edge levels specific\nText:'` |
| worst | `' orig val sales std' [·] ' throws uses js'` | 32 | `' прогн'` (0.493) | **correct** | 1.95 | 0.493 | 0.014 | `' прогн throws uses js invest cut determi'` |
| best | `' between item slow pet' [·] ' mit dans photo'` | 64 | `' прогн'` (0.995) | **correct** | 0.07 | 0.995 | -0.054 | `' прогн mit dans photo cell object stud d'` |

### `'慊'` — id 121316

single-probe lp -0.155 · fragility 0.17 · mean lp -0.19 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' lib didn' [·] ' bed another column'` | 8 | `'慊'` (0.211) | **correct** | 7.04 | 0.211 | 0.015 | `'慊 bed another column int soon\nText'` |
| worst | `' orig val sales std' [·] ' throws uses js'` | 32 | `' throws'` (0.521) | **deletion** | 1.90 | 0.382 | 0.001 | `' throws uses js invest cut determin mate'` |
| worst | `' random chat incre' [·] ' across edge levels'` | 8 | `'慊'` (0.371) | **correct** | 3.58 | 0.371 | 0.019 | `'慊 across edge levels specific\nText:'` |
| best | `' nil gives compl money' [·] ' fem specified es'` | 16 | `'慊'` (1.000) | **correct** | 0.01 | 1.000 | 0.014 | `'慊 fem specified es argument agre liter c'` |

### `'ee'` — id 2127

single-probe lp -0.552 · fragility 0.17 · mean lp -0.31 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' manufact ter ide arr' [·] ' greater catch deb'` | 32 | `'er'` (0.283) | **substitution** | 6.97 | 0.006 | 0.021 | `'er greater catch deb move cause aud iss'` |
| worst | `' variable sense except bul' [·] ' wom tem agreed'` | 16 | `'ee'` (0.334) | **correct** | 2.95 | 0.334 | 0.010 | `'ee wom tem agreed point recently fun com'` |
| worst | `' walk directly ent sur' [·] ' major press one'` | 16 | `'ee'` (0.511) | **correct** | 2.55 | 0.512 | 0.046 | `'ee major press one ey arch controller ma'` |
| best | `' fa let btn comput' [·] ' inj oper import'` | 8 | `'ee'` (1.000) | **correct** | 0.00 | 1.000 | 0.018 | `'ee inj oper import\nText: '` |

### `'נע'` — id 134089

single-probe lp -0.668 · fragility 0.12 · mean lp -0.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' favor times chang single' [·] ' blue index layout'` | 8 | `' blue'` (0.525) | **deletion** | 1.65 | 0.409 | 0.081 | `' blue index layout\nText: 1'` |
| worst | `' told text happened owner' [·] ' out calls upon'` | 64 | `'נע'` (0.494) | **correct** | 1.95 | 0.494 | 0.062 | `' out calls upon recomm kn link hon solid'` |
| worst | `' der saved ren political' [·] ' exc police'` | 64 | `'נע'` (0.570) | **correct** | 2.72 | 0.569 | 0.002 | `'נע exc police\nText: 1'` |
| best | `' lib didn' [·] ' bed another column'` | 8 | `'נע'` (0.996) | **correct** | 0.07 | 0.996 | 0.041 | `'נע bed another column int soon\nText'` |

### `' اليمن'` — id 131194

single-probe lp -0.259 · fragility 0.12 · mean lp -0.23 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' random chat incre' [·] ' across edge levels'` | 8 | `' Yemen'` (0.717) | **substitution** | 1.10 | 0.264 | 0.107 | `' Yemen across edge levels specific\nText:'` |
| worst | `' that extends deter ev' [·] ' pract camp pie'` | 32 | `' Yemen'` (0.499) | **substitution** | 1.03 | 0.499 | 0.022 | `' Yemen pract camp pie calcul state pred\n'` |
| worst | `' lib didn' [·] ' bed another column'` | 8 | `' اليمن'` (0.550) | **correct** | 1.83 | 0.551 | 0.050 | `' اليمن bed another column int soon\nText'` |
| best | `' redu raise mus order' [·] ' rights energ send'` | 64 | `' اليمن'` (0.981) | **correct** | 0.15 | 0.981 | 0.049 | `' اليمن rights energ send frame allow hop'` |

### `'stdexcept'` — id 94513

single-probe lp -0.386 · fragility 0.12 · mean lp -0.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cal them' [·] ' someone damage educ'` | 32 | `'\n'` (0.280) | **substitution** | 4.82 | 0.132 | 0.052 | `'\nText: 1234'` |
| worst | `' favor times chang single' [·] ' blue index layout'` | 8 | `'stdexcept'` (0.516) | **correct** | 3.89 | 0.516 | 0.013 | `'stdexcept blue index layout\nText: '` |
| worst | `' house' [·] ' tried win far'` | 16 | `'stdexcept'` (0.620) | **correct** | 2.83 | 0.620 | -0.008 | `'stdexcept tried win far core hor sens go'` |
| best | `' orig val sales std' [·] ' throws uses js'` | 32 | `'stdexcept'` (1.000) | **correct** | 0.00 | 1.000 | 0.037 | `'stdexcept throws uses js invest cut dete'` |

### `'𝓪'` — id 147910

single-probe lp -0.370 · fragility 0.12 · mean lp -0.24 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' told text happened owner' [·] ' out calls upon'` | 64 | `'𝓪'` (0.357) | **correct** | 3.93 | 0.357 | 0.031 | `'𝓪 out calls upon recomm kn link hon'` |
| worst | `' les rev into browser' [·] ' path arch we'` | 32 | `'𝓪'` (0.484) | **correct** | 2.98 | 0.484 | 0.083 | `'𝓪 path arch we prevent anything appear e'` |
| worst | `' ar ref det ren' [·] ' pos click both'` | 64 | `'𝓪'` (0.552) | **correct** | 3.53 | 0.552 | -0.021 | `'𝓪 pos click both die isn empty matter'` |
| best | `' that extends deter ev' [·] ' pract camp pie'` | 32 | `'𝓪'` (0.982) | **correct** | 0.28 | 0.981 | -0.024 | `'𝓪 pract camp pie calcul state pred\n'` |

### `'ঙ'` — id 148571

single-probe lp -0.293 · fragility 0.12 · mean lp -0.41 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' lib didn' [·] ' bed another column'` | 8 | `' bed'` (0.867) | **deletion** | 1.21 | 0.001 | 0.014 | `' bed another column int soon\nText:'` |
| worst | `' manufact ter ide arr' [·] ' greater catch deb'` | 32 | `' greater'` (0.539) | **deletion** | 1.42 | 0.420 | 0.032 | `' greater catch deb move cause aud iss th'` |
| worst | `' told text happened owner' [·] ' out calls upon'` | 64 | `'ঙ'` (0.543) | **correct** | 1.74 | 0.543 | 0.020 | `'ঙ out calls upon recomm kn link hon'` |
| best | `' fa let btn comput' [·] ' inj oper import'` | 8 | `'ঙ'` (0.995) | **correct** | 0.08 | 0.995 | -0.002 | `'ঙ inj oper import\nText: '` |

### `'ANGER'` — id 92368

single-probe lp -0.406 · fragility 0.08 · mean lp -0.13 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' measure below invol chance' [·] ' enter email weap'` | 16 | `' AN'` (0.748) | **truncation** | 0.91 | 0.243 | 0.106 | `' ANGER enter email weap proper plan alig'` |
| worst | `' favor times chang single' [·] ' blue index layout'` | 8 | `'ANGER'` (0.591) | **correct** | 1.43 | 0.591 | 0.084 | `'ANGER blue index layout\nText: '` |
| worst | `' very maybe check classes' [·] ' requ charge clean'` | 16 | `'ANGER'` (0.691) | **correct** | 1.08 | 0.691 | 0.039 | `'ANGER requ charge clean\nText: '` |
| best | `' walk directly ent sur' [·] ' major press one'` | 16 | `'ANGER'` (1.000) | **correct** | 0.01 | 1.000 | 0.057 | `'ANGER major press one ey arch controller'` |

### `'鳢'` — id 122085

single-probe lp -0.293 · fragility 0.08 · mean lp -0.10 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' lib didn' [·] ' bed another column'` | 8 | `'鳢'` (0.360) | **correct** | 4.74 | 0.360 | 0.014 | `'鳢 bed another column int soon\nText'` |
| worst | `' random chat incre' [·] ' across edge levels'` | 8 | `'鳢'` (0.392) | **correct** | 2.95 | 0.392 | 0.043 | `'鳢 across edge levels specific\nText:'` |
| worst | `' variable sense except bul' [·] ' wom tem agreed'` | 16 | `'鳢'` (0.892) | **correct** | 0.97 | 0.892 | -0.059 | `'鳢 wom tem agreed point recently fun comm'` |
| best | `' redu raise mus order' [·] ' rights energ send'` | 64 | `'鳢'` (1.000) | **correct** | 0.00 | 1.000 | 0.047 | `'鳢 rights energ send frame allow hope sal'` |

### `' ๆ'` — id 128630

single-probe lp -0.202 · fragility 0.08 · mean lp -0.16 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' lib didn' [·] ' bed another column'` | 8 | `' doesnt'` (0.580) | **substitution** | 1.88 | 0.242 | 0.049 | `' doesnt bed another column int soon\nText'` |
| worst | `' random chat incre' [·] ' across edge levels'` | 8 | `' ๆ'` (0.325) | **correct** | 3.18 | 0.325 | 0.094 | `' ๆ across edge levels specific\nText:'` |
| worst | `' don hot being there' [·] ' loaded night der'` | 64 | `' ๆ'` (0.791) | **correct** | 0.86 | 0.791 | -0.001 | `' ๆ loaded night der learn any setup phot'` |
| best | `' nil gives compl money' [·] ' fem specified es'` | 16 | `' ๆ'` (0.996) | **correct** | 0.06 | 0.996 | 0.017 | `' ๆ fem specified es argument agre liter '` |

### `'וצ'` — id 123926

single-probe lp -0.317 · fragility 0.08 · mean lp -0.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' random chat incre' [·] ' across edge levels'` | 8 | `'\n\n'` (0.293) | **substitution** | 4.98 | 0.138 | 0.079 | `"\n\nOkay, let's see. The"` |
| worst | `' der saved ren political' [·] ' exc police'` | 64 | `'וצ'` (0.578) | **correct** | 4.58 | 0.578 | 0.009 | `'וצ exc police\nText: 1'` |
| worst | `' lib didn' [·] ' bed another column'` | 8 | `'וצ'` (0.765) | **correct** | 1.93 | 0.766 | 0.021 | `'וצ bed another column int soon\nText'` |
| best | `' redu raise mus order' [·] ' rights energ send'` | 64 | `'וצ'` (0.995) | **correct** | 0.08 | 0.995 | 0.057 | `'וצ rights energ send frame allow hope sa'` |

### `'👕'` — id 146416

single-probe lp -0.400 · fragility 0.08 · mean lp -0.22 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' random chat incre' [·] ' across edge levels'` | 8 | `'👕'` (0.296) | **correct** | 5.37 | 0.296 | 0.166 | `'👕 across edge levels specific\nText:'` |
| worst | `' child objects' [·] ' long my pur'` | 8 | `'👕'` (0.506) | **correct** | 2.71 | 0.507 | 0.192 | `'👕 long my pur we tour\nText'` |
| worst | `' even' [·] ' necessary es imp'` | 32 | `'👕'` (0.631) | **correct** | 2.36 | 0.631 | 0.139 | `'👕 necessary es imp cast long prevent flo'` |
| best | `' that extends deter ev' [·] ' pract camp pie'` | 32 | `'👕'` (0.979) | **correct** | 0.24 | 0.979 | -0.009 | `'👕 pract camp pie calcul state pred\n'` |

### `'饩'` — id 119734

single-probe lp -0.144 · fragility 0.08 · mean lp -0.13 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' don hot being there' [·] ' loaded night der'` | 64 | `' loaded'` (0.611) | **deletion** | 1.72 | 0.164 | -0.029 | `' loaded night der learn any setup phot r'` |
| worst | `' random chat incre' [·] ' across edge levels'` | 8 | `'饩'` (0.452) | **correct** | 4.18 | 0.452 | 0.035 | `'饩 across edge levels specific\nText:'` |
| worst | `' told text happened owner' [·] ' out calls upon'` | 64 | `'饩'` (0.792) | **correct** | 1.08 | 0.792 | 0.007 | `'饩 out calls upon recomm kn link hon'` |
| best | `' nil gives compl money' [·] ' fem specified es'` | 16 | `'饩'` (1.000) | **correct** | 0.01 | 1.000 | 0.003 | `'饩 fem specified es argument agre liter c'` |

### `' كذلك'` — id 135283

single-probe lp -0.504 · fragility 0.08 · mean lp -0.19 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' walk directly ent sur' [·] ' major press one'` | 16 | `' كذلك'` (0.498) | **correct** | 3.30 | 0.498 | 0.019 | `' كذلك major press one ey arch controller'` |
| worst | `' child objects' [·] ' long my pur'` | 8 | `' كذلك'` (0.586) | **correct** | 3.67 | 0.586 | 0.043 | `' كذلك long my pur we tour\nText'` |
| worst | `' between item slow pet' [·] ' mit dans photo'` | 64 | `' كذلك'` (0.632) | **correct** | 2.89 | 0.632 | 0.008 | `' كذلك mit dans photo cell object stud di'` |
| best | `' very maybe check classes' [·] ' requ charge clean'` | 16 | `' كذلك'` (0.967) | **correct** | 0.43 | 0.967 | -0.010 | `' كذلك requ charge clean\nText: '` |

### `' важно'` — id 135497

single-probe lp -0.295 · fragility 0.08 · mean lp -0.19 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' that extends deter ev' [·] ' pract camp pie'` | 32 | `' importante'` (0.444) | **substitution** | 1.76 | 0.444 | -0.000 | `' важно pract camp pie calcul state pred\n'` |
| worst | `' ar ref det ren' [·] ' pos click both'` | 64 | `' важно'` (0.586) | **correct** | 2.17 | 0.586 | -0.007 | `' важно pos click both die isn empty matt'` |
| worst | `' cal them' [·] ' someone damage educ'` | 32 | `' важно'` (0.619) | **correct** | 2.12 | 0.619 | 0.031 | `' важно someone damage educ things plus s'` |
| best | `' don hot being there' [·] ' loaded night der'` | 64 | `' важно'` (0.992) | **correct** | 0.10 | 0.992 | -0.024 | `' важно loaded night der learn any setup '` |

### `'shed'` — id 69461

single-probe lp -0.509 · fragility 0.08 · mean lp -0.08 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' child objects' [·] ' long my pur'` | 8 | `'hed'` (0.658) | **substitution** | 1.19 | 0.311 | 0.056 | `'hed long my pur we tour\nText'` |
| worst | `' don hot being there' [·] ' loaded night der'` | 64 | `' shed'` (0.481) | **substitution** | 1.30 | 0.481 | 0.022 | `'shed loaded night der learn any setup ph'` |
| worst | `' der saved ren political' [·] ' exc police'` | 64 | `'shed'` (0.911) | **correct** | 0.47 | 0.911 | 0.058 | `'shed exc police\nText: 1'` |
| best | `' manufact ter ide arr' [·] ' greater catch deb'` | 32 | `'shed'` (1.000) | **correct** | 0.00 | 1.000 | 0.015 | `'shed greater catch deb move cause aud is'` |

### `' dây'` — id 132625

single-probe lp -0.412 · fragility 0.08 · mean lp -0.26 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' random chat incre' [·] ' across edge levels'` | 8 | `' day'` (0.129) | **substitution** | 7.02 | 0.022 | 0.070 | `"\nOkay, let's see. The"` |
| worst | `' variable sense except bul' [·] ' wom tem agreed'` | 16 | `' dây'` (0.283) | **correct** | 6.63 | 0.283 | -0.013 | `' dây wom tem agreed point recently fun c'` |
| worst | `' walk directly ent sur' [·] ' major press one'` | 16 | `' dây'` (0.626) | **correct** | 1.70 | 0.626 | 0.025 | `' dây major press one ey arch controller '` |
| best | `' don hot being there' [·] ' loaded night der'` | 64 | `' dây'` (0.998) | **correct** | 0.04 | 0.998 | 0.008 | `' dây loaded night der learn any setup ph'` |

## Verified glitch tokens (reference)


### `'ớ'` — id 141628  (single-probe lp -20.98, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.808) | substitution | 0.82 | 0.000 | 0.006 |
| worst | `' cal them' [·] ' someone damage educ'` | `'<|im_end|>'` (0.424) | substitution | 2.09 | 0.000 | 0.044 |
| worst | `' requ ang to concern' [·] ' professional'` | `' professional'` (0.528) | deletion | 2.38 | 0.000 | -0.075 |
| best | `' orig val sales std' [·] ' throws uses js'` | `'<|im_end|>'` (0.225) | substitution | 7.39 | 0.000 | -0.006 |

### `'อังกฤษ'` — id 127887  (single-probe lp -20.44, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' favor times chang single' [·] ' blue index layout'` | `' blue'` (0.913) | deletion | 0.66 | 0.000 | 0.073 |
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.628) | substitution | 1.45 | 0.000 | -0.045 |
| worst | `' random chat incre' [·] ' across edge levels'` | `'<|im_end|>'` (0.498) | substitution | 2.29 | 0.000 | 0.171 |
| best | `' lib didn' [·] ' bed another column'` | `'<tool_call>'` (0.319) | substitution | 6.48 | 0.000 | -0.072 |

### `'ใช่'` — id 126984  (single-probe lp -20.35, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' requ ang to concern' [·] ' professional'` | `' professional'` (0.321) | deletion | 2.72 | 0.000 | -0.012 |
| worst | `' ar ref det ren' [·] ' pos click both'` | `'<|im_end|>'` (0.611) | substitution | 2.14 | 0.000 | -0.045 |
| worst | `' don hot being there' [·] ' loaded night der'` | `' loaded'` (0.451) | deletion | 2.68 | 0.000 | -0.146 |
| best | `' orig val sales std' [·] ' throws uses js'` | `'<|im_end|>'` (0.062) | substitution | 10.14 | 0.000 | -0.019 |

### `'ล่า'` — id 126892  (single-probe lp -20.28, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.877) | substitution | 0.58 | 0.000 | -0.041 |
| worst | `' cal them' [·] ' someone damage educ'` | `'<|im_end|>'` (0.433) | substitution | 2.53 | 0.000 | 0.047 |
| worst | `' lib didn' [·] ' bed another column'` | `'<|im_start|>'` (0.863) | substitution | 1.21 | 0.000 | -0.094 |
| best | `' between item slow pet' [·] ' mit dans photo'` | `'<|im_end|>'` (0.408) | substitution | 4.83 | 0.000 | -0.168 |

### `' สิงหาคม'` — id 142447  (single-probe lp -19.99, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' ar ref det ren' [·] ' pos click both'` | `'<|im_start|>'` (0.885) | substitution | 1.05 | 0.000 | -0.092 |
| worst | `' random chat incre' [·] ' across edge levels'` | `'\n'` (0.287) | substitution | 4.11 | 0.000 | 0.101 |
| worst | `' don hot being there' [·] ' loaded night der'` | `'<|im_end|>'` (0.454) | substitution | 2.37 | 0.000 | -0.136 |
| best | `' nil gives compl money' [·] ' fem specified es'` | `'f'` (0.098) | substitution | 7.46 | 0.000 | -0.059 |

### `'𬸪'` — id 123637  (single-probe lp -19.97, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' cal them' [·] ' someone damage educ'` | `'<|im_end|>'` (0.512) | substitution | 1.92 | 0.000 | 0.019 |
| worst | `' don hot being there' [·] ' loaded night der'` | `'<|im_end|>'` (0.451) | substitution | 1.76 | 0.000 | -0.057 |
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.843) | substitution | 0.89 | 0.000 | -0.002 |
| best | `' manufact ter ide arr' [·] ' greater catch deb'` | `'<|im_start|>'` (0.416) | substitution | 2.68 | 0.000 | -0.032 |

### `'웛'` — id 149983  (single-probe lp -19.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' don hot being there' [·] ' loaded night der'` | `'<|im_end|>'` (0.961) | substitution | 0.31 | 0.000 | -0.081 |
| worst | `' cal them' [·] ' someone damage educ'` | `'<|im_end|>'` (0.707) | substitution | 1.51 | 0.000 | 0.010 |
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.986) | substitution | 0.16 | 0.000 | -0.053 |
| best | `' variable sense except bul' [·] ' wom tem agreed'` | `'<|im_end|>'` (0.575) | substitution | 3.97 | 0.000 | -0.165 |

### `'แก้ปัญหา'` — id 143828  (single-probe lp -19.42, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' don hot being there' [·] ' loaded night der'` | `' loaded'` (0.522) | deletion | 2.20 | 0.000 | -0.154 |
| worst | `' cal them' [·] ' someone damage educ'` | `'<|im_end|>'` (0.453) | substitution | 2.52 | 0.000 | 0.043 |
| worst | `' requ ang to concern' [·] ' professional'` | `'<|im_end|>'` (0.689) | substitution | 1.79 | 0.000 | -0.020 |
| best | `' between item slow pet' [·] ' mit dans photo'` | `'apl'` (0.231) | substitution | 7.90 | 0.000 | -0.137 |

### `'อัพ'` — id 140665  (single-probe lp -19.39, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `' tried'` (0.738) | deletion | 1.38 | 0.000 | -0.028 |
| worst | `' don hot being there' [·] ' loaded night der'` | `'\n'` (0.376) | substitution | 2.43 | 0.000 | -0.155 |
| worst | `' walk directly ent sur' [·] ' major press one'` | `'orge'` (0.425) | substitution | 4.33 | 0.000 | -0.082 |
| best | `' that extends deter ev' [·] ' pract camp pie'` | `'\n'` (0.044) | substitution | 10.19 | 0.000 | -0.197 |

### `'ค้น'` — id 133355  (single-probe lp -19.27, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.663) | substitution | 1.38 | 0.000 | -0.082 |
| worst | `' cal them' [·] ' someone damage educ'` | `' someone'` (0.320) | deletion | 2.74 | 0.000 | 0.084 |
| worst | `' random chat incre' [·] ' across edge levels'` | `'\n\n'` (0.407) | substitution | 2.66 | 0.000 | 0.132 |
| best | `' measure below invol chance' [·] ' enter email weap'` | `'<|im_end|>'` (0.285) | substitution | 6.61 | 0.000 | -0.087 |

### `'กระเป๋า'` — id 136695  (single-probe lp -19.25, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.777) | substitution | 0.97 | 0.000 | -0.020 |
| worst | `' cal them' [·] ' someone damage educ'` | `' someone'` (0.574) | deletion | 2.01 | 0.000 | 0.103 |
| worst | `' requ ang to concern' [·] ' professional'` | `' professional'` (0.418) | deletion | 2.07 | 0.000 | -0.001 |
| best | `' nil gives compl money' [·] ' fem specified es'` | `'<tool_call>'` (0.339) | substitution | 6.39 | 0.000 | -0.075 |

### `' แต่ก็'` — id 139599  (single-probe lp -19.16, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' cal them' [·] ' someone damage educ'` | `' someone'` (0.697) | deletion | 1.54 | 0.000 | 0.021 |
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.788) | substitution | 1.05 | 0.000 | -0.022 |
| worst | `' requ ang to concern' [·] ' professional'` | `'<|im_end|>'` (0.371) | substitution | 2.33 | 0.000 | -0.045 |
| best | `' redu raise mus order' [·] ' rights energ send'` | `'<|im_end|>'` (0.736) | substitution | 2.84 | 0.000 | -0.111 |

### `'พ่อ'` — id 130320  (single-probe lp -18.89, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `' tried'` (0.483) | deletion | 1.28 | 0.000 | -0.041 |
| worst | `' requ ang to concern' [·] ' professional'` | `' professional'` (0.499) | deletion | 2.33 | 0.000 | -0.043 |
| worst | `' random chat incre' [·] ' across edge levels'` | `'<|im_end|>'` (0.358) | substitution | 2.36 | 0.000 | 0.118 |
| best | `' lib didn' [·] ' bed another column'` | `'<tool_call>'` (0.066) | substitution | 9.37 | 0.000 | -0.119 |

### `'𝇗'` — id 151297  (single-probe lp -18.86, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' favor times chang single' [·] ' blue index layout'` | `' mon'` (0.979) | substitution | 0.26 | 0.000 | 0.038 |
| worst | `' that extends deter ev' [·] ' pract camp pie'` | `' mon'` (0.576) | substitution | 1.51 | 0.000 | -0.114 |
| worst | `' don hot being there' [·] ' loaded night der'` | `' mon'` (0.805) | substitution | 1.43 | 0.000 | -0.093 |
| best | `' measure below invol chance' [·] ' enter email weap'` | `' mon'` (0.430) | substitution | 3.79 | 0.000 | 0.098 |

### `'บุรี'` — id 133153  (single-probe lp -18.84, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `' tried'` (0.544) | deletion | 2.47 | 0.000 | -0.006 |
| worst | `' even' [·] ' necessary es imp'` | `'<|im_start|>'` (0.344) | substitution | 4.19 | 0.000 | 0.031 |
| worst | `' requ ang to concern' [·] ' professional'` | `'<|im_start|>'` (0.398) | substitution | 3.10 | 0.000 | -0.047 |
| best | `' nil gives compl money' [·] ' fem specified es'` | `'<tool_call>'` (0.267) | substitution | 6.39 | 0.000 | -0.018 |

### `'สมบู'` — id 138059  (single-probe lp -18.68, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.755) | substitution | 1.01 | 0.000 | -0.053 |
| worst | `' even' [·] ' necessary es imp'` | `'<|im_end|>'` (0.793) | substitution | 1.50 | 0.000 | 0.025 |
| worst | `' requ ang to concern' [·] ' professional'` | `' professional'` (0.733) | deletion | 1.71 | 0.000 | -0.040 |
| best | `' fa let btn comput' [·] ' inj oper import'` | `'<|im_end|>'` (0.204) | substitution | 7.30 | 0.000 | -0.172 |

### `'ราคาถูก'` — id 138415  (single-probe lp -18.65, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `' tried'` (0.824) | deletion | 0.79 | 0.000 | -0.003 |
| worst | `' cal them' [·] ' someone damage educ'` | `' someone'` (0.354) | deletion | 2.82 | 0.000 | 0.048 |
| worst | `' random chat incre' [·] ' across edge levels'` | `'\n\n'` (0.383) | substitution | 3.23 | 0.000 | 0.255 |
| best | `' nil gives compl money' [·] ' fem specified es'` | `'<|im_end|>'` (0.385) | substitution | 5.83 | 0.000 | -0.100 |

### `'เต็ม'` — id 129910  (single-probe lp -18.29, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `'<|im_end|>'` (0.554) | substitution | 1.14 | 0.000 | -0.091 |
| worst | `' cal them' [·] ' someone damage educ'` | `'<|im_end|>'` (0.335) | substitution | 2.62 | 0.000 | 0.063 |
| worst | `' don hot being there' [·] ' loaded night der'` | `'<|im_end|>'` (0.771) | substitution | 1.50 | 0.000 | -0.163 |
| best | `' fa let btn comput' [·] ' inj oper import'` | `'<|im_start|>'` (0.290) | substitution | 6.34 | 0.000 | -0.162 |

### `'แข็'` — id 127932  (single-probe lp -18.20, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' house' [·] ' tried win far'` | `' tried'` (0.571) | deletion | 1.40 | 0.000 | -0.053 |
| worst | `' don hot being there' [·] ' loaded night der'` | `'<|im_start|>'` (0.406) | substitution | 2.36 | 0.000 | -0.155 |
| worst | `' random chat incre' [·] ' across edge levels'` | `'\n\n'` (0.199) | substitution | 4.34 | 0.000 | 0.215 |
| best | `' orig val sales std' [·] ' throws uses js'` | `'\n'` (0.137) | substitution | 7.91 | 0.000 | -0.012 |

### `'บอกว่า'` — id 133219  (single-probe lp -18.18, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' cal them' [·] ' someone damage educ'` | `'\n'` (0.279) | substitution | 2.96 | 0.000 | 0.017 |
| worst | `' random chat incre' [·] ' across edge levels'` | `'\n'` (0.150) | substitution | 6.70 | 0.000 | 0.238 |
| worst | `' house' [·] ' tried win far'` | `' tried'` (0.637) | deletion | 2.88 | 0.000 | 0.012 |
| best | `' between item slow pet' [·] ' mit dans photo'` | `'<|im_end|>'` (0.245) | substitution | 7.79 | 0.000 | -0.110 |
