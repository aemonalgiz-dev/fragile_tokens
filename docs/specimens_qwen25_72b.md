# Confident-substitution specimens

Every example in this file was produced by greedy decoding; `p` is the model's probability for the token it emitted, `H` the entropy in bits at that position, `p(target)` its probability for the correct token, `ret` the cosine between the last-layer slot state and the correct token's unembedding row (identity still pointing at its readout while the readout chose otherwise).

## Model

- **model**: `Qwen/Qwen2.5-72B-Instruct`
- **commit**: `495f39366efef23836d0cfae4fbe635880d2be31`
- **dtype**: `torch.bfloat16`
- **num_layers**: `80`
- **hidden_size**: `8192`
- **vocab_size**: `152064`
- **tie_word_embeddings**: `False`
- **tokenizer_class**: `Qwen2Tokenizer`
- **architecture**: `Qwen2ForCausalLM`
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


### `' الاسلام'` — id 139559

single-probe lp -0.086 · fragility 1.00 · mean lp -3.82 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' elements data instance it' [·] ' when said named'` | 16 | `' ا'` (0.467) | **truncation** | 4.24 | 0.001 | -0.095 | `' اسلام when said named account\nText'` |
| worst | `' stat something javax to' [·] ' mot files reach'` | 64 | `' إ'` (0.418) | **substitution** | 2.84 | 0.001 | -0.071 | `' اسلام mot files reach sure quite media'` |
| worst | `' later allow param dist' [·] ' appro effect title'` | 64 | `' ا'` (0.552) | **truncation** | 1.91 | 0.002 | -0.049 | `' اسلام appro effect title again di state'` |
| best | `' len' [·] ' rais std systems'` | 8 | `' الاسلام'` (0.453) | **correct** | 4.31 | 0.453 | -0.010 | `' الإسلام rais std systems img sun treatm'` |

### `'跸'` — id 121195

single-probe lp -0.005 · fragility 1.00 · mean lp -4.63 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.991) | **deletion** | 0.10 | 0.000 | -0.012 | `' included'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.999) | **deletion** | 0.02 | 0.000 | -0.071 | `' not million provided personal sus conte'` |
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `' result'` (0.859) | **deletion** | 1.02 | 0.000 | -0.054 | `' result heart'` |
| best | `' there lives node theme' [·] ' open ass dir'` | 16 | `'跸'` (0.386) | **correct** | 7.01 | 0.386 | -0.060 | `'跸 open ass dir case figure char byte'` |

### `' şü'` — id 142507

single-probe lp -0.050 · fragility 0.96 · mean lp -2.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' won exit itself word' [·] ' national cur'` | 32 | `' ş'` (0.945) | **truncation** | 0.73 | 0.005 | -0.133 | `' ş national cur\n\nIt appears there might'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' ş'` (0.896) | **truncation** | 1.24 | 0.016 | -0.111 | `' şü not million provided personal sus co'` |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' ş'` (0.318) | **truncation** | 5.09 | 0.023 | -0.126 | `' şü dist conditions vote results cr chan'` |
| best | `' bad reports' [·] ' assert signific cert'` | 8 | `' şü'` (0.658) | **correct** | 3.54 | 0.658 | -0.091 | `' şü assert signific cert vote pol\nText'` |

### `'╋'` — id 149042

single-probe lp -0.088 · fragility 0.96 · mean lp -1.86 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.611) | **deletion** | 2.35 | 0.006 | -0.055 | `' included'` |
| worst | `' increased download' [·] ' unit rep saying'` | 32 | `' unit'` (0.688) | **deletion** | 2.32 | 0.018 | -0.074 | `' unit rep saying fl cons ro shift photo'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.440) | **deletion** | 2.72 | 0.017 | -0.059 | `'(hours implements your http came sleep g'` |
| best | `' idx min activity parts' [·] ' develop'` | 8 | `'╋'` (0.641) | **correct** | 2.17 | 0.641 | -0.048 | `'➕ develop\n\nIt appears there was a'` |

### `'螈'` — id 121762

single-probe lp -0.037 · fragility 0.96 · mean lp -5.11 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.997) | **deletion** | 0.04 | 0.000 | -0.093 | `' not million provided personal sus conte'` |
| worst | `' req did las through' [·] ' except anyone placeholder'` | 32 | `' except'` (0.639) | **deletion** | 1.62 | 0.000 | -0.034 | `'除外 anyone placeholder top sl\n\nIt appears'` |
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.926) | **deletion** | 0.59 | 0.000 | -0.083 | `' policy sure cred\n\nIt looks like there'` |
| best | `' len' [·] ' rais std systems'` | 8 | `'螈'` (0.750) | **correct** | 2.51 | 0.750 | -0.055 | `'螈 rais std systems img sun treatment\n\n'` |

### `'睬'` — id 119279

single-probe lp -0.051 · fragility 0.92 · mean lp -4.51 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.954) | **deletion** | 0.44 | 0.000 | -0.025 | `' hours implements your http came sleep g'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.999) | **deletion** | 0.03 | 0.000 | -0.057 | `' not million provided personal sus conte'` |
| worst | `' bad reports' [·] ' assert signific cert'` | 8 | `' assert'` (0.996) | **deletion** | 0.07 | 0.000 | -0.058 | `' assert significant vote policy\n\nIt appe'` |
| best | `' rais veh tor man' [·] ' enough problems max'` | 32 | `'睬'` (0.835) | **correct** | 1.94 | 0.834 | -0.054 | `'睬 enough problems max offer session name'` |

### `'禘'` — id 123271

single-probe lp -0.003 · fragility 0.92 · mean lp -4.63 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.982) | **deletion** | 0.14 | 0.000 | -0.066 | `' not million provided personal sus conte'` |
| worst | `' increased download' [·] ' unit rep saying'` | 32 | `' unit'` (0.905) | **deletion** | 0.61 | 0.000 | -0.076 | `' unit rep saying fl cons ro shift photo'` |
| worst | `' las goal logger hard' [·] ' beh func eff'` | 8 | `'<|im_start|>'` (0.551) | **substitution** | 1.32 | 0.000 | -0.043 | `' beh func eff'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `'禘'` (0.914) | **correct** | 1.12 | 0.914 | -0.059 | `'禘 art human ver scope effects dans fig'` |

### `' לציין'` — id 140066

single-probe lp -0.093 · fragility 0.92 · mean lp -5.00 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' thought weight shown gives' [·] ' sn seg head'` | 64 | `'<|im_end|>'` (0.684) | **substitution** | 1.31 | 0.000 | -0.148 | `''` |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `'<|im_start|>'` (0.492) | **substitution** | 2.40 | 0.000 | -0.141 | `' dist conditions vote results cr channel'` |
| worst | `' stat something javax to' [·] ' mot files reach'` | 64 | `'<|im_start|>'` (0.505) | **substitution** | 2.29 | 0.000 | -0.145 | `' mot files reach sure quite media popula'` |
| best | `' idx min activity parts' [·] ' develop'` | 8 | `' לציין'` (0.699) | **correct** | 3.48 | 0.699 | -0.083 | `' לציין develop\nIt appears there might be'` |

### `'鹪'` — id 121910

single-probe lp -0.036 · fragility 0.92 · mean lp -4.59 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.998) | **deletion** | 0.02 | 0.000 | -0.032 | `' not million provided personal sus conte'` |
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.979) | **deletion** | 0.17 | 0.000 | -0.043 | `' policy sure cred\n\nIt seems there was'` |
| worst | `' increased download' [·] ' unit rep saying'` | 32 | `' unit'` (0.998) | **deletion** | 0.02 | 0.000 | -0.024 | `' unit rep saying fl cons ro shift photo'` |
| best | `' len' [·] ' rais std systems'` | 8 | `'鹪'` (0.750) | **correct** | 3.22 | 0.750 | -0.000 | `'鹪 rais std systems img sun treatment\n\n'` |

### `' למעלה'` — id 138103

single-probe lp -0.018 · fragility 0.88 · mean lp -2.50 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' dist'` (0.707) | **deletion** | 1.76 | 0.000 | -0.060 | `' dist conditions vote results cr channel'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.569) | **deletion** | 3.67 | 0.004 | -0.086 | `' not million provided personal sus conte'` |
| worst | `' protected lives management pred' [·] ' deg'` | 8 | `' ↑'` (0.561) | **substitution** | 4.19 | 0.006 | -0.061 | `' ↑ deg\n\nIt seems like there was'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `' למעלה'` (0.943) | **correct** | 0.79 | 0.943 | -0.017 | `' למעלה art human ver scope effects dans '` |

### `'蜍'` — id 121201

single-probe lp -0.003 · fragility 0.88 · mean lp -5.23 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.990) | **deletion** | 0.12 | 0.000 | -0.065 | `' policy sure cred\n\nIt seems there was'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.997) | **deletion** | 0.03 | 0.000 | -0.044 | `' not million provided personal sus conte'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.991) | **deletion** | 0.10 | 0.000 | -0.043 | `' hours implements your http came sleep g'` |
| best | `' len' [·] ' rais std systems'` | 8 | `'蜍'` (0.916) | **correct** | 0.67 | 0.916 | -0.033 | `'蜍 rais std systems img sun treatment\n\n'` |

### `'蔸'` — id 121371

single-probe lp -0.006 · fragility 0.88 · mean lp -2.66 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.993) | **deletion** | 0.09 | 0.000 | -0.068 | `' not million provided personal sus conte'` |
| worst | `' increased download' [·] ' unit rep saying'` | 32 | `' unit'` (0.988) | **deletion** | 0.15 | 0.000 | -0.066 | `' unit rep saying fl cons ro shift photo'` |
| worst | `' won exit itself word' [·] ' national cur'` | 32 | `' national'` (0.945) | **deletion** | 0.64 | 0.001 | -0.085 | `' national cur'` |
| best | `' len' [·] ' rais std systems'` | 8 | `'蔸'` (0.910) | **correct** | 0.91 | 0.910 | -0.028 | `'蔸 rais std systems img sun treatment\n\n'` |

### `'悫'` — id 120580

single-probe lp -0.024 · fragility 0.83 · mean lp -3.12 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.999) | **deletion** | 0.01 | 0.000 | -0.034 | `' not million provided personal sus conte'` |
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.997) | **deletion** | 0.05 | 0.000 | -0.051 | `' policy sure cred\n\nIt seems there was'` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.988) | **deletion** | 0.15 | 0.001 | 0.014 | `' included'` |
| best | `' thought weight shown gives' [·] ' sn seg head'` | 64 | `'悫'` (0.965) | **correct** | 0.45 | 0.965 | -0.059 | `'悫 sn seg head'` |

### `'圊'` — id 120644

single-probe lp -0.007 · fragility 0.83 · mean lp -3.46 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.982) | **deletion** | 0.17 | 0.000 | -0.120 | `' not million provided personal sus conte'` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.930) | **deletion** | 0.58 | 0.000 | -0.033 | `' included'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.911) | **deletion** | 0.92 | 0.001 | -0.063 | `' hours implements your http came sleep g'` |
| best | `' las goal logger hard' [·] ' beh func eff'` | 8 | `'圊'` (0.906) | **correct** | 1.14 | 0.906 | -0.065 | `'圊 beh func eff\n\nIt seems like'` |

### `'疬'` — id 120201

single-probe lp -0.016 · fragility 0.79 · mean lp -3.11 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.998) | **deletion** | 0.03 | 0.000 | -0.091 | `' policy sure cred'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.986) | **deletion** | 0.12 | 0.000 | -0.079 | `' not million provided personal sus conte'` |
| worst | `' increased download' [·] ' unit rep saying'` | 32 | `' unit'` (0.973) | **deletion** | 0.26 | 0.000 | -0.081 | `' unit rep saying fl cons ro shift photo'` |
| best | `' tre under available star' [·] ' del aria list'` | 16 | `'疬'` (0.985) | **correct** | 0.23 | 0.985 | -0.059 | `'疬 del aria list node meas ed short'` |

### `' הן'` — id 130032

single-probe lp -0.067 · fragility 0.79 · mean lp -3.60 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' henne'` (0.067) | **substitution** | 10.37 | 0.000 | -0.101 | `' henne dist conditions vote results cr c'` |
| worst | `' thought weight shown gives' [·] ' sn seg head'` | 64 | `' henne'` (0.093) | **substitution** | 10.47 | 0.000 | -0.099 | `' henne sn seg head\n\nIt appears there'` |
| worst | `' won exit itself word' [·] ' national cur'` | 32 | `' henne'` (0.043) | **substitution** | 10.65 | 0.001 | -0.104 | `' henne national cur\nIt appears there was'` |
| best | `' bad reports' [·] ' assert signific cert'` | 8 | `' הן'` (0.984) | **correct** | 0.25 | 0.984 | -0.097 | `' הן assert signific cert vote pol\nIt'` |

### `'邡'` — id 122171

single-probe lp -0.026 · fragility 0.75 · mean lp -1.98 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.879) | **deletion** | 0.74 | 0.001 | -0.102 | `' not million provided personal sus conte'` |
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `' result'` (0.454) | **deletion** | 3.25 | 0.011 | -0.064 | `''` |
| worst | `' won exit itself word' [·] ' national cur'` | 32 | `'<|im_end|>'` (0.474) | **substitution** | 4.70 | 0.012 | -0.094 | `''` |
| best | `' len' [·] ' rais std systems'` | 8 | `'邡'` (0.884) | **correct** | 1.32 | 0.884 | -0.034 | `'邡 rais std systems img sun treatment\n'` |

### `'蟪'` — id 121977

single-probe lp -0.002 · fragility 0.71 · mean lp -2.61 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `'<|im_end|>'` (0.501) | **substitution** | 1.87 | 0.000 | -0.042 | `''` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.958) | **deletion** | 0.37 | 0.000 | 0.012 | `' included'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.961) | **deletion** | 0.31 | 0.001 | -0.067 | `' not million provided personal sus conte'` |
| best | `' protected lives management pred' [·] ' deg'` | 8 | `'蟪'` (0.941) | **correct** | 0.88 | 0.941 | -0.049 | `'蟪 deg\n\nIt appears there may have'` |

### `' trwał'` — id 141124

single-probe lp -0.027 · fragility 0.71 · mean lp -1.89 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' thought weight shown gives' [·] ' sn seg head'` | 64 | `' tr'` (0.720) | **truncation** | 3.15 | 0.006 | -0.135 | `' trwał sn seg head\n\nIt'` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `' tr'` (0.336) | **truncation** | 6.31 | 0.008 | -0.066 | `' trwal included\n\nIt seems there was'` |
| worst | `' stat something javax to' [·] ' mot files reach'` | 64 | `' tr'` (0.763) | **truncation** | 2.90 | 0.008 | -0.121 | `' trwal mot files reach sure quite media'` |
| best | `' len' [·] ' rais std systems'` | 8 | `' trwał'` (0.965) | **correct** | 0.45 | 0.965 | -0.056 | `' trwał rais std systems img sun treatmen'` |

### `'洑'` — id 122525

single-probe lp -0.007 · fragility 0.71 · mean lp -2.67 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.990) | **deletion** | 0.10 | 0.000 | -0.055 | `' not million provided personal sus conte'` |
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `' result'` (0.525) | **deletion** | 1.80 | 0.001 | -0.031 | `''` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.924) | **deletion** | 0.65 | 0.001 | -0.057 | `' hours implements your http came sleep g'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `'洑'` (0.936) | **correct** | 0.78 | 0.936 | -0.070 | `'洑 art human ver scope effects dans fig'` |

### `'浯'` — id 120495

single-probe lp -0.086 · fragility 0.71 · mean lp -3.70 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.993) | **deletion** | 0.07 | 0.000 | -0.066 | `' policy sure cred\n\nIt seems there was'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (1.000) | **deletion** | 0.01 | 0.000 | -0.074 | `' not million provided personal sus conte'` |
| worst | `' won exit itself word' [·] ' national cur'` | 32 | `' national'` (0.988) | **deletion** | 0.16 | 0.000 | -0.081 | `' national cur'` |
| best | `' water long sol investig' [·] ' filter brand natural'` | 16 | `'浯'` (0.908) | **correct** | 1.53 | 0.908 | -0.065 | `'浯 filter brand natural design\n\nIt appear'` |

### `' أنحاء'` — id 143400

single-probe lp -0.006 · fragility 0.71 · mean lp -1.38 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' water long sol investig' [·] ' filter brand natural'` | 16 | `'🔍'` (0.077) | **substitution** | 9.68 | 0.004 | -0.091 | `'🔍 filter brand natural design\n\nIt appear'` |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `'👋'` (0.101) | **substitution** | 10.41 | 0.006 | -0.066 | `'👋dist conditions vote results cr channel'` |
| worst | `' protected lives management pred' [·] ' deg'` | 8 | `'🎉'` (0.111) | **substitution** | 9.45 | 0.012 | -0.035 | `'🎉deg\n\nIt appears there was a'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `' أنحاء'` (0.914) | **correct** | 1.28 | 0.914 | -0.030 | `' أنحاء art human ver scope effects dans '` |

### `'🛏'` — id 148729

single-probe lp -0.053 · fragility 0.71 · mean lp -1.68 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `'<|im_end|>'` (0.803) | **substitution** | 1.68 | 0.000 | -0.094 | `''` |
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `'<|im_end|>'` (0.930) | **substitution** | 0.78 | 0.010 | -0.041 | `''` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `'<|im_end|>'` (0.407) | **substitution** | 2.94 | 0.026 | -0.156 | `''` |
| best | `' las goal logger hard' [·] ' beh func eff'` | 8 | `'🛏'` (0.858) | **correct** | 1.66 | 0.858 | -0.063 | `'🛏 beh func eff\nText: The'` |

### `'剞'` — id 120354

single-probe lp -0.004 · fragility 0.67 · mean lp -4.50 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.997) | **deletion** | 0.04 | 0.000 | -0.005 | `' included'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.991) | **deletion** | 0.10 | 0.000 | -0.019 | `' hours implements your http came sleep g'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.996) | **deletion** | 0.05 | 0.000 | -0.068 | `' not million provided personal sus conte'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `'剞'` (0.993) | **correct** | 0.12 | 0.993 | -0.052 | `'剞 art human ver scope effects dans fig'` |

### `'胬'` — id 120822

single-probe lp -0.016 · fragility 0.67 · mean lp -2.30 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (1.000) | **deletion** | 0.00 | 0.000 | -0.068 | `' not million provided personal sus conte'` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.989) | **deletion** | 0.13 | 0.003 | -0.003 | `' included'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.979) | **deletion** | 0.24 | 0.002 | -0.026 | `' hours implements your http came sleep g'` |
| best | `' idx min activity parts' [·] ' develop'` | 8 | `'胬'` (0.989) | **correct** | 0.13 | 0.989 | -0.041 | `'胬 develop\n\nIt appears there might be'` |

### `'荑'` — id 120030

single-probe lp -0.036 · fragility 0.67 · mean lp -1.89 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.999) | **deletion** | 0.02 | 0.000 | -0.053 | `' not million provided personal sus conte'` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.920) | **deletion** | 0.91 | 0.005 | -0.011 | `' included'` |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' dist'` (0.861) | **deletion** | 1.84 | 0.023 | -0.044 | `' dist conditions vote results cr channel'` |
| best | `' len' [·] ' rais std systems'` | 8 | `'荑'` (0.970) | **correct** | 0.49 | 0.970 | -0.008 | `'荑 rais std systems img sun treatment\n\n'` |

### `'鸨'` — id 120185

single-probe lp -0.023 · fragility 0.62 · mean lp -2.57 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (1.000) | **deletion** | 0.01 | 0.000 | -0.063 | `' not million provided personal sus conte'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.992) | **deletion** | 0.09 | 0.001 | -0.036 | `' hours implements your http came sleep g'` |
| worst | `' increased download' [·] ' unit rep saying'` | 32 | `' unit'` (0.957) | **deletion** | 0.44 | 0.002 | -0.071 | `' unit rep saying fl cons ro shift photo'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `'鸨'` (0.998) | **correct** | 0.03 | 0.998 | -0.045 | `'鸨 art human ver scope effects dans fig'` |

### `'郏'` — id 119850

single-probe lp -0.000 · fragility 0.62 · mean lp -1.51 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.993) | **deletion** | 0.08 | 0.001 | -0.074 | `' not million provided personal sus conte'` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.963) | **deletion** | 0.38 | 0.014 | -0.021 | `' included'` |
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `' result'` (0.850) | **deletion** | 1.55 | 0.033 | -0.026 | `' result heart\n\nIt seems like there was'` |
| best | `' len' [·] ' rais std systems'` | 8 | `'郏'` (0.978) | **correct** | 0.30 | 0.978 | -0.023 | `'郏 rais std systems img sun treatment\n\n'` |

### `' העסק'` — id 131713

single-probe lp -0.066 · fragility 0.62 · mean lp -2.34 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `'<|im_end|>'` (0.800) | **substitution** | 2.58 | 0.000 | -0.122 | `''` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `'<|im_end|>'` (0.544) | **substitution** | 5.27 | 0.001 | -0.012 | `''` |
| worst | `' stat something javax to' [·] ' mot files reach'` | 64 | `'<|im_end|>'` (0.529) | **substitution** | 5.93 | 0.001 | -0.117 | `''` |
| best | `' len' [·] ' rais std systems'` | 8 | `' העסק'` (0.876) | **correct** | 1.94 | 0.876 | -0.031 | `' העסק rais std systems img sun treatment'` |

### `'噢'` — id 118401

single-probe lp -0.038 · fragility 0.62 · mean lp -1.98 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' oh'` (0.755) | **substitution** | 0.81 | 0.000 | -0.046 | `' oh not million provided personal sus co'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.940) | **deletion** | 0.41 | 0.003 | -0.022 | `' hours implements your http came sleep g'` |
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `' oh'` (0.903) | **substitution** | 0.53 | 0.010 | -0.031 | `" oh result heart\n\nNote: I've"` |
| best | `' protected lives management pred' [·] ' deg'` | 8 | `'噢'` (0.997) | **correct** | 0.04 | 0.997 | -0.011 | `'噢 deg\n\nIt seems like there might'` |

### `' אלינו'` — id 141006

single-probe lp -0.009 · fragility 0.62 · mean lp -2.52 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `'-END'` (0.362) | **substitution** | 8.21 | 0.000 | -0.176 | `'-END-olums dist conditions vote results'` |
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `' b'` (0.791) | **substitution** | 2.97 | 0.000 | -0.189 | `' bize result heart\n\nIt appears there'` |
| worst | `' protected lives management pred' [·] ' deg'` | 8 | `' b'` (0.283) | **substitution** | 6.15 | 0.001 | -0.040 | `' bize yakın\n\nIt appears there was'` |
| best | `' rais veh tor man' [·] ' enough problems max'` | 32 | `' אלינו'` (0.980) | **correct** | 0.28 | 0.980 | -0.043 | `' אלינו enough problems max offer session'` |

### `'滹'` — id 121503

single-probe lp -0.008 · fragility 0.58 · mean lp -2.09 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.999) | **deletion** | 0.02 | 0.000 | -0.051 | `' not million provided personal sus conte'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.934) | **deletion** | 0.57 | 0.002 | -0.023 | `' hours implements your http came sleep g'` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.977) | **deletion** | 0.24 | 0.003 | 0.020 | `' included'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `'滹'` (0.993) | **correct** | 0.13 | 0.993 | -0.046 | `'滹 art human ver scope effects dans fig'` |

### `'魃'` — id 121466

single-probe lp -0.047 · fragility 0.58 · mean lp -2.34 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.995) | **deletion** | 0.06 | 0.000 | -0.068 | `' not million provided personal sus conte'` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.951) | **deletion** | 0.42 | 0.000 | 0.001 | `' included'` |
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.972) | **deletion** | 0.27 | 0.001 | -0.064 | `' policy sure cred\n\nIt seems there was'` |
| best | `' las goal logger hard' [·] ' beh func eff'` | 8 | `'魃'` (0.981) | **correct** | 0.34 | 0.981 | -0.031 | `'魃 beh func eff\nIt seems like'` |

### `'ው'` — id 148996

single-probe lp -0.091 · fragility 0.58 · mean lp -0.76 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.580) | **deletion** | 4.12 | 0.054 | -0.025 | `' hours implements your http came sleep g'` |
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `'<|im_end|>'` (0.251) | **substitution** | 6.05 | 0.152 | -0.042 | `''` |
| worst | `' req did las through' [·] ' except anyone placeholder'` | 32 | `'除外'` (0.623) | **substitution** | 3.35 | 0.123 | -0.029 | `'除外 anyone placeholder top sl\n\nIt appears'` |
| best | `' there lives node theme' [·] ' open ass dir'` | 16 | `'ው'` (0.847) | **correct** | 2.31 | 0.847 | -0.117 | `'ው open ass dir case figure char byte'` |

### `'扃'` — id 120246

single-probe lp -0.004 · fragility 0.58 · mean lp -1.99 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `' result'` (0.924) | **deletion** | 0.84 | 0.001 | 0.005 | `' result heart\n\nIt seems like there was'` |
| worst | `' water long sol investig' [·] ' filter brand natural'` | 16 | `'shutdown'` (0.226) | **substitution** | 6.16 | 0.002 | -0.030 | `'shutdown filter brand natural design\nIt '` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.949) | **deletion** | 0.60 | 0.005 | -0.076 | `' not million provided personal sus conte'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `'扃'` (0.957) | **correct** | 0.41 | 0.957 | -0.056 | `'扃 art human ver scope effects dans fig'` |

### `'�'` — id 56842

single-probe lp -0.038 · fragility 0.58 · mean lp -1.87 · canonical False · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.991) | **deletion** | 0.09 | 0.002 | -0.032 | `' policy sure cred\n\nIt seems there was'` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.983) | **deletion** | 0.16 | 0.005 | -0.004 | `' included\n\nIt appears there was a minor'` |
| worst | `' won exit itself word' [·] ' national cur'` | 32 | `' national'` (0.991) | **deletion** | 0.09 | 0.005 | -0.019 | `' national cur\n\nIt seems there might be'` |
| best | `' las goal logger hard' [·] ' beh func eff'` | 8 | `'�'` (1.000) | **correct** | 0.01 | 1.000 | -0.021 | `'� beh func eff\nIt appears there'` |

### `' العسكري'` — id 140421

single-probe lp -0.002 · fragility 0.58 · mean lp -1.15 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' elements data instance it' [·] ' when said named'` | 16 | `' ع'` (0.876) | **substitution** | 1.37 | 0.030 | -0.090 | `' عسكري when said named account\n'` |
| worst | `' stat something javax to' [·] ' mot files reach'` | 64 | `' ع'` (0.842) | **substitution** | 1.75 | 0.037 | -0.057 | `' عسكري mot files reach sure quite'` |
| worst | `' later allow param dist' [·] ' appro effect title'` | 64 | `' ع'` (0.854) | **substitution** | 1.14 | 0.062 | -0.042 | `' عسكري appro effect title again di'` |
| best | `' las goal logger hard' [·] ' beh func eff'` | 8 | `' العسكري'` (0.996) | **correct** | 0.05 | 0.996 | -0.054 | `' العسكري beh func eff\nText: '` |

### `' الفلسطيني'` — id 134037

single-probe lp -0.003 · fragility 0.54 · mean lp -0.96 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' printf ne throw parents' [·] ' art human ver'` | 16 | `' ف'` (0.919) | **substitution** | 0.67 | 0.052 | -0.079 | `' فلسطيني art human ver scope'` |
| worst | `' stat something javax to' [·] ' mot files reach'` | 64 | `' ف'` (0.871) | **substitution** | 1.54 | 0.034 | -0.081 | `' فلسطيني mot files reach sure'` |
| worst | `' later allow param dist' [·] ' appro effect title'` | 64 | `' ف'` (0.759) | **substitution** | 2.05 | 0.071 | -0.051 | `' فلسطيني appro effect title again'` |
| best | `' rais veh tor man' [·] ' enough problems max'` | 32 | `' الفلسطيني'` (0.996) | **correct** | 0.07 | 0.996 | -0.023 | `' الفلسطيني enough problems max offer ses'` |

### `'協助'` — id 111172

single-probe lp -0.011 · fragility 0.54 · mean lp -1.99 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' bad reports' [·] ' assert signific cert'` | 8 | `' assert'` (0.588) | **deletion** | 1.54 | 0.000 | -0.040 | `' assist assert significant vote policy\n\n'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' assistance'` (0.610) | **substitution** | 1.48 | 0.004 | -0.048 | `' assistance not million provided persona'` |
| worst | `' there lives node theme' [·] ' open ass dir'` | 16 | `' assisting'` (0.598) | **substitution** | 1.80 | 0.008 | -0.040 | `' assisting open ass dir case figure char'` |
| best | `' len' [·] ' rais std systems'` | 8 | `'協助'` (0.994) | **correct** | 0.06 | 0.994 | -0.013 | `'協助 rais std systems img sun treatment\n'` |

### `'旆'` — id 120479

single-probe lp -0.055 · fragility 0.54 · mean lp -2.21 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' dist'` (0.975) | **deletion** | 0.38 | 0.000 | -0.064 | `' dist conditions vote results cr channel'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.985) | **deletion** | 0.22 | 0.000 | -0.079 | `' not million provided personal sus conte'` |
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.991) | **deletion** | 0.10 | 0.000 | -0.051 | `' policy sure cred\n\nIt seems there was'` |
| best | `' rais veh tor man' [·] ' enough problems max'` | 32 | `'旆'` (0.985) | **correct** | 0.23 | 0.985 | -0.059 | `'旆 enough problems max offer session name'` |

### `' الأجنبية'` — id 140225

single-probe lp -0.007 · fragility 0.54 · mean lp -3.76 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `' extr'` (0.651) | **substitution** | 1.64 | 0.000 | -0.105 | `' extranjera result heart\n\nNote:'` |
| worst | `' water long sol investig' [·] ' filter brand natural'` | 16 | `' estr'` (0.496) | **substitution** | 2.37 | 0.000 | -0.104 | `' estrangeira filter brand natural design'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' foreigners'` (0.500) | **substitution** | 2.51 | 0.000 | -0.092 | `' foreigners not million provided persona'` |
| best | `' rais veh tor man' [·] ' enough problems max'` | 32 | `' الأجنبية'` (0.997) | **correct** | 0.05 | 0.996 | -0.057 | `' الأجنبية enough problems max offer sess'` |

### `'霭'` — id 122014

single-probe lp -0.021 · fragility 0.54 · mean lp -1.15 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.983) | **deletion** | 0.17 | 0.002 | -0.006 | `' hours implements your http came sleep g'` |
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.904) | **deletion** | 0.57 | 0.017 | -0.050 | `' policy sure cred\n\nIt seems there was'` |
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `' result'` (0.937) | **deletion** | 0.62 | 0.025 | 0.002 | `' result heart\n\nIt seems there was an'` |
| best | `' there lives node theme' [·] ' open ass dir'` | 16 | `'霭'` (0.993) | **correct** | 0.09 | 0.993 | -0.039 | `'霭 open ass dir case figure char byte'` |

### `'踽'` — id 121758

single-probe lp -0.011 · fragility 0.50 · mean lp -1.44 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.990) | **deletion** | 0.15 | 0.000 | -0.074 | `' not million provided personal sus conte'` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.947) | **deletion** | 0.62 | 0.006 | -0.008 | `' included'` |
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.903) | **deletion** | 1.18 | 0.010 | -0.060 | `' policy sure cred\n\nIt seems there was'` |
| best | `' stat something javax to' [·] ' mot files reach'` | 64 | `'踽'` (0.983) | **correct** | 0.30 | 0.983 | -0.059 | `'踽 mot files reach sure quite media popul'` |

### `' toplantı'` — id 135369

single-probe lp -0.013 · fragility 0.50 · mean lp -0.65 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' toplant'` (0.827) | **truncation** | 1.12 | 0.144 | -0.159 | `' toplant dist conditions vote results cr'` |
| worst | `' won exit itself word' [·] ' national cur'` | 32 | `' toplant'` (0.759) | **truncation** | 1.66 | 0.169 | -0.133 | `' toplantı national cur\n\nIt seems there'` |
| worst | `' protected lives management pred' [·] ' deg'` | 8 | `' toplant'` (0.799) | **truncation** | 1.04 | 0.178 | -0.064 | `' toplantısı deg\n\nIt appears there was'` |
| best | `' len' [·] ' rais std systems'` | 8 | `' toplantı'` (0.977) | **correct** | 0.28 | 0.977 | -0.058 | `' toplantı rais std systems img sun treat'` |

### `'主流'` — id 105276

single-probe lp -0.016 · fragility 0.50 · mean lp -1.21 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' mainstream'` (0.992) | **substitution** | 0.07 | 0.008 | -0.034 | `' mainstream policy sure cred\n\nIt seems t'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' mainstream'` (0.987) | **substitution** | 0.10 | 0.012 | -0.036 | `' mainstream not million provided persona'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' mainstream'` (0.958) | **substitution** | 0.25 | 0.042 | -0.008 | `' mainstream hours implements your http c'` |
| best | `' protected lives management pred' [·] ' deg'` | 8 | `'主流'` (0.997) | **correct** | 0.03 | 0.997 | 0.007 | `'主流 deg\n\nIt appears there may be'` |

### `' الخم'` — id 133448

single-probe lp -0.037 · fragility 0.50 · mean lp -1.00 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' stre me filename known' [·] ' dans box mobile'` | 16 | `' חמ'` (0.133) | **substitution** | 6.74 | 0.063 | -0.021 | `'خام dans box mobile well charge protect '` |
| worst | `' rais veh tor man' [·] ' enough problems max'` | 32 | `'خم'` (0.148) | **substitution** | 9.57 | 0.079 | -0.071 | `'خم enough problems max offer session nam'` |
| worst | `' elements data instance it' [·] ' when said named'` | 16 | `'خم'` (0.252) | **substitution** | 4.56 | 0.072 | -0.070 | `'خم when said named account\nIt seems'` |
| best | `' len' [·] ' rais std systems'` | 8 | `' الخم'` (0.967) | **correct** | 0.38 | 0.967 | -0.009 | `' الخم rais std systems img sun treatment'` |

### `'肄'` — id 121264

single-probe lp -0.070 · fragility 0.50 · mean lp -1.13 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `' result'` (0.989) | **deletion** | 0.16 | 0.004 | -0.011 | `' result heart\n\nIt appears that the last'` |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' dist'` (0.971) | **deletion** | 0.33 | 0.007 | -0.035 | `' dist conditions vote results cr channel'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.950) | **deletion** | 0.35 | 0.037 | -0.061 | `' not million provided personal sus conte'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `'肄'` (0.999) | **correct** | 0.02 | 0.999 | -0.057 | `'肄 art human ver scope effects dans fig'` |

### `' السياسي'` — id 133527

single-probe lp -0.003 · fragility 0.50 · mean lp -2.23 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' elements data instance it' [·] ' when said named'` | 16 | `' سياسي'` (0.955) | **substitution** | 0.45 | 0.002 | -0.061 | `' سياسي when said named account\nIt seems'` |
| worst | `' printf ne throw parents' [·] ' art human ver'` | 16 | `' سياسي'` (0.994) | **substitution** | 0.07 | 0.003 | -0.099 | `' سياسي art human ver scope effects dans '` |
| worst | `' later allow param dist' [·] ' appro effect title'` | 64 | `' سياسي'` (0.991) | **substitution** | 0.10 | 0.005 | -0.036 | `' سياسي appro effect title again di state'` |
| best | `' las goal logger hard' [·] ' beh func eff'` | 8 | `' السياسي'` (0.993) | **correct** | 0.09 | 0.993 | -0.060 | `' السياسي beh func eff\nText: '` |

### `'おそらく'` — id 143993

single-probe lp -0.056 · fragility 0.50 · mean lp -1.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' presumably'` (0.709) | **substitution** | 2.22 | 0.004 | -0.092 | `' presumably not million provided persona'` |
| worst | `' won exit itself word' [·] ' national cur'` | 32 | `' perhaps'` (0.577) | **substitution** | 3.31 | 0.009 | -0.100 | `' perhaps national cur\nIt seems there was'` |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' dist'` (0.099) | **deletion** | 11.20 | 0.012 | -0.108 | `' dist conditions vote results cr channel'` |
| best | `' len' [·] ' rais std systems'` | 8 | `'おそらく'` (0.998) | **correct** | 0.04 | 0.997 | -0.034 | `'おそらく rais std systems img sun treatment\n'` |

### `'卮'` — id 119570

single-probe lp -0.006 · fragility 0.50 · mean lp -2.74 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' increased download' [·] ' unit rep saying'` | 32 | `' unit'` (0.999) | **deletion** | 0.02 | 0.000 | -0.061 | `' unit rep saying fl cons ro shift photo'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.988) | **deletion** | 0.12 | 0.000 | -0.064 | `' not million provided personal sus conte'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.987) | **deletion** | 0.14 | 0.000 | -0.025 | `' hours implements your http came sleep g'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `'卮'` (0.998) | **correct** | 0.04 | 0.998 | -0.054 | `'卮 art human ver scope effects dans fig'` |

### `'蜱'` — id 121419

single-probe lp -0.010 · fragility 0.50 · mean lp -1.76 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' increased download' [·] ' unit rep saying'` | 32 | `' unit'` (0.999) | **deletion** | 0.02 | 0.000 | -0.069 | `' unit rep saying fl cons ro shift photo'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.998) | **deletion** | 0.03 | 0.001 | -0.070 | `' not million provided personal sus conte'` |
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.980) | **deletion** | 0.19 | 0.005 | -0.049 | `' policy sure cred\n\nIt appears there was'` |
| best | `' las goal logger hard' [·] ' beh func eff'` | 8 | `'蜱'` (0.989) | **correct** | 0.21 | 0.989 | -0.031 | `'蜱 beh func eff\nIt seems there'` |

### `'獄'` — id 119211

single-probe lp -0.080 · fragility 0.46 · mean lp -1.09 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.989) | **deletion** | 0.13 | 0.000 | -0.082 | `' not million provided personal sus conte'` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.894) | **deletion** | 1.04 | 0.019 | -0.021 | `' included'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.914) | **deletion** | 0.62 | 0.046 | -0.033 | `' hours implements your http came sleep g'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `'獄'` (0.999) | **correct** | 0.01 | 0.999 | -0.065 | `'獄 art human ver scope effects dans fig'` |

### `'╮'` — id 146219

single-probe lp -0.029 · fragility 0.46 · mean lp -1.09 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.943) | **deletion** | 0.80 | 0.004 | -0.074 | `' not million provided personal sus conte'` |
| worst | `' won exit itself word' [·] ' national cur'` | 32 | `' national'` (0.958) | **deletion** | 0.50 | 0.006 | -0.052 | `' national cur\n\nIt seems there might be'` |
| worst | `' increased download' [·] ' unit rep saying'` | 32 | `' unit'` (0.626) | **deletion** | 2.16 | 0.058 | -0.069 | `' unit rep saying fl cons ro shift photo'` |
| best | `' las goal logger hard' [·] ' beh func eff'` | 8 | `'╮'` (0.993) | **correct** | 0.11 | 0.993 | -0.021 | `'╮ beh func eff\nText: '` |

### `'欠缺'` — id 116936

single-probe lp -0.000 · fragility 0.46 · mean lp -1.91 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.992) | **deletion** | 0.08 | 0.000 | -0.019 | `' policy sure cred\n\nIt seems there was'` |
| worst | `' increased download' [·] ' unit rep saying'` | 32 | `' unit'` (0.997) | **deletion** | 0.04 | 0.001 | -0.022 | `' unit rep saying fl cons ro shift photo'` |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' dist'` (0.982) | **deletion** | 0.21 | 0.001 | -0.036 | `' dist conditions vote results cr channel'` |
| best | `' las goal logger hard' [·] ' beh func eff'` | 8 | `'欠缺'` (0.999) | **correct** | 0.02 | 0.999 | -0.009 | `'欠缺 beh func eff\nIt seems like'` |

### `'怦'` — id 119960

single-probe lp -0.005 · fragility 0.46 · mean lp -1.16 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.999) | **deletion** | 0.02 | 0.000 | -0.058 | `' not million provided personal sus conte'` |
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.967) | **deletion** | 0.33 | 0.008 | -0.061 | `' policy sure cred\n\nIt looks like there'` |
| worst | `' bad reports' [·] ' assert signific cert'` | 8 | `' assert'` (0.891) | **deletion** | 0.72 | 0.083 | -0.049 | `' assert significant certainty vote polit'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `'怦'` (1.000) | **correct** | 0.01 | 0.999 | -0.047 | `'怦 art human ver scope effects dans fig'` |

### `'увеличен'` — id 139720

single-probe lp -0.028 · fragility 0.46 · mean lp -2.75 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' bad reports' [·] ' assert signific cert'` | 8 | `' exaggerated'` (0.998) | **substitution** | 0.02 | 0.000 | -0.112 | `' exaggerated assert significance vote po'` |
| worst | `' protected lives management pred' [·] ' deg'` | 8 | `'ex'` (0.728) | **substitution** | 0.90 | 0.000 | -0.075 | `'exaggerated deg\n\nIt appears there'` |
| worst | `' req did las through' [·] ' except anyone placeholder'` | 32 | `' exaggerated'` (0.719) | **substitution** | 1.16 | 0.000 | -0.123 | `' exaggerated except anyone placeholder t'` |
| best | `' increased download' [·] ' unit rep saying'` | 32 | `'увеличен'` (1.000) | **correct** | 0.00 | 1.000 | -0.062 | `'увеличен unit rep saying fl cons ro shif'` |

### `' בסדר'` — id 136099

single-probe lp -0.007 · fragility 0.46 · mean lp -1.16 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `'🔍'` (0.118) | **substitution** | 9.92 | 0.012 | -0.135 | `'🔍not million provided personal sus conte'` |
| worst | `' protected lives management pred' [·] ' deg'` | 8 | `'👌'` (0.185) | **substitution** | 8.34 | 0.022 | -0.048 | `'👌👌👌👌👌👌👌👌'` |
| worst | `' bad reports' [·] ' assert signific cert'` | 8 | `'👌'` (0.053) | **substitution** | 11.16 | 0.022 | -0.111 | `'👌assert signific cert vote pol\nIt'` |
| best | `' rais veh tor man' [·] ' enough problems max'` | 32 | `' בסדר'` (0.960) | **correct** | 0.61 | 0.960 | -0.035 | `' בסדר enough problems max offer session '` |

### `'玟'` — id 122279

single-probe lp -0.080 · fragility 0.46 · mean lp -1.57 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.995) | **deletion** | 0.06 | 0.000 | -0.075 | `' not million provided personal sus conte'` |
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.957) | **deletion** | 0.35 | 0.006 | -0.081 | `' policy sure cred\n\nIt seems there was'` |
| worst | `' won exit itself word' [·] ' national cur'` | 32 | `' national'` (0.976) | **deletion** | 0.26 | 0.006 | -0.069 | `' national cur\n\nIt seems there might be'` |
| best | `' len' [·] ' rais std systems'` | 8 | `'玟'` (1.000) | **correct** | 0.01 | 1.000 | -0.032 | `'玟 rais std systems img sun treatment\n\n'` |

### `'されました'` — id 135877

single-probe lp -0.018 · fragility 0.46 · mean lp -1.07 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' won exit itself word' [·] ' national cur'` | 32 | `'<|im_end|>'` (0.080) | **substitution** | 9.77 | 0.012 | -0.114 | `''` |
| worst | `' later allow param dist' [·] ' appro effect title'` | 64 | `' wurden'` (0.492) | **substitution** | 5.87 | 0.015 | -0.042 | `' wurden appro effect title again di stat'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.367) | **deletion** | 7.77 | 0.030 | -0.101 | `' not million provided personal sus conte'` |
| best | `' idx min activity parts' [·] ' develop'` | 8 | `'されました'` (0.984) | **correct** | 0.22 | 0.984 | -0.081 | `'されました develop\nIt seems like there might'` |

### `'掂'` — id 119401

single-probe lp -0.010 · fragility 0.46 · mean lp -1.77 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' increased download' [·] ' unit rep saying'` | 32 | `' unit'` (0.998) | **deletion** | 0.02 | 0.000 | -0.062 | `' unit rep saying fl cons ro shift photo'` |
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.954) | **deletion** | 0.30 | 0.001 | -0.051 | `' policy sure cred\n\nIt seems there was'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.991) | **deletion** | 0.10 | 0.003 | -0.043 | `' not million provided personal sus conte'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `'掂'` (0.998) | **correct** | 0.03 | 0.998 | -0.039 | `'掂 art human ver scope effects dans fig'` |

### `' follando'` — id 71579

single-probe lp -0.034 · fragility 0.46 · mean lp -1.53 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' fol'` (0.360) | **truncation** | 4.39 | 0.001 | -0.128 | `' folloing dist conditions vote results c'` |
| worst | `' later allow param dist' [·] ' appro effect title'` | 64 | `' fol'` (0.982) | **truncation** | 0.23 | 0.001 | -0.073 | `' folloing appro effect title again di'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' foll'` (0.497) | **truncation** | 2.05 | 0.004 | -0.113 | `' folling not million provided personal s'` |
| best | `' idx min activity parts' [·] ' develop'` | 8 | `' follando'` (0.999) | **correct** | 0.02 | 0.999 | -0.101 | `' follando develop\nText: The cat in'` |

### `' sobą'` — id 138495

single-probe lp -0.003 · fragility 0.42 · mean lp -0.90 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' rais veh tor man' [·] ' enough problems max'` | 32 | `' Selbst'` (0.034) | **substitution** | 10.33 | 0.004 | -0.043 | `' Selbst enough problems max offer sessio'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' sam'` (0.397) | **substitution** | 4.64 | 0.022 | -0.105 | `' sam not million provided personal sus c'` |
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `' sama'` (0.291) | **substitution** | 6.43 | 0.065 | -0.100 | `' sama result heart\n\nIt appears there may'` |
| best | `' there lives node theme' [·] ' open ass dir'` | 16 | `' sobą'` (0.999) | **correct** | 0.01 | 0.999 | -0.056 | `' sobą open ass dir case figure char byte'` |

### `' oluşan'` — id 139527

single-probe lp -0.073 · fragility 0.42 · mean lp -0.60 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' oluştur'` (0.207) | **substitution** | 7.35 | 0.183 | -0.103 | `' oluştur not million provided personal s'` |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' oluşan'` (0.256) | **correct** | 8.61 | 0.256 | -0.108 | `' oluşturulan dist conditions vote result'` |
| worst | `' protected lives management pred' [·] ' deg'` | 8 | `' oluş'` (0.288) | **truncation** | 5.46 | 0.288 | -0.010 | `' oluşan deg\n\nIt appears there might'` |
| best | `' stre me filename known' [·] ' dans box mobile'` | 16 | `' oluşan'` (0.865) | **correct** | 1.66 | 0.865 | 0.019 | `' oluşan dans box mobile well charge prot'` |

### `' הנוכחי'` — id 141090

single-probe lp -0.060 · fragility 0.42 · mean lp -0.47 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' costs nor full able' [·] ' included'` | 32 | `' הנוכחי'` (0.133) | **correct** | 9.95 | 0.133 | -0.050 | `' הנוכחי included\nText: The cat in'` |
| worst | `' elements data instance it' [·] ' when said named'` | 16 | `' הנוכחי'` (0.322) | **correct** | 7.50 | 0.322 | -0.126 | `'.GetCurrentMethod when said named accoun'` |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' הנוכחי'` (0.362) | **correct** | 6.52 | 0.362 | -0.136 | `' הנוכחי dist conditions vote results cr '` |
| best | `' idx min activity parts' [·] ' develop'` | 8 | `' הנוכחי'` (0.971) | **correct** | 0.51 | 0.971 | -0.074 | `' הנוכחי develop\nIt seems like there migh'` |

### `'见效'` — id 116147

single-probe lp -0.084 · fragility 0.42 · mean lp -2.02 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' stre me filename known' [·] ' dans box mobile'` | 16 | `' dans'` (0.999) | **deletion** | 0.02 | 0.000 | 0.010 | `' dans box mobile well charge protect awa'` |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' dist'` (0.520) | **deletion** | 1.27 | 0.000 | 0.001 | `'成效 dist conditions vote results cr chann'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.998) | **deletion** | 0.03 | 0.001 | -0.040 | `' not million provided personal sus conte'` |
| best | `' las goal logger hard' [·] ' beh func eff'` | 8 | `'见效'` (0.999) | **correct** | 0.01 | 0.999 | 0.007 | `'见效 beh func eff\nIt seems like'` |

### `'笕'` — id 120421

single-probe lp -0.021 · fragility 0.42 · mean lp -2.37 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.998) | **deletion** | 0.03 | 0.000 | -0.058 | `' not million provided personal sus conte'` |
| worst | `' won exit itself word' [·] ' national cur'` | 32 | `' national'` (0.943) | **deletion** | 0.57 | 0.001 | -0.085 | `' national cur'` |
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `' result'` (0.808) | **deletion** | 1.35 | 0.001 | -0.043 | `' result heart\n\nIt appears there may have'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `'笕'` (0.993) | **correct** | 0.11 | 0.993 | -0.033 | `'笕 art human ver scope effects dans fig'` |

### `'雊'` — id 123230

single-probe lp -0.003 · fragility 0.42 · mean lp -1.62 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.998) | **deletion** | 0.02 | 0.000 | -0.088 | `' not million provided personal sus conte'` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.949) | **deletion** | 0.46 | 0.004 | -0.004 | `' included'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.910) | **deletion** | 0.81 | 0.021 | -0.065 | `' hours implements your http came sleep g'` |
| best | `' las goal logger hard' [·] ' beh func eff'` | 8 | `'雊'` (0.989) | **correct** | 0.18 | 0.989 | -0.078 | `'雊 beh func eff\nIt seems like'` |

### `' oldukça'` — id 132821

single-probe lp -0.018 · fragility 0.42 · mean lp -0.63 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' thought weight shown gives' [·] ' sn seg head'` | 64 | `' oldukça'` (0.139) | **correct** | 9.41 | 0.139 | -0.089 | `'👋👋👋👋👋👋👋👋'` |
| worst | `' later allow param dist' [·] ' appro effect title'` | 64 | `' oldukça'` (0.179) | **correct** | 8.76 | 0.179 | -0.053 | `''` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `'👋'` (0.245) | **substitution** | 7.33 | 0.191 | -0.079 | `'👋quite hours implements your http came s'` |
| best | `' len' [·] ' rais std systems'` | 8 | `' oldukça'` (0.962) | **correct** | 0.61 | 0.962 | -0.031 | `' oldukça rais std systems img sun treatm'` |

### `' כדאי'` — id 132804

single-probe lp -0.008 · fragility 0.42 · mean lp -1.30 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `'<|im_end|>'` (0.201) | **substitution** | 7.90 | 0.003 | -0.141 | `''` |
| worst | `' there lives node theme' [·] ' open ass dir'` | 16 | `' kell'` (0.026) | **substitution** | 11.62 | 0.009 | -0.132 | `' kell open ass dir case figure char byte'` |
| worst | `' thought weight shown gives' [·] ' sn seg head'` | 64 | `'<|im_end|>'` (0.185) | **substitution** | 9.21 | 0.021 | -0.142 | `''` |
| best | `' idx min activity parts' [·] ' develop'` | 8 | `' כדאי'` (0.981) | **correct** | 0.37 | 0.981 | -0.059 | `' כדאי develop\nIt seems like there might'` |

### `'麼'` — id 100283

single-probe lp -0.039 · fragility 0.42 · mean lp -1.97 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.995) | **deletion** | 0.06 | 0.000 | -0.080 | `' not million provided personal sus conte'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.919) | **deletion** | 0.86 | 0.000 | -0.063 | `' hours implements your http came sleep g'` |
| worst | `' stat something javax to' [·] ' mot files reach'` | 64 | `'<|im_end|>'` (0.413) | **substitution** | 2.99 | 0.005 | -0.070 | `''` |
| best | `' len' [·] ' rais std systems'` | 8 | `'麼'` (0.997) | **correct** | 0.06 | 0.997 | -0.023 | `'麼 rais std systems img sun treatment\n\n'` |

### `'吭'` — id 119315

single-probe lp -0.006 · fragility 0.42 · mean lp -1.02 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.996) | **deletion** | 0.06 | 0.002 | -0.046 | `' not million provided personal sus conte'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.938) | **deletion** | 0.47 | 0.032 | -0.019 | `' hours implements your http came sleep g'` |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' dist'` (0.533) | **deletion** | 5.73 | 0.093 | -0.030 | `' dist conditions vote results cr channel'` |
| best | `' len' [·] ' rais std systems'` | 8 | `'吭'` (1.000) | **correct** | 0.00 | 1.000 | -0.015 | `'吭 rais std systems img sun treatment\n\n'` |

### `'嬿'` — id 123733

single-probe lp -0.051 · fragility 0.42 · mean lp -0.69 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.752) | **deletion** | 1.83 | 0.007 | -0.094 | `' not million provided personal sus conte'` |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `'<|im_end|>'` (0.447) | **substitution** | 6.47 | 0.113 | -0.072 | `''` |
| worst | `' bad reports' [·] ' assert signific cert'` | 8 | `' assert'` (0.532) | **deletion** | 2.60 | 0.285 | -0.071 | `' assert significant certainty vote polit'` |
| best | `' there lives node theme' [·] ' open ass dir'` | 16 | `'嬿'` (0.997) | **correct** | 0.06 | 0.997 | -0.086 | `'嬿 open ass dir case figure char byte'` |

### `'溉'` — id 103595

single-probe lp -0.049 · fragility 0.42 · mean lp -1.11 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.995) | **deletion** | 0.06 | 0.002 | -0.062 | `' not million provided personal sus conte'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' hours'` (0.967) | **deletion** | 0.25 | 0.023 | -0.039 | `' hours implements your http came sleep g'` |
| worst | `' users hear works file' [·] ' policy sure cred'` | 8 | `' policy'` (0.947) | **deletion** | 0.38 | 0.037 | -0.059 | `' policy sure cred\n\nIt seems there was'` |
| best | `' printf ne throw parents' [·] ' art human ver'` | 16 | `'溉'` (0.998) | **correct** | 0.03 | 0.998 | -0.057 | `'溉 art human ver scope effects dans fig'` |

### `'隱私權'` — id 103496

single-probe lp -0.035 · fragility 0.42 · mean lp -0.49 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' las goal logger hard' [·] ' beh func eff'` | 8 | `'<|im_start|>'` (0.192) | **substitution** | 6.91 | 0.116 | -0.062 | `'🔍 beh func eff\n\nIt seems there'` |
| worst | `' thought weight shown gives' [·] ' sn seg head'` | 64 | `'<|im_end|>'` (0.379) | **substitution** | 4.71 | 0.261 | -0.116 | `''` |
| worst | `' req did las through' [·] ' except anyone placeholder'` | 32 | `'隱私權'` (0.320) | **correct** | 4.99 | 0.320 | -0.097 | `' 개인정보 except anyone placeholder top sl\n\n'` |
| best | `' stay cho after specific' [·] ' hours implements your'` | 32 | `'隱私權'` (0.985) | **correct** | 0.19 | 0.985 | -0.073 | `'隱私權 hours implements your http came slee'` |

### `' والف'` — id 135635

single-probe lp -0.021 · fragility 0.42 · mean lp -0.97 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `'&W'` (0.079) | **substitution** | 9.66 | 0.040 | -0.072 | `'&W dist conditions vote results cr chann'` |
| worst | `' stat something javax to' [·] ' mot files reach'` | 64 | `'AndFeel'` (0.083) | **substitution** | 9.54 | 0.042 | -0.026 | `'AndFeel mot files reach sure quite media'` |
| worst | `' stay cho after specific' [·] ' hours implements your'` | 32 | `' والف'` (0.081) | **correct** | 9.77 | 0.081 | -0.041 | `'&W hours implements your http came sleep'` |
| best | `' bad reports' [·] ' assert signific cert'` | 8 | `' والف'` (0.993) | **correct** | 0.12 | 0.993 | -0.055 | `' والف assert signific cert vote pol\nIt'` |

### `'並同意'` — id 118224

single-probe lp -0.002 · fragility 0.42 · mean lp -0.74 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `'<|im_end|>'` (0.434) | **substitution** | 4.75 | 0.017 | -0.148 | `''` |
| worst | `' stat something javax to' [·] ' mot files reach'` | 64 | `'<|im_start|>'` (0.341) | **substitution** | 5.85 | 0.032 | -0.107 | `'同意義見 mot files reach sure'` |
| worst | `' thought weight shown gives' [·] ' sn seg head'` | 64 | `'<|im_end|>'` (0.276) | **substitution** | 6.91 | 0.101 | -0.126 | `''` |
| best | `' idx min activity parts' [·] ' develop'` | 8 | `'並同意'` (0.999) | **correct** | 0.03 | 0.999 | -0.081 | `'並同意 develop\nIt appears that the last'` |

### `'垴'` — id 122409

single-probe lp -0.044 · fragility 0.38 · mean lp -1.45 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' increased download' [·] ' unit rep saying'` | 32 | `' unit'` (0.959) | **deletion** | 0.45 | 0.001 | -0.082 | `' unit rep saying fl cons ro shift photo'` |
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.990) | **deletion** | 0.14 | 0.002 | -0.067 | `' not million provided personal sus conte'` |
| worst | `' won exit itself word' [·] ' national cur'` | 32 | `' national'` (0.914) | **deletion** | 0.83 | 0.002 | -0.078 | `' national cur'` |
| best | `' rais veh tor man' [·] ' enough problems max'` | 32 | `'垴'` (0.999) | **correct** | 0.01 | 0.999 | -0.084 | `'垴 enough problems max offer session name'` |

### `' Украин'` — id 137829

single-probe lp -0.084 · fragility 0.38 · mean lp -0.63 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' won exit itself word' [·] ' national cur'` | 32 | `' Ukrainian'` (0.819) | **substitution** | 0.94 | 0.126 | -0.084 | `' Ukrainian national cur\n\nNote: The last'` |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `' Ukrainian'` (0.813) | **substitution** | 0.87 | 0.160 | -0.102 | `' Ukrainian dist conditions vote results '` |
| worst | `' elements data instance it' [·] ' when said named'` | 16 | `' Укра'` (0.750) | **truncation** | 1.06 | 0.190 | -0.104 | `' Україн when said named account\n\n'` |
| best | `' later allow param dist' [·] ' appro effect title'` | 64 | `' Украин'` (0.970) | **correct** | 0.23 | 0.970 | -0.055 | `' Украин appro effect title again di stat'` |

### `' חיפה'` — id 139676

single-probe lp -0.007 · fragility 0.38 · mean lp -0.93 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' thought weight shown gives' [·] ' sn seg head'` | 64 | `'<|im_end|>'` (0.404) | **substitution** | 7.48 | 0.006 | -0.123 | `''` |
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `'<|im_end|>'` (0.413) | **substitution** | 7.10 | 0.021 | -0.129 | `''` |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | 64 | `'<|im_end|>'` (0.151) | **substitution** | 10.21 | 0.056 | -0.097 | `''` |
| best | `' rais veh tor man' [·] ' enough problems max'` | 32 | `' חיפה'` (0.972) | **correct** | 0.48 | 0.972 | -0.079 | `' חיפה enough problems max offer session '` |

### `'辚'` — id 121746

single-probe lp -0.002 · fragility 0.38 · mean lp -1.37 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | 64 | `' not'` (0.999) | **deletion** | 0.02 | 0.000 | -0.056 | `' not million provided personal sus conte'` |
| worst | `' sh messages communic una' [·] ' result heart'` | 64 | `' result'` (0.981) | **deletion** | 0.24 | 0.001 | -0.013 | `' result heart'` |
| worst | `' costs nor full able' [·] ' included'` | 32 | `' included'` (0.983) | **deletion** | 0.18 | 0.007 | 0.019 | `' included'` |
| best | `' len' [·] ' rais std systems'` | 8 | `'辚'` (1.000) | **correct** | 0.01 | 1.000 | -0.029 | `'辚 rais std systems img sun treatment\n\n'` |

## Verified glitch tokens (reference)


### `'_ghost'` — id 98784  (single-probe lp -24.56, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | `'_ghost'` (0.995) | correct | 0.05 | 0.995 | -0.068 |
| worst | `' there lives node theme' [·] ' open ass dir'` | `'_ghost'` (0.999) | correct | 0.02 | 0.998 | -0.076 |
| worst | `' users hear works file' [·] ' policy sure cred'` | `'_ghost'` (0.999) | correct | 0.01 | 0.999 | -0.084 |
| best | `' stat something javax to' [·] ' mot files reach'` | `'_ghost'` (1.000) | correct | 0.00 | 1.000 | -0.052 |

### `'_CHAN'` — id 80324  (single-probe lp -24.13, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' bad reports' [·] ' assert signific cert'` | `'_CHAN'` (0.995) | correct | 0.06 | 0.995 | -0.062 |
| worst | `' increased download' [·] ' unit rep saying'` | `'_CHAN'` (0.996) | correct | 0.04 | 0.996 | -0.066 |
| worst | `' cult sure files selection' [·] ' not million provided'` | `'_CHAN'` (0.997) | correct | 0.04 | 0.997 | -0.079 |
| best | `' stre me filename known' [·] ' dans box mobile'` | `'_CHAN'` (1.000) | correct | 0.00 | 1.000 | -0.013 |

### `'/topics'` — id 57662  (single-probe lp -22.25, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | `'/topics'` (1.000) | correct | 0.00 | 1.000 | -0.085 |
| worst | `' water long sol investig' [·] ' filter brand natural'` | `'/topics'` (1.000) | correct | 0.00 | 1.000 | -0.067 |
| worst | `' protected lives management pred' [·] ' deg'` | `'/topics'` (1.000) | correct | 0.00 | 1.000 | -0.055 |
| best | `' later allow param dist' [·] ' appro effect title'` | `'/topics'` (1.000) | correct | 0.00 | 1.000 | -0.035 |

### `'_latitude'` — id 80364  (single-probe lp -21.69, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' increased download' [·] ' unit rep saying'` | `'_latitude'` (1.000) | correct | 0.00 | 1.000 | -0.053 |
| worst | `' stay cho after specific' [·] ' hours implements your'` | `'_latitude'` (1.000) | correct | 0.00 | 1.000 | -0.060 |
| worst | `' costs nor full able' [·] ' included'` | `'_latitude'` (1.000) | correct | 0.00 | 1.000 | -0.035 |
| best | `' later allow param dist' [·] ' appro effect title'` | `'_latitude'` (1.000) | correct | 0.00 | 1.000 | -0.034 |

### `'$filter'` — id 90969  (single-probe lp -21.56, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' won exit itself word' [·] ' national cur'` | `'$filter'` (0.933) | correct | 0.46 | 0.933 | -0.081 |
| worst | `' rais veh tor man' [·] ' enough problems max'` | `'$filter'` (0.988) | correct | 0.12 | 0.988 | -0.075 |
| worst | `' water long sol investig' [·] ' filter brand natural'` | `'$filter'` (0.991) | correct | 0.09 | 0.991 | -0.035 |
| best | `' stat something javax to' [·] ' mot files reach'` | `'$filter'` (0.999) | correct | 0.01 | 0.999 | -0.092 |

### `'$product'` — id 68173  (single-probe lp -21.56, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' increased download' [·] ' unit rep saying'` | `'$product'` (0.997) | correct | 0.04 | 0.997 | -0.090 |
| worst | `' las goal logger hard' [·] ' beh func eff'` | `'$product'` (0.997) | correct | 0.04 | 0.997 | -0.070 |
| worst | `' sh messages communic una' [·] ' result heart'` | `'$product'` (0.997) | correct | 0.04 | 0.997 | -0.104 |
| best | `' printf ne throw parents' [·] ' art human ver'` | `'$product'` (1.000) | correct | 0.01 | 0.999 | -0.105 |

### `'/widget'` — id 33336  (single-probe lp -21.38, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' users hear works file' [·] ' policy sure cred'` | `'/widget'` (0.999) | correct | 0.01 | 0.999 | -0.077 |
| worst | `' las goal logger hard' [·] ' beh func eff'` | `'/widget'` (1.000) | correct | 0.01 | 1.000 | -0.069 |
| worst | `' won exit itself word' [·] ' national cur'` | `'/widget'` (1.000) | correct | 0.01 | 1.000 | -0.087 |
| best | `' len' [·] ' rais std systems'` | `'/widget'` (1.000) | correct | 0.00 | 1.000 | -0.029 |

### `'눠'` — id 144478  (single-probe lp -21.04, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' cult sure files selection' [·] ' not million provided'` | `' not'` (0.986) | deletion | 0.21 | 0.000 | -0.078 |
| worst | `' stay cho after specific' [·] ' hours implements your'` | `' hours'` (0.551) | deletion | 5.04 | 0.000 | -0.048 |
| worst | `' stat something javax to' [·] ' mot files reach'` | `'<|im_end|>'` (0.256) | substitution | 7.91 | 0.000 | -0.076 |
| best | `' idx min activity parts' [·] ' develop'` | `' modalità'` (0.082) | substitution | 10.37 | 0.000 | -0.040 |

### `'ﯴ'` — id 149852  (single-probe lp -20.69, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' increased download' [·] ' unit rep saying'` | `' unit'` (0.993) | deletion | 0.07 | 0.000 | -0.051 |
| worst | `' stre me filename known' [·] ' dans box mobile'` | `' dans'` (0.996) | deletion | 0.05 | 0.000 | -0.027 |
| worst | `' costs nor full able' [·] ' included'` | `' included'` (0.846) | deletion | 0.67 | 0.000 | 0.027 |
| best | `' protected lives management pred' [·] ' deg'` | `'<|im_end|>'` (0.935) | substitution | 0.95 | 0.000 | -0.054 |

### `'_fact'` — id 47805  (single-probe lp -20.25, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' water long sol investig' [·] ' filter brand natural'` | `'_fact'` (0.996) | correct | 0.04 | 0.996 | -0.074 |
| worst | `' costs nor full able' [·] ' included'` | `'_fact'` (0.999) | correct | 0.01 | 0.999 | -0.049 |
| worst | `' there lives node theme' [·] ' open ass dir'` | `'_fact'` (0.999) | correct | 0.01 | 0.999 | -0.068 |
| best | `' printf ne throw parents' [·] ' art human ver'` | `'_fact'` (1.000) | correct | 0.00 | 1.000 | -0.067 |

### `'윁'` — id 147390  (single-probe lp -20.03, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' increased download' [·] ' unit rep saying'` | `' unit'` (0.989) | deletion | 0.10 | 0.000 | -0.058 |
| worst | `' bad reports' [·] ' assert signific cert'` | `' assert'` (0.993) | deletion | 0.07 | 0.000 | -0.092 |
| worst | `' stre me filename known' [·] ' dans box mobile'` | `' dans'` (0.991) | deletion | 0.10 | 0.000 | -0.033 |
| best | `' protected lives management pred' [·] ' deg'` | `'<|im_end|>'` (0.941) | substitution | 0.75 | 0.000 | -0.042 |

### `'_pool'` — id 15709  (single-probe lp -20.00, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' water long sol investig' [·] ' filter brand natural'` | `'_pool'` (0.998) | correct | 0.03 | 0.998 | -0.060 |
| worst | `' costs nor full able' [·] ' included'` | `'_pool'` (0.999) | correct | 0.01 | 0.999 | -0.035 |
| worst | `' there lives node theme' [·] ' open ass dir'` | `'_pool'` (1.000) | correct | 0.00 | 1.000 | -0.052 |
| best | `' tre under available star' [·] ' del aria list'` | `'_pool'` (1.000) | correct | 0.00 | 1.000 | -0.061 |

### `'㭕'` — id 122431  (single-probe lp -19.97, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' increased download' [·] ' unit rep saying'` | `' unit'` (0.981) | deletion | 0.16 | 0.000 | -0.072 |
| worst | `' stre me filename known' [·] ' dans box mobile'` | `' dans'` (0.990) | deletion | 0.11 | 0.000 | -0.031 |
| worst | `' cult sure files selection' [·] ' not million provided'` | `' not'` (0.914) | deletion | 0.43 | 0.000 | -0.083 |
| best | `' protected lives management pred' [·] ' deg'` | `'<|im_end|>'` (0.829) | substitution | 1.69 | 0.000 | -0.024 |

### `'䨰'` — id 150811  (single-probe lp -19.91, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' increased download' [·] ' unit rep saying'` | `' unit'` (0.980) | deletion | 0.16 | 0.000 | -0.050 |
| worst | `' bad reports' [·] ' assert signific cert'` | `' assert'` (0.994) | deletion | 0.06 | 0.000 | -0.096 |
| worst | `' costs nor full able' [·] ' included'` | `' included'` (0.844) | deletion | 0.68 | 0.000 | 0.027 |
| best | `' stat something javax to' [·] ' mot files reach'` | `'<|im_end|>'` (0.757) | substitution | 1.36 | 0.000 | -0.162 |

### `'ﯭ'` — id 151192  (single-probe lp -19.91, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' costs nor full able' [·] ' included'` | `' included'` (0.726) | deletion | 0.91 | 0.000 | 0.043 |
| worst | `' cult sure files selection' [·] ' not million provided'` | `' not'` (0.904) | deletion | 0.46 | 0.000 | -0.065 |
| worst | `' increased download' [·] ' unit rep saying'` | `' unit'` (0.945) | deletion | 0.37 | 0.000 | -0.048 |
| best | `' protected lives management pred' [·] ' deg'` | `'<|im_end|>'` (0.875) | substitution | 1.31 | 0.000 | -0.019 |

### `'_ATOMIC'` — id 87843  (single-probe lp -19.88, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' protected lives management pred' [·] ' deg'` | `'_ATOMIC'` (0.986) | correct | 0.15 | 0.986 | -0.038 |
| worst | `' water long sol investig' [·] ' filter brand natural'` | `'_ATOMIC'` (0.990) | correct | 0.09 | 0.990 | -0.063 |
| worst | `' stay cho after specific' [·] ' hours implements your'` | `'_ATOMIC'` (0.996) | correct | 0.05 | 0.996 | -0.059 |
| best | `' later allow param dist' [·] ' appro effect title'` | `'_ATOMIC'` (1.000) | correct | 0.00 | 1.000 | -0.042 |

### `'.scss'` — id 17696  (single-probe lp -19.88, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' stat something javax to' [·] ' mot files reach'` | `'.scss'` (0.999) | correct | 0.01 | 0.999 | -0.066 |
| worst | `' sometimes home request obt' [·] ' dist conditions vote'` | `'.scss'` (0.999) | correct | 0.01 | 0.999 | -0.059 |
| worst | `' stay cho after specific' [·] ' hours implements your'` | `'.scss'` (0.999) | correct | 0.01 | 0.999 | -0.019 |
| best | `' stre me filename known' [·] ' dans box mobile'` | `'.scss'` (1.000) | correct | 0.00 | 1.000 | -0.008 |

### `'𠳐'` — id 120101  (single-probe lp -19.81, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' increased download' [·] ' unit rep saying'` | `' unit'` (0.962) | deletion | 0.27 | 0.000 | -0.058 |
| worst | `' stre me filename known' [·] ' dans box mobile'` | `' dans'` (0.986) | deletion | 0.14 | 0.000 | -0.023 |
| worst | `' bad reports' [·] ' assert signific cert'` | `' assert'` (0.976) | deletion | 0.18 | 0.000 | -0.079 |
| best | `' protected lives management pred' [·] ' deg'` | `'<|im_end|>'` (0.862) | substitution | 1.56 | 0.000 | -0.029 |

### `'_gate'` — id 54798  (single-probe lp -19.75, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' costs nor full able' [·] ' included'` | `'_gate'` (0.999) | correct | 0.02 | 0.998 | -0.049 |
| worst | `' water long sol investig' [·] ' filter brand natural'` | `'_gate'` (0.999) | correct | 0.02 | 0.999 | -0.069 |
| worst | `' las goal logger hard' [·] ' beh func eff'` | `'_gate'` (0.999) | correct | 0.01 | 0.999 | -0.054 |
| best | `' stat something javax to' [·] ' mot files reach'` | `'_gate'` (1.000) | correct | 0.00 | 1.000 | -0.061 |

### `'닠'` — id 149083  (single-probe lp -19.66, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' increased download' [·] ' unit rep saying'` | `' unit'` (0.987) | deletion | 0.11 | 0.000 | -0.047 |
| worst | `' bad reports' [·] ' assert signific cert'` | `' assert'` (0.995) | deletion | 0.05 | 0.000 | -0.076 |
| worst | `' costs nor full able' [·] ' included'` | `' included'` (0.771) | deletion | 0.83 | 0.000 | 0.040 |
| best | `' protected lives management pred' [·] ' deg'` | `'<|im_end|>'` (0.924) | substitution | 0.75 | 0.000 | -0.043 |
