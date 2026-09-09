# Confident-substitution specimens

Every example in this file was produced by greedy decoding; `p` is the model's probability for the token it emitted, `H` the entropy in bits at that position, `p(target)` its probability for the correct token, `ret` the cosine between the last-layer slot state and the correct token's unembedding row (identity still pointing at its readout while the readout chose otherwise).

## Model

- **model**: `Qwen/Qwen3.8-27B`
- **commit**: `None`
- **dtype**: `torch.bfloat16`
- **num_layers**: `64`
- **hidden_size**: `5120`
- **vocab_size**: `248320`
- **tie_word_embeddings**: `False`
- **tokenizer_class**: `Qwen2Tokenizer`
- **architecture**: `?`
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


### `' оказаться'` — id 216231

single-probe lp -0.118 · fragility 0.96 · mean lp -4.48 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' industry full mark love' [·] ' where word column'` | 8 | `' occur'` (0.476) | **substitution** | 3.92 | 0.001 | -0.006 | `' occur where word column'` |
| worst | `' life dig think piece' [·] ' or company connection'` | 64 | `' or'` (0.870) | **deletion** | 1.63 | 0.002 | 0.024 | `' or company connection session makes suc'` |
| worst | `' serv grow ex form' [·] ' website'` | 8 | `'<|im_end|>'` (0.500) | **substitution** | 3.41 | 0.003 | -0.015 | `''` |
| best | `' um' [·] ' parameters months cor'` | 16 | `' оказаться'` (0.974) | **correct** | 0.37 | 0.974 | 0.001 | `' оказаться parameters months cor shows p'` |

### `' Председа'` — id 216698

single-probe lp -0.446 · fragility 0.96 · mean lp -1.35 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' power asked descri' [·] ' equipment physical kn'` | 32 | `'-Cds'` (0.185) | **substitution** | 8.43 | 0.030 | -0.014 | `'-Cds equipment physical kn public natura'` |
| worst | `' header coming clo users' [·] ' under si power'` | 32 | `' Председа'` (0.134) | **correct** | 9.26 | 0.134 | 0.006 | `' Председа under si power define mention '` |
| worst | `' serv grow ex form' [·] ' website'` | 8 | `' Председа'` (0.154) | **correct** | 8.69 | 0.154 | 0.044 | `' Председа website'` |
| best | `' post double dise rock' [·] ' announced pro current'` | 64 | `' Председа'` (0.663) | **correct** | 4.19 | 0.663 | -0.008 | `' Председа announced pro current shall pl'` |

### `'стит'` — id 158693

single-probe lp -0.622 · fragility 0.96 · mean lp -2.66 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' expert became token record' [·] ' date decided country'` | 16 | `'stit'` (0.989) | **substitution** | 0.12 | 0.007 | -0.029 | `'stit date decided country govern recentl'` |
| worst | `' far instead pur commit' [·] ' arguments learning bo'` | 16 | `'stit'` (0.988) | **substitution** | 0.11 | 0.009 | -0.009 | `'stit arguments learning bo messages terr'` |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `'stit'` (0.973) | **substitution** | 0.26 | 0.016 | -0.012 | `'stit activity decl flex method crit mind'` |
| best | `' materials' [·] ' middle tot decided'` | 8 | `'стит'` (0.981) | **correct** | 0.28 | 0.981 | 0.015 | `'стит middle tot decided lost govern fore'` |

### `' 아버지'` — id 228388

single-probe lp -0.007 · fragility 0.92 · mean lp -3.14 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' df' [·] ' influ address claim'` | 8 | `'아버'` (0.991) | **truncation** | 0.09 | 0.007 | 0.011 | `'아버지 influ address claim act direction de'` |
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `'아버'` (0.989) | **truncation** | 0.11 | 0.008 | 0.024 | `'아버지 announced pro current shall pl charg'` |
| worst | `' far instead pur commit' [·] ' arguments learning bo'` | 16 | `'아버'` (0.979) | **truncation** | 0.20 | 0.012 | 0.030 | `'아버지 arguments learning bo messages terr'` |
| best | `' life dig think piece' [·] ' or company connection'` | 64 | `' 아버지'` (0.950) | **correct** | 0.31 | 0.950 | 0.044 | `' 아버지 or company connection session makes'` |

### `'мой'` — id 170870

single-probe lp -0.418 · fragility 0.88 · mean lp -2.83 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' icon sometimes mom interview' [·] ' fac taking site'` | 32 | `'my'` (0.906) | **substitution** | 0.64 | 0.001 | -0.017 | `'my fac taking site func orig try os'` |
| worst | `' on package political into' [·] ' tw exam match'` | 16 | `'<|im_end|>'` (0.777) | **substitution** | 1.17 | 0.001 | -0.002 | `''` |
| worst | `' expert became token record' [·] ' date decided country'` | 16 | `'my'` (0.914) | **substitution** | 0.76 | 0.005 | -0.032 | `'my date decided country govern recently '` |
| best | `' um' [·] ' parameters months cor'` | 16 | `'мой'` (0.999) | **correct** | 0.02 | 0.999 | -0.017 | `'мой parameters months cor shows proble h'` |

### `' грунт'` — id 212272

single-probe lp -0.176 · fragility 0.79 · mean lp -3.87 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' led act cit install' [·] ' give seek man'` | 64 | `' grunt'` (0.995) | **substitution** | 0.06 | 0.000 | -0.019 | `' grunt give seek man costs goes size pro'` |
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `' grunt'` (0.997) | **substitution** | 0.04 | 0.000 | -0.029 | `' grunt announced pro current shall pl ch'` |
| worst | `' life dig think piece' [·] ' or company connection'` | 64 | `' grunt'` (0.989) | **substitution** | 0.10 | 0.000 | -0.001 | `' grunt or company connection session mak'` |
| best | `' on package political into' [·] ' tw exam match'` | 16 | `' грунт'` (0.912) | **correct** | 0.45 | 0.912 | -0.005 | `' грунт tw exam match often'` |

### `' adecu'` — id 160392

single-probe lp -0.357 · fragility 0.75 · mean lp -0.89 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' serv grow ex form' [·] ' website'` | 8 | `' adequ'` (0.848) | **substitution** | 0.84 | 0.115 | 0.077 | `' adequ website'` |
| worst | `' industry full mark love' [·] ' where word column'` | 8 | `' adequ'` (0.824) | **substitution** | 0.88 | 0.143 | 0.028 | `' adequ where word column'` |
| worst | `' re lot distributed rather' [·] ' times needed layout'` | 8 | `' adequ'` (0.711) | **substitution** | 1.28 | 0.231 | 0.010 | `' adequ times needed layout'` |
| best | `' df' [·] ' influ address claim'` | 8 | `' adecu'` (0.847) | **correct** | 0.86 | 0.847 | 0.044 | `' adecu influ address claim act direction'` |

### `' 않습니다'` — id 172716

single-probe lp -0.446 · fragility 0.75 · mean lp -1.56 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' called period fam life' [·] ' comple into must'` | 64 | `'합니다'` (0.472) | **substitution** | 3.10 | 0.010 | -0.003 | `'합니다 comple into must weeks higher fire k'` |
| worst | `' far instead pur commit' [·] ' arguments learning bo'` | 16 | `'합니다'` (0.634) | **substitution** | 2.18 | 0.019 | 0.041 | `'합니다 arguments learning bo messages terr'` |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `'합니다'` (0.313) | **substitution** | 4.86 | 0.019 | 0.007 | `'합니다 activity decl flex method crit mind '` |
| best | `' power asked descri' [·] ' equipment physical kn'` | 32 | `' 않습니다'` (0.914) | **correct** | 0.98 | 0.914 | 0.028 | `' 않습니다 equipment physical kn public natur'` |

### `' individuales'` — id 210286

single-probe lp -0.402 · fragility 0.71 · mean lp -0.67 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' re lot distributed rather' [·] ' times needed layout'` | 8 | `' individuelles'` (0.580) | **substitution** | 2.21 | 0.274 | 0.031 | `' individuelles times needed layout'` |
| worst | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `' individuelles'` (0.534) | **substitution** | 2.22 | 0.324 | -0.015 | `' individuelles fully inst behind areas p'` |
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `' individuelles'` (0.397) | **substitution** | 2.93 | 0.310 | 0.008 | `' individuelles offers like ax ver young '` |
| best | `' called period fam life' [·] ' comple into must'` | 64 | `' individuales'` (0.767) | **correct** | 1.21 | 0.767 | 0.030 | `' individuales comple into must weeks hig'` |

### `' espectacular'` — id 220643

single-probe lp -0.184 · fragility 0.62 · mean lp -0.59 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `' spectacular'` (0.649) | **substitution** | 1.18 | 0.307 | -0.004 | `' spectacular offers like ax ver young lo'` |
| worst | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `' spectacular'` (0.497) | **substitution** | 1.59 | 0.341 | -0.005 | `' spectacular fully inst behind areas pra'` |
| worst | `' icon sometimes mom interview' [·] ' fac taking site'` | 32 | `' spectacular'` (0.643) | **substitution** | 1.04 | 0.344 | 0.022 | `' spectacular fac taking site func orig t'` |
| best | `' post double dise rock' [·] ' announced pro current'` | 64 | `' espectacular'` (0.863) | **correct** | 0.69 | 0.863 | -0.031 | `' espectacular announced pro current shal'` |

### `' половину'` — id 223231

single-probe lp -0.065 · fragility 0.62 · mean lp -0.83 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' icon sometimes mom interview' [·] ' fac taking site'` | 32 | `' половины'` (0.718) | **substitution** | 1.26 | 0.110 | 0.011 | `' половины fac taking site func orig try '` |
| worst | `' life dig think piece' [·] ' or company connection'` | 64 | `' половины'` (0.608) | **substitution** | 1.64 | 0.224 | 0.030 | `' половины or company connection session '` |
| worst | `' materials' [·] ' middle tot decided'` | 8 | `' половины'` (0.542) | **substitution** | 1.86 | 0.226 | 0.009 | `' половины middle tot decided lost govern'` |
| best | `' um' [·] ' parameters months cor'` | 16 | `' половину'` (0.946) | **correct** | 0.40 | 0.946 | 0.033 | `' половину parameters months cor shows pr'` |

### `' الديمق'` — id 158107

single-probe lp -0.062 · fragility 0.58 · mean lp -0.58 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' header coming clo users' [·] ' under si power'` | 32 | `' الديمق'` (0.272) | **correct** | 5.69 | 0.272 | 0.008 | `' الديمق under si power define mention at'` |
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `' الديمق'` (0.327) | **correct** | 4.50 | 0.327 | -0.007 | `' الديمق announced pro current shall pl c'` |
| worst | `' apply benef comp' [·] ' distrib below forward'` | 16 | `' الديمق'` (0.337) | **correct** | 5.05 | 0.337 | 0.033 | `' الديمق distrib below forward die added '` |
| best | `' df' [·] ' influ address claim'` | 8 | `' الديمق'` (0.982) | **correct** | 0.22 | 0.982 | 0.028 | `' الديمق influ address claim act directio'` |

### `' المناسب'` — id 195690

single-probe lp -0.022 · fragility 0.58 · mean lp -1.43 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `' مناسب'` (0.988) | **substitution** | 0.13 | 0.008 | 0.014 | `' مناسب fully inst behind areas pract gr '` |
| worst | `' header coming clo users' [·] ' under si power'` | 32 | `' مناسب'` (0.974) | **substitution** | 0.24 | 0.016 | -0.000 | `' مناسب under si power define mention at '` |
| worst | `' expert became token record' [·] ' date decided country'` | 16 | `' مناسب'` (0.979) | **substitution** | 0.17 | 0.018 | 0.019 | `' مناسب date decided country govern recen'` |
| best | `' df' [·] ' influ address claim'` | 8 | `' المناسب'` (0.993) | **correct** | 0.08 | 0.992 | 0.009 | `' المناسب influ address claim act directi'` |

### `' وته'` — id 232267

single-probe lp -0.153 · fragility 0.58 · mean lp -1.22 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `' ته'` (0.973) | **substitution** | 0.27 | 0.011 | -0.008 | `' ته conditions respect word constant day'` |
| worst | `' header coming clo users' [·] ' under si power'` | 32 | `' ته'` (0.954) | **substitution** | 0.35 | 0.037 | -0.001 | `' ته under si power define mention at pro'` |
| worst | `' called period fam life' [·] ' comple into must'` | 64 | `' ته'` (0.494) | **substitution** | 1.50 | 0.046 | 0.001 | `'ته comple into must weeks higher fire kn'` |
| best | `' expert became token record' [·] ' date decided country'` | 16 | `' وته'` (0.945) | **correct** | 0.34 | 0.945 | 0.019 | `' وته date decided country govern recentl'` |

### `' директор'` — id 180444

single-probe lp -0.013 · fragility 0.54 · mean lp -0.69 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' far instead pur commit' [·] ' arguments learning bo'` | 16 | `'д'` (0.817) | **truncation** | 0.91 | 0.161 | 0.058 | `'директор arguments learning bo messages'` |
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `'д'` (0.782) | **truncation** | 1.14 | 0.175 | 0.036 | `'директор conditions respect word constan'` |
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `'д'` (0.766) | **truncation** | 0.97 | 0.220 | 0.017 | `'директор announced pro current shall'` |
| best | `' df' [·] ' influ address claim'` | 8 | `' директор'` (1.000) | **correct** | 0.01 | 1.000 | 0.026 | `' директор influ address claim act direct'` |

### `' certamente'` — id 188176

single-probe lp -0.120 · fragility 0.54 · mean lp -0.65 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' home wall dat ke' [·] ' plus line lik'` | 64 | `' certainly'` (0.695) | **substitution** | 1.33 | 0.199 | -0.012 | `' certainly plus line lik energ playing c'` |
| worst | `' far instead pur commit' [·] ' arguments learning bo'` | 16 | `' certainly'` (0.778) | **substitution** | 1.14 | 0.135 | 0.017 | `' certainly arguments learning bo message'` |
| worst | `' on package political into' [·] ' tw exam match'` | 16 | `' certainly'` (0.417) | **substitution** | 1.92 | 0.253 | -0.010 | `' certainly tw exam match often'` |
| best | `' df' [·] ' influ address claim'` | 8 | `' certamente'` (0.985) | **correct** | 0.15 | 0.985 | 0.028 | `' certamente influ address claim act dire'` |

### `' здании'` — id 245308

single-probe lp -0.105 · fragility 0.50 · mean lp -1.12 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' materials' [·] ' middle tot decided'` | 8 | `' здания'` (0.959) | **substitution** | 0.37 | 0.009 | -0.012 | `' здания middle tot decided lost govern f'` |
| worst | `' called period fam life' [·] ' comple into must'` | 64 | `' здания'` (0.953) | **substitution** | 0.37 | 0.022 | -0.008 | `' здания comple into must weeks higher fi'` |
| worst | `' expert became token record' [·] ' date decided country'` | 16 | `' здание'` (0.918) | **substitution** | 0.53 | 0.059 | -0.019 | `' здание date decided country govern rece'` |
| best | `' um' [·] ' parameters months cor'` | 16 | `' здании'` (0.958) | **correct** | 0.38 | 0.958 | -0.022 | `' здании parameters months cor shows prob'` |

### `' проводиться'` — id 231029

single-probe lp -0.118 · fragility 0.46 · mean lp -0.53 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' life dig think piece' [·] ' or company connection'` | 64 | `' проводится'` (0.609) | **substitution** | 1.72 | 0.326 | 0.009 | `' проводится or company connection sessio'` |
| worst | `' far instead pur commit' [·] ' arguments learning bo'` | 16 | `' проводится'` (0.566) | **substitution** | 1.73 | 0.343 | -0.012 | `' проводится arguments learning bo messag'` |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `' проводится'` (0.548) | **substitution** | 1.70 | 0.377 | -0.003 | `' проводится activity decl flex method cr'` |
| best | `' led act cit install' [·] ' give seek man'` | 64 | `' проводиться'` (0.947) | **correct** | 0.40 | 0.947 | -0.007 | `' проводиться give seek man costs goes si'` |

### `' diinginkan'` — id 202522

single-probe lp -0.001 · fragility 0.46 · mean lp -1.49 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' on package political into' [·] ' tw exam match'` | 16 | `' desired'` (0.916) | **substitution** | 0.69 | 0.002 | 0.010 | `' desired tw exam match often'` |
| worst | `' safe mod edit history' [·] ' seg'` | 16 | `' desired'` (0.800) | **substitution** | 1.04 | 0.003 | 0.042 | `' desired seg'` |
| worst | `' icon sometimes mom interview' [·] ' fac taking site'` | 32 | `' desired'` (0.961) | **substitution** | 0.41 | 0.011 | 0.018 | `' desired fac taking site func orig try o'` |
| best | `' um' [·] ' parameters months cor'` | 16 | `' diinginkan'` (0.999) | **correct** | 0.01 | 0.999 | 0.005 | `' diinginkan parameters months cor shows '` |

### `' وزيادة'` — id 230548

single-probe lp -0.063 · fragility 0.42 · mean lp -0.63 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `' زيادة'` (0.495) | **substitution** | 1.64 | 0.098 | 0.003 | `' زيادة conditions respect word constant '` |
| worst | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `' زيادة'` (0.794) | **substitution** | 1.01 | 0.177 | -0.026 | `' زيادة fully inst behind areas pract gr '` |
| worst | `' header coming clo users' [·] ' under si power'` | 32 | `' زيادة'` (0.729) | **substitution** | 1.32 | 0.163 | -0.016 | `' زيادة under si power define mention at '` |
| best | `' um' [·] ' parameters months cor'` | 16 | `' وزيادة'` (0.979) | **correct** | 0.24 | 0.979 | 0.001 | `' وزيادة parameters months cor shows prob'` |

### `'ņem'` — id 219021

single-probe lp -0.602 · fragility 0.42 · mean lp -0.55 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' event even giving' [·] ' is route full'` | 8 | `'ņ'` (0.774) | **truncation** | 0.83 | 0.222 | 0.032 | `'ņem is route full last'` |
| worst | `' industry full mark love' [·] ' where word column'` | 8 | `'ņ'` (0.752) | **truncation** | 0.86 | 0.244 | 0.008 | `'ņ where word column'` |
| worst | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `'ņ'` (0.644) | **truncation** | 1.08 | 0.344 | -0.012 | `'ņ fully inst behind areas pract gr polit'` |
| best | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `'ņem'` (0.929) | **correct** | 0.56 | 0.929 | 0.001 | `'ņem activity decl flex method crit mind '` |

### `' acidentes'` — id 241582

single-probe lp -0.092 · fragility 0.38 · mean lp -0.52 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `' accidentes'` (0.841) | **substitution** | 0.91 | 0.129 | -0.004 | `' accidentes offers like ax ver young los'` |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `' accidentes'` (0.647) | **substitution** | 1.02 | 0.347 | 0.011 | `' accidentes activity decl flex method cr'` |
| worst | `' re lot distributed rather' [·] ' times needed layout'` | 8 | `' accidentes'` (0.605) | **substitution** | 1.84 | 0.324 | 0.020 | `' accidentes times needed layout'` |
| best | `' um' [·] ' parameters months cor'` | 16 | `' acidentes'` (0.974) | **correct** | 0.21 | 0.973 | 0.022 | `' acidentes parameters months cor shows p'` |

### `' przedsta'` — id 214254

single-probe lp -0.147 · fragility 0.33 · mean lp -0.47 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' apply benef comp' [·] ' distrib below forward'` | 16 | `' predsta'` (0.606) | **substitution** | 1.82 | 0.223 | -0.029 | `' predsta distrib below forward die added'` |
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `' przedsta'` (0.318) | **correct** | 2.63 | 0.318 | -0.048 | `' предста offers like ax ver young loss v'` |
| worst | `' home wall dat ke' [·] ' plus line lik'` | 64 | `' predsta'` (0.541) | **substitution** | 1.79 | 0.328 | -0.030 | `' predsta plus line lik energ playing cor'` |
| best | `' post double dise rock' [·] ' announced pro current'` | 64 | `' przedsta'` (0.962) | **correct** | 0.38 | 0.962 | -0.022 | `' przedsta announced pro current shall pl'` |

### `' универса'` — id 180211

single-probe lp -0.060 · fragility 0.33 · mean lp -0.57 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' life dig think piece' [·] ' or company connection'` | 64 | `' универса'` (0.282) | **correct** | 3.68 | 0.282 | 0.052 | `' универса or company connection session '` |
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `' универса'` (0.265) | **correct** | 4.14 | 0.265 | -0.027 | `' универса offers like ax ver young loss '` |
| worst | `' expert became token record' [·] ' date decided country'` | 16 | `' универса'` (0.276) | **correct** | 4.00 | 0.276 | 0.016 | `' универса date decided country govern re'` |
| best | `' um' [·] ' parameters months cor'` | 16 | `' универса'` (0.916) | **correct** | 0.88 | 0.916 | 0.009 | `' универса parameters months cor shows pr'` |

### `' సి'` — id 235354

single-probe lp -0.025 · fragility 0.29 · mean lp -0.54 · canonical False · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' led act cit install' [·] ' give seek man'` | 64 | `' সি'` (0.491) | **substitution** | 3.09 | 0.117 | -0.011 | `' সি give seek man costs goes size produc'` |
| worst | `' header coming clo users' [·] ' under si power'` | 32 | `' সি'` (0.334) | **substitution** | 3.39 | 0.123 | 0.022 | `' সি under si power define mention at pro'` |
| worst | `' safe mod edit history' [·] ' seg'` | 16 | `' সি'` (0.458) | **substitution** | 3.31 | 0.179 | 0.039 | `' সি seg'` |
| best | `' serv grow ex form' [·] ' website'` | 8 | `' సి'` (0.994) | **correct** | 0.09 | 0.994 | 0.088 | `' సి website'` |

### `' anzus'` — id 242290

single-probe lp -0.115 · fragility 0.29 · mean lp -0.48 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' expert became token record' [·] ' date decided country'` | 16 | `' anzus'` (0.121) | **correct** | 11.05 | 0.121 | -0.007 | `' anzus date decided country govern recen'` |
| worst | `' power asked descri' [·] ' equipment physical kn'` | 32 | `' anzus'` (0.196) | **correct** | 10.57 | 0.196 | -0.014 | `' anzus equipment physical kn public natu'` |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `' anzus'` (0.272) | **correct** | 9.57 | 0.271 | -0.021 | `' anzus activity decl flex method crit mi'` |
| best | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `' anzus'` (0.981) | **correct** | 0.32 | 0.981 | 0.005 | `' anzus fully inst behind areas pract gr '` |

### `' 불'` — id 149829

single-probe lp -0.018 · fragility 0.29 · mean lp -0.83 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' um' [·] ' parameters months cor'` | 16 | `'불'` (0.994) | **substitution** | 0.06 | 0.006 | 0.060 | `'불 parameters months cor shows proble han'` |
| worst | `' event even giving' [·] ' is route full'` | 8 | `'불'` (0.988) | **substitution** | 0.11 | 0.011 | 0.058 | `'불 is route full last'` |
| worst | `' safe mod edit history' [·] ' seg'` | 16 | `'불'` (0.952) | **substitution** | 0.28 | 0.047 | 0.073 | `'불 seg'` |
| best | `' industry full mark love' [·] ' where word column'` | 8 | `' 불'` (0.999) | **correct** | 0.01 | 0.999 | 0.059 | `' 불 where word column'` |

### `'രു'` — id 162723

single-probe lp -0.687 · fragility 0.25 · mean lp -0.75 · canonical False · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `'రు'` (0.910) | **substitution** | 0.96 | 0.003 | -0.031 | `'రు offers like ax ver young loss via'` |
| worst | `' um' [·] ' parameters months cor'` | 16 | `'uru'` (0.786) | **substitution** | 1.59 | 0.021 | -0.017 | `'uru parameters months cor shows proble h'` |
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `'ರು'` (0.467) | **substitution** | 2.85 | 0.063 | -0.024 | `'ರು conditions respect word constant day'` |
| best | `' serv grow ex form' [·] ' website'` | 8 | `'രു'` (0.984) | **correct** | 0.22 | 0.984 | 0.045 | `'രു website'` |

### `' 영업이익'` — id 237261

single-probe lp -0.179 · fragility 0.25 · mean lp -0.30 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' event even giving' [·] ' is route full'` | 8 | `'영업'` (0.791) | **truncation** | 0.84 | 0.200 | 0.046 | `'영업이익 is route full last'` |
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `'영업'` (0.615) | **truncation** | 1.15 | 0.373 | 0.004 | `'영업이익 conditions respect word constant da'` |
| worst | `' home wall dat ke' [·] ' plus line lik'` | 64 | `'영업'` (0.619) | **truncation** | 1.02 | 0.375 | -0.014 | `'영업이익 plus line lik energ playing core'` |
| best | `' led act cit install' [·] ' give seek man'` | 64 | `' 영업이익'` (0.988) | **correct** | 0.11 | 0.988 | -0.007 | `' 영업이익 give seek man costs goes size prod'` |

### `'ตะวันออก'` — id 165184

single-probe lp -0.378 · fragility 0.25 · mean lp -0.32 · canonical False · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' power asked descri' [·] ' equipment physical kn'` | 32 | `'ตะวันออก'` (0.431) | **correct** | 5.30 | 0.431 | -0.005 | `'ตะวันออก equipment physical kn public na'` |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `'ตะวันออก'` (0.534) | **correct** | 2.79 | 0.534 | -0.009 | `'ตะวันออก activity decl flex method crit '` |
| worst | `' expert became token record' [·] ' date decided country'` | 16 | `'ตะวันออก'` (0.550) | **correct** | 4.17 | 0.550 | -0.007 | `'ตะวันออก date decided country govern rec'` |
| best | `' called period fam life' [·] ' comple into must'` | 64 | `'ตะวันออก'` (0.990) | **correct** | 0.16 | 0.990 | 0.015 | `'ตะวันออก comple into must weeks higher f'` |

### `' 세상'` — id 199215

single-probe lp -0.006 · fragility 0.25 · mean lp -0.45 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' safe mod edit history' [·] ' seg'` | 16 | `'세'` (0.757) | **truncation** | 1.25 | 0.169 | 0.056 | `'세상 seg'` |
| worst | `' event even giving' [·] ' is route full'` | 8 | `'세'` (0.755) | **truncation** | 1.22 | 0.191 | 0.035 | `'세상은 route full last'` |
| worst | `' header coming clo users' [·] ' under si power'` | 32 | `'세'` (0.545) | **truncation** | 1.71 | 0.375 | -0.006 | `'세상 under si power define mention at'` |
| best | `' apply benef comp' [·] ' distrib below forward'` | 16 | `' 세상'` (1.000) | **correct** | 0.00 | 1.000 | 0.036 | `' 세상 distrib below forward die added engi'` |

### `' eksklus'` — id 228778

single-probe lp -0.270 · fragility 0.21 · mean lp -0.30 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' um' [·] ' parameters months cor'` | 16 | `' exklus'` (0.715) | **substitution** | 1.08 | 0.263 | 0.019 | `' exklus parameters months cor shows prob'` |
| worst | `' led act cit install' [·] ' give seek man'` | 64 | `' exklus'` (0.580) | **substitution** | 1.15 | 0.399 | 0.003 | `' exklus give seek man costs goes size pr'` |
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `' exklus'` (0.584) | **substitution** | 1.12 | 0.401 | -0.007 | `' exklus announced pro current shall pl c'` |
| best | `' industry full mark love' [·] ' where word column'` | 8 | `' eksklus'` (0.973) | **correct** | 0.23 | 0.973 | 0.016 | `' eksklus where word column'` |

### `' membut'` — id 174370

single-probe lp -0.034 · fragility 0.21 · mean lp -0.35 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `' membut'` (0.262) | **correct** | 9.10 | 0.262 | -0.040 | `' membut conditions respect word constant'` |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `' membut'` (0.381) | **correct** | 8.29 | 0.381 | 0.009 | `' membut activity decl flex method crit m'` |
| worst | `' far instead pur commit' [·] ' arguments learning bo'` | 16 | `' membut'` (0.367) | **correct** | 8.07 | 0.367 | -0.012 | `' membut arguments learning bo messages t'` |
| best | `' df' [·] ' influ address claim'` | 8 | `' membut'` (0.969) | **correct** | 0.42 | 0.969 | 0.024 | `' membut influ address claim act directio'` |

### `'ೈ'` — id 169654

single-probe lp -0.114 · fragility 0.21 · mean lp -0.47 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `'इ'` (0.095) | **substitution** | 8.04 | 0.069 | -0.045 | `'इ offers like ax ver young loss via'` |
| worst | `' on package political into' [·] ' tw exam match'` | 16 | `' tw'` (0.781) | **deletion** | 2.26 | 0.088 | -0.007 | `' tw exam match often'` |
| worst | `' home wall dat ke' [·] ' plus line lik'` | 64 | `'ਈ'` (0.282) | **substitution** | 6.43 | 0.111 | -0.021 | `'ਈ plus line lik energ playing core limit'` |
| best | `' df' [·] ' influ address claim'` | 8 | `'ೈ'` (0.997) | **correct** | 0.06 | 0.997 | -0.016 | `'ೈ influ address claim act direction deg'` |

### `' الصحف'` — id 228023

single-probe lp -0.012 · fragility 0.21 · mean lp -0.26 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' materials' [·] ' middle tot decided'` | 8 | `' صح'` (0.603) | **substitution** | 1.87 | 0.285 | -0.006 | `' صحف middle tot decided lost govern fore'` |
| worst | `' serv grow ex form' [·] ' website'` | 8 | `' الصحف'` (0.443) | **correct** | 2.65 | 0.443 | 0.044 | `' الصحف website'` |
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `' الصحف'` (0.509) | **correct** | 2.55 | 0.509 | 0.000 | `' الصحف announced pro current shall pl ch'` |
| best | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `' الصحف'` (0.997) | **correct** | 0.06 | 0.997 | -0.012 | `' الصحف offers like ax ver young loss via'` |

### `' ближайшее'` — id 203000

single-probe lp -0.019 · fragility 0.21 · mean lp -1.30 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' event even giving' [·] ' is route full'` | 8 | `' nearest'` (0.808) | **substitution** | 1.38 | 0.000 | 0.007 | `' nearest is route full last'` |
| worst | `' industry full mark love' [·] ' where word column'` | 8 | `' where'` (0.534) | **deletion** | 2.31 | 0.000 | -0.009 | `' where word column'` |
| worst | `' icon sometimes mom interview' [·] ' fac taking site'` | 32 | `' nearest'` (0.737) | **substitution** | 2.26 | 0.003 | -0.008 | `' nearest fac taking site func orig try o'` |
| best | `' post double dise rock' [·] ' announced pro current'` | 64 | `' ближайшее'` (0.998) | **correct** | 0.03 | 0.998 | -0.001 | `' ближайшее announced pro current shall p'` |

### `' โร'` — id 156266

single-probe lp -0.058 · fragility 0.21 · mean lp -0.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' home wall dat ke' [·] ' plus line lik'` | 64 | `'โร'` (0.741) | **substitution** | 1.36 | 0.212 | 0.066 | `'โร plus line lik energ playing core limi'` |
| worst | `' serv grow ex form' [·] ' website'` | 8 | `' โร'` (0.669) | **correct** | 1.56 | 0.669 | 0.072 | `' โร website'` |
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `' โร'` (0.498) | **correct** | 2.55 | 0.498 | 0.051 | `' โร conditions respect word constant day'` |
| best | `' on package political into' [·] ' tw exam match'` | 16 | `' โร'` (0.995) | **correct** | 0.08 | 0.995 | 0.053 | `' โร tw exam match often'` |

### `' الملكي'` — id 230547

single-probe lp -0.003 · fragility 0.21 · mean lp -0.28 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' far instead pur commit' [·] ' arguments learning bo'` | 16 | `' ملك'` (0.802) | **substitution** | 0.90 | 0.179 | -0.002 | `' ملكي arguments learning bo messages ter'` |
| worst | `' serv grow ex form' [·] ' website'` | 8 | `' ملك'` (0.489) | **substitution** | 1.26 | 0.489 | 0.059 | `' ملكي website'` |
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `' الملكي'` (0.525) | **correct** | 3.25 | 0.525 | -0.015 | `' الملكي announced pro current shall pl c'` |
| best | `' df' [·] ' influ address claim'` | 8 | `' الملكي'` (0.999) | **correct** | 0.01 | 0.999 | 0.000 | `' الملكي influ address claim act directio'` |

### `' investitori'` — id 246694

single-probe lp -0.032 · fragility 0.21 · mean lp -0.29 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' event even giving' [·] ' is route full'` | 8 | `' investitori'` (0.309) | **correct** | 4.14 | 0.309 | 0.036 | `' investitori is route full last'` |
| worst | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `' investitori'` (0.447) | **correct** | 3.66 | 0.447 | -0.009 | `' investitori fully inst behind areas pra'` |
| worst | `' header coming clo users' [·] ' under si power'` | 32 | `' investitori'` (0.545) | **correct** | 2.88 | 0.545 | 0.009 | `' investitori under si power define menti'` |
| best | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `' investitori'` (0.983) | **correct** | 0.18 | 0.983 | 0.007 | `' investitori conditions respect word con'` |

### `'чаем'` — id 244295

single-probe lp -0.512 · fragility 0.17 · mean lp -0.44 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `'챔'` (0.218) | **substitution** | 7.70 | 0.026 | -0.016 | `'챔 announced pro current shall pl charg d'` |
| worst | `' home wall dat ke' [·] ' plus line lik'` | 64 | `'çam'` (0.674) | **substitution** | 2.56 | 0.133 | -0.029 | `'çam plus line lik energ playing core lim'` |
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `'чаем'` (0.419) | **correct** | 4.28 | 0.419 | -0.036 | `'чаем conditions respect word constant da'` |
| best | `' materials' [·] ' middle tot decided'` | 8 | `'чаем'` (0.998) | **correct** | 0.03 | 0.998 | -0.030 | `'чаем middle tot decided lost govern fore'` |

### `' persegu'` — id 197003

single-probe lp -0.091 · fragility 0.17 · mean lp -0.33 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `' persec'` (0.435) | **substitution** | 2.41 | 0.339 | -0.025 | `' persecute fully inst behind areas pract'` |
| worst | `' led act cit install' [·] ' give seek man'` | 64 | `' persegu'` (0.445) | **correct** | 2.30 | 0.445 | 0.021 | `' persegu give seek man costs goes size p'` |
| worst | `' safe mod edit history' [·] ' seg'` | 16 | `' persegu'` (0.500) | **correct** | 2.27 | 0.501 | 0.021 | `' persegu seg'` |
| best | `' df' [·] ' influ address claim'` | 8 | `' persegu'` (0.950) | **correct** | 0.52 | 0.950 | 0.038 | `' persegu influ address claim act directi'` |

### `' територ'` — id 167221

single-probe lp -0.089 · fragility 0.17 · mean lp -0.25 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `' территор'` (0.485) | **substitution** | 1.98 | 0.378 | -0.008 | `' территор announced pro current shall pl'` |
| worst | `' icon sometimes mom interview' [·] ' fac taking site'` | 32 | `' територ'` (0.559) | **correct** | 2.32 | 0.559 | -0.005 | `' територ fac taking site func orig try o'` |
| worst | `' life dig think piece' [·] ' or company connection'` | 64 | `' територ'` (0.579) | **correct** | 1.84 | 0.579 | 0.012 | `' територ or company connection session m'` |
| best | `' materials' [·] ' middle tot decided'` | 8 | `' територ'` (0.960) | **correct** | 0.39 | 0.960 | 0.013 | `' територ middle tot decided lost govern '` |

### `' اليومية'` — id 206475

single-probe lp -0.001 · fragility 0.17 · mean lp -0.29 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' event even giving' [·] ' is route full'` | 8 | `' '` (0.834) | **substitution** | 1.29 | 0.094 | 0.031 | `' 日常 is route full last'` |
| worst | `' on package political into' [·] ' tw exam match'` | 16 | `' '` (0.687) | **substitution** | 2.49 | 0.163 | 0.006 | `' 每日 tw exam match often'` |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `' اليومية'` (0.334) | **correct** | 5.09 | 0.334 | -0.008 | `' اليومية activity decl flex method crit '` |
| best | `' df' [·] ' influ address claim'` | 8 | `' اليومية'` (1.000) | **correct** | 0.01 | 1.000 | -0.009 | `' اليومية influ address claim act directi'` |

### `' まと'` — id 207371

single-probe lp -0.096 · fragility 0.17 · mean lp -0.20 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `'まと'` (0.550) | **substitution** | 1.69 | 0.378 | -0.029 | `'まと announced pro current shall pl charg '` |
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `'まと'` (0.586) | **substitution** | 1.64 | 0.356 | 0.002 | `'まと conditions respect word constant day'` |
| worst | `' home wall dat ke' [·] ' plus line lik'` | 64 | `' まと'` (0.535) | **correct** | 1.54 | 0.535 | -0.006 | `' まと plus line lik energ playing core lim'` |
| best | `' df' [·] ' influ address claim'` | 8 | `' まと'` (0.996) | **correct** | 0.05 | 0.996 | -0.013 | `' まと influ address claim act direction de'` |

### `'ிய'` — id 153396

single-probe lp -0.389 · fragility 0.17 · mean lp -0.56 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `'ಿಯ'` (0.886) | **substitution** | 1.16 | 0.007 | -0.000 | `'ಿಯ conditions respect word constant day'` |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `'iy'` (0.574) | **substitution** | 2.72 | 0.047 | -0.008 | `'iy activity decl flex method crit mind t'` |
| worst | `' header coming clo users' [·] ' under si power'` | 32 | `'iy'` (0.604) | **substitution** | 2.89 | 0.099 | -0.002 | `'iy under si power define mention at prom'` |
| best | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `'ிய'` (0.980) | **correct** | 0.19 | 0.980 | 0.021 | `'ிய fully inst behind areas pract gr poli'` |

### `' 있었다'` — id 174011

single-probe lp -0.061 · fragility 0.17 · mean lp -0.36 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `'었다'` (0.425) | **substitution** | 2.25 | 0.037 | -0.027 | `'었다 conditions respect word constant day'` |
| worst | `' safe mod edit history' [·] ' seg'` | 16 | `'있'` (0.364) | **truncation** | 2.48 | 0.134 | 0.012 | `'있었다 seg'` |
| worst | `' re lot distributed rather' [·] ' times needed layout'` | 8 | `'있'` (0.478) | **truncation** | 1.64 | 0.422 | 0.002 | `'있었다 times needed layout'` |
| best | `' called period fam life' [·] ' comple into must'` | 64 | `' 있었다'` (0.995) | **correct** | 0.08 | 0.995 | -0.029 | `' 있었다 comple into must weeks higher fire '` |

### `' bestimm'` — id 91446

single-probe lp -0.099 · fragility 0.17 · mean lp -0.24 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `' bestem'` (0.485) | **substitution** | 2.08 | 0.378 | -0.011 | `' bestem offers like ax ver young loss vi'` |
| worst | `' power asked descri' [·] ' equipment physical kn'` | 32 | `' bestem'` (0.424) | **substitution** | 2.15 | 0.375 | 0.005 | `' bestem equipment physical kn public nat'` |
| worst | `' materials' [·] ' middle tot decided'` | 8 | `' bestimm'` (0.568) | **correct** | 1.82 | 0.568 | 0.027 | `' bestimm middle tot decided lost govern '` |
| best | `' um' [·] ' parameters months cor'` | 16 | `' bestimm'` (0.971) | **correct** | 0.34 | 0.971 | 0.016 | `' bestimm parameters months cor shows pro'` |

### `' 웹사이트가'` — id 220550

single-probe lp -0.170 · fragility 0.17 · mean lp -0.26 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `' 웹사이트가'` (0.396) | **correct** | 3.63 | 0.396 | -0.018 | `' 웹사이트가 conditions respect word constant '` |
| worst | `' life dig think piece' [·] ' or company connection'` | 64 | `' 웹사이트가'` (0.466) | **correct** | 3.92 | 0.466 | 0.025 | `' 웹사이트가 or company connection session mak'` |
| worst | `' led act cit install' [·] ' give seek man'` | 64 | `' 웹사이트가'` (0.572) | **correct** | 3.22 | 0.572 | -0.012 | `' 웹사이트가 give seek man costs goes size pro'` |
| best | `' df' [·] ' influ address claim'` | 8 | `' 웹사이트가'` (0.981) | **correct** | 0.16 | 0.981 | -0.022 | `' 웹사이트가 influ address claim act direction'` |

### `'ನ್ನ'` — id 161539

single-probe lp -0.125 · fragility 0.12 · mean lp -0.26 · canonical False · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `'ன்ன'` (0.324) | **substitution** | 6.96 | 0.119 | -0.027 | `'ன்ன activity decl flex method crit mind '` |
| worst | `' power asked descri' [·] ' equipment physical kn'` | 32 | `'ன்ன'` (0.420) | **substitution** | 4.25 | 0.271 | -0.003 | `'ன்ன equipment physical kn public natural'` |
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `'ನ್ನ'` (0.346) | **correct** | 4.65 | 0.346 | -0.018 | `'ನ್ನ offers like ax ver young loss via'` |
| best | `' df' [·] ' influ address claim'` | 8 | `'ನ್ನ'` (0.991) | **correct** | 0.13 | 0.991 | 0.024 | `'ನ್ನ influ address claim act direction de'` |

### `' ricette'` — id 211396

single-probe lp -0.010 · fragility 0.12 · mean lp -0.15 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' power asked descri' [·] ' equipment physical kn'` | 32 | `' ricette'` (0.337) | **correct** | 2.64 | 0.336 | 0.002 | `' ricette equipment physical kn public na'` |
| worst | `' event even giving' [·] ' is route full'` | 8 | `' ricette'` (0.462) | **correct** | 2.29 | 0.462 | 0.044 | `' ricette is route full last'` |
| worst | `' industry full mark love' [·] ' where word column'` | 8 | `' ricette'` (0.580) | **correct** | 1.83 | 0.580 | 0.020 | `' ricette where word column'` |
| best | `' home wall dat ke' [·] ' plus line lik'` | 64 | `' ricette'` (0.995) | **correct** | 0.07 | 0.995 | -0.018 | `' ricette plus line lik energ playing cor'` |

### `' 주도'` — id 207225

single-probe lp -0.014 · fragility 0.12 · mean lp -0.26 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' home wall dat ke' [·] ' plus line lik'` | 64 | `'주'` (0.900) | **truncation** | 0.71 | 0.074 | 0.012 | `'주도 plus line lik energ playing core'` |
| worst | `' event even giving' [·] ' is route full'` | 8 | `'주'` (0.879) | **truncation** | 0.69 | 0.105 | 0.032 | `'주도 is route full last'` |
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `'주'` (0.824) | **truncation** | 0.82 | 0.162 | 0.020 | `'주도 announced pro current shall pl charg'` |
| best | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `' 주도'` (0.998) | **correct** | 0.03 | 0.998 | 0.005 | `' 주도 fully inst behind areas pract gr pol'` |

### `'лага'` — id 150062

single-probe lp -0.549 · fragility 0.12 · mean lp -0.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' safe mod edit history' [·] ' seg'` | 16 | `'ला'` (0.306) | **substitution** | 3.26 | 0.164 | 0.041 | `'लाग seg'` |
| worst | `' far instead pur commit' [·] ' arguments learning bo'` | 16 | `'лага'` (0.447) | **correct** | 3.24 | 0.447 | -0.015 | `'лага arguments learning bo messages terr'` |
| worst | `' um' [·] ' parameters months cor'` | 16 | `'лага'` (0.612) | **correct** | 2.00 | 0.612 | -0.020 | `'лага parameters months cor shows proble '` |
| best | `' df' [·] ' influ address claim'` | 8 | `'лага'` (0.999) | **correct** | 0.01 | 0.999 | 0.003 | `'лага influ address claim act direction d'` |

### `' стране'` — id 176203

single-probe lp -0.010 · fragility 0.12 · mean lp -0.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `' страны'` (0.588) | **substitution** | 1.09 | 0.404 | -0.021 | `' страны announced pro current shall pl c'` |
| worst | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `' страны'` (0.559) | **substitution** | 1.06 | 0.436 | -0.019 | `' страны fully inst behind areas pract gr'` |
| worst | `' home wall dat ke' [·] ' plus line lik'` | 64 | `' стране'` (0.566) | **correct** | 1.31 | 0.566 | 0.002 | `' стране plus line lik energ playing core'` |
| best | `' df' [·] ' influ address claim'` | 8 | `' стране'` (0.995) | **correct** | 0.06 | 0.995 | 0.006 | `' стране influ address claim act directio'` |

### `' удобным'` — id 243621

single-probe lp -0.004 · fragility 0.12 · mean lp -0.34 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' called period fam life' [·] ' comple into must'` | 64 | `' удобный'` (0.986) | **substitution** | 0.15 | 0.005 | -0.016 | `' удобный comple into must weeks higher f'` |
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `' удобный'` (0.734) | **substitution** | 1.21 | 0.210 | -0.008 | `' удобный conditions respect word constan'` |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `' удобный'` (0.496) | **substitution** | 1.10 | 0.496 | -0.008 | `' удобный activity decl flex method crit '` |
| best | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `' удобным'` (0.997) | **correct** | 0.05 | 0.997 | -0.019 | `' удобным offers like ax ver young loss v'` |

### `' bertem'` — id 181347

single-probe lp -0.188 · fragility 0.12 · mean lp -0.30 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `' bertem'` (0.369) | **correct** | 7.47 | 0.369 | -0.022 | `' bertem activity decl flex method crit m'` |
| worst | `' re lot distributed rather' [·] ' times needed layout'` | 8 | `' bertem'` (0.415) | **correct** | 6.46 | 0.415 | -0.020 | `' bertem times needed layout'` |
| worst | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `' bertem'` (0.537) | **correct** | 5.55 | 0.537 | -0.027 | `' bertem fully inst behind areas pract gr'` |
| best | `' materials' [·] ' middle tot decided'` | 8 | `' bertem'` (0.981) | **correct** | 0.34 | 0.981 | 0.005 | `' bertem middle tot decided lost govern f'` |

### `' todėl'` — id 228762

single-probe lp -0.105 · fragility 0.12 · mean lp -0.16 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `' therefore'` (0.431) | **substitution** | 3.03 | 0.296 | -0.023 | `' therefore announced pro current shall p'` |
| worst | `' led act cit install' [·] ' give seek man'` | 64 | `' todėl'` (0.406) | **correct** | 3.07 | 0.406 | -0.061 | `' todėl give seek man costs goes size pro'` |
| worst | `' life dig think piece' [·] ' or company connection'` | 64 | `' todėl'` (0.608) | **correct** | 2.26 | 0.608 | 0.007 | `' todėl or company connection session mak'` |
| best | `' safe mod edit history' [·] ' seg'` | 16 | `' todėl'` (0.996) | **correct** | 0.06 | 0.996 | 0.032 | `' todėl seg'` |

### `' costantemente'` — id 229162

single-probe lp -0.239 · fragility 0.12 · mean lp -0.35 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' re lot distributed rather' [·] ' times needed layout'` | 8 | `' constantemente'` (0.636) | **substitution** | 1.50 | 0.234 | -0.002 | `' constantemente times needed layout'` |
| worst | `' on package political into' [·] ' tw exam match'` | 16 | `' constantemente'` (0.628) | **substitution** | 1.29 | 0.297 | -0.010 | `' constantemente tw exam match often'` |
| worst | `' um' [·] ' parameters months cor'` | 16 | `' constantemente'` (0.611) | **substitution** | 1.09 | 0.370 | 0.008 | `' constantemente parameters months cor sh'` |
| best | `' post double dise rock' [·] ' announced pro current'` | 64 | `' costantemente'` (0.912) | **correct** | 0.46 | 0.912 | 0.015 | `' costantemente announced pro current sha'` |

### `'ренда'` — id 234472

single-probe lp -0.120 · fragility 0.08 · mean lp -0.10 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' apply benef comp' [·] ' distrib below forward'` | 16 | `'rensa'` (0.658) | **substitution** | 1.41 | 0.274 | -0.043 | `'rensa distrib below forward die added en'` |
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `'ренда'` (0.585) | **correct** | 2.76 | 0.585 | -0.019 | `'ренда conditions respect word constant d'` |
| worst | `' power asked descri' [·] ' equipment physical kn'` | 32 | `'ренда'` (0.833) | **correct** | 1.54 | 0.833 | -0.008 | `'ренда equipment physical kn public natur'` |
| best | `' expert became token record' [·] ' date decided country'` | 16 | `'ренда'` (0.999) | **correct** | 0.02 | 0.999 | -0.007 | `'ренда date decided country govern recent'` |

### `' объемом'` — id 232917

single-probe lp -0.012 · fragility 0.08 · mean lp -0.22 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' re lot distributed rather' [·] ' times needed layout'` | 8 | `' объемом'` (0.163) | **correct** | 9.80 | 0.163 | -0.006 | `' объемом times needed layout'` |
| worst | `' power asked descri' [·] ' equipment physical kn'` | 32 | `' объемом'` (0.303) | **correct** | 9.16 | 0.303 | -0.015 | `' объемом equipment physical kn public na'` |
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `' объемом'` (0.762) | **correct** | 3.30 | 0.762 | 0.003 | `' объемом offers like ax ver young loss v'` |
| best | `' called period fam life' [·] ' comple into must'` | 64 | `' объемом'` (0.962) | **correct** | 0.43 | 0.962 | 0.002 | `' объемом comple into must weeks higher f'` |

### `' WebDriver'` — id 40515

single-probe lp -0.010 · fragility 0.08 · mean lp -0.10 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' on package political into' [·] ' tw exam match'` | 16 | `'WebDriver'` (0.864) | **substitution** | 0.60 | 0.133 | 0.050 | `'WebDriver tw exam match often'` |
| worst | `' home wall dat ke' [·] ' plus line lik'` | 64 | `'WebDriver'` (0.834) | **substitution** | 0.67 | 0.164 | 0.017 | `'WebDriver plus line lik energ playing co'` |
| worst | `' life dig think piece' [·] ' or company connection'` | 64 | `' WebDriver'` (0.934) | **correct** | 0.39 | 0.934 | 0.090 | `' WebDriver or company connection session'` |
| best | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `' WebDriver'` (1.000) | **correct** | 0.00 | 1.000 | 0.041 | `' WebDriver fully inst behind areas pract'` |

### `' sindac'` — id 196948

single-probe lp -0.050 · fragility 0.08 · mean lp -0.19 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' expert became token record' [·] ' date decided country'` | 16 | `' sindac'` (0.382) | **correct** | 2.04 | 0.382 | 0.023 | `' sindac date decided country govern rece'` |
| worst | `' life dig think piece' [·] ' or company connection'` | 64 | `' sindac'` (0.467) | **correct** | 1.83 | 0.467 | 0.036 | `' sindac or company connection session ma'` |
| worst | `' called period fam life' [·] ' comple into must'` | 64 | `' sindac'` (0.739) | **correct** | 1.50 | 0.739 | 0.009 | `' sindac comple into must weeks higher fi'` |
| best | `' far instead pur commit' [·] ' arguments learning bo'` | 16 | `' sindac'` (0.980) | **correct** | 0.21 | 0.980 | 0.029 | `' sindac arguments learning bo messages t'` |

### `' cambiato'` — id 224214

single-probe lp -0.097 · fragility 0.08 · mean lp -0.14 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `' cambi'` (0.466) | **truncation** | 2.74 | 0.249 | -0.030 | `' cambiata offers like ax ver young loss'` |
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `' cambiato'` (0.515) | **correct** | 1.95 | 0.515 | 0.008 | `' cambiato conditions respect word consta'` |
| worst | `' materials' [·] ' middle tot decided'` | 8 | `' cambiato'` (0.737) | **correct** | 2.03 | 0.737 | 0.022 | `' cambiato middle tot decided lost govern'` |
| best | `' icon sometimes mom interview' [·] ' fac taking site'` | 32 | `' cambiato'` (0.997) | **correct** | 0.04 | 0.997 | 0.032 | `' cambiato fac taking site func orig try '` |

### `' présentent'` — id 228465

single-probe lp -0.029 · fragility 0.08 · mean lp -0.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' event even giving' [·] ' is route full'` | 8 | `' présent'` (0.417) | **truncation** | 2.63 | 0.368 | 0.035 | `' présent is route full last'` |
| worst | `' expert became token record' [·] ' date decided country'` | 16 | `' présentent'` (0.543) | **correct** | 1.84 | 0.543 | -0.013 | `' présentent date decided country govern '` |
| worst | `' on package political into' [·] ' tw exam match'` | 16 | `' présentent'` (0.686) | **correct** | 1.75 | 0.686 | 0.004 | `' présentent tw exam match often'` |
| best | `' post double dise rock' [·] ' announced pro current'` | 64 | `' présentent'` (0.989) | **correct** | 0.13 | 0.989 | -0.016 | `' présentent announced pro current shall '` |

### `' الخدمة'` — id 185367

single-probe lp -0.119 · fragility 0.08 · mean lp -0.12 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `'ال'` (0.482) | **truncation** | 1.83 | 0.331 | 0.003 | `'الخدمة offers like ax ver young loss'` |
| worst | `' home wall dat ke' [·] ' plus line lik'` | 64 | `'ال'` (0.492) | **truncation** | 1.47 | 0.434 | 0.019 | `'الخدمة plus line lik energ playing core'` |
| worst | `' on package political into' [·] ' tw exam match'` | 16 | `' الخدمة'` (0.751) | **correct** | 1.03 | 0.751 | 0.038 | `' الخدمة tw exam match often'` |
| best | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `' الخدمة'` (0.999) | **correct** | 0.02 | 0.999 | -0.002 | `' الخدمة fully inst behind areas pract gr'` |

### `' конкурса'` — id 185790

single-probe lp -0.004 · fragility 0.08 · mean lp -0.11 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' called period fam life' [·] ' comple into must'` | 64 | `' конкурс'` (0.604) | **truncation** | 1.23 | 0.366 | 0.006 | `' конкурс comple into must weeks higher f'` |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `' конкурса'` (0.629) | **correct** | 1.28 | 0.629 | 0.017 | `' конкурса activity decl flex method crit'` |
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `' конкурса'` (0.645) | **correct** | 1.39 | 0.645 | 0.020 | `' конкурса conditions respect word consta'` |
| best | `' home wall dat ke' [·] ' plus line lik'` | 64 | `' конкурса'` (0.998) | **correct** | 0.03 | 0.998 | 0.021 | `' конкурса plus line lik energ playing co'` |

### `' dichiarazioni'` — id 221213

single-probe lp -0.009 · fragility 0.08 · mean lp -0.15 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' life dig think piece' [·] ' or company connection'` | 64 | `' dichiarazione'` (0.772) | **substitution** | 0.85 | 0.221 | 0.035 | `' dichiarazione or company connection ses'` |
| worst | `' header coming clo users' [·] ' under si power'` | 32 | `' dichiarazione'` (0.667) | **substitution** | 1.09 | 0.315 | 0.021 | `' dichiarazione under si power define men'` |
| worst | `' safe mod edit history' [·] ' seg'` | 16 | `' dichiarazioni'` (0.660) | **correct** | 1.22 | 0.660 | 0.039 | `' dichiarazioni seg'` |
| best | `' expert became token record' [·] ' date decided country'` | 16 | `' dichiarazioni'` (0.999) | **correct** | 0.02 | 0.999 | 0.011 | `' dichiarazioni date decided country gove'` |

### `' ежегодно'` — id 217214

single-probe lp -0.009 · fragility 0.08 · mean lp -0.78 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' re lot distributed rather' [·] ' times needed layout'` | 8 | `' annually'` (1.000) | **substitution** | 0.00 | 0.000 | 0.032 | `' annually times needed layout'` |
| worst | `' event even giving' [·] ' is route full'` | 8 | `' annually'` (0.998) | **substitution** | 0.03 | 0.000 | 0.053 | `' annually is route full last'` |
| worst | `' on package political into' [·] ' tw exam match'` | 16 | `' ежегодно'` (0.932) | **correct** | 0.37 | 0.932 | 0.031 | `' ежегодно tw exam match often'` |
| best | `' led act cit install' [·] ' give seek man'` | 64 | `' ежегодно'` (0.998) | **correct** | 0.03 | 0.998 | 0.011 | `' ежегодно give seek man costs goes size '` |

### `' kokemuks'` — id 230321

single-probe lp -0.019 · fragility 0.08 · mean lp -0.27 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `' kokemuks'` (0.368) | **correct** | 4.72 | 0.368 | -0.034 | `' kokemuks conditions respect word consta'` |
| worst | `' led act cit install' [·] ' give seek man'` | 64 | `' kokemuks'` (0.626) | **correct** | 3.72 | 0.626 | -0.047 | `' kokemuks give seek man costs goes size '` |
| worst | `' apply benef comp' [·] ' distrib below forward'` | 16 | `' kokemuks'` (0.667) | **correct** | 4.35 | 0.667 | -0.030 | `' kokemuks distrib below forward die adde'` |
| best | `' df' [·] ' influ address claim'` | 8 | `' kokemuks'` (0.956) | **correct** | 0.61 | 0.956 | -0.008 | `' kokemuks influ address claim act direct'` |

### `' Китая'` — id 213001

single-probe lp -0.068 · fragility 0.08 · mean lp -0.22 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' industry full mark love' [·] ' where word column'` | 8 | `' China'` (0.370) | **substitution** | 3.57 | 0.239 | -0.005 | `' China where word column'` |
| worst | `' re lot distributed rather' [·] ' times needed layout'` | 8 | `' Китая'` (0.501) | **correct** | 2.76 | 0.501 | 0.021 | `' Китая times needed layout'` |
| worst | `' power asked descri' [·] ' equipment physical kn'` | 32 | `' Китая'` (0.630) | **correct** | 1.84 | 0.630 | 0.018 | `' Китая equipment physical kn public natu'` |
| best | `' um' [·] ' parameters months cor'` | 16 | `' Китая'` (0.998) | **correct** | 0.03 | 0.998 | -0.007 | `' Китая parameters months cor shows probl'` |

### `' تحد'` — id 159888

single-probe lp -0.072 · fragility 0.08 · mean lp -0.17 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' safe mod edit history' [·] ' seg'` | 16 | `'ت'` (0.807) | **truncation** | 1.08 | 0.140 | 0.032 | `'تحد seg'` |
| worst | `' home wall dat ke' [·] ' plus line lik'` | 64 | `'ت'` (0.730) | **truncation** | 1.62 | 0.196 | -0.006 | `'تحد plus line lik energ playing core'` |
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `' تحد'` (0.659) | **correct** | 1.30 | 0.659 | -0.007 | `' تحد announced pro current shall pl char'` |
| best | `' um' [·] ' parameters months cor'` | 16 | `' تحد'` (0.998) | **correct** | 0.04 | 0.998 | 0.015 | `' تحد parameters months cor shows proble '` |

### `' النقد'` — id 195827

single-probe lp -0.003 · fragility 0.08 · mean lp -0.11 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `' نقد'` (0.791) | **substitution** | 0.85 | 0.200 | -0.015 | `' نقد announced pro current shall pl char'` |
| worst | `' serv grow ex form' [·] ' website'` | 8 | `' النقد'` (0.650) | **correct** | 0.97 | 0.650 | 0.059 | `' النقد website'` |
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `' النقد'` (0.951) | **correct** | 0.31 | 0.951 | -0.002 | `' النقد conditions respect word constant '` |
| best | `' constructor ist comp environment' [·] ' fully inst behind'` | 32 | `' النقد'` (1.000) | **correct** | 0.00 | 1.000 | -0.001 | `' النقد fully inst behind areas pract gr '` |

### `' الهي'` — id 161542

single-probe lp -0.234 · fragility 0.08 · mean lp -0.25 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' post double dise rock' [·] ' announced pro current'` | 64 | `' هي'` (0.441) | **substitution** | 2.13 | 0.343 | -0.003 | `' هي announced pro current shall pl charg'` |
| worst | `' um' [·] ' parameters months cor'` | 16 | `' الهي'` (0.551) | **correct** | 1.21 | 0.551 | 0.004 | `' الهي parameters months cor shows proble'` |
| worst | `' serv grow ex form' [·] ' website'` | 8 | `' الهي'` (0.641) | **correct** | 1.13 | 0.641 | 0.045 | `' الهي website'` |
| best | `' re lot distributed rather' [·] ' times needed layout'` | 8 | `' الهي'` (0.957) | **correct** | 0.33 | 0.957 | 0.022 | `' الهي times needed layout'` |

### `'кладки'` — id 162926

single-probe lp -0.517 · fragility 0.08 · mean lp -0.10 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' home wall dat ke' [·] ' plus line lik'` | 64 | `'klad'` (0.376) | **substitution** | 2.82 | 0.331 | -0.018 | `'kladkas plus line lik energ playing core'` |
| worst | `' um' [·] ' parameters months cor'` | 16 | `'кладки'` (0.513) | **correct** | 1.74 | 0.513 | -0.017 | `'кладки parameters months cor shows probl'` |
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `'кладки'` (0.827) | **correct** | 1.44 | 0.827 | -0.019 | `'кладки offers like ax ver young loss via'` |
| best | `' re lot distributed rather' [·] ' times needed layout'` | 8 | `'кладки'` (1.000) | **correct** | 0.01 | 1.000 | 0.014 | `'кладки times needed layout'` |

### `' أخر'` — id 215613

single-probe lp -0.126 · fragility 0.08 · mean lp -0.28 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | 64 | `' أخر'` (0.582) | **correct** | 2.10 | 0.582 | -0.016 | `' أخر conditions respect word constant da'` |
| worst | `' power asked descri' [·] ' equipment physical kn'` | 32 | `' أخر'` (0.616) | **correct** | 1.68 | 0.616 | -0.007 | `' أخر equipment physical kn public natura'` |
| worst | `' apply benef comp' [·] ' distrib below forward'` | 16 | `' أخر'` (0.642) | **correct** | 1.31 | 0.643 | -0.012 | `' أخر distrib below forward die added eng'` |
| best | `' life dig think piece' [·] ' or company connection'` | 64 | `' أخر'` (0.903) | **correct** | 0.64 | 0.903 | 0.002 | `' أخر or company connection session makes'` |

### `' фруктов'` — id 214915

single-probe lp -0.002 · fragility 0.08 · mean lp -0.35 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' industry full mark love' [·] ' where word column'` | 8 | `' fruits'` (0.973) | **substitution** | 0.26 | 0.001 | 0.021 | `' fruits where word column'` |
| worst | `' called period fam life' [·] ' comple into must'` | 64 | `' frutas'` (0.400) | **substitution** | 3.02 | 0.275 | -0.017 | `' frutas comple into must weeks higher fi'` |
| worst | `' far instead pur commit' [·] ' arguments learning bo'` | 16 | `' фруктов'` (0.924) | **correct** | 0.47 | 0.924 | 0.010 | `' фруктов arguments learning bo messages '` |
| best | `' um' [·] ' parameters months cor'` | 16 | `' фруктов'` (0.999) | **correct** | 0.01 | 0.999 | -0.003 | `' фруктов parameters months cor shows pro'` |

### `' Copies'` — id 76827

single-probe lp -0.373 · fragility 0.08 · mean lp -0.12 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' industry full mark love' [·] ' where word column'` | 8 | `' Copies'` (0.522) | **correct** | 1.19 | 0.522 | 0.072 | `' Copies where word column'` |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `' Copies'` (0.608) | **correct** | 1.17 | 0.608 | 0.054 | `' Copies activity decl flex method crit m'` |
| worst | `' called period fam life' [·] ' comple into must'` | 64 | `' Copies'` (0.705) | **correct** | 0.90 | 0.705 | 0.002 | `' Copies comple into must weeks higher fi'` |
| best | `' icon sometimes mom interview' [·] ' fac taking site'` | 32 | `' Copies'` (0.999) | **correct** | 0.02 | 0.999 | 0.032 | `' Copies fac taking site func orig try os'` |

### `'ார'` — id 178568

single-probe lp -0.610 · fragility 0.08 · mean lp -0.30 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | 32 | `'ාර'` (0.779) | **substitution** | 2.79 | 0.014 | -0.044 | `'ාර offers like ax ver young loss via'` |
| worst | `' apply benef comp' [·] ' distrib below forward'` | 16 | `'ார'` (0.510) | **correct** | 3.68 | 0.510 | 0.011 | `'ார distrib below forward die added engin'` |
| worst | `' on package political into' [·] ' tw exam match'` | 16 | `'ார'` (0.642) | **correct** | 2.85 | 0.642 | 0.028 | `'ார tw exam match often'` |
| best | `' df' [·] ' influ address claim'` | 8 | `'ார'` (0.996) | **correct** | 0.07 | 0.996 | 0.005 | `'ார influ address claim act direction deg'` |

### `' unseres'` — id 183207

single-probe lp -0.007 · fragility 0.08 · mean lp -0.11 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' re lot distributed rather' [·] ' times needed layout'` | 8 | `' unseres'` (0.583) | **correct** | 2.85 | 0.583 | 0.031 | `' unseres times needed layout'` |
| worst | `' home wall dat ke' [·] ' plus line lik'` | 64 | `' unseres'` (0.571) | **correct** | 2.73 | 0.571 | 0.003 | `' unseres plus line lik energ playing cor'` |
| worst | `' power asked descri' [·] ' equipment physical kn'` | 32 | `' unseres'` (0.642) | **correct** | 3.45 | 0.642 | 0.003 | `' unseres equipment physical kn public na'` |
| best | `' safe mod edit history' [·] ' seg'` | 16 | `' unseres'` (0.998) | **correct** | 0.02 | 0.998 | 0.038 | `' unseres seg'` |

### `' رو'` — id 172919

single-probe lp -0.088 · fragility 0.08 · mean lp -0.28 · canonical True · Magikarp `?`

| cell | context (…left [·] right…) | K | emitted (p) | mode | H bits | p(target) | ret | continuation |
|---|---|---|---|---|---|---|---|---|
| worst | `' called period fam life' [·] ' comple into must'` | 64 | `'رو'` (0.960) | **substitution** | 0.39 | 0.021 | 0.012 | `'رو comple into must weeks higher fire kn'` |
| worst | `' safe mod edit history' [·] ' seg'` | 16 | `'رو'` (0.844) | **substitution** | 0.98 | 0.061 | 0.037 | `'رو seg'` |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | 32 | `' رو'` (0.950) | **correct** | 0.50 | 0.950 | 0.028 | `' رو activity decl flex method crit mind '` |
| best | `' re lot distributed rather' [·] ' times needed layout'` | 8 | `' رو'` (1.000) | **correct** | 0.01 | 1.000 | 0.048 | `' رو times needed layout'` |

## Verified glitch tokens (reference)


### `' ForCanBeConverted'` — id 76549  (single-probe lp -25.07, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' constructor ist comp environment' [·] ' fully inst behind'` | `' fully'` (1.000) | deletion | 0.00 | 0.000 | -0.065 |
| worst | `' far instead pur commit' [·] ' arguments learning bo'` | `' arguments'` (1.000) | deletion | 0.00 | 0.000 | -0.030 |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | `' activity'` (1.000) | deletion | 0.01 | 0.000 | -0.061 |
| best | `' mult feed lock writ' [·] ' conditions respect word'` | `' backpage'` (0.112) | substitution | 10.39 | 0.000 | -0.068 |

### `'echslungs'` — id 221404  (single-probe lp -22.97, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' life dig think piece' [·] ' or company connection'` | `'arnings'` (0.957) | substitution | 0.44 | 0.000 | -0.028 |
| worst | `' home wall dat ke' [·] ' plus line lik'` | `'arnings'` (0.828) | substitution | 1.00 | 0.000 | -0.094 |
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | `'arnings'` (0.974) | substitution | 0.27 | 0.000 | -0.114 |
| best | `' header coming clo users' [·] ' under si power'` | `'arnings'` (0.386) | substitution | 3.78 | 0.000 | -0.063 |

### `' szexf'` — id 214963  (single-probe lp -16.50, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' industry full mark love' [·] ' where word column'` | `' where'` (0.895) | deletion | 1.38 | 0.000 | -0.046 |
| worst | `' um' [·] ' parameters months cor'` | `' parameters'` (0.870) | deletion | 1.62 | 0.000 | -0.063 |
| worst | `' life dig think piece' [·] ' or company connection'` | `' or'` (0.825) | deletion | 2.42 | 0.000 | -0.016 |
| best | `' post double dise rock' [·] ' announced pro current'` | `' announced'` (0.040) | deletion | 12.64 | 0.000 | -0.073 |

### `' Несмотр'` — id 183442  (single-probe lp -15.53, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' led act cit install' [·] ' give seek man'` | `'不折不扣'` (0.164) | substitution | 6.58 | 0.004 | -0.057 |
| worst | `' power asked descri' [·] ' equipment physical kn'` | `'不折不扣'` (0.056) | substitution | 10.15 | 0.004 | -0.023 |
| worst | `' constructor ist comp environment' [·] ' fully inst behind'` | `'不折不扣'` (0.396) | substitution | 6.18 | 0.004 | -0.062 |
| best | `' safe mod edit history' [·] ' seg'` | `' Несмотр'` (0.328) | correct | 8.27 | 0.328 | -0.038 |

### `' долгове'` — id 202834  (single-probe lp -15.29, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' event even giving' [·] ' is route full'` | `' '` (0.174) | substitution | 8.85 | 0.000 | 0.006 |
| worst | `' constructor ist comp environment' [·] ' fully inst behind'` | `' '` (0.350) | substitution | 7.76 | 0.000 | -0.084 |
| worst | `' far instead pur commit' [·] ' arguments learning bo'` | `' '` (0.092) | substitution | 9.48 | 0.000 | -0.075 |
| best | `' df' [·] ' influ address claim'` | `' долгосро'` (0.095) | substitution | 10.70 | 0.001 | -0.062 |

### `'<|im_start|>'` — id 248045  (single-probe lp -14.76, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' mult feed lock writ' [·] ' conditions respect word'` | `'+'` (0.444) | substitution | 3.01 | 0.000 | 0.005 |
| worst | `' um' [·] ' parameters months cor'` | `' parameters'` (0.471) | deletion | 1.60 | 0.000 | 0.040 |
| worst | `' power asked descri' [·] ' equipment physical kn'` | `' equipment'` (0.681) | deletion | 2.09 | 0.000 | 0.006 |
| best | `' expert became token record' [·] ' date decided country'` | `' date'` (0.743) | deletion | 2.33 | 0.001 | 0.009 |

### `'@index'` — id 44238  (single-probe lp -14.61, fragility 0.04)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' auto comm it sl' [·] ' activity decl flex'` | `'@index'` (0.437) | correct | 1.65 | 0.437 | -0.019 |
| worst | `' power asked descri' [·] ' equipment physical kn'` | `'@index'` (0.956) | correct | 0.43 | 0.956 | -0.040 |
| worst | `' expert became token record' [·] ' date decided country'` | `'@index'` (0.978) | correct | 0.25 | 0.978 | -0.032 |
| best | `' life dig think piece' [·] ' or company connection'` | `'@index'` (0.999) | correct | 0.02 | 0.999 | -0.020 |

### `'снове'` — id 244397  (single-probe lp -14.13, fragility 0.96)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' life dig think piece' [·] ' or company connection'` | `'dessen'` (0.082) | substitution | 9.97 | 0.006 | -0.012 |
| worst | `' auto comm it sl' [·] ' activity decl flex'` | `'slave'` (0.108) | substitution | 8.70 | 0.011 | -0.051 |
| worst | `' apply benef comp' [·] ' distrib below forward'` | `'sdale'` (0.042) | substitution | 10.28 | 0.019 | -0.066 |
| best | `' materials' [·] ' middle tot decided'` | `'снове'` (0.734) | correct | 3.58 | 0.734 | -0.013 |

### `' โรงแรมบรรยากาศ'` — id 236333  (single-probe lp -13.33, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' life dig think piece' [·] ' or company connection'` | `' atmosphere'` (0.235) | substitution | 7.18 | 0.000 | -0.035 |
| worst | `' expert became token record' [·] ' date decided country'` | `' atmosphere'` (0.233) | substitution | 7.81 | 0.000 | -0.034 |
| worst | `' industry full mark love' [·] ' where word column'` | `' atmosphere'` (0.215) | substitution | 8.75 | 0.000 | -0.033 |
| best | `' home wall dat ke' [·] ' plus line lik'` | `' atmosfera'` (0.048) | substitution | 10.78 | 0.000 | -0.078 |

### `'(stypy'` — id 93656  (single-probe lp -13.16, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' industry full mark love' [·] ' where word column'` | `'<|im_end|>'` (0.758) | substitution | 2.86 | 0.026 | 0.003 |
| worst | `' on package political into' [·] ' tw exam match'` | `'<|im_end|>'` (0.675) | substitution | 3.46 | 0.052 | 0.003 |
| worst | `' led act cit install' [·] ' give seek man'` | `'(st'` (0.499) | truncation | 4.98 | 0.082 | -0.018 |
| best | `' expert became token record' [·] ' date decided country'` | `'(stypy'` (0.369) | correct | 6.37 | 0.369 | -0.018 |

### `'$category'` — id 89283  (single-probe lp -12.68, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' auto comm it sl' [·] ' activity decl flex'` | `'$category'` (0.968) | correct | 0.39 | 0.968 | -0.002 |
| worst | `' power asked descri' [·] ' equipment physical kn'` | `'$category'` (0.975) | correct | 0.39 | 0.975 | 0.007 |
| worst | `' on package political into' [·] ' tw exam match'` | `'$category'` (0.980) | correct | 0.21 | 0.980 | 0.025 |
| best | `' df' [·] ' influ address claim'` | `'$category'` (1.000) | correct | 0.01 | 1.000 | 0.004 |

### `'.xticks'` — id 79164  (single-probe lp -12.49, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' far instead pur commit' [·] ' arguments learning bo'` | `'.xticks'` (0.989) | correct | 0.14 | 0.989 | -0.010 |
| worst | `' deb wasn ball inde' [·] ' offers like ax'` | `'.xticks'` (0.990) | correct | 0.12 | 0.990 | -0.039 |
| worst | `' industry full mark love' [·] ' where word column'` | `'.xticks'` (0.994) | correct | 0.08 | 0.994 | -0.008 |
| best | `' materials' [·] ' middle tot decided'` | `'.xticks'` (1.000) | correct | 0.01 | 1.000 | -0.004 |

### `'.byId'` — id 90547  (single-probe lp -12.46, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' industry full mark love' [·] ' where word column'` | `'.byId'` (0.996) | correct | 0.06 | 0.996 | -0.026 |
| worst | `' um' [·] ' parameters months cor'` | `'.byId'` (1.000) | correct | 0.01 | 1.000 | -0.013 |
| worst | `' on package political into' [·] ' tw exam match'` | `'.byId'` (1.000) | correct | 0.01 | 1.000 | -0.003 |
| best | `' apply benef comp' [·] ' distrib below forward'` | `'.byId'` (1.000) | correct | 0.00 | 1.000 | -0.011 |

### `'.icon'` — id 20360  (single-probe lp -12.31, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' expert became token record' [·] ' date decided country'` | `'.icon'` (0.998) | correct | 0.02 | 0.998 | -0.047 |
| worst | `' serv grow ex form' [·] ' website'` | `'.icon'` (0.999) | correct | 0.02 | 0.999 | 0.008 |
| worst | `' industry full mark love' [·] ' where word column'` | `'.icon'` (0.999) | correct | 0.02 | 0.999 | -0.041 |
| best | `' apply benef comp' [·] ' distrib below forward'` | `'.icon'` (1.000) | correct | 0.00 | 1.000 | -0.014 |

### `'.fromFunction'` — id 83194  (single-probe lp -12.28, fragility 0.04)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' industry full mark love' [·] ' where word column'` | `'.fromFunction'` (0.565) | correct | 1.41 | 0.565 | 0.028 |
| worst | `' called period fam life' [·] ' comple into must'` | `'.fromFunction'` (0.844) | correct | 0.82 | 0.844 | -0.017 |
| worst | `' far instead pur commit' [·] ' arguments learning bo'` | `'.fromFunction'` (0.926) | correct | 0.62 | 0.926 | 0.030 |
| best | `' apply benef comp' [·] ' distrib below forward'` | `'.fromFunction'` (1.000) | correct | 0.01 | 1.000 | -0.011 |

### `'.syn'` — id 34068  (single-probe lp -12.21, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' industry full mark love' [·] ' where word column'` | `'.syn'` (0.999) | correct | 0.02 | 0.999 | 0.001 |
| worst | `' on package political into' [·] ' tw exam match'` | `'.syn'` (0.999) | correct | 0.02 | 0.999 | -0.016 |
| worst | `' post double dise rock' [·] ' announced pro current'` | `'.syn'` (0.999) | correct | 0.02 | 0.999 | -0.035 |
| best | `' home wall dat ke' [·] ' plus line lik'` | `'.syn'` (1.000) | correct | 0.00 | 1.000 | 0.014 |

### `'_ET'` — id 71247  (single-probe lp -12.00, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' industry full mark love' [·] ' where word column'` | `'_ET'` (0.999) | correct | 0.01 | 0.999 | 0.046 |
| worst | `' event even giving' [·] ' is route full'` | `'_ET'` (0.999) | correct | 0.01 | 0.999 | 0.046 |
| worst | `' on package political into' [·] ' tw exam match'` | `'_ET'` (0.999) | correct | 0.02 | 0.999 | 0.020 |
| best | `' home wall dat ke' [·] ' plus line lik'` | `'_ET'` (1.000) | correct | 0.00 | 1.000 | 0.010 |

### `'赞赏功能被关闭'` — id 141785  (single-probe lp -11.88, fragility 1.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' serv grow ex form' [·] ' website'` | `' '` (0.990) | substitution | 0.21 | 0.001 | -0.002 |
| worst | `' life dig think piece' [·] ' or company connection'` | `' '` (0.981) | substitution | 0.37 | 0.001 | 0.025 |
| worst | `' industry full mark love' [·] ' where word column'` | `' '` (0.988) | substitution | 0.24 | 0.001 | 0.022 |
| best | `' auto comm it sl' [·] ' activity decl flex'` | `' '` (0.215) | substitution | 8.70 | 0.102 | 0.013 |

### `'_SU'` — id 54720  (single-probe lp -11.85, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' event even giving' [·] ' is route full'` | `'_SU'` (0.995) | correct | 0.06 | 0.995 | 0.031 |
| worst | `' power asked descri' [·] ' equipment physical kn'` | `'_SU'` (0.995) | correct | 0.06 | 0.995 | 0.018 |
| worst | `' safe mod edit history' [·] ' seg'` | `'_SU'` (0.996) | correct | 0.05 | 0.996 | 0.043 |
| best | `' constructor ist comp environment' [·] ' fully inst behind'` | `'_SU'` (1.000) | correct | 0.01 | 1.000 | 0.005 |

### `'.react'` — id 52844  (single-probe lp -11.68, fragility 0.00)

| cell | context | emitted (p) | mode | H bits | p(target) | ret |
|---|---|---|---|---|---|---|
| worst | `' serv grow ex form' [·] ' website'` | `'.react'` (0.989) | correct | 0.10 | 0.989 | 0.031 |
| worst | `' safe mod edit history' [·] ' seg'` | `'.react'` (0.998) | correct | 0.03 | 0.998 | -0.014 |
| worst | `' industry full mark love' [·] ' where word column'` | `'.react'` (0.998) | correct | 0.03 | 0.998 | 0.009 |
| best | `' df' [·] ' influ address claim'` | `'.react'` (1.000) | correct | 0.00 | 1.000 | 0.024 |
