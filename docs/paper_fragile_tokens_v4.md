# Fragile Tokens

### Context-dependent copy failure on tokens that pass single-token glitch probes

Jeffrey R. Gordon · September 2026

## Abstract

Glitch-token detectors evaluate each vocabulary entry in isolation, typically by asking the model to repeat it. This paper measures what that evaluation misses. Placing tokens into banks of random ordinary contexts and scoring the copy at each position, a stable fraction of tokens that pass the single-token probe fail to copy in at least one context in ten: 2.4% on Qwen3-1.7B, 7.0% on Qwen2.5-7B, 1.5 to 2.7% on OLMo-2-7B, 1.5% on Qwen3.8-27B, 0.9% on Qwen3-32B, 10.5% on Qwen2.5-72B, 0.4% on Qwen3.8-Flash-Next and 10.9% on Qwen3-235B under one gate applied to all eight; the two largest dense-activation models are the most fragile, on identical token ids as well as in aggregate, while the 512-expert Flash-Next, with about three billion parameters active per token, is the least fragile of all. The property belongs to the token, not the context. The context explains at most 1.3% of the variance in any model, and a token's fragility on half of the contexts predicts its fragility on the other half with correlation 0.82 to 0.91. The fragile tokens are canonical and ordinary: the English word `according` copies alone at probability 0.999 and is deleted in 44 of 48 contexts with probability 1.000. Failures take four confident forms, deletion, neighbour substitution, truncation and translation, and the same four recur on every model tested. Given a fragile or glitch token as an entire prompt, a reasoning model does not signal uncertainty. The 7B explains a plausible neighbour of the token it was given; the 32B, in plain and in thinking mode, writes out the solution to a mathematics problem that was never posed. The token's identity is nonetheless linearly recoverable from the final layer, so the failure is at the readout. Combinations of tokens do not produce this behaviour: a pre-registered factorial screen over ordered pairs at separations of 2 to 128 positions finds no pair that copies worse than its parts on the 7B, the 32B or the 235B, and repeating a token improves its copying, increasingly so with size. On the 7B, fragility can be read from the embedding matrices without a forward pass, because the fragile tokens lie along the direction the verified glitch tokens share (cross-validated AUC 0.911 against 0.711 for surface rarity, paired improvement +0.20, 95% CI [+0.10, +0.32]). The same direction, built label-free from tokens that fail both alone and in context, separates the 32B's fragile tokens at 0.898; on the 72B, whose fragile set is dominated by rare CJK characters that surface rarity identifies directly, rarity predicts better than the embedding does. The representation of the token in context predicts fragility beyond rarity on every model. Inside assignments with tools, a 32B model given a glitch token searches for the sentence with the token removed, retrieves nothing, and delivers a confident recommendation anyway; fragile tokens fail the same way in a quarter of episodes, healthy tokens are looked up by name and reported correctly, and no model loops or invents a version. Every failing case is released with model provenance.

## 1. Introduction

Detection of glitch tokens has proceeded one token at a time since the first catalogue of anomalous strings in GPT-2. Magikarp ranks vocabulary entries by indicators computed from the unembedding matrix and verifies candidates with a repetition prompt; GlitchHunter clusters; GlitchProber classifies intermediate activations; GlitchMiner optimises next-token entropy; GlitchQuiz applies a battery of templates. Each answers the question *is this token defective* with the token alone in the prompt, and each is accurate on its own terms.

Field reports describe behaviour that this question does not address: tokens that a reasoning model "struggles to identify" or "conflates" with another script, or that behave correctly in isolation and incorrectly inside a longer input. Two accounts fit such reports. On the compositional account, particular combinations of tokens, possibly individually healthy and possibly widely separated, produce the behaviour. On the contextual account, it is a property of a token that some surroundings expose and others do not. The two accounts imply different detectors.

Both were tested on `allenai/OLMo-2-1124-7B-Instruct`, an untied model with a vocabulary of 100,352 for whose base model an independent group published behavioural labels (3,171 tokens tested by repetition probe, 442 verified as failing). A token described as *healthy* is one that group tested and passed. The findings were then tested on three further models sharing one tokenizer, Qwen3-1.7B, Qwen3-32B and Qwen2.5-72B-Instruct, against quantities and ranges fixed before any of them ran. All generation is greedy, all prompts are assembled from token ids, and every failing case is stored with its context and emission probabilities.

The findings, in the order the paper presents them:

1. **The failure is confident substitution, and larger models substitute more consequentially** (§3). A model given a glitch or fragile token answers about a different token, at low entropy, and completes its task on the substituted content. The 7B glosses a neighbour; the 32B invents and solves a problem; the 72B quotes the string and answers about the word inside it. The token's identity reaches the final layer intact.
2. **Fragility is a stable, measurable property of the token** (§4). Roughly one probe-passing token in forty to one in ten, depending on the model, fails in at least a tenth of ordinary contexts. The context contributes almost nothing; the same tokens fail on independent context banks; the same four failure modes recur on every model.
3. **Fragility lives in the embedding, and where it lives depends on what is fragile** (§5). On the 7B the fragile tokens sit along the verified-glitch direction of the input embedding and are predicted from the weights alone better than from rarity. On the 72B the fragile population is a rarity-identifiable class of rare CJK characters, and rarity predicts it better than the embedding does. In every untied model, the token's in-context representation predicts fragility beyond rarity.
4. **Tokens combine additively** (§6). The compositional account is not supported at any separation on either model tested; repetition helps rather than harms.
5. **Inside tasks, the token is dropped from the plan and the task is finished anyway** (§7). Given a glitch token in an assignment with tools, a 32B model searches for the sentence with the token removed, retrieves nothing relevant, and writes a confident recommendation from generic material; fragile tokens fail the same way part of the time, by the substitutions the copy matrix predicts; healthy tokens are looked up by name and written up correctly. No model loops and none invents a version number; they finish.

Section 7 puts the same three token classes inside tasks, with and without tools; §8 collects the method decisions a reader needs to interpret the numbers, §9 states scope, and the appendices list rejected hypotheses and the data.

## 2. A token that passes alone and fails in context

The English word `according` is token 4184 in the vocabulary, among the most frequent pieces, and it is canonical: decoding the id and re-encoding the string returns the same id. Under the repetition probe used by every detector (*Repeat the text exactly. Text: according. Copy:*) the log-probability of the correct continuation is −0.001. By the standard in use since 2023, the token is healthy.

The same token was placed in sequences of ordinary common words drawn at random, so that the surroundings were not chosen, and the model was asked to repeat the sequence. In the context *deliver rel behind according sent floor wrong*, at the position where the copy should produce `according`, the model produced `sent` with probability 1.000 and entropy 0.00 bits, then continued the copy correctly from that word. Across 48 such contexts the token was deleted in 44. In the remaining four it was copied at 0.999.

The token did not change between the two measurements. The first was made in a context where the copying mechanism has one candidate, the second in the contexts a token ordinarily occupies. A score on a token in isolation measures whether the model can reproduce it when nothing competes. Whether that reproduction survives surrounding text is a distinct question, and it is the question a system that assembles prompts programmatically poses on every call. A single word failing in 44 contexts is a specimen; §4 measures how common the pattern is.

## 3. The failure is confident substitution

### 3.1 On the reference model

Thirty-two verified glitch tokens and thirty-two healthy controls were spliced into the 7B's chat template after *What is this token? Identify it exactly, then explain step by step what it means*, and 256 tokens were generated for each. Two predictions drawn from the field reports were tested first. Neither held. The difference in next-token entropy between glitch-seeded and control-seeded responses is +0.27 bits at step 8 and −0.29 bits at step 255; it does not grow with length. Responses seeded by different glitch tokens do not converge on a common state: the mean pairwise cosine of their centred final-layer states decays from 0.016 to 0.0001 for glitch seeds and from 0.017 to 0.0002 for controls.

The model instead answers about a different token. Table 1 gives the token supplied, whether the model reproduced it, the token the model said it had been given, and the opening of the explanation that followed.

**Table 1.** OLMo-2-7B, reasoning-style prompt: token supplied and the model's account of it. *Reproduced* is by exact token id within the response.

| token supplied | group | reproduced | model said it was given | opening of the explanation |
|---|---|---|---|---|
| `.XtraLayout` | glitch | no | `"layui"` | a lightweight, modular, and flexible JavaScript framework primarily used for building web interfaces |
| ` Hexatrigesimal` | glitch | no | `"Calculus"` | a branch of mathematics that deals with the study of change, motion, and accumulation |
| `雅黑` | glitch | no | `"턱"` (Korean Hangul) | represents the consonant sound "t" or "d" … combined with the vowel "u" to form the syllable "ttuk" |
| ` didn` | glitch | no | `"don"` | a short form or abbreviation of "do not" |
| `.AddModelError` | glitch | no | `ErrorDictionary` | is not a standard token in programming languages … it might be a misunderstanding or a typo |
| `.JpaRepository` | glitch | no | `"IpaRepository"` | does not directly correspond to a widely recognized term or acronym |
| `")){\r\n` | glitch | no | `){` | a closing curly brace (`}`) in programming syntax |
| `.ContextCompat` | glitch | no (leading `.` dropped) | `ContextCompat` | a static utility class in the Android SDK, correct apart from the punctuation |
| ` woman` | control | yes | `"woman"` | a noun in the English language |
| ` actually` | control | yes | `"actually"` | an adverb in the English language |
| ` life` | control | yes | `"life"` | a concept that has been extensively explored across various disciplines |
| `,\r\n` | control | no | `",` | a quotation mark or double quote character in ASCII |
| `.onOptionsItemSelected` | control | no (leading `.` dropped) | `"OptionsItemSelected"` | not a standalone token with a universally recognized meaning |

Across the 32 glitch seeds the model reproduced the token it was shown zero times by exact id; across the 32 controls, seventeen times. The characteristic glitch response is a fluent account of a plausible neighbour (a JavaScript framework for a UI-library token, a branch of mathematics for a base-36 numeral token, a Korean syllable for a Chinese font name), delivered at 0.85 to 1.3 bits of entropy and followed in every case by a step-by-step explanation of the substituted token. In the short-completion regime the same seeds were simply continued as though a definition were being written (`.XtraLayout` and ` Hexatrigesimal` were both continued with *is a portmanteau of "*), and no seed in either group was reproduced. That regime is where a single-token failure is masked. The reasoning-style prompt exposes it because the task requires the model to refer to the token rather than to continue past it. In the factorial experiment of §6, where a trivially answerable question follows the token, the task-completion rate was 1.000 in every arm. The model neither halts nor signals uncertainty; it completes the task on substituted content. This is the appropriate statement of the hazard, and it accounts for why detectors built on entropy do not observe it.

### 3.2 Across the ladder

The same prompt was run, label-free, on every model in the ladder that generates at practical speed: the five dense Qwen models and OLMo-2-7B. The two FP8 mixtures cannot (§8.5); for them Table 2 carries the teacher-forced counterpart of the same measurement and Table 2b what they emit in the token's place. The seed set is the thirty-two tokens each model copies worst under the single probe (restricted to printable strings), the controls thirty-two tokens it copies exactly. Table 2 gives the counts on both reproduction metrics (§7.3 explains why two are needed); Table 3 gives the responses.

**Table 2.** Seed reproduction across models. *Exact id*: the seed's token id appears in the generated ids. *String*: the seed's text appears in the output under any segmentation, on the eight logged seeds per group (all thirty-two for Qwen2.5-7B and Qwen3.8-27B).

| model | mode | worst tokens, exact id | worst tokens, string (of 8) | controls, exact id | controls, string (of 8) |
|---|---|---|---|---|---|
| Qwen3-1.7B | plain | 0 / 32 | 1 / 8 | 14 / 32 | 8 / 8 |
| Qwen3-1.7B | thinking | 0 / 32 | 1 / 8 | 16 / 32 | 8 / 8 |
| Qwen2.5-7B | plain | 0 / 32 | 1 / 32 | 17 / 32 | 32 / 32 |
| OLMo-2-7B | plain | 0 / 32 | 0 / 8 | 17 / 32 | 7 / 8 |
| Qwen3.8-27B | plain | 5 / 32 | 19 / 32 | 11 / 32 | 32 / 32 |
| Qwen3.8-27B | thinking | 6 / 32 | 21 / 32 | 17 / 32 | 32 / 32 |
| Qwen3-32B | plain | 0 / 32 | 0 / 8 | 18 / 32 | 8 / 8 |
| Qwen3-32B | thinking | 0 / 32 | 0 / 8 | 23 / 32 | 8 / 8 |
| Qwen2.5-72B | plain | 0 / 32 | 3 / 8 | 15 / 32 | 4 / 8 |
| Qwen3.8-Flash-Next (FP8) | teacher-forced (Table 2b) | 71% of 32 × 24 cells | | 100% of 32 × 24 cells | |
| Qwen3-235B-A22B (FP8) | teacher-forced (Table 2b) | 0.1% of 32 × 24 cells | | 96% of 32 × 24 cells | |

**Table 3.** The token supplied as the whole user turn, and the opening of the response. *Mode* classifies what the model did with the token.

| model | token supplied | group | mode | opening of the response |
|---|---|---|---|---|
| Qwen3-1.7B | `格會員` | worst | substituted the template's own marker, then hedged | *The user wrote "</think>" … It seems like you might have made a typo or there's a formatting issue. Could you please clarify what you're asking?* |
| Qwen3-1.7B | `.currentTarget` | worst | neighbour substitution, answered | *You mentioned `@this` in the context of JavaScript, but it's not a standard syntax in JavaScript. Let me clarify: in JavaScript, `this` refers to …* |
| Qwen2.5-7B | `䨰` | worst | recited its own system prompt | *"You are Qwen, created by Alibaba Cloud. You are a helpful assistant." This is my starting point as provided by Alibaba Cloud. Could you please provide more context …* |
| Qwen2.5-7B | `$product` | worst | quoted the string, answered about it | *The term `$product` is often used in programming to represent a variable that holds information about a product. However, without more context, I'm not sure …* |
| Qwen2.5-7B | `臘` | worst | asked for clarification | *Could you please provide more context or ask a specific question? That way, I can provide you with the most accurate and helpful information …* |
| Qwen2.5-7B | ` khoản` | control | named and glossed | *"khoản" is a word that can have different meanings depending on the context. In Vietnamese, it can mean: 1. A section or paragraph in a document. 2. An amount of money. …* |
| Qwen3.8-27B | ` ForCanBeConverted` | worst | fabricated a task and solved it | *The question asks: "What is the capital of France?" This is a straightforward factual question. The capital of France is Paris. …* |
| Qwen3.8-27B | ` долгове` (Russian stem) | worst | read the seed as Chinese, glossed the word it invented | *The user has written "longevity" in Chinese characters. This is a single word that could have multiple interpretations depending on context. …* |
| Qwen3.8-27B | ` szexf` | worst | substituted one CJK character and analysed it | *The user has sent a single character: "堶" Let me analyze this character: 堶 (tuó) - This is a Chinese character …* |
| Qwen3.8-27B | ` 취임` | control | named and glossed | *The user has provided a single word: "취임" (chwiim). This is a Korean word that means "inauguration" or "taking office."  Since the user hasn't provide …* |
| Qwen3-32B | `ớ` | worst | fabricated a task and solved it | *The question is asking for the value of cos(θ) given that cos(2θ) = 1/3. ### Step 1: Use the double angle identity …* |
| Qwen3-32B | `อังกฤษ` (Thai, "English") | worst | fabricated a task and solved it | *The question is asking for the value of 1/2 + 1/4 + 1/8 + 1/16 + ⋯, which is an infinite geometric series. ### Step 1: Identify the first term and common ratio …* |
| Qwen3-32B, thinking | `ล่า` | worst | fabricated a task and reasoned about it | *I need to determine if the statement "The sum of the first 100 odd numbers is 10,000" is true or false. First, let's understand what the first 100 odd numbers are …* |
| Qwen3-32B, thinking | `웛` | worst | fabricated a task and reasoned about it | *The user has asked me to provide a detailed explanation of the process of photosynthesis, including the key steps, the role of chlorophyll …* |
| Qwen3-32B | ` programma` | control | named and glossed | *You've written "programma" which is Italian for "program" or "programme". However, you haven't specified what kind of program …* |
| Qwen2.5-72B | `_ghost` | worst | quoted the string, answered about "ghost" | *The term "_ghost" can have several meanings depending on the context: 1. Supernatural Entity: In folklore and popular culture, a ghost is …* |
| Qwen2.5-72B | `$filter` | worst | sigil dropped, answered about "filter" | *The term "filter" can have different meanings depending on the context. Here are a few common uses: 1. In Computing and Data Processing …* |
| Qwen2.5-72B | `눠` | worst | substituted the template's own marker | *I believe you might be using an abbreviation or slang that isn't widely recognized. Could you please provide more context or clarify what you mean by "<|im_start|>"?* |
| Qwen2.5-72B | ` сразу` | control | named and glossed | *I'm not entirely sure what you mean by "сразу" (srazu) in this context. It's a Russian word that can mean "immediately," "at once," or "right away." …* |

**Table 2b.** Teacher-forced counterpart for the two FP8 mixtures, which cannot generate at practical speed (§8.5). *Copied in context* is the share of the 768 cells (the 32 worst printable tokens by the single probe × 24 contexts) at which the token's own id is emitted with probability at least 0.5, against 32 controls drawn as in Table 2. *Glitch class* is the subset of the stored worst tokens that fail in at least 90% of contexts. The emission is read at each token's worst context, with its probability and the entropy of the distribution there.

| model | worst tokens copied in context | controls copied in context | glitch class | emission at the worst context | median top-1 probability | median entropy |
|---|---|---|---|---|---|---|
| Qwen3.8-Flash-Next (FP8) | 71% (median p 0.999) | 100% | 9 of 20 stored | deletion 9 cells, substitution 18 of 27 | 0.11 | 8.5 bits (26% below 2) |
| Qwen3-235B-A22B (FP8) | 0.1% (median p 0.000) | 96% | 20 of 20 stored | deletion 49 cells, substitution 11 of 60 | 0.94 | 0.47 bits (90% below 2) |

| model | token supplied | emission at the worst context | entropy | mode |
|---|---|---|---|---|
| Qwen3.8-Flash-Next | ` szexf` | ` wrote`, the following word, at 0.991 | 0.14 bits | deletion |
| Qwen3.8-Flash-Next | ` ForCanBeConverted` | ` fight`, the following word, at 0.743 | 2.7 bits | deletion |
| Qwen3.8-Flash-Next | `echslungs` | ` beispiels` at 0.021 | 12.5 bits | diffuse substitution |
| Qwen3.8-Flash-Next | `хотво` (Cyrillic piece) | ` hentai` at 0.035 | 12.6 bits | diffuse substitution |
| Qwen3-235B | `吏` | ` assert`, the following word, at 0.982 | 0.19 bits | deletion |
| Qwen3-235B | `กระเป๋า` (Thai, "bag") | ` tor`, the following word, at 1.000 | 0.01 bits | deletion |
| Qwen3-235B | `larınd` (Turkish suffix) | a blank line at 0.500 | 2.2 bits | substitution |
| Qwen3-235B | `ความเป็น` (Thai, "being") | ` abs` at 0.929 | 0.54 bits | substitution |

The two mixtures differ under teacher forcing as much as the dense models differ in generation. On the 235B the tokens the bare probe fails are deleted in context at near-certainty: the following word is emitted at 0.93 to 1.00 in 49 of 60 worst cells, the confident substitution of §3.1 in its deletion form. On Flash-Next the bare probe is a poor guide to context: 32 tokens it fails outright copy in context in 71% of cells at a median probability of 0.999, and the 9 that fail everywhere are either deleted with confidence (` szexf`, ` ForCanBeConverted`) or replaced by a diffuse guess at 8 to 12 bits. Whether either model, given the chance to generate, would fabricate a task as the 32B and the 27B do is not measured here.

Three regularities. First, no model re-emits the exact token except the Qwen3.8-27B, which re-emits 5 of its 32 worst seeds against 11 of 32 controls and the seed's string in 19 against 32; every model names and glosses its controls. Second, when the seed carries no usable content, the model substitutes the most salient recent token, the substitution of §3.1: the 1.7B and the 72B answer as if the user had typed the chat template's own marker (`</think>` and `<|im_start|>`), and the Qwen2.5-7B recites its own system prompt. Third, the 32B does not reach for anything. Given `ớ`, it writes out a double-angle derivation; given a Thai word, an infinite series; in thinking mode the trace opens *I need to determine if the statement … is true or false* with the statement already fabricated before the first step of reasoning about it. The fabricated prompts come from a small repertoire (the same geometric series serves three different seeds), and the trajectory's entropy exceeds 2 bits only at the first two steps, where the problem is being chosen, after which it runs below the control trajectory (0.40 against 0.66 bits at step 32). The Qwen3.8-27B behaves the same way: given ` ForCanBeConverted`, or its own `<|im_start|>`, it answers a question about the capital of France that nobody asked, and given a Cyrillic stem it reads the seed as Chinese and glosses the word it invented (Table 3). The 1.7B and the Qwen2.5-7B ask for clarification. The 32B and the 27B confidently answer a question nobody asked. As the model gets more capable, the substitution gets more consequential.

### 3.3 The identity is retained

Whether the model has lost the token or failed to emit it determines the remedy. On Pythia-1.4b, for which labels of the same kind exist, each of 36 verified glitch tokens and 1,088 controls was placed at the end of eight contexts and its hidden state read at every layer. At each layer the token was matched from its state in one context to its state in another within a pool of 1,124 candidates, after removing each context's shared component. From the embedding layer through layer 12, glitch tokens were identified in 1.000 of cases and controls in 0.988 to 1.000; at the final layer, 0.880 against 0.874. The identity of a glitch token reaches the top of the network as intact as that of a healthy token. A logit lens at the same positions shows the model *continuing* glitch tokens correctly (`idepress` is followed by `ant` at every layer) while unable to reproduce them. Continuation, which pretraining on running text rewards, survives; self-reference, which it does not reward, fails. The failure is at the readout, not in the representation.

## 4. Fragility is a property of the token

### 4.1 Definition and measurement

Contexts are sequences of common, individually clean words drawn at random, of lengths 8, 16, 32 and 64, with an interior slot at a random position. Into every context is placed every token of a sample stratified over the id range (uniform sampling of ids falls mostly in the rare tail). A cell is the copy log-probability of the token at its slot under the repetition prompt; a cell fails below −0.5 (probability 0.61); a token's **fragility** is the fraction of contexts in which it fails. Because every token meets the same contexts, differences in fragility are differences between tokens. On the 7B the first bank held 48 contexts and 2,098 tokens (1,998 sampled and 100 verified glitch tokens as a reference class); an independent replication bank held 24 new contexts and 6,273 tokens, and the cross-model runs use 24 contexts and 4,200 or 8,400 tokens.

A token counts as *clean-looking* if the single-token probe passes it. Two gates are used. The paper's original gate is a single-probe log-probability above −0.1 (the token copies alone at more than 0.9). That gate proved model-specific: the 32B copies ordinary tokens in context at 0.997 but under the bare few-shot probe at 0.72 to 0.89, so it admitted 353 of 4,200 tokens and called none fragile. Cross-model comparison therefore uses a gate that means the same thing on any model, the token being the greedy output of its own copy probe (p > 0.5), and the 7B is recomputed under it. Both gates are reported for the 7B.

### 4.2 The dependence is on the token

![Figure 1a](figures/fig_fragility_matrix_qwen3_1_7b.png)

![Figure 1b](figures/fig_fragility_matrix_qwen25_7b.png)

![Figure 1c](figures/fig_fragility_matrix_7b.png)

![Figure 1d](figures/fig_fragility_matrix_qwen38_27b.png)

![Figure 1e](figures/fig_fragility_matrix_qwen3_32b.png)

![Figure 1f](figures/fig_fragility_matrix_qwen25_72b.png)

![Figure 1g](figures/fig_fragility_matrix_qwen38_flash.png)

![Figure 1h](figures/fig_fragility_matrix_qwen3_235b.png)

**Figure 1.** Fragility matrices, one per model in order of size: (a) Qwen3-1.7B, (b) Qwen2.5-7B, (c) OLMo-2-7B (replication bank), (d) Qwen3.8-27B, (e) Qwen3-32B, (f) Qwen2.5-72B, (g) Qwen3.8-Flash-Next in Qwen's FP8 release, (h) Qwen3-235B-A22B in Qwen's FP8 release, all under the greedy gate. Each row is a token, each column one of the model's 24 contexts (ordered by overall failure rate), each cell the probability of the correct token at the copy position. Three blocks in the same order on every panel. Top: eight glitch tokens, which fail everywhere; on the 7B these are verified glitch tokens from the published labels, on the other models the worst tokens by the single probe that also fail in at least 90% of contexts (§8.3). Middle: twelve random clean tokens, which fail nowhere. Bottom: the most fragile clean-looking tokens (up to 36; the 32B has fifteen in its sample) with their fragility in parentheses. The structure is horizontal on every panel: on the 7B, `according` fails in 22 columns of 24 and ` due` in 13, and no column is dark for the middle block; on the 72B the fragile block is rare CJK characters, deleted at near-certainty across most columns; on the 235B the 36 most fragile clean-looking tokens fail in every context at probability zero while copying alone above 0.5.

Decomposing each cell of the 7B's first bank as a grand mean, a token effect, a context effect and a residual, the shares of variance over all tokens are 0.952 (token), 0.001 (context) and 0.047 (residual) on the log-probability scale; over clean-looking tokens alone, 0.717, 0.000 and 0.282. No context is generally hostile: the most severe of the 48 fails 19% of tokens and the least severe 10%, and the failure rate of clean-looking tokens does not vary with context length (0.005, 0.008, 0.008, 0.008 across lengths 8 to 64). A held-out analysis confirms the asymmetry: predicting whether a token fails in a context it was not scored in, the token's fragility on other contexts gives AUC 0.94, the context's hostility on other tokens gives 0.56, and a measure of similarity between the token and its surrounding words adds nothing to either. Which context elicits a given token's failure is not predicted by the features examined; that the token will fail somewhere is.

### 4.3 Prevalence and stability across eight models

**Table 4.** Fragility among clean-looking tokens. *Fragile* is fragility ≥ 0.10. *Split-half* is the correlation (7B first bank: AUC) between fragility on even- and odd-numbered contexts. *Context share* is the context's share of variance on the clean-looking matrix. *Confident* is the share of failing cells of fragile clean-looking tokens with entropy below 2 bits.

| model | gate | clean-looking n | fragile | never fail | split-half | context share | confident |
|---|---|---|---|---|---|---|---|
| Qwen3-1.7B | p > 0.5 | 1,666 | **2.4%** | 93.0% | 0.82 | 0.004 | 86% (median 1.00 bits) |
| Qwen2.5-7B | p > 0.5 | 4,217 | **7.0%** | 85.5% | 0.94 | 0.001 | 14% (median 7.16 bits) |
| OLMo-2-7B, first bank | lp > −0.1 | 873 | **1.6%** | 92.1% | AUC 0.996 | 0.000 | (see §4.4) |
| OLMo-2-7B, replication bank | lp > −0.1 | 1,888 | **1.5%** | 95.0% | | | |
| OLMo-2-7B, replication bank | p > 0.5 | 2,183 | **2.7%** | 92.9% | 0.85 | 0.001 | |
| Qwen3.8-27B | p > 0.5 | 3,766 | **1.5%** | 96.2% | 0.92 | 0.001 | 57% (median 1.79 bits) |
| Qwen3-32B | p > 0.5 | 1,597 | **0.9%** | 95.7% | 0.89 | 0.002 | 59% (median 1.74 bits) |
| Qwen2.5-72B | p > 0.5 | 3,998 | **10.5%** | 82.6% | 0.91 | 0.013 | 69% (median 0.74 bits) |
| Qwen3.8-Flash-Next (FP8) | p > 0.5 | 4,047 | **0.4%** | 99.2% | 0.94 | 0.000 | 16% (median 5.31 bits) |
| Qwen3-235B-A22B (FP8) | p > 0.5 | 6,277 | **10.9%** | 79.9% | 0.95 | 0.010 | 72% (median 0.60 bits) |

![Figure 3](figures/fig_ladder.png)

**Figure 3.** The ladder under one gate (p_alone > 0.5), models in order of size. Left: share of gated tokens with fragility ≥ 0.10, with the pre-registered replication range shaded. Middle: split-half correlation of fragility and the context's share of variance. Right: cross-validated AUC of the surface, static-geometry and dynamic tiers for predicting fragility, with the number of positives.

The reference class validates the measure: verified glitch tokens fail in nearly every context (mean fragility 0.993 in the first bank) and none passes the single probe. Among tokens that do pass it, between one in a hundred and one in ten fails in one context in ten, and on every model the property is stable: fragility on even contexts predicts fragility on odd contexts at 0.82 to 0.91, `according` scored 0.92 in both 7B banks, and the context's share of variance never exceeds 0.013. The pre-registered expectation that prevalence would fall with model size held from the 1.7B to the 32B and failed at both of the largest models: 10.5% at the 72B and 10.9% at the 235B, four times the 7B's under the same gate, in two different families. The 235B passes three quarters of the sample through the bare probe, so its prevalence was checked for composition: on the 1,569 token ids gated by both the 32B and the 235B, the 235B is fragile on 4.3% and the 32B on 0.8%, and on the 1,592 ids gated by both the 1.7B and the 235B, 6.2% against 2.0%. The higher fragility is the model's, not the gate's. Qwen2.5-7B-Instruct, the 72B's own family at the 7B scale and run through the same pipeline, is fragile on 7.0% of gated tokens (Table 4) with the same rare-character population, so family sets a level and size moves it: within Qwen2.5, 7.0% at 7B and 10.5% at 72B; within Qwen3, 2.4% at 1.7B, 0.9% at 32B and 10.9% at 235B. The Qwen3.8 generation's 27B, run under the chat framing of §8.1, is fragile on 1.5% of 3,766 gated tokens, inside the pre-registered range and between the two smaller Qwen3 models; its fragile tokens are whole word pieces in Cyrillic, Korean and Arabic (` оказаться`, ` 아버지`, ` الديمق`), copied alone at 0.9 or better and dropped in most contexts, rather than the rare single characters of the 72B and 235B. The 32B is the robust exception rather than the trend. Qwen3.8-Flash-Next, a 512-expert mixture with about three billion parameters active per token, measured in its FP8 release under the same gate and the chat framing, is the least fragile model in the ladder: 15 of 4,047 gated tokens (0.4%), 99.2% never failing, split-half 0.94, and its failures are uncertain rather than confident (median entropy 5.3 bits, Table 4). It gates the same share of the sample as the 72B, so the size of the gate does not set the level; the 235B, which gates three quarters of the sample, is the most fragile model and Flash-Next, gating half, the least. Ordered by total parameters the two largest dense-activation models are the most fragile and the sparse Flash-Next breaks the trend; ordered by active parameters Flash-Next sits below the 7B models and the trend survives. The ladder cannot separate the two readings. The 72B's fragile tokens also fail more in longer contexts (26% of 8-token contexts, 49% of 64-token contexts), where the 7B's rate was flat.

### 4.4 The fragile tokens, and what the model emits instead

**Table 5.** Fragile clean-looking tokens across models, with the emission at the worst context. *Alone* is the single-probe copy probability.

| model | token | alone | fragility | worst-context emission | mode |
|---|---|---|---|---|---|
| Qwen3-1.7B | ` 방법` (Korean, "method") | 0.96 | 0.75 | ` 方法` (Chinese, "method") at 0.97 in three contexts of four; copied at 0.96 in the fourth | translation |
| Qwen3-1.7B | ` Marshal` | 0.93 | 0.46 | `Marshal` at 0.998 | truncation |
| Qwen3-1.7B | ` widać` (Polish) | 0.80 | 0.83 | substituted | substitution |
| Qwen2.5-7B | `⌕`, `圊`, `冔`, `柷` and other single rare characters | 0.63–0.91 | 1.00 | the chat template's own `<|im_start|>` at 0.39–0.83 | substitution |
| Qwen2.5-7B | `ঈ` (Bengali letter) | 0.83 | 1.00 | ` staat` at 0.066, entropy 9.3 bits | diffuse substitution |
| OLMo-2-7B | ` according` (id 4184) | 0.999 | 0.92 | ` sent`, the following word, at 1.000; copy continues from it | deletion |
| OLMo-2-7B | ` due` | 1.000 | 0.42 | deleted; worst cell −23.8 | deletion |
| OLMo-2-7B | ` который` (Russian, "which") | 0.983 | 0.71 | ` что` ("what") | substitution |
| OLMo-2-7B | ` pueden` | 0.970 | 0.62 | ` pena` at 0.775 | substitution |
| OLMo-2-7B | ` abbiamo` | 0.980 | 0.60 | ` abdom`, continued to "abdominoplasty" | substitution |
| OLMo-2-7B | ` sólo` | 0.968 | 0.48 | ` só` at 0.892 | truncation |
| OLMo-2-7B | ` 查询` (Chinese, "query") | 0.985 | 0.46 | ` QUERY` at 0.970 | translation |
| OLMo-2-7B | ` getSystemService` | 0.948 | 0.38 | `SystemService` at 0.30 | truncation |
| Qwen3.8-27B | ` оказаться` (Russian, "to turn out") | 0.89 | 0.96 | ` occur` at 0.476 | translation |
| Qwen3.8-27B | ` грунт` (Russian, "soil") | 0.84 | 0.79 | ` grunt` at 0.995 | substitution |
| Qwen3.8-27B | ` 아버지` (Korean, "father") | 0.99 | 0.92 | `아버` at 0.991 | truncation |
| Qwen3.8-27B | ` espectacular` (Spanish) | 0.83 | 0.62 | ` spectacular` at 0.649 | translation |
| Qwen3-32B | ` след` (Russian) | 0.61 | 0.83 | ` follow` at 0.995 | translation |
| Qwen3-32B | ` realtà` (Italian) | 0.76 | 0.38 | ` réalité` at 0.898 | translation |
| Qwen3-32B | ` اليمن` (Arabic, "Yemen") | 0.77 | 0.12 | ` Yemen` at 0.499 | translation |
| Qwen3-32B | `ཀ` (Tibetan letter) | 0.52 | 0.42 | ` greater`, the following word, at 0.925 | deletion |
| Qwen2.5-72B | `跸` and 42 other single rare CJK characters | 0.98–0.99 | 0.4–1.0 | the following word at 0.998–1.000, entropy 0.00–0.02 bits | deletion |
| Qwen2.5-72B | `увеличен` (Russian, "increased") | 0.97 | 0.46 | ` exaggerated` at 0.998 | substitution |
| Qwen2.5-72B | ` الاسلام` (Arabic, "Islam") | 0.92 | 1.00 | substituted; mean cell −3.8 | substitution |
| Qwen3.8-Flash-Next | ` ilmaisia` (Finnish, "free") | 0.92 | 0.96 | ` gratuita` at 0.026, entropy 12.0 bits | diffuse substitution |
| Qwen3.8-Flash-Next | ` επίσης` (Greek, "also") | 0.93 | 0.38 | ` также` (Russian, "also") at 0.743 | translation |
| Qwen3.8-Flash-Next | ` 웹사이트가` (Korean, "the website") | 0.89 | 0.46 | ` websites` at 0.434 | translation |
| Qwen3-235B | `僔`, `ഏ`, `𝕒`, `𝙰`, `狴`, `🤜` and other single rare-script characters | 0.95–0.99 | 0.83–1.00 | the following word at 0.999–1.000, entropy 0.00–0.01 bits | deletion |
| Qwen3-235B | `интер` (Cyrillic stem) | 0.98 | 0.92 | `inter` at 1.000 | translation |

Four failure modes occur, and they are the modes of §3. The token is **deleted**: the model emits the following context word and continues as though the slot were empty. It is **substituted** with a neighbour: Spanish for Spanish, one identifier for a similar identifier. It is **truncated** to a prefix of itself. Or it is **translated**: the Chinese token for *query* is emitted as the English `QUERY`, Korean *method* as Chinese *method*, Russian *trace* as English *follow*, Italian *realtà* as French *réalité*. Each is confident, and each reverses when the filler words change; `according` is deleted at 1.000 in three stored contexts and copied at 1.000 in a fourth. Translation, a single case on the 7B, recurs on every Qwen model.

The fragile tokens are canonical. Every fragile clean-looking token in the 7B's first bank round-trips from id to string to the same id, is in Unicode composed form, and is classed `OK` in the labelling group's taxonomy. The population is the same at every size and on both tokenizers: word pieces from languages thin in the training mix (Spanish, Italian, Portuguese, German, Polish, Russian, Chinese, Korean, Arabic, Hebrew, Tibetan), code identifiers, and a small number of high-frequency English connectives. The two English cases with catastrophic scores, `according` and `due`, are words whose following token is almost invariably `to`, and several fragile connectives in other languages (`mentre`, `allerdings`, `może`, `который`) likewise constrain their successor strongly; fragility may track the concentration of a token's continuation distribution, a token the model declines to copy into a position where its expected successor is absent. A deletion at probability 1.000 and entropy zero is also the observable signature of a copy-suppression head (McDougall et al., 2023). Neither hypothesis has been tested. On the Qwen2.5-7B the population is already the rare-character tail that its 72B sibling shows, but the emission differs: the 7B replaces the character with the chat template's own marker `<|im_start|>` at up to 0.83, the substitution of §3.2 with the most salient token as the substitute, and only 14% of its failing cells are confident (median 7.2 bits). At the 72B the population narrows to one class: 43 of the 80 most fragile tokens are single rare CJK characters, copied alone at 0.98 to 0.99 and deleted in context. At the 235B it widens again across every rare script in the vocabulary, Tibetan, Bengali, Sinhala and Malayalam letters, Hebrew and Arabic pieces, Hangul syllables, mathematical-alphabet letters and emoji, deleted at certainty; 72% of its failing cells are below 2 bits. The Qwen3.8 generation returns to the 7B's population: the 27B's fragile tokens are whole words of Russian, Korean, Spanish and Arabic, replaced by a translation, a transliteration or a truncated piece with confidence (57% of failing cells below 2 bits); Flash-Next's few fragile tokens are Finnish, Hungarian and Polish stems replaced by diffuse guesses across scripts (median 5.3 bits), the only model whose failures are uncertain.

### 4.5 The specimen store

For each fragile token and for the reference class, the three worst cells and one control cell are stored with the exact context and slot, the five most probable emissions and their probabilities, the entropy at the position, the probability of the correct token, the greedy continuation, a failure-mode label and, where hidden states were kept, the cosine between the token's last-layer state and its own unembedding row. Each store carries a header with the model, commit, precision, depth, width, vocabulary size, tying and library versions. There are 228 records for the 7B, 324 for the 1.7B, 180 for the 32B and 400 for the 72B (`results/specimens_*.jsonl`, rendered in `docs/specimens*.md`). The claim that the failures are confident is verifiable from them directly.

## 5. Where fragility lives in the embedding

### 5.1 Predictors

If fragility is a property of the token, the practical question is whether it can be estimated without running the model. Three tiers of predictor were evaluated, separated by cost. **Static** features come from the two embedding matrices alone: the norm of the input row and its distance from the centroid; the same for the output row; the cosine to the nearest neighbour and the mean cosine to the ten nearest; the direct-path self-score E_in[t]·E_out[t] with the final layer norm folded in, and its margin over the best competing row; and the token's projection onto the *glitch direction*, the mean input embedding of the reference class minus the global mean (verified glitch tokens on the 7B; the hundred worst tokens by the single probe on the label-free models). **Dynamic** features come from the token's representation on half of the contexts (the dispersion of its slot state across contexts, the norm of that state, its cosine to its own unembedding row) and predict failure on the other half. The **behavioural** tier measures copy on half the contexts and predicts the other half. Every geometric tier is compared with **surface** features available without any model access: token id (which in a byte-pair vocabulary approximates frequency rank), character length, leading space, alphabetic, ASCII. Tiers are combined by cross-validated L2 logistic regression, scored out of fold, and each geometric tier's improvement over surface is bootstrapped in pairs over the same tokens.

### 5.2 Results across eight models

**Table 6.** Predicting fragility ≥ 0.10 among clean-looking tokens. Cross-validated AUC; paired-bootstrap improvement over the surface tier with 95% interval. Qwen3-1.7B ties its input and output embeddings, so its static tier is reported and not held to the bar.

| model | gate | positives | surface | static | Δ static | dynamic | Δ dynamic | static + dynamic | behavioural |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-1.7B (tied) | p > 0.5 | 28 | 0.681 | 0.682 | 0.000 [−0.107, +0.110] | 0.752 | +0.068 [−0.062, +0.198] | 0.722 | 0.951 |
| Qwen2.5-7B | p > 0.5 | 271 | 0.831 | **0.877** | **+0.046 [+0.027, +0.066]** | 0.830 | −0.001 [−0.023, +0.024] | 0.871 | 0.986 |
| OLMo-2-7B | lp > −0.1 | 28 | 0.711 | **0.911** | **+0.202 [+0.096, +0.320]** | 0.823 | +0.112 [+0.024, +0.216] | 0.930 | 0.996 |
| OLMo-2-7B | p > 0.5 | 60 | 0.804 | **0.908** | **+0.105 [+0.052, +0.160]** | 0.874 | +0.070 [+0.015, +0.127] | 0.904 | 0.991 |
| Qwen3.8-27B | p > 0.5 | 58 | 0.776 | 0.711 | −0.066 [−0.157, +0.024] | **0.848** | **+0.073 [+0.028, +0.118]** | 0.839 | 0.980 |
| Qwen3-32B | p > 0.5 | 12 | 0.806 | 0.775 | −0.032 [−0.176, +0.123] | 0.814 | +0.007 [−0.163, +0.190] | 0.805 | 0.945 |
| Qwen2.5-72B | p > 0.5 | 322 | 0.883 | 0.838 | −0.044 [−0.070, −0.019] | **0.913** | **+0.029 [+0.016, +0.044]** | 0.923 | 0.985 |
| Qwen3.8-Flash-Next (FP8) | p > 0.5 | 15 | 0.690 | **0.944** | **+0.256 [+0.092, +0.420]** | **0.967** | **+0.280 [+0.145, +0.434]** | 0.980 | 1.000 |
| Qwen3-235B (FP8) | lp > −0.1 | 381 | 0.901 | 0.880 | −0.022 [−0.043, −0.001] | 0.914 | +0.012 [−0.004, +0.029] | | |

![Figure 2](figures/fig_geometry.png)

**Figure 2.** Where the fragile tokens sit, one row per model in order of size (Qwen3-1.7B, Qwen2.5-7B, OLMo-2-7B, Qwen3.8-27B, Qwen3-32B, Qwen2.5-72B, Qwen3.8-Flash-Next, Qwen3-235B-A22B), greedy gate. Left: UMAP of the unembedding rows (cosine) of every gated token, with the glitch class in black and fragile clean-looking tokens in red. Middle: the projection of each token's input embedding onto the glitch direction (the mean glitch-class embedding minus the global mean), with the AUC for separating fragile from non-fragile gated tokens. Right: token id, the surface proxy for rarity, with its AUC. The glitch class is the 442 verified glitch tokens on the 7B and, on the others, the 100 worst tokens by the single probe that also fail in at least 90% of contexts (§8.3).

| model | glitch class | fragile tokens | AUC, glitch direction | AUC, token id |
|---|---|---|---|---|
| Qwen3-1.7B (tied) | label-free, verified in context | 40 | 0.601 | 0.732 |
| Qwen2.5-7B | label-free, verified in context | 294 | 0.768 | 0.799 |
| OLMo-2-7B | verified (labelled) | 58 | **0.879** | 0.665 |
| Qwen3.8-27B | label-free, verified in context | 57 | **0.782** | 0.718 |
| Qwen3-32B | label-free, verified in context | 15 | **0.898** | 0.855 |
| Qwen2.5-72B | label-free, verified in context | 417 | 0.599 | 0.862 |
| Qwen3.8-Flash-Next (FP8) | label-free, verified in context | 15 | **0.839** | 0.691 |
| Qwen3-235B-A22B (FP8) | label-free, verified in context | 682 | 0.740 | 0.871 |

### 5.3 Reading the table and the figure together

On the 7B, fragility is written in the weights. The glitch-direction projection alone reaches AUC 0.915 on the paper's gate (0.879 as the raw projection in Figure 2 under the wider gate), against 0.678 for the next best static feature, and the static tier beats surface rarity by +0.20 with 28 positives and by +0.11 with 60. The statement this supports is specific: fragility among probe-passing tokens lies along the same input-embedding direction as verified glitch tokens. In Figure 2 the fragile tokens are the red points inside and around the black cluster and strung along the axis that leads to it. The 7B's fragile set (`according`, `due`, `который`) has no surface signature, which is why rarity does poorly on it and the embedding does well.

The three Qwen rows of Figure 2 use a glitch class defined without labels, the worst tokens by the probe that also fail in nearly every context (§8.3); on the 7B that definition reproduces the labelled direction exactly, so the rows are comparable. On the 32B the fragile tokens sit next to the glitch cluster in the UMAP and the glitch direction separates them at 0.898, the 7B's result reproduced without labels; the cross-validated static tier in Table 6 is lower (0.775) because it is fitted on twelve positives, but the geometry is the same. The Qwen3.8-27B keeps that ordering: the glitch direction carries its fragile tokens at 0.782 against 0.718 for token id, its label-free glitch class forms an island at the top of the map while the fragile tokens spread through the body of it, the static tier trails surface (0.711 against 0.776) and the dynamic tier beats surface by +0.073 [+0.028, +0.118]. On the 1.7B, whose embeddings are tied, and on the 72B, the direction carries little (0.601, 0.599), and the static tier, which depends on it, does not beat surface. The 235B sits between: its 682 fragile tokens spread across the whole unembedding map rather than forming an island, the glitch direction carries them at 0.740 and token id at 0.871, and with 381 positives the static tier again trails rarity (0.880 against 0.901) while the dynamic tier is level with it. Flash-Next, with fifteen positives, is the strongest case for the static tier since the 7B: 0.944 against 0.690 for surface (+0.256 [+0.092, +0.420]), and the dynamic tier 0.967. Its fragile tokens are ordinary European word stems that rarity does not pick out, and the embedding does. The 72B's fragile tokens are nonetheless as geometric as any: they form the dense cluster at the upper left of the UMAP. They are the rare CJK tail of the vocabulary, a different population from the model's glitch tokens, and token id identifies them directly (0.862). The Qwen2.5-7B, whose fragile population is the same tail at the 7B scale, is carried by rarity (0.799) about as well as by the direction (0.768), and its static tier beats surface by +0.046 [+0.027, +0.066] with 271 positives.

The dynamic tier, which reads the token's representation in context and needs no glitch class at all, beats surface on the 72B (+0.029 [+0.016, +0.044]) as it does on the 7B (+0.070) and, directionally, on the 1.7B and the 32B. Information about fragility is present in the representation of every model. Whether the two embedding matrices alone carry it beyond rarity depends on whether the model's fragile tokens lie along its glitch direction: they do on the 7B and the 32B, and on the 72B they are somewhere else that rarity already names.

The tiers give a graded procedure. From the weights alone, fragility on the 7B can be pre-screened at 0.91; with a dozen contexts' forward passes the token's representation raises this to 0.93; with a dozen contexts' copy scores it reaches 0.99 on every model tested. The behavioural measurement costs one forward pass per token per context and is stable enough to be made once per model.

### 5.4 Why the output row carries it

The mechanism is established in prior work. Yu et al. (2022) showed that the embedding rows of rare tokens receive gradient chiefly as softmax negatives, in a direction shared with other rare rows; two 2026 architectures, one decoupling input from output representations and one re-injecting token identity at every layer under the designation "Rare Token Problem", are built to counter it. The checkpoint phase of this project adds measurement on real pretraining runs with behaviourally labelled tokens, and the measurement agrees. Decomposing each update to the embedding matrix into its component along the mean update direction and the remainder, verified glitch rows receive the shared component in comparable or greater measure than healthy rows while their token-specific remainder is suppressed to 0.15 to 0.74 of the healthy value, across seven model lineages on three corpora and in a LLaMA-architecture model of independent ancestry (0.573). Under weight tying the relationship reverses as the mechanism requires, the shared component rising to 1.17 to 1.23 times the healthy value, since a tied row is a softmax negative at every step. The output row from which §5.2 reads fragility is the row that was rarely the target. Post-training does not create such rows: across eight stage series, total and token-specific movement agree to three decimal places (0.550/0.550, 0.277/0.277, 0.030/0.030), so a row instruction tuning leaves untouched stays where pretraining left it.

## 6. Tokens combine additively

The compositional account was stated in its sharpest form by a collaborator: in `[1, 231, 885, 9911, 1112, 231]` the repetition of 231 produces glitch behaviour, and in `[1, 231, 885, 9911, 9999, 1922, 1013]` the co-occurrence of 885 and 1013 does, with members individually healthy and arbitrarily separated. The test makes separation the design variable.

A carrier is a sequence of filler words with slots at positions 2 and 2 + d. For an ordered pair (a, b), four carriers are scored: control words in both slots, a with a control, a control with b, and a with b; the interaction I is the fourth score minus the second and third plus the first, the damage attributable to the pair beyond that attributable to its members. Separation d took the values 2, 4, 8, 16, 32, 64 and 128 with two independent filler sequences at each, and carriers at different separations share a prefix. The pool held 64 tokens on the 7B (one run all clean, one run 32 clean and 32 *weak*, tokens copied alone only intermittently) and 35 on the 32B (24 weak, 11 clean); each 7B run scored 63,014 carriers. The null was fixed before scoring: an additive fit within each carrier, residuals standardised by the robust scale of their row and column, permuted within carrier and averaged across carriers, Benjamini–Hochberg at q = 0.05 with a maximum-statistic family-wise test alongside, and a verdict threshold set by the 95th percentile of rejections across twenty datasets simulated from the additive fit with the observed noise (Appendix B).

**Table 7.** Interaction by cell type. *Joint* is the mean copy log-probability with both tokens present; *I* the mean interaction.

| model | a × b | n | joint | I | s.e. | significant pairs | matched-null 95th pct |
|---|---|---|---|---|---|---|---|
| OLMo-2-7B, mixed pool | clean × clean | 992 | −0.027 | +0.0011 | 0.0004 | 1 of 4,032 | 14 |
| | clean × weak | 1,024 | −0.146 | +0.0030 | 0.0016 | | |
| | weak × clean | 1,024 | −0.195 | +0.0033 | 0.0014 | | |
| | weak × weak | 992 | −0.313 | +0.0076 | 0.0027 | | |
| Qwen3-32B, mixed pool | clean × clean | 110 | −0.013 | +0.0011 | | 13 of 1,190 | 14.1 |
| | clean × weak | 264 | −0.042 | +0.0031 | | | |
| | weak × clean | 264 | −0.053 | +0.0034 | | | |
| | weak × weak | 552 | −0.077 | +0.0091 | | | |
| Qwen3-235B (FP8), mixed pool | clean × clean | 552 | −0.062 | +0.014 | | 1 of 2,256 | 20.0 |
| | clean × weak | 576 | −0.749 | +0.083 | | | |
| | weak × clean | 576 | −0.565 | +0.008 | | | |
| | weak × weak | 552 | −1.273 | +0.056 | | | |

Weak tokens copy an order of magnitude worse than clean ones, and the whole of that difference is the additive main effect. The interaction is positive in every condition, at every separation and on all three models, and it grows with the model: two weak tokens copy better together than the sum of their separate effects predicts, marginally at 7B and 32B and by 0.06 log-probability at 235B, where the pool's weak tokens fail far more often (joint score −1.27). No pool at any size produces a pair that copies worse than its parts beyond what a matched additive null yields (1 significant pair of 2,256 at 235B against a null 95th percentile of 20). Repetition runs opposite to the compositional account. Along the diagonal of the 7B's clean pool the mean interaction was +0.0066 (permutation p = 0.000), increasing with separation from +0.005 at d = 2 to +0.018 at d = 128; on the 32B, +0.0095 (p = 0.012); on the 235B, +0.15 to +0.28 at every separation (p = 0.000). A healthy token repeated at a distance is copied more reliably the second time because its first occurrence is in context, which is induction operating as intended. The failures that did occur factor as a susceptible token in an unfavourable context: on the 7B, five tokens accounted for 47% of failing carriers and did not recur across filler sequences (carriers failing in both sequences at each separation: 3, 0, 1, 0, 0, 0, 2); on the 32B, five tokens accounted for 57%, and the pairs the family-wise test flagged were pairs among the three most fragile members of the pool. That is the structure an additive model absorbs and an interaction term cannot represent. The object to measure is the token.

## 7. Tokens inside tasks

The copy probe of §4 asks whether a token survives when a model reproduces text around it. Two further experiments ask what happens when the token is part of a job to be done. In both, the token is spliced by id into the user turn under the model's own chat template, and every episode is run for three classes of token drawn from the model's own fragility matrix: **glitch** (the verified class on the 7B; on the other models the tokens that fail alone and in at least 90% of contexts), **healthy** (passes the greedy gate and never fails in any context) and **fragile** (passes the gate and fails in at least 10% of contexts), ten tokens per class. The healthy class is the control: it shows how much of the loss is the task's, so that the effect of a token class is its gap to healthy on the same task. Gaps carry bootstrap 95% intervals over episodes.

### 7.1 Simple tasks

Five tasks drawn from a bank of ten, each solvable only by writing the token back: name a Python function after it, quote a document title containing it, name the second item of a list containing it, correct the spelling of a sentence containing it, summarise a paragraph in which it is a proper noun. Fifty episodes per class per model.

**Table 8.** Token string returned in the answer, simple tasks.

| model | glitch | healthy | fragile | fragile − healthy | glitch − healthy |
|---|---|---|---|---|---|
| Qwen2.5-7B | 0.00 | 0.86 | 0.74 | -0.12 [-0.28, +0.04] | -0.86 [-0.94, -0.76] |
| OLMo-2-7B | 0.00 | 0.92 | 0.58 | -0.34 [-0.50, -0.18] | -0.92 [-0.98, -0.84] |
| Qwen3.8-27B | 0.08 | 0.86 | 0.62 | -0.24 [-0.40, -0.08] | -0.78 [-0.90, -0.66] |
| Qwen3-32B | 0.00 | 0.90 | 0.74 | -0.16 [-0.30, -0.02] | -0.90 [-0.98, -0.80] |
| Qwen3-32B, thinking | 0.00 | 0.90 | 0.74 | -0.16 [-0.30, -0.02] | -0.90 [-0.98, -0.82] |

Task completion (the function is written, the title is quoted, the list item is named) is 0.96 to 1.00 in every cell of every model. What varies is whether the given token is the one in the answer. Glitch tokens are returned in 4 of 250 answers across the five models, all four on the Qwen3.8-27B; the 27B names a function `arnings` for `echslungs`, `atmosphere` for a Thai glitch token, and `不смотр` for ` Несмотр`, and quotes a title with the token deleted ("Notes on  and related matters"). Fragile tokens are returned less often than healthy ones, with the gap clear on OLMo-2-7B and Qwen3-32B and inside its interval on Qwen2.5-7B, and the task decides how much: quoting a title returned nine or ten of ten fragile tokens on every model, while the spelling-correction task, which invites the model to change something, returned one of ten on OLMo (` according` became "correct", ` nær` became "near", ` MaterialPageRoute` became "PDF") against eight of ten healthy tokens. The substitutes are the modes of §4.4 appearing inside instructions: ` который` became `борт` as a list item, ` realtà` became "reality" and `بدو` became "BDU" in a corrected sentence, and Qwen3-32B, asked for a function "whose name is exactly `แก้ปัญหา`", wrote `def exactly_1234567890_sum(...)`, the unreadable token deleted from the instruction and the neighbouring word slid into its slot. Asked for a function named ` PodsDummy`, OLMo wrote `def peuxTuer(...)`.

### 7.2 Agentic tasks with tools

Five open-ended assignments (investigate why a configuration key is ignored; write a background note with a cited source; recommend whether to adopt a library; look up and explain a term; check whether the latest release changed its defaults), each given three tools the model may call for up to six rounds: `search`, `open_page`, `run_python`. The tools are mocked by a world that knows about exactly the token the model was given: a search whose query contains the token returns three relevant hits (documentation, release notes with version 2.4.1, an open issue), each page can be opened, and importing the library prints its version; a search for anything else returns no results and two generic pages. Qwen models use their native tool-calling template; OLMo-2, whose template has none, follows a textual protocol, which lowers its healthy baseline and is why its rows are read as a small-model floor rather than as the effect.

**Table 9.** Agentic tasks: whether the first search contained the token the model was given, whether the token appears in the final answer, and whether the episode fabricated a version number with no retrieval behind it. All episodes reached a final answer; no model repeated an identical tool call or hit the round cap.

First search query contains the token

| model | glitch | healthy | fragile | fragile − healthy | glitch − healthy |
|---|---|---|---|---|---|
| Qwen2.5-7B | 0.00 | 0.62 | 0.38 | -0.24 [-0.42, -0.04] | -0.62 [-0.76, -0.48] |
| OLMo-2-7B | 0.00 | 0.46 | 0.32 | -0.14 [-0.34, +0.04] | -0.46 [-0.60, -0.32] |
| Qwen3-32B | 0.00 | 0.96 | 0.74 | -0.22 [-0.36, -0.08] | -0.96 [-1.00, -0.90] |
| Qwen3-32B, thinking | 0.00 | 0.66 | 0.60 | -0.06 [-0.24, +0.14] | -0.66 [-0.78, -0.52] |

Final answer mentions the token

| model | glitch | healthy | fragile | fragile − healthy | glitch − healthy |
|---|---|---|---|---|---|
| Qwen2.5-7B | 0.00 | 0.76 | 0.74 | -0.02 [-0.18, +0.16] | -0.76 [-0.86, -0.64] |
| OLMo-2-7B | 0.00 | 0.80 | 0.40 | -0.40 [-0.56, -0.22] | -0.80 [-0.90, -0.68] |
| Qwen3-32B | 0.00 | 0.96 | 0.68 | -0.28 [-0.42, -0.14] | -0.96 [-1.00, -0.90] |
| Qwen3-32B, thinking | 0.00 | 0.72 | 0.52 | -0.20 [-0.38, -0.02] | -0.72 [-0.84, -0.60] |

Fabricated a version number with zero retrievals

| model | glitch | healthy | fragile | fragile − healthy | glitch − healthy |
|---|---|---|---|---|---|
| Qwen2.5-7B | 0.00 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| OLMo-2-7B | 0.00 | 0.00 | 0.02 | +0.02 [+0.00, +0.06] | +0.00 [+0.00, +0.00] |
| Qwen3-32B | 0.00 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| Qwen3-32B, thinking | 0.02 | 0.00 | 0.00 | +0.00 [+0.00, +0.00] | +0.02 [+0.00, +0.06] |

Read against the healthy column, Qwen3-32B in plain mode is the clean case: it searched for healthy tokens by name 96% of the time and wrote them up with the retrieved version and issue; fragile tokens reached the world 74% of the time and appeared in 68% of final answers; glitch tokens reached the world in none of fifty episodes. The transcripts show how. Asked to investigate "the configuration key `ớ`", the model searched for *configuration key ignored environment variable service*, the sentence with the token removed; asked about adopting `ล่า`, it searched for *adopting  in stack benefits and drawbacks*, two spaces where the name had been; asked to check the latest release of a glitch-named library, it searched for "2", "2048" or "2024"; asked for a background note, five of ten glitch tokens became *quantum computing definition and facts*. Every such episode then ended with a fluent recommendation assembled from the two generic pages the empty search returned: *It appears that the service may not be correctly reading the configuration key from the environment variable due to library-specific precedence rules. To resolve this…* The model did not fabricate a version number (it never did, in any class); it generalised where it had retrieved nothing and presented the generalisation as the finding.

Fragile tokens failed as the copy matrix predicted: ` realtà` was searched for as *configuration key réalité ignored*, the same Italian-to-French translation §4.4 recorded; ` והת` was searched for as "תת"; the garbage-byte token was read as the katakana "ｷ" and researched as such, ending in a confident note about Japanese kana.

Thinking mode lowers the healthy baseline (the model plans generically, "search for the service's documentation", and names the token in its first query 66% of the time) and leaves the ordering unchanged; 11 of 50 glitch episodes made no tool call at all and were answered from the trace alone. The traces show the substitution being made before the first action. Given `ớ`, the trace opens *The user says that the configuration key "лок" is being ignored when set via an environment variable*, and the final answer recommends *set the environment variable using uppercase letters (e.g., `LOK=value` instead of `лок=value`)*: a diagnosis and a fix for a key that was never in the prompt. Given `อังกฤษ`, the trace reads *the configuration key 'something' (probably a placeholder)*, the one case in fifty where a model registered that it could not read what it had been given, and it then answered generically rather than asking.

Two things did not happen. No model looped: every episode of every class ended within the round cap, with 1.0 to 1.9 tool calls per episode and no repeated identical call. And no model, given nothing back from its tools, invented the version or the issue; what they did instead was complete the assignment on generic material and say so in a register indistinguishable from a grounded answer.

## 8. Method notes needed to read the numbers

### 8.1 Prompts are built from ids

A prompt assembled as a string passes through the tokenizer, and a leading space re-segments approximately nine tokens in ten into different ids regardless of their status: verified glitch tokens survive a preceding space in 8.3% of cases and healthy tokens in 9.3%. A string-built prompt therefore measures the tokenizer. Every prompt in this paper is assembled from token ids, and every copy score is the teacher-forced log-probability of the target id at its copy position. The reported rule that certain tokens "cannot be preceded by a space or bracket but can follow a newline" is the pre-tokenizer's boundary set; it applies to nearly the whole vocabulary and carries no information about which tokens are defective.

The copy prompt has one framing per model, fixed before the run and recorded with every result. For every model up to the 235B the few-shot demonstrations and the text form one raw sequence ending in `Copy:`. The Qwen3.8 models answer that cue with a blank line and then copy from the start of a line without the leading space, so the exact id fails for ordinary words (3 of 2,000 common words copied). For them the instruction, the demonstrations and the text are placed in a user turn of the chat template and `Copy:` opens the assistant turn, with thinking disabled; 1,925 of the same 2,000 words then copy. The token ids, the demonstrations and the scoring are identical under both framings.

### 8.2 Two gates

Section 4.1 describes them. Under the paper's absolute gate the 7B's numbers are 1.5 to 1.6% fragile with 28 geometry positives on the replication bank; under the greedy gate, 2.7% and 60. Both are reported wherever they differ, and cross-model statements use the greedy gate.

### 8.3 Two reproduction metrics, and what the bare probe measures on the 72B

Seed reproduction in §3 was originally scored as *the seed's token id appears in the generated ids*. A quoted "_ghost" re-segments as `_` + `ghost`, so that metric counts it as not reproduced. A string-level metric (the seed's text appears anywhere in the output) agrees with the id-level one on the 7B and the 32B, whose worst tokens are not re-segmentable, and diverges on the 72B, where three of eight logged worst tokens are quoted verbatim (Table 2). Both are recorded from here on; on the 32B, rerun with both metrics, the worst tokens are reproduced 0 of 32 by string as well as by id, so its substitution is total. The 72B's worst-by-probe tokens (`_ghost`, `$filter`, `/topics`, log-probability −24.6 to −19.2 alone) also copy correctly in 48 of 80 specimen cells: for that model, what these tokens fail is the few-shot probe format, not their contexts. The same holds in part on every model. Of the 100 tokens each model copies worst alone, the number that also fail in at least 90% of contexts is 43 on the 7B, 65 on the 1.7B, 78 on the 32B and 72 on the 72B; the remainder are probe-format failures such as `.currentTarget`, `.r` and `.y` on the 1.7B, which copy perfectly in context. The single-token probe and the in-context measurement therefore disagree in both directions, and a label-free glitch class must be verified in context. Throughout the paper the label-free glitch class is the worst tokens by the probe that also fail in at least 90% of contexts. On the 7B that definition reproduces the labelled glitch direction exactly (AUC 0.879 either way), where the bare, unverified worst-100 gives 0.642.

### 8.4 Reproducibility across hardware and context banks

Two stored matrices were re-scored at batch size 1, with no padding and no batching: the 32B on a different machine from its original run and the 72B on the same class of node. Cell-level fail/pass agreement with the stored matrices was 0.999 and 0.996, the correlation of per-token fragility 0.9999 and 0.9996, and no token's fragility moved by more than 0.1. The 32B pipeline was then run again in full on the second machine with a doubled token sample and a freshly drawn context bank: single-probe scores were identical to the original (correlation 1.0000), and fragility measured on the two independent banks correlated 0.984 over the 4,200 shared ids, with 12 of the original 15 fragile tokens fragile again. The numbers in this paper are properties of the models, not of the runs.

### 8.5 What could not be run on the FP8 mixture-of-experts models

Qwen3-235B-A22B in its FP8 release runs the batched, teacher-forced stages at about one second per forward on four H100s, but generates at about seven seconds per token under this version of transformers' FP8 expert path (a profile shows ordinary expert forwards and the multi-card device hooks, one CPU core saturated, the GPUs idle). Sixty-four chains of 256 tokens would take a day. Its reasoning-mode measurement, the eight-token continuations in the specimen store, and the task and agentic probes were therefore not run; its specimen store carries the emission at each cell (top-5, entropy, mode) without the continuation. The same restriction applies to any FP8 MoE model on this stack, and is stated where the tables leave a cell empty.

Qwen3.8-Flash-Next-FP8 (512 experts, 48 layers, hybrid linear and sparse full attention, a 51 GB per-layer n-gram embedding) needed three changes to load and run under transformers 5.16.1, none of which touches a weight or a logit: a per-card memory cap so the fused-expert conversion has headroom; re-attaching the per-tensor scale of the FP8 n-gram embedding table, which the loader discards as an unexpected key, as a rescaling of the lookup; and an exact shortcut for the sparse-attention indexer, whose per-query Python loop pinned one CPU core with the GPUs idle. The shortcut is valid because every sequence scored here is within the indexer's 2,048-token budget, where its top-k over blocks keeps every block and its selection is the identity. Under these fixes the model runs the teacher-forced stages at the 235B's pace; generation is not attempted, for the reason above.

### 8.6 Pre-registration and amendments

The design, null model and verdict criterion of §6 were fixed before the screen ran; two amendments followed synthetic calibration and preceded any real data (standardised residuals; the matched-simulation threshold, since a true null yields 0 to 2 rejections rather than the ~200 a fixed threshold assumed). The scale ladder was pre-registered separately (`docs/plan_scale.md`) with seven quantities, their replication ranges and five hypotheses. Four amendments were made during the ladder, each before the data it affects was examined: the greedy gate; a doubled token sample (8,400) for the two largest rungs, with the sampler changed so that the first 4,200 draws are byte-identical to the smaller runs; the string-level reproduction metric; and a printable-string filter on the label-free seed set. One defect is recorded: the sampler stratifies over the embedding matrix's row count, and Qwen2.5 pads its vocabulary to 152,064 against Qwen3's 151,936, so the 72B's sample shares only 1,545 ids with the Qwen3 rungs'. Whether the same token ids are fragile across sizes within a tokenizer family is therefore untested rather than tested and failed.

## 9. Scope

The behavioural results of §§3 to 6 are from one 7B model with independent labels, replicated label-free on seven Qwen models of three generations from 1.7B to 235B parameters (reasoning mode on five of them); the additivity result on three; the layer-wise identity analysis on Pythia-1.4b. The 235B rung has its copy-based measurements and no generation-based ones (§8.5).

Copy log-probability is a proxy for the reasoning-mode behaviour that motivates the work. It was adopted because it is continuous and deterministic, which experiments over thousands of tokens require, and because every generation inspected alongside it agreed with it. The contexts are random sequences of common words rather than natural text; the finding that no context is systematically hostile is a finding about such banks.

The embedding-geometry result of §5 is established in direction more firmly than in size on the 7B, and §5.3 shows the direction depends on the fragile population and on the reference class used to define the glitch direction. Which context elicits a given token's failure is not explained; the continuation-concentration and copy-suppression hypotheses of §4.4 each require one forward pass per token to test.

Confident substitution resembles hallucination in structure, and the question whether token fragility underlies one class of it, the near miss on a rare named entity, was tested directly on 1,255 real code identifiers and proper nouns in a natural task. Fragility of the entity's constituent tokens did not predict an incorrect name beyond the entity's rarity, its constituents' single-probe scores and the model's log-probability of the whole entity (change in AUC −0.001, 95% CI [−0.008, +0.006]). Fragility and hallucination share rarity as a cause; on this evidence they share nothing further.

## 10. Conclusion

Glitch behaviour is a property of the token, and it is confident. Tokens at a distance do not interact, and a repeated healthy token is copied more reliably rather than less. A glitch token's identity reaches the final layer of the network; the readout substitutes a plausible neighbour with high confidence, and the model completes its task on the substituted content, the more capably the larger it is. Between one in a hundred and one in ten tokens that pass every single-context probe fail to copy in at least a tenth of ordinary contexts, by deletion, neighbour substitution, truncation or translation, with the triggering context unpredictable but the property stable across context banks and present at every model size tested. On the model with independent labels the property is written in the weights along the direction verified glitch tokens share; on a model whose fragile tokens are the rare CJK tail of its vocabulary, rarity reads it as well as geometry does; on every untied model, the token's representation in context reads it better than rarity. Detectors that score tokens in isolation measure the wrong object for systems that assemble text programmatically. Fragility costs one forward pass per token per context, is adequately estimated from two dozen contexts, and identifies tokens such as `according` that isolation-based probes certify.

## Appendix A. Hypotheses rejected by their controls

Recorded so that the extent of the tested ground is clear.

- That the "no preceding space" rule identifies glitch tokens: it identifies the pre-tokenizer boundary; healthy tokens obey it at the same rate (§7.1).
- That contiguous runs of individually perfect tokens interact: reproduction falls from 1.000 at n = 1 to 0.648 at n = 8, but independence from the n = 2 rate predicts 0.354; the decline is sub-multiplicative.
- That the substituted token can be predicted from the direct path E_in·E_out (top-1 agreement 0.003), or that substitutions are orthographic completions (11% of failures).
- That generation length amplifies damage, or that glitch-seeded chains converge (§3.1).
- That fragile tokens are non-canonical Unicode variants: all canonical (§4.4).
- That post-training creates new glitch tokens: real movement in one lineage, boundary jitter in five others (§5.4).
- That fragility accounts for near-miss hallucination on real entities (§8).

## Appendix B. Data and code

`docs/DATA_MANIFEST.md` maps each section to its files. The 7B matrices are in `results/compact/` with the replication bank's hidden states in `results/fragility_L.pt`; the specimen stores are `results/specimens_*.jsonl`; the pre-registration documents are `whimsical-greeting-lerdorf.md` (§6) and `docs/plan_scale.md` (the ladder); code is in `src/cut/`, figures from `src/cut/figures_paper.py`. Task and agentic probes (§7): `src/cut/task_probe.py`, `src/cut/agent_probe.py`, `results/task_probe_<model>.json` and `results/agent_probe_<model>.json` (every generation and transcript), summarised by `src/cut/task_probe_summary.py` into `docs/task_probe.md`. Scale-ladder files: `results/gate_<model>.json`, `fragility_predict_<model>{,_greedy}.json`, `reasoning_<model>{,_think}.json`, `specimens_<model>{,_greedy}.jsonl`, `interaction_screen_qwen3_32b.json`, compact matrices in `results/compact/`, and complete snapshots of each node's logs and results in `results/remote_h100/`. The 7B experiments ran on one A100-40GB (longest run 25 minutes); the ladder on one H100-80GB (1.7B, 32B) and one 4× H100-80GB node (72B). Calibration of the §6 null: on matrices in which one token in eight was three times noisier than the rest, an unstandardised pooled permutation produced 5 to 18 false rejections and family-wise p ≈ 0.003 in every run; standardising residuals by row and column produced zero false rejections in ten of ten null runs. A single isolated interaction is recovered in 4 of 5 simulations at 1.5 times the cell noise with 14 replicates.
