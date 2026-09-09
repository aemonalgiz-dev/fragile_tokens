# The token that passes every test and fails in a sentence

## Glitch-token detectors check one token at a time. That is not where language models use them.

The English word "according" is token 4184 in the vocabulary of OLMo-2-7B. It is one of the most common pieces in the language, and by every published test it is healthy: ask the model to repeat it on its own and it does so with probability 0.999.

Now put it in a sentence of random ordinary words and ask the model to repeat the sentence. In the context *deliver rel behind according sent floor wrong*, at the position where the copy should produce "according", the model produces "sent" with probability 1.000 and entropy of zero bits, then carries on copying correctly from that word. Across 48 such contexts the token was deleted in 44. In the other four it was copied at 0.999.

Nothing about the token changed between the two measurements. The first was made in a setting where the copying mechanism has one candidate. The second was made in the settings a token ordinarily occupies. Every glitch-token detector I know of makes the first measurement, and every system that assembles prompts programmatically makes the second on every call.

I spent the last weeks measuring how much the first measurement misses, on eight open models from 1.7 billion to 235 billion parameters. This is what I found.

## What a fragile token is

A glitch token, in the sense the field has used since the SolidGoldMagikarp discovery in GPT-2, is a vocabulary entry the model cannot reproduce even on its own. Detectors such as Magikarp, GlitchHunter, GlitchProber, GlitchMiner and GlitchQuiz find these well, and they agree with each other.

A fragile token is different. It passes the single-token probe, so every detector calls it healthy, and it fails when it appears inside ordinary text. To measure this I built banks of 24 random sequences of common words, placed every token of a large sample into every sequence, and scored the probability of the correct token at the copy position under a fixed prompt:

```
Repeat the text exactly.
Text: apple pie is good
Copy: apple pie is good
Text: the quick brown fox
Copy: the quick brown fox
Text: <the context words, with the token in its slot>
Copy: <scored here, token by token>
```

A cell fails when the log-probability drops below −0.5. A token's fragility is the fraction of contexts in which it fails. Because every token meets the same contexts, differences in fragility are differences between tokens, not between contexts.

![Fragility matrix for Qwen3-32B](figures/fig_fragility_matrix_qwen3_32b.png)

The picture above is one model's matrix. Each row is a token, each column one of the 24 contexts, and each cell the probability of the correct token at the copy position. The top block is the glitch class, tokens that fail alone and everywhere. The middle block is stable tokens, which fail nowhere. The bottom block is the fragile tokens: they passed the probe, and the rows are dark. The structure is horizontal on every model. Rows are dark or bright; columns are not. The context explains at most 1.3 percent of the variance on any model, and a token's fragility on half the contexts predicts its fragility on the other half at 0.82 to 0.95.

## How common it is

Under one gate applied to all eight models (the token must be the greedy output of its own probe), the share of clean-looking tokens that fail in at least one context in ten:

- Qwen3-1.7B: 2.4 percent
- Qwen2.5-7B: 7.0 percent
- OLMo-2-7B: 2.7 percent
- Qwen3.8-27B: 1.5 percent
- Qwen3-32B: 0.9 percent
- Qwen2.5-72B: 10.5 percent
- Qwen3.8-Flash-Next: 0.4 percent
- Qwen3-235B: 10.9 percent

![The ladder](figures/fig_ladder.png)

Two things stand out. The two largest models by total parameters are the most fragile, and that holds on identical token ids as well as in aggregate: on the ids that both the 32B and the 235B accept, the 235B is fragile on 4.3 percent and the 32B on 0.8 percent. And the 512-expert Flash-Next, with about three billion parameters active per token, is the least fragile model of all. Whether size makes things worse depends on whether you count total or active parameters, and this ladder cannot separate the two readings.

The fragile tokens themselves are ordinary. They are word pieces from languages thin in the training mix (Spanish, Polish, Russian, Korean, Arabic, Finnish), code identifiers, and on some models the rare-character tail of the vocabulary. They are canonical strings in Unicode composed form. On the three models for which Magikarp has published its verdicts, every fragile token is in its "OK" category, its indicators never single them out, and where its prompts were run on one they passed it. The disagreement runs one way only: every glitch token Magikarp verified fails my gate and my contexts.

## What the model does instead

The failures are confident. On most models the majority of failing cells have less than two bits of entropy. Four modes recur everywhere. The token is deleted: the model emits the following word and continues as though the slot were empty. It is substituted with a neighbour. It is truncated to a prefix of itself. Or it is translated: the Chinese token for "query" comes out as the English "QUERY", Korean "method" as Chinese "method", Italian "realtà" as French "réalité".

Handing a model a fragile token as its entire prompt shows the same thing in generation. No model asks what it was given. Qwen3-1.7B copies the Japanese token 続きを読 back correctly and then explains that it is Korean. Qwen2.5-7B, given the Hebrew word לציין, explains the Chinese word for "pointer". Qwen3.8-27B, given the Russian stem Председа, transliterates it to "Predseda" and treats it as a generic Slavic word; given the Latin string "anzus" it produces Arabic script and an Arabic girl's name. OLMo-2-7B, given the German "allerdings", writes an essay on artificial intelligence.

Glitch tokens produce the extreme version. Qwen3-32B, given five different Thai words as the whole prompt, evaluates a continued fraction, solves Ramanujan's nested radical, proves that the first n odd numbers sum to n², and reports that the user "sent a single X character". Qwen3.8-27B answers "What is the capital of France?" when given a token no human has ever typed. Nothing in the trace signals that the input was unreadable. On Pythia-1.4b, where labels exist, the token's identity is recoverable from the final-layer hidden state as reliably as a healthy token's, so the failure is at the readout, not in the representation.

## Where fragile tokens live

The detectors' geometric intuition is right, but the geometry they use is too simple. Magikarp scores a token by its position relative to the mean of the under-trained rows. Fragile tokens do not sit along that single direction on every model; on Qwen2.5-72B the projection separates them from stable tokens at an AUC of 0.60, which is barely better than chance.

They do sit next to glitch tokens. Score each token by its cosine proximity to the model's glitch class in embedding space and the separation is 0.74 to 0.90 on seven of the eight models, 0.80 on that same 72B. A random reference set of the same size gives 0.41 to 0.55, so the relation is to glitch tokens specifically. The relation is a neighbourhood, not an axis, which is what the UMAP of the unembedding shows and a one-dimensional projection cannot.

![Where the fragile tokens sit](figures/fig_geometry.png)

Each row is a model. The first column is a UMAP of the unembedding rows with the glitch class in black and fragile tokens in red. The other columns show one score each, as box plots for the three classes: glitch on top, stable, fragile. A fragile box that sits away from the stable box and towards the glitch box is the relation the text describes, and on most models it does so on the proximity score even where it does not on the projection.

## Tokens do not combine to do this

One account of glitch behaviour in the wild is compositional: particular pairs or repetitions of otherwise healthy tokens, possibly far apart, produce it. I tested this with a pre-registered factorial screen over ordered pairs at separations of 2 to 128 positions on three models. No screen produced more interacting pairs than a matched additive null yields, and a repeated token copies better the second time, not worse, because its first occurrence is in context. The object to measure is the token.

## Inside real tasks

Copying is a proxy. Inside instructions the same tokens fail the same way. Asked to write a Python function named exactly after a fragile token, Qwen3.8-27B named it Przewodniczący, the Polish for chairman, when the token was the Russian stem Председа. Asked to correct the spelling of a sentence containing "realtà", Qwen3-32B returned the sentence with "reality" in its place. Asked to quote a title containing the Hebrew piece והת, it quoted the title with the real tax form "I-9" in its place. Every task was completed. Only the token changed.

With tools it gets more consequential. In a mocked research environment whose search index knows exactly one library, the one named by the token in the assignment, Qwen3-32B given a glitch token searched for the sentence with the token removed, retrieved nothing, and delivered a confident recommendation assembled from generic pages, in 50 of 50 episodes. Fragile tokens reached the index in three quarters of episodes; stable ones in 96 percent. Qwen3.8-27B did something different: it kept calling tools. Given a glitch token it repeated identical calls in 28 percent of episodes and hit the six-round cap in 86 percent. No model, in any class, invented a version number when its tools returned nothing. They finished the job on generic material and said so in a register indistinguishable from a grounded answer.

## What to do about it

Fragility costs one forward pass per token per context, is adequately estimated from two dozen contexts, and is stable across independent context banks and across hardware. It identifies tokens such as "according" that isolation-based probes certify. If you build systems that assemble prompts programmatically, the token you should worry about is not the one that fails alone. It is the one that passes alone and fails in the sentence you happened to build.

The full paper, with every table, every prompt and the model provenance of every failing case, and the datasets behind it (fragility matrices, specimen stores, reasoning-mode generations and agent transcripts for all eight models) are available from me on request.
