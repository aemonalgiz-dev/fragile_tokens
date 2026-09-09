# Prior Art Survey: Glitch Token Detection (as of 2026-09)

## 1. The core glitch-token literature (complete; arXiv full-text search for
##    "glitch token" returns exactly 6 papers)

| Work | ID / venue | Signal used | Requires | Granularity |
|---|---|---|---|---|
| Coercing LLMs to do and reveal (almost) anything | 2402.14020 | ad-hoc, security framing | trained model | single token |
| GlitchHunter (Categorization Taxonomy & Effective Detection) | 2404.09894, FSE/PACMSE 2024 | embedding-space clustering (iterative token-embedding graph) | trained model | single token |
| Fishing for Magikarp | 2405.05417, EMNLP 2024 | tokenizer analysis + unembedding-weight indicators + prompting | final checkpoint | single token |
| AnomaLLMy | 2406.19840 | low-confidence single-token predictions, black-box API | deployed model | single token |
| GlitchProber | 2408.04905, ASE 2024 | mid-layer activations + attention deviation, PCA + SVM | trained model, white-box | single token |
| GlitchMiner | 2410.15052, AAAI 2026 | entropy maximization + gradient-guided local search | trained model, gradients | single token |
| One Bad Token Spoils the Barrel (GlitchQuiz / GlitchEdit) | USENIX Sec 2026 | 8 behavioral task templates; repair by embedding interpolation | trained model | single token |
| Check Your LLM's Secret Dictionary | 2605.22005 | SVD of `lm_head`, no inference | final weights | single token |

**Invariant across all eight: post-hoc, single final checkpoint, one token at a time.**

## 2. Adjacent: prevention at tokenizer-training time
- BPE Gets Picky (2409.04599) — prunes under-trained intermediate merges *during tokenizer training*.
- From Where Words Come / SA-BPE (2604.14053, Apr 2026) — source-attribution regularization of code tokenizers to minimize under-trained tokens.
- Problematic Tokens: Tokenizer Bias (2406.11214) — tokenizer/model corpus mismatch, multilingual.
- UTF (2410.12318) — *uses* under-trained tokens as fingerprints.
Both prevention works act before LLM pretraining on static corpus statistics. Neither observes the model learning.

## 3. Adjacent: training dynamics of embeddings (no glitch framing)
- Rare Tokens Degenerate All Tokens (2109.03127) — training dynamics of rare-token embeddings; adaptive gradient gating. Closest mechanistic ancestor.
- A Hub of Short Rows Inflates ID Estimation (2608.29702, Aug 2026) — **caveat**: the low-norm hub overlaps under-trained-token detector flags, but the authors find those rows *were* updated during training; "what seems to characterize these rows is simply their length, not an absence of updates."
- Sticking to the Mean: Sticky Tokens in Text Embedding Models (2507.18171) — 40 checkpoints, but embedding models, cross-model not cross-time.
- Pythia (154 ckpts/model) and OLMo checkpoint suites make trajectory work tractable.

## 4. Adjacent: multi-token phenomena (three distinct bodies, none = under-trained n-grams)
(a) **Adversarial** — Universal Adversarial Triggers (1908.07125), GCG, 2404.16020 (triggers do not transfer, even within a family). Multi-token, but *synthesized by gradient descent against an objective*; not attributed to under-training.
(b) **Non-canonical tokenization** — sequences the tokenizer can never emit, hence exactly zero frequency in pretraining:
    - Broken Tokens? (2506.19004) — **models are surprisingly robust**: up to 93.4% of original performance on random tokenizations never seen in training.
    - LMs over Canonical BPE (2506.07956); Where is the signal in tokenization space? (2408.08541); Tokenization as FST (2410.15696); Not Equally Robust Across Languages (2607.26831); Emergent retokenization symmetry (2606.15521).
(c) **Prompt boundary / token healing** — practitioner-level, single boundary, no detection framework.

## 5. Verdict
- **Idea 1 (glitch formation across training):** no prior work. Open.
- **Idea 2 (n-gram / compositional glitch):** open in the *under-training* framing; must be
  positioned against (a) and (b) above. (b) partially falsifies the naive hypothesis.
