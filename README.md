# fragile-tokens

Find the tokens a language model cannot copy inside ordinary text, including the ones that pass every
single-token glitch probe.

A *glitch token* fails when the model is asked to repeat it on its own. A *fragile token* passes that
probe and fails when it appears inside a sentence: the model deletes it, substitutes a neighbour,
truncates it or translates it, usually with high confidence, and then carries on with the task as if the
substituted token were the input. Every published detector measures the first case. This tool measures
the second, the way a system that assembles prompts programmatically encounters tokens on every call.

The method is the one in *Fragile Tokens: context-dependent copy failure on tokens that pass
single-token glitch probes* (Gordon, 2026; `docs/paper_fragile_tokens_v6.md`, arXiv package in `arxiv/`).
On eight open models from 1.7B to 235B parameters, between 0.4% and 10.9% of tokens that pass the probe
are fragile, the property belongs to the token rather than the context, and fragile tokens sit next to
glitch tokens in embedding space.

## Install

```bash
pip install -e .            # library + the fragile-tokens command
pip install -e .[gpu]       # adds accelerate and bitsandbytes for device_map / 4-bit loading
pip install -e .[dev]       # pytest and mypy
```

Python 3.10+, PyTorch, transformers. A GPU is not required but a scan of thousands of tokens on a 7B
model is a GPU job; a scan of the tokens in one prompt is not.

## Command line

Scan a stratified sample of 4,200 tokens on 24 random contexts (the paper's setting) and write the
scan, a CSV and a Markdown report:

```bash
fragile-tokens scan --model Qwen/Qwen3-1.7B --out qwen3_1_7b.json --csv qwen3_1_7b.csv --markdown qwen3_1_7b.md
```

Scan only the tokens of a prompt you are about to ship, or an explicit id list:

```bash
fragile-tokens scan --model Qwen/Qwen3-1.7B --text "Set the configuration key realtà to false" --out prompt_scan.json
fragile-tokens scan --model Qwen/Qwen3-1.7B --token-ids ids.json --out ids_scan.json
```

Read a scan back:

```bash
fragile-tokens report qwen3_1_7b.json --top 50                     # fragile tokens, Markdown to stdout
fragile-tokens report qwen3_1_7b.json --classes glitch probe_only  # other classes
fragile-tokens report qwen3_1_7b.json --csv all.csv --json summary.json
```

Check a text against a scan, measuring any token the scan did not cover on the same context bank:

```bash
fragile-tokens check --scan qwen3_1_7b.json --model Qwen/Qwen3-1.7B --text "Quote the title: Przewodniczący"
```

Static geometry, no forward pass: score the scanned tokens by proximity to the scan's own glitch class
and rank the whole vocabulary to find candidates the scan did not reach:

```bash
fragile-tokens geometry qwen3_1_7b.json --rank-vocabulary --top 500 --out geometry.json
```

Options worth knowing: `--framing chat` for models that need the copy prompt inside their chat template
(Qwen3.5 and Qwen3.8; `auto` chooses chat whenever the tokenizer has a template), `--gate absolute` for the
stricter probe gate, `--contexts-per-length` and `--context-lengths` to change the bank, `--cpu` to keep
the model off any GPU, `--max-gpu-memory 56GiB` to spread a large model across cards. `-v`/`-vv`/`-q`
control logging; all diagnostics go to stderr through the `fragile_tokens` logger.

## Library

```python
from fragile_tokens import FragileTokenDetector, ScanConfig, TokenClass

det = FragileTokenDetector.from_pretrained("Qwen/Qwen3-1.7B")           # ScanConfig() defaults
result = det.scan(n_tokens=4200)                                        # or det.scan(token_ids=[...])
result.save("scan.json")

for r in result.by_class(TokenClass.FRAGILE)[:10]:
    print(r.text, r.fragility, r.worst.mode.value, r.worst.emitted_text)  # your code may print; the library logs

reports = det.check_text("Set the configuration key realtà to false", result)
```

`ScanResult` holds the provenance, the configuration, the context bank, the full matrix (copy log-probability,
emitted id and next-token entropy for every token at every context) and one `TokenReport` per token.
`FragileTokenDetector.geometry(result, load_embeddings(name))` returns the static scores.

## What is measured

1. **Copy prompt.** `Repeat the text exactly.` followed by two demonstrations and `Text: ... Copy: ...`,
   assembled from token ids so no re-tokenization can hide a token. Raw or chat framing.
2. **Single-token probe.** The token alone as text and copy. It passes the *greedy gate* when its
   probability exceeds 0.5 (it is the model's own greedy output), the *absolute gate* at log-probability
   above -0.1.
3. **Context bank.** Common words the model copies exactly on its own, drawn into 24 random sequences of
   8, 16, 32 and 64 tokens with one interior slot each. Every token meets the same contexts.
4. **Fragility matrix.** Teacher-forced log-probability of the token at its copy position in every context.
   A cell fails below -0.5 (probability 0.61). Fragility is the share of failing contexts.
5. **Classes.** `glitch` fails the probe and at least 90% of contexts; `fragile` passes the probe and fails
   at least 10% of contexts; `intermittent` passes and fails somewhere below that; `stable` never fails;
   `probe_only` fails the probe but copies in context (a probe-format failure); `ungated` fails the probe
   with mixed behaviour.
6. **Worst cell.** For every token, the context with the lowest copy log-probability, the id the model
   emitted there, its failure mode (deletion of the slot, truncation to a prefix, substitution) and whether
   the emission was confident (entropy below 2 bits).
7. **Geometry.** Projection of the input embedding onto the glitch direction, and mean cosine of the
   (gain-scaled) output embedding to its five nearest glitch-class rows. The AUC of each score for
   separating fragile from stable tokens is reported; the proximity score ranks the full vocabulary.

## Layout

- `fragile_tokens/`: the package (typed, `mypy --strict` clean, logging only).
  `config.py` thresholds, `prompt.py` the copy prompt, `contexts.py` banks, `scoring.py` the probe and matrix,
  `classify.py` classes and modes, `geometry.py` static scores, `report.py` results and serialization,
  `detector.py` the orchestrating object, `models.py` loading (with the FP8 mixture-of-experts fixes the
  paper's runs needed), `cli.py`.
- `tests/`: pytest suite on a fake tokenizer and a fake copying model; no downloads, no GPU.
- `src/cut/`, `scratch/`, `results/`, `docs/`: the research code, launch scripts, result files and the paper
  as they were used for the publication. They are kept as the record of the runs; the package above is the
  maintained interface.

## Citation

Gordon, Jeffrey R. *Fragile Tokens: Context-Dependent Copy Failure on Tokens That Pass Single-Token Glitch
Probes.* 2026. https://medium.com/p/0270e70c72a6
