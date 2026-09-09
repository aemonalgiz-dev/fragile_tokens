# Scale ladder: do the fragile-token results hold on larger and more capable models?

Written before any of these models were run, 6 September 2026. Reference row: `allenai/OLMo-2-1124-7B-Instruct` (results in `docs/paper_fragile_tokens_v3.md`).

## The ladder

Four rungs at roughly 2B, 30B, 100B and 300B parameters, one model each, all on **one tokenizer** (Qwen2.5 and Qwen3 share their 151,936-entry vocabulary), so that whether the *same token ids* stay fragile across a 140× range in parameters is a direct question. The OLMo-2-7B result already in hand is the cross-family reference row. Every rung is bf16 or an official FP8 release; no rung is int4, so no quantisation control is required.

| rung | model | params (active) | precision | fits on | thinking | embeddings |
|---|---|---|---|---|---|---|
| ~2B | Qwen3-1.7B | 1.7B | bf16 | 1× H100 | yes | **tied** — static tier reported, not held to the bar |
| ~30B | Qwen3-32B | 32B | bf16 (64 GB) | 1× H100 | yes | untied |
| ~100B | Qwen2.5-72B-Instruct | 72B dense | bf16 (145 GB) | 4× H100 | no | untied |
| ~300B | Qwen3-235B-A22B-FP8 | 235B (22B active) | Qwen's official FP8 (235 GB) | 4× H100 | yes | untied |

Repository ids verified 6 September 2026. Qwen2.5 pads its vocabulary to 152,064 against Qwen3's 151,936; the real token ids are identical, and cross-rung comparisons are made on ids, not vocabulary size. No official FP8 exists for Qwen2.5-72B; rather than a third-party `compressed-tensors` build, it runs in bf16 on the four-card node, so the ladder contains no quantisation that is not the vendor's own release.

Label-free operation throughout: the fragility matrix needs no labels; the reasoning-mode and specimen stages take their glitch set as the worst tokens by the single probe from that model's own matrix, and controls from tokens it copies perfectly. The interaction pool (N = 48, mixed) runs at the 32B and 235B rungs.

If a dense model is preferred at the top rung, Llama-3.1-405B in Meta's own FP8 (405 GB, 8× H100) substitutes for Qwen3-235B at the cost of the tokenizer chain and roughly triple the node cost.

## What "the results stay the same" means, fixed in advance

For each model, the quantities and the range that counts as replication:

1. **Fragility prevalence** among clean-looking tokens (single probe > −0.1), threshold frag ≥ 0.10 over 24 contexts: **0.5–3%**. Below 0.5% is "fragility vanishes with scale"; above 3% is "worse than the 7B", either of which is a finding.
2. **Context share of variance** on the clean-looking log-probability matrix: **< 0.01**. Above that, some contexts are hostile in general for this model, which would be new.
3. **Stability**: fragility on even contexts predicts odd at AUC **≥ 0.95**.
4. **Failure character**: mean entropy at failing cells among fragile clean-looking tokens **< 2 bits**, and deletion/substitution/truncation/translation modes present in the specimen store. Entropy ≥ 2 bits would mean larger models fail *uncertainly*, which changes the safety story.
5. **Geometry**: static CV AUC vs surface, paired bootstrap Δ **> 0 with CI excluding 0**; for tied-embedding models (Gemma, some Qwen sizes) the static tier is reported but not held to this bar, since E_out = E_in.
6. **Reasoning-mode substitution**: seed reproduction glitch ≪ control, and glitch-seed entropy **< 2 bits**. With thinking on: the same, or a documented change.
7. **Additivity** (13B and 32B only, mixed pool N = 48): significant pairs ≤ matched-null 95th percentile, and mean I(weak × weak) within **±0.01**.

## Scale hypotheses (stated so they can fail)

- **H1** Prevalence falls with parameter count within a family (13B < 7B; 32B < 14B) but does not reach zero; the confident character (item 4) persists at every size.
- **H2** The same token ids are fragile across sizes within a family sharing a tokenizer: Jaccard of the fragile sets (OLMo-2 7B vs 13B) **> 0.3**. Across tokenizers, overlap by *string* among the top-50 fragile tokens is small.
- **H3** Thinking mode reduces *deletion* (the model has room to notice the token) but not *substitution*; seed reproduction with thinking on is higher than off but remains below the control rate.
- **H4** Static geometry keeps its advantage over surface rarity at every size; the advantage narrows as the surface tier's AUC rises with scale.
- **H5** Additivity holds at every size. A positive interaction at 32B would be the first compositional effect seen and would take precedence over everything above.

## Compute

One **4× H100-80GB** node (~$12/h) runs the whole ladder; the 235B FP8 checkpoint is what requires the fourth card. Estimated wall time per rung, at 4,200 tokens × 24 contexts plus reasoning (×2 where thinking exists), specimens, and the interaction pool where scheduled: 1.7B ≈ 15 min; 32B ≈ 2 h; 72B bf16 ≈ 3 h; 235B-A22B FP8 ≈ 2.5 h (inference scales with the 22B active). **Total ≈ 8–9 h, about $100**, plus downloads (the 235B checkpoint is 235 GB).

Cheaper split: run the first two rungs on a 1× H100 (~$3/h, ~2.5 h), then a 4× node for the 72B and 235B only (~5.5 h). Same total time, about $75.

Instance setup: `pip install -U torch transformers accelerate safetensors scipy`; sync `src/` and `external/`; no HF token needed for any rung. FP8 checkpoints load through the repository's own `quantization_config`; pass `--dtype bf16` if a loader complains.

## Files

`src/cut/loadmodel.py` (bf16 / device_map / NF4 loader, safetensors embedding reader), `run_scale.sh` (the ladder, resumable), `scale_summary.py` (the cross-model table), and the label-optional versions of `fragility.py`, `fragility_predict.py`, `reasoning_drift.py`, `specimens.py`, `interaction_screen.py`.

## Status (6 September 2026, evening)

Rungs 1–2 ran on a 1× H100 (`docs/findings_scale.md`). The absolute −0.1 gate proved model-specific (it admits 353 tokens and no fragile one on the 32B); cross-model comparison uses the greedy gate p_alone > 0.5 (`gate_summary.py`, `--gate-lp -0.693`), under which items 1, 2, 6 and 7 hold at both rungs, item 5 holds on the 7B reference with 60 positives and is inconclusive at 32B with 12. **Amendment for rungs 3–4:** raise `N_TOK` to 8,400 so the greedy-gated positive count reaches ~30 at ~1% prevalence; the fragility matrix cost doubles, nothing else changes.
