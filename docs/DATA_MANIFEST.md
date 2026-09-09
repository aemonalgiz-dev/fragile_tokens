# Data manifest — everything pulled from the A100 before shutdown (2026-09-06)

The 8xH100 Lambda instance was shut down after this pull. Everything the
papers cite is in this repository. Model: `allenai/OLMo-2-1124-7B-Instruct`,
commit `470b1fba1ae01581f270116362ee4aa1b97f4c84`, fp16, unless a file says
otherwise. All behavioural numbers use greedy decoding.

## Where each paper section's numbers live

| paper section | primary files | notes |
|---|---|---|
| §3.1–3.2 confident substitution | `results/reasoning_drift.json`, `logs/drift.log` | 32 glitch vs 32 healthy; per-step entropy, alignment, 8 sample generations |
| §3.3 identity retention | `results/forward_geom.json` | Pythia-1.4b; retrieval by layer, logit-lens specimens |
| §4 interaction screen (clean pool) | `results/interaction_screen.json`, `results/compact/interaction_screen_compact.pt`, `logs/interaction_screen.log` | 63,014 cells; S00/S10/S01/S11 tensors, I, p, padj, rej, band tests, per-cell `lp` vectors |
| §4 mixed pool | `results/interaction_screen_mixed.json`, `results/compact/interaction_screen_mixed_compact.pt`, `logs/interaction_screen_mixed.log` | includes `kind` (clean/weak) per pool token |
| §4 confirm | `results/interaction_confirm_v2.json` (clean, normalised scorer), `results/interaction_confirm_mixed.json` | all 225 / 240 reasoning outputs saved |
| §4 geometry | `results/interaction_geometry*.json` | |
| §4 null calibration | `logs/` has no calibration log — it ran locally; rerun `python -m src.cut.calibrate_null` (numpy only) | |
| §5 fragility, pilot | `results/fragility.json`, `results/compact/fragility_compact.pt`, `logs/fragility.log` | 2,098 × 48 matrix `M`, `bank`, `single`, `tokens`, `is_glitch` |
| §5 fragility, replication | `results/fragility_L.json`, `results/compact/fragility_L_compact.pt`, `logs/fragility_L.log` | 6,273 × 24; includes the 1,973 entity constituent tokens |
| §6 geometry vs fragility | `results/fragility_predict.json` (pilot), `results/fragility_predict_t05.json`, `results/fragility_predict_L.json` (replication, with `vs_surface` paired bootstrap), `logs/predict_L.log` | |
| §5.4 canonicality check | `logs/` (run as `fragility_diag.py`, output printed) — rerun from compact pt | |
| §5.4 / specimens | `results/specimens_confident_substitution.jsonl` (228 records + `_model_info` header), `docs/specimens.md` | top-5 emissions, entropy, p(target), mode, continuation |
| §7 pretraining mechanism | `results/traj_*.pt` (8 models), `results/f2_*.json` (Amber + 4 FineWeb ablations), `results/cross_family.json` if present, `docs/findings_cross_family.md` | trajectories are 20 × V indicator tensors, no embeddings |
| §7 post-training / tying | `results/sw_*.json` (9 series), `results/stage_families.json`, `logs/sw_*.log` | boundary-jitter control in `rank_based`, `newly_base_rank_median`, `newly_beyond_5x` |
| §8.1 segmentation | `results/segmentation.json`, `logs/` (o200k output printed in transcript) | |
| §8.2 unreachable adjacency | `results/ngram_glitch.json` | |
| §8.3 n-gram reproduction | `results/ni_tested.json`, `results/ni_common.json` (decode-fixed), superseded `ngram_identify*.json` | |
| §8.4 direct path | `results/embed_predict.json` | |
| §8.5 orthographic | `results/orthographic.json` | |
| §8.6 trajectory homology | `results/traj_topology.json` (n=32; the powered rerun was never launched) | |
| §8.7 stage-relative abandonment | `results/sw_*.json`, `docs/findings_stagewise.md` | |
| §8.9 hallucination bridge | `results/entities.json` (1,255 entities, ids), `results/hallucination_bridge.json` (first scorer; all outputs), `results/hallucination_bridge_rescored.json` (final-component scorer; `cls2`, `span2`), `logs/bridge.log` | |
| earlier mechanism paper | `results/day1*.json`, `strength.json`, `attractor.json`, `battery.json`, `verification*.json`, `repair_*.json`, `repetition_*.json`, `ladder_rungs.json`, `rig_aggregate.json`, `results/remote_root/runs_small.tgz` (per-seed rig eval/logs) | see `docs/paper_draft.md` |

## Compact `.pt` files (`results/compact/`)

Identical to the remote originals with the `states` dict removed (slot hidden
states at layers L/2 and L, fp16). Everything else — matrices, ids, contexts,
statistics — is intact. Sizes: fragility 0.7 MB, fragility_L 1.1 MB,
interaction_screen 27 MB, interaction_screen_mixed 27 MB.

`results/fragility_L.pt` (3.70 GB, **with** states) was pulled in full and
verified: `M` (6273 × 24) plus `text_mid`, `text_last`, `copy_last` each
(6273 × 24 × 4096) fp16. It is what the *dynamic* geometry features of §6
(`text_norm_mean`, spreads, retention) are computed from; the static and
behavioural tiers need nothing from it.

## Not pulled, and why

- `results/fragility.pt` (2.4 GB), `results/interaction_screen*.pt` (3.0 GB each)
  with states: dynamic features for §4 geometry and the §6 pilot only; the
  compacts hold every other field. Regenerate with `run_interaction.sh` /
  `run_fragility.sh` (~15–25 min each on one A100).
- `runs/` (3.8 GB): from-scratch rig checkpoints, 6 seeds × arms. Reproducible
  with `src/rig/run_program.sh` (exact seeds in the script). Small outputs are in
  `results/remote_root/runs_small.tgz`.
- `data/` (755 MB): tokenized rig corpora; regenerated by `src/rig/build_corpus.py`
  → `train_tokenizer.py` → `tokenize_corpus.py`.
- HuggingFace cache.

## Also archived

- `results/remote_root/`: root-level logs from the earliest phases (`corpus.log`,
  `prep.log`, `families.log`, `fam2.log`, `program.log`), ad-hoc scripts
  (`ent_split.py`, `sb.py`, `tax_olmo.py`, `ws.py`, `fam2.sh`), `cf.txt`,
  `results_candidates.json`, and `src_remote.tgz` — a snapshot of the remote
  `src/` (the local `src/` is canonical; every edit was made locally and synced).
- `logs/`: all 118 run logs. These are the authoritative record of printed
  tables where a JSON lacks a field.
- `results/remote_docs/specimens.md`: the remote copy of the specimen document
  (local `docs/specimens.md` has the corrected header and is canonical).

## Regeneration cost, if ever needed

One A100-40GB. Fragility replication 25 min; each interaction pool 15 min;
bridge 2 min; specimens 2 min; stage sweep (8 families) ~1 h; pretraining
harvests depend on checkpoint download bandwidth (~10–30 min per family).

## Scale ladder (6 September 2026, 1× H100, `docs/findings_scale.md`)

Label-free runs on Qwen3-1.7B (tied) and Qwen3-32B (untied); rungs 3–4 not run.

- `results/fragility_qwen3_{1_7b,32b}.json`: fragility matrices summarised at the
  paper's −0.1 gate (model-specific; see the gate files).
- `results/gate_{L,qwen3_1_7b,qwen3_32b}.json`: greedy gate (p_alone > 0.5),
  both failure thresholds, decomposition, split-half r — the cross-model numbers.
- `results/fragility_predict_{L,qwen3_1_7b,qwen3_32b}_greedy.json`: geometry
  under the greedy gate (7B: 60 positives, Δ +0.105 [+0.052, +0.160]).
- `results/reasoning_qwen3_1_7b{,_think}_v2.json` (supersede the non-`_v2`
  files, whose "plain" run was in thinking mode), `reasoning_qwen3_32b{,_think}.json`:
  worst-32 vs 32 controls, generations in `reasoning.samples`.
- `results/specimens_qwen3_{1_7b,32b}{,_greedy}.jsonl`, `docs/specimens_qwen3_*.md`:
  specimen stores with model provenance.
- `results/interaction_screen_qwen3_32b.json`: mixed pool N = 35 at 32B.
- `logs/qwen3_*.log`: every stage's printed tables.
- Full matrices with hidden states, pulled and md5-verified against the node before
  shutdown: `results/fragility_qwen3_1_7b.pt` (1.24 GB, a5b91752…), `fragility_qwen3_32b.pt`
  (3.10 GB, 5a284b6d…), `interaction_screen_qwen3_32b.pt` (1.25 GB, e566cde6…); compacts
  without states in `results/compact/*qwen3*_compact.pt`.
- `results/remote_h100/`: complete snapshot of the node (results, docs, logs, src, and
  `logs/env_h100_*` — driver 580.105.08, CUDA 13.0, torch 2.14.0, transformers 5.16.1,
  pip freeze), plus the staged scripts `rerun_17b.sh`, `greedy_specimens.sh`, `bootstrap.sh`.
  The 1x H100 was fully extracted 6 September 2026 20:45 UTC.

## Scale ladder, rung 3 (Qwen2.5-72B-Instruct, 4x H100, 6-7 September 2026)

- `results/fragility_qwen25_72b.json`, `gate_qwen25_72b.json`, `fragility_predict_qwen25_72b.json`
  (absolute gate, 154 positives), `fragility_predict_qwen25_72b_greedy.json` (322 positives),
  `reasoning_qwen25_72b.json` (id-level metric; 16 saved generations), `specimens_qwen25_72b.jsonl`,
  `docs/specimens_qwen25_72b.md`; `results/compact/fragility_qwen25_72b_compact.pt` (matrix without
  states; the 9.9 GB full matrix was on the node). Logs `logs/qwen25_72b_*.log`, `logs/post_72b_cpu.log`.
- Not obtained: `specimens_qwen25_72b_greedy.jsonl`, `reasoning_qwen25_72b_v2.json` (string metric)
  -- queued for the replacement node (`scratch/launch_rung4.sh`, works from the compact matrix).
- Rung 4 (Qwen3-235B-A22B-FP8): first attempt lost with the node at 00:52 UTC 7 September after its
  fragility matrix (22:58-00:48) and geometry had completed; nothing transferred. Re-run pending.

## Paper v4 and figures (7 September 2026)

- `docs/paper_fragile_tokens_v4.md`: current draft (findings-first, no em dashes,
  consolidated cross-model tables, figures). `v3` retained for history.
- `docs/figures/fig_fragility_matrix_7b.{png,pdf}`: 7B replication-bank matrix, 36 fragile
  + 12 clean + 8 verified glitch tokens x 24 contexts.
- `docs/figures/fig_geometry.{png,pdf}` (+ `.json` with the panel AUCs): UMAP of unembedding
  rows, glitch-direction projection, token id, for 7B and 72B (greedy gate).
- `docs/figures/fig_ladder.{png,pdf}`: prevalence, split-half r, context share, and
  predictor-tier AUCs per model.
- `results/emb_qwen25_72b_sample.npz`: 72B input/output embedding rows (gain folded) for
  the 8,400 sampled ids, plus centroids; lets the 72B geometry be re-plotted without the
  145 GB checkpoint. Generated by `src/cut/figures_paper.py` inputs (see script header).
- Update (7 September, later): figures now cover every model. `fig_fragility_matrix_{qwen3_1_7b,qwen3_32b,qwen25_72b}.{png,pdf}`
  (one per model, same layout as the 7B figure); `fig_geometry.{png,pdf}` has four rows (7B, 1.7B, 32B, 72B) with
  per-model AUCs in `fig_geometry.json`; embedding rows cached in `results/emb_{qwen3_1_7b,qwen3_32b,qwen25_72b}_sample.npz`.
  Self-contained builds: `docs/paper_fragile_tokens_v4.html` (images embedded), `docs/paper_fragile_tokens_v4.docx`
  (pandoc 3.5 from the Markdown; rebuild with the two commands in the session log).
- Update (7 September, figures final): matrices are one per model in one layout, blocks
  Glitch -> Stable -> Fragile. The label-free glitch class (Qwen models, Fig. 2 and Fig. 4)
  is "worst by single probe AND fragility >= 0.9"; on the 7B this reproduces the labelled
  glitch direction (0.879). Per-model AUCs in `docs/figures/fig_geometry.json`.
