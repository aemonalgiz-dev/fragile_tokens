# arXiv submission package: Fragile Tokens

Built from `docs/paper_fragile_tokens_v6.md`, which carries the wording and organization of the Medium
version (first person, American spelling, captions below tables and figures, Section 11 Future Work).

## Files

- `main.tex`: the paper, XeLaTeX. Compile with `xelatex main.tex` twice, or `tectonic main.tex` (verified with
  tectonic 0.17.0: 46 pages, no errors, no missing glyphs). No BibTeX step: the bibliography is a `thebibliography`
  block inside `main.tex`, cited author-year through natbib.
- `main.pdf`: the compiled paper. `fragile_tokens_arxiv_submission.zip`: `main.tex` plus `figures/`, ready to upload.
- `figures/`: 8 fragility-matrix PDFs (Figure 1), `fig_ladder.pdf` (Figure 2, stacked print layout),
  `fig_geometry_a..d.pdf` (Figure 3, two models per landscape page), and `glyph_*.pdf`, nine single characters
  (three emoji, two Tibetan letters, an Arabic presentation form, three rare Han characters) that no TeX Live font
  carries, drawn as tiny images where they occur in the tables.
- Fonts are taken from TeX Live by file name, so nothing needs installing on arXiv:
  TeX Gyre Termes and Heros (text), DejaVu Sans Mono (code), FreeSerif (Cyrillic, Greek, Hebrew, Arabic, Thai, Indic),
  FandolSong (Chinese), IPAexMincho (Japanese), UnBatang (Korean).

## Submission metadata (paste into the arXiv form)

**Title**
Fragile Tokens: Context-Dependent Copy Failure on Tokens That Pass Single-Token Glitch Probes

**Authors**
Jeffrey R. Gordon

**Abstract** (1,876 characters; the arXiv limit is 1,920)

Glitch-token detectors evaluate each vocabulary entry in isolation, usually by asking the model to repeat it. This paper measures what that evaluation misses. I placed tokens into banks of random ordinary contexts and scored the copy at each position. Under one gate applied to eight open models from 1.7B to 235B parameters, between 0.4% (Qwen3.8-Flash-Next) and 10.9% (Qwen3-235B) of tokens that pass the single-token probe fail to copy in at least one of ten contexts. The property belongs to the token, not the context: the context explains at most 0.013 of the variance on any model, and a token's fragility on half of the contexts predicts its fragility on the other half at 0.82 to 0.95. The fragile tokens are canonical and ordinary; the English word "according" copies alone at probability 0.999 and is deleted in 44 of 48 contexts. Failures take four forms, deletion, neighbor substitution, truncation, and translation, and most failing cells are confident, below 2 bits of entropy. Given a fragile or glitch token as an entire prompt, no model asks about the token it was given; the smaller models ask for clarification about a token they have already replaced, and the larger ones write out the solution to a problem that was never posed. On Pythia-1.4b the token's identity is recoverable from its final-layer state, so the failure is at the readout. Tokens do not combine to produce the behavior: a pre-registered factorial screen finds no more interacting pairs than an additive null yields. Fragile tokens sit next to glitch tokens in embedding space on every model with untied embeddings, with cosine proximity to the glitch class separating them from stable tokens at AUC 0.74 to 0.90, where a single glitch direction misses the relation. Inside tool-using assignments, models given a glitch token search for the sentence with the token removed, retrieve nothing, and deliver a recommendation anyway. Every failing case is released with model provenance.

**Comments**
46 pages, 3 figures (13 figure pages), 26 tables. Datasets: fragility matrices, specimen stores, reasoning-mode generations and agent transcripts for eight models.

**Primary category**: cs.CL (Computation and Language)
**Cross-lists**: cs.LG (Machine Learning), cs.AI (Artificial Intelligence)

**License**: arXiv.org perpetual, non-exclusive license is the minimum; CC BY 4.0 if you want others to reuse the tables and figures freely.

## Before uploading

1. Upload `main.tex` and the `figures/` directory (or the zip). arXiv detects XeLaTeX from `fontspec`; the first line of `main.tex` also carries the `% !TEX TS-program = xelatex` hint.
2. Check the compiled PDF that arXiv shows before announcing: the landscape figure pages (Figures 1 and 3) and the wide tables (5, 9a, 9b) are the pages to look at.
3. Update the "released with the paper" sentences with a link once the datasets are hosted.
