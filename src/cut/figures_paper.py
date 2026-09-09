"""Figures for the paper: the fragility matrices, the embedding geometry, and the ladder,
for every model that has a compact fragility matrix in results/compact/.

CPU only. Embedding rows come either from results/emb_<tag>_sample.npz (rows for the
sampled ids, written once from the safetensors shards) or, for the 7B, straight from the
Hugging Face cache via loadmodel.load_embeddings.

  python -m src.cut.figures_paper                # everything that has data
  python -m src.cut.figures_paper --only matrix  # one figure family

Outputs docs/figures/*.png (and .pdf).
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np
import torch

os.environ.setdefault("GLITCH_FORCE_CPU", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path("docs/figures")
FS = 1.0          # font scale; --print sets it
PRINT = False
GREEDY = float(np.log(0.5))
COL = {"clean": "#9aa0a6", "fragile": "#d62728", "glitch": "#111111"}

# tag, display name, HF id, embedding cache (None: read the shards from the HF cache).
# Ordered by parameter count; every multi-model figure follows this order.
MODELS = [
    ("qwen3_1_7b", "Qwen3-1.7B (tied)", "Qwen/Qwen3-1.7B", "results/emb_qwen3_1_7b_sample.npz"),
    ("qwen25_7b", "Qwen2.5-7B", "Qwen/Qwen2.5-7B-Instruct", "results/emb_qwen25_7b_sample.npz"),
    ("L", "OLMo-2-7B", "allenai/OLMo-2-1124-7B-Instruct", None),
    ("qwen38_27b", "Qwen3.8-27B", "Qwen/Qwen3.8-27B", "results/emb_qwen38_27b_sample.npz"),
    ("qwen3_32b", "Qwen3-32B", "Qwen/Qwen3-32B", "results/emb_qwen3_32b_sample.npz"),
    ("qwen25_72b", "Qwen2.5-72B", "Qwen/Qwen2.5-72B-Instruct", "results/emb_qwen25_72b_sample.npz"),
    ("qwen38_flash", "Qwen3.8-Flash-Next (FP8)", "Qwen/Qwen3.8-Flash-Next-FP8", "results/emb_qwen38_flash_sample.npz"),
    ("qwen3_235b", "Qwen3-235B-A22B (FP8)", "Qwen/Qwen3-235B-A22B-FP8", "results/emb_qwen3_235b_sample.npz"),
]


def _font():
    # font fallback chain so token labels in CJK / Thai / Arabic / Hebrew render (matplotlib >= 3.6)
    have = {f.name for f in font_manager.fontManager.ttflist}
    chain = [n for n in ("Segoe UI", "Microsoft YaHei", "Microsoft JhengHei", "Malgun Gothic", "Leelawadee UI",
                         "Segoe UI Historic", "Ebrima", "SimSun-ExtB", "Segoe UI Symbol", "Arial Unicode MS",
                         "Noto Sans", "DejaVu Sans") if n in have]
    plt.rcParams["font.family"] = chain or ["DejaVu Sans"]
    plt.rcParams.update({"font.size": 12 * FS, "axes.titlesize": 12 * FS, "axes.labelsize": 11 * FS, "xtick.labelsize": 10 * FS, "ytick.labelsize": 10 * FS, "legend.fontsize": 10 * FS, "axes.spines.top": False, "axes.spines.right": False})


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, score))


def load_compact(tag):
    d = torch.load(Path(f"results/compact/fragility_{tag}_compact.pt"), weights_only=False, map_location="cpu")
    tok = np.asarray(d["tokens"]); s = np.asarray(d["single"], np.float64)
    M = np.asarray(d["M"], np.float64); thr = float(d["fail_thr"]); g = np.asarray(d["is_glitch"])
    return tok, s, M, thr, g, (M < thr).mean(1)


_DEC = {}


def decode_fn(model):
    if model not in _DEC:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        _DEC[model] = lambda i: tok.decode([int(i)])
    return _DEC[model]


def _printable(dec, tok, i):
    st = dec(tok[i])
    return bool(st.strip()) and st.isprintable()


def reference_idx(s, g, dec, tok, frag, n):
    """The labelled reference class where labels exist; otherwise a label-free one that is
    verified in context: the worst printable tokens by the single probe that also fail in
    at least 90% of contexts. (The bare probe alone mislabels code prefixes such as
    `_ghost` that copy perfectly in context; see probe_only_idx.)"""
    ref = np.where(g == 1)[0]
    if len(ref):
        return list(ref[:n]), "verified glitch"
    order = [i for i in np.argsort(s) if _printable(dec, tok, i) and frag[i] >= 0.9]
    return order[:n], "glitch, label-free:\nfails alone and in\n≥ 90% of contexts"


def probe_only_idx(s, dec, tok, frag, n, pool=200):
    """Tokens the bare probe fails badly (among the `pool` worst) that copy in context."""
    order = [i for i in np.argsort(s)[:pool] if _printable(dec, tok, i) and frag[i] <= 0.10]
    return order[:n]


# ------------------------------------------------------------------------------------------
def _matrix_rows(tag, model, n_frag=36, n_ok=12, n_ref=8, seed=0):
    tok, s, M, thr, g, frag = load_compact(tag)
    dec = decode_fn(model)
    clean = (s > GREEDY) & (g == 0)
    fr_idx = [i for i in np.argsort(-frag) if clean[i] and frag[i] >= 0.10][:n_frag]
    rng = np.random.default_rng(seed)
    ok_pool = np.where(clean & (frag == 0))[0]
    ok_idx = list(rng.choice(ok_pool, min(n_ok, len(ok_pool)), replace=False))
    ref_idx, ref_name = reference_idx(s, g, dec, tok, frag, n_ref)
    # block order, top to bottom, identical on every model: glitch -> stable -> fragile
    blocks = [(ref_idx, ref_name), (ok_idx, "stable:\nrandom clean,\nnever fail"),
              (fr_idx, "fragile,\nclean-looking\n(frag ≥ 0.10)")]
    blocks = [(idx, lab) for idx, lab in blocks if len(idx)]
    rows = [i for idx, _ in blocks for i in idx]
    P = np.exp(np.clip(M[rows], -12, 0))
    order = np.argsort(-(M[clean] < thr).mean(0))
    def short(s, n=16):
        return s if len(s) <= n else s[:n - 1] + "…"
    labels = [f"{short(dec(tok[i]))!r}  ({frag[i]:.2f})" for i in rows]
    return P[:, order], labels, [(len(idx), lab) for idx, lab in blocks], ref_name


def _draw_matrix(ax, P, labels, blocks, ref_name=None, fontsize=10.5, side_labels=True):
    im = ax.imshow(P, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_xticks(range(P.shape[1])); ax.set_xticklabels([str(c + 1) for c in range(P.shape[1])], fontsize=10)
    y0 = 0
    for k, (n, lab) in enumerate(blocks):
        if k:
            ax.axhline(y0 - 0.5, color="white", lw=1.5)
        if side_labels:
            ax.text(P.shape[1] + 0.4, y0 + n / 2 - 0.5, lab, va="center", fontsize=11)
        y0 += n
    return im


MATRIX_FILES = {"L": "fig_fragility_matrix_7b", "qwen25_7b": "fig_fragility_matrix_qwen25_7b", "qwen3_1_7b": "fig_fragility_matrix_qwen3_1_7b",
                "qwen3_32b": "fig_fragility_matrix_qwen3_32b", "qwen25_72b": "fig_fragility_matrix_qwen25_72b",
                "qwen3_235b": "fig_fragility_matrix_qwen3_235b",
                "qwen38_27b": "fig_fragility_matrix_qwen38_27b",
                "qwen38_flash": "fig_fragility_matrix_qwen38_flash"}


def fig_matrix(tag, name, model):
    """One model per figure, identical layout: fragile block, random-clean block, reference block."""
    P, labels, blocks, ref_name = _matrix_rows(tag, model)
    nrows = len(labels); sizes = [n for n, _ in blocks]
    fig, ax = plt.subplots(figsize=(8.6, 2.6 + 0.15 * nrows))
    im = _draw_matrix(ax, P, labels, blocks)
    ax.set_xlabel("context (24 random common-word sequences, ordered by overall failure rate)")
    # shorter figures (few fragile rows) need more room between the x-label and the colorbar
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.025, pad=0.06 if nrows > 45 else 0.11, aspect=45)
    cb.set_label("p(correct token at the copy position)")
    bank = "replication bank, greedy gate" if tag == "L" else "greedy gate"
    ref_note = ("verified glitch" if tag == "L" else "glitch, label-free (fails alone and in context)")
    ax.set_title(name, fontsize=14, loc="left", fontweight="bold")
    fig.tight_layout()
    stem = MATRIX_FILES[tag]
    fig.savefig(OUT / f"{stem}.png", dpi=180); fig.savefig(OUT / f"{stem}.pdf")
    plt.close(fig)
    print("wrote", stem, "rows", sizes)


def fig_matrix_all():
    for tag, name, model, _ in MODELS:
        fig_matrix(tag, name, model)


# ------------------------------------------------------------------------------------------
def _geometry_arrays(tag, model, npz=None):
    tok, s, M, thr, g, frag = load_compact(tag)
    if npz and Path(npz).exists():
        z = np.load(npz, allow_pickle=True)
        E_in, E_out, mu_in, mu_out = z["E_in"], z["E_out"], z["mu_in"], z["mu_out"]
        assert np.array_equal(z["tokens"], tok)
    else:
        from src.cut.loadmodel import load_embeddings
        Ei, Eo, gain, tied = load_embeddings(model)
        Eo = Eo * gain.unsqueeze(0)
        idx = torch.as_tensor(tok, dtype=torch.long)
        E_in = Ei[idx].float().numpy(); E_out = Eo[idx].float().numpy()
        mu_in = Ei.float().mean(0).numpy(); mu_out = Eo.float().mean(0).numpy()
    clean = (s > GREEDY) & (g == 0)
    fragile = clean & (frag >= 0.10)
    dec = decode_fn(model)
    ref, ref_name = reference_idx(s, g, dec, tok, frag, 100)
    ref = np.asarray(ref)
    gdir = E_in[ref].mean(0) - mu_in
    gdir = gdir / (np.linalg.norm(gdir) + 1e-9)
    proj = (E_in - mu_in) @ gdir
    return dict(tok=tok, s=s, frag=frag, clean=clean, fragile=fragile, ref=ref, ref_name=ref_name,
                E_out=E_out, E_in=E_in, mu_out=mu_out, proj=proj)



def _class_boxes(ax, score, A, name, xlabel, a, legend=False, bottom=False):
    """One score, three classes: box plots in the order glitch (top), stable, fragile, with the fragile
    tokens and the glitch class drawn as points so small classes stay visible. The title carries the
    AUC for separating fragile from stable clean-looking tokens on this score alone."""
    groups = [("glitch class", score[A["ref"]], COL["glitch"]),
              ("stable", score[A["clean"] & ~A["fragile"]], COL["clean"]),
              ("fragile", score[A["fragile"]], COL["fragile"])]
    bp = ax.boxplot([g[1] for g in groups], vert=False, widths=0.58, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.3), whiskerprops=dict(color="#666"), capprops=dict(color="#666"))
    for patch, (_, _, col) in zip(bp["boxes"], groups):
        patch.set_facecolor(col); patch.set_alpha(0.55); patch.set_edgecolor("#444")
    rng = np.random.default_rng(0)
    for k, (gname, vals, col) in enumerate(groups, start=1):
        if gname != "stable":
            n = len(vals); sub = vals if n <= 400 else rng.choice(vals, 400, replace=False)
            ax.scatter(sub, k + rng.uniform(-0.22, 0.22, len(sub)), s=7, c=col, alpha=0.75, linewidths=0, zorder=3)
    ax.set_yticks([1, 2, 3]); ax.set_yticklabels([g[0] for g in groups], fontsize=11 * FS)
    ax.invert_yaxis()
    if PRINT:
        from matplotlib.ticker import MaxNLocator, FuncFormatter
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if score.max() > 1000:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    if bottom:
        ax.set_xlabel(xlabel, fontsize=11 * FS)
    ax.set_title(name, fontsize=13 * FS, loc="left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

def fig_geometry(part=None):
    """For every model with embeddings available: UMAP of the unembedding rows (gated tokens
    + reference class), the projection onto the reference-class direction, and token id.
    part: None for all models in one figure; 'a'..'d' for two models each (the print version)."""
    import umap
    panels = [m for m in MODELS if m[3] is None or Path(m[3]).exists()]
    suffix = ""
    if part is not None:
        k = "abcd".index(part); panels = panels[2 * k:2 * k + 2]; suffix = "_" + part
    fig, axes = plt.subplots(len(panels), 4, figsize=(20.5, (4.6 if part is None else 6.2) * len(panels)))
    axes = np.atleast_2d(axes)
    summary = {}
    for r, (tag, name, model, npz) in enumerate(panels):
        A = _geometry_arrays(tag, model, npz)
        n_ref_lab = ("442 verified glitch tokens" if tag == "L"
                     else f"{len(A['ref'])} label-free glitch tokens, fail alone and in ≥ 90% of contexts")
        title = f"{name}  (glitch class: {n_ref_lab})"
        sel = A["clean"] | np.isin(np.arange(len(A["tok"])), A["ref"])
        X = A["E_out"][sel] - A["mu_out"]
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        emb = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine", random_state=0).fit_transform(X)
        kind = np.where(A["fragile"][sel], "fragile", np.where(np.isin(np.where(sel)[0], A["ref"]), "glitch", "clean"))
        ax = axes[r, 0]
        for k, z, sz, al in (("clean", 1, 6, 0.35), ("glitch", 2, 10, 0.8), ("fragile", 3, 22, 0.95)):
            m = kind == k
            ax.scatter(emb[m, 0], emb[m, 1], s=sz, c=COL[k], alpha=al, zorder=z, linewidths=0,
                       label={"clean": "gated, never fragile", "glitch": "glitch class",
                              "fragile": "fragile clean-looking (frag ≥ 0.10)"}[k])
        short = name.split(" (")[0]
        ax.set_title(short, fontsize=13 * FS, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
        y = A["fragile"][A["clean"]].astype(int)
        # proximity to the glitch class in the UMAP's own space: mean cosine to the five nearest
        # glitch-class rows. The projection is one axis; the UMAP shows neighbourhoods, and the
        # proximity score reads the neighbourhood directly.
        Xo = A["E_out"] - A["mu_out"]
        Xo = Xo / (np.linalg.norm(Xo, axis=1, keepdims=True) + 1e-9)
        ref_rows = Xo[A["ref"]]
        S = Xo @ ref_rows.T
        S[A["ref"], np.arange(len(A["ref"]))] = -2          # a glitch row is not its own neighbour
        prox = np.sort(S, axis=1)[:, -5:].mean(1)
        ids = A["tok"].astype(float)
        a_dir = auc(A["proj"][A["clean"]], y); a_prox = auc(prox[A["clean"]], y); a_id = auc(ids[A["clean"]], y)
        scores = [(A["proj"], "input row projected on the glitch direction (mean glitch row minus global mean)", a_dir),
                  (prox, "mean cosine to the 5 nearest glitch-class unembedding rows", a_prox),
                  (ids, "token id (about frequency rank)", a_id)]
        if PRINT:   # the rotated print page has no room for the long labels
            scores = [(sc, lab, au) for (sc, _, au), lab in zip(scores, ["projection on the glitch direction", "mean cosine to the 5 nearest glitch rows", "token id (frequency rank)"])]
        for c, (score, xlabel, a) in enumerate(scores, start=1):
            _class_boxes(axes[r, c], score, A, short, xlabel, a, legend=(r == 0 and c == 1),
                         bottom=(r == len(panels) - 1))
        summary[tag] = {"n_gated": int(A["clean"].sum()), "n_fragile": int(A["fragile"].sum()),
                        "auc_glitch_direction": a_dir, "auc_glitch_proximity": a_prox,
                        "auc_token_id": a_id, "reference": A["ref_name"]}
    fig.tight_layout(rect=[0, 0, 1, 0.962 if part is None else 0.77])
    headers = ["Embedding projections", "Projection onto the glitch direction",
               "Proximity to the glitch class", "Surface rarity (token id)"]
    if part is not None:
        headers = ["Embedding\nprojections", "Projection onto\nthe glitch direction",
                   "Proximity to\nthe glitch class", "Surface rarity\n(token id)"]
    ytop, yleg, ynote = (0.994, 0.9865, 0.9735) if part is None else (0.99, 0.905, 0.845)
    for c, h in enumerate(headers):
        pos = axes[0, c].get_position()
        fig.text(pos.x0, ytop, h, fontsize=16 * FS, fontweight="bold", ha="left", va="top")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=COL["clean"], ms=7, label="gated, never fragile"),
               Line2D([], [], marker="o", ls="", color=COL["glitch"], ms=8, label="glitch class"),
               Line2D([], [], marker="o", ls="", color=COL["fragile"], ms=9, label="fragile clean-looking (fragility ≥ 0.10)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(axes[0, 0].get_position().x0, yleg),
               ncol=3, frameon=False, fontsize=12.5 * FS, handletextpad=0.4, columnspacing=1.6)
    note = ("Box plots: each box spans the middle half of the class, the line is its median, "
            "and every token of the glitch and fragile classes is drawn as a point.")
    if part is not None:
        note = note.replace("median, and", "median,\nand")
    fig.text(axes[0, 0].get_position().x0, ynote, note, fontsize=12.5 * FS, ha="left", va="top", color="#333")
    fig.savefig(OUT / f"fig_geometry{suffix}.png", dpi=180); fig.savefig(OUT / f"fig_geometry{suffix}.pdf")
    plt.close(fig)
    json.dump(summary, open(OUT / f"fig_geometry{suffix}.json", "w"), indent=1)
    print("wrote fig_geometry", json.dumps(summary))


# ------------------------------------------------------------------------------------------
def fig_ladder():
    """The ladder in one figure: prevalence and stability under the greedy gate, and the
    geometry tiers against surface, per model."""
    models = [(n.replace(" (tied)", "").replace("-A22B (FP8)", "").replace(" (FP8)", ""), t) for t, n, _, _ in MODELS]
    gate = {t: json.load(open(f"results/gate_{t}.json", encoding="utf-8")) for _, t in models}
    pred = {t: json.load(open(f"results/fragility_predict_{t}_greedy.json", encoding="utf-8")) for _, t in models}
    names = [m for m, _ in models]; x = np.arange(len(models))

    if PRINT:
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 12.5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    legend_kw = dict(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=11 * FS, handlelength=1.4, columnspacing=1.4, ncol=2 if not PRINT else 3)
    ax = axes[0]
    prev = [100 * gate[t]["fail_p61"]["ge10"] for _, t in models]
    ax.bar(x, prev, color=COL["fragile"], alpha=0.85, label="fragile share")
    ax.axhspan(0.5, 3.0, color="#cccccc", alpha=0.35, lw=0, label="pre-registered range 0.5–3%")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11 * FS, rotation=30, ha="right"); ax.set_ylabel("% of gated tokens with fragility ≥ 0.10")
    ax.set_title("Prevalence", fontsize=14 * FS, fontweight="bold", loc="left", pad=46 * FS); ax.legend(**legend_kw)
    ax.set_ylim(0, max(prev) * 1.12)

    ax = axes[1]
    hc = [gate[t]["fail_p61"]["half_corr"] for _, t in models]
    ctx = [gate[t]["shares_clean"]["context"] for _, t in models]
    ax.bar(x - 0.2, hc, width=0.4, color="#1f77b4", label="split-half correlation of fragility")
    ax.bar(x + 0.2, ctx, width=0.4, color="#ff7f0e", label="context share of variance")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11 * FS, rotation=30, ha="right"); ax.set_ylim(0, 1.0)
    ax.set_title("Stability across contexts", fontsize=14 * FS, fontweight="bold", loc="left", pad=46 * FS)
    ax.legend(**legend_kw)

    ax = axes[2]
    tiers = [("surface only", "#9aa0a6", "surface (rarity)"), ("static geometry", "#d62728", "static geometry"),
             ("dynamic (train half)", "#2ca02c", "dynamic (slot state)")]
    w = 0.26
    for j, (key, c, lab) in enumerate(tiers):
        vals = [pred[t]["cv"][key]["auc"] for _, t in models]
        ax.bar(x + (j - 1) * w, vals, width=w, color=c, label=lab)
    ax.set_ylim(0.5, 1.0); ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11 * FS, rotation=30, ha="right"); ax.set_ylabel("cross-validated AUC")
    ax.set_title("Predicting fragility without a forward pass", fontsize=14 * FS, fontweight="bold", loc="left", pad=46 * FS)
    ax.legend(**legend_kw)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ladder.png", dpi=180); fig.savefig(OUT / "fig_ladder.pdf")
    plt.close(fig)
    print("wrote fig_ladder")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["matrix", "geometry", "ladder"], default=None)
    ap.add_argument("--print", action="store_true", help="larger type, docs/figures_print, geometry in four parts")
    a = ap.parse_args()
    global OUT, FS, PRINT
    if a.print:
        # the geometry parts are scaled down to a rotated page and need larger type; the
        # stacked ladder is printed at close to full size and keeps normal type
        OUT = Path("docs/figures_print"); FS = 1.8 if a.only == "geometry" else 1.0; PRINT = True
    OUT.mkdir(parents=True, exist_ok=True)
    _font()
    if a.only in (None, "matrix"):
        fig_matrix_all()
    if a.only in (None, "ladder"):
        fig_ladder()
    if a.only in (None, "geometry"):
        fig_geometry()


if __name__ == "__main__":
    main()
