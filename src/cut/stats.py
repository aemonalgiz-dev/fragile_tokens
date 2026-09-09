"""Dependency-light stats: paired bootstrap CI, Welch's t, and the permutation /
FDR machinery for the non-contiguous interaction screen."""
from __future__ import annotations
import numpy as np


def paired_boot(x, y, n=10000, seed=0, stat="mean"):
    """Bootstrap CI on mean(x-y) or median(x-y) for paired samples.

    The median variant matters here: shuffled-control copy-logprobs have a heavy
    left tail, so the mean is outlier-dominated and understates the typical case.
    """
    rng = np.random.default_rng(seed)
    d = np.asarray(x) - np.asarray(y)
    f = np.median if stat == "median" else np.mean
    idx = rng.integers(0, len(d), size=(n, len(d)))
    vals = f(d[idx], axis=1)
    return float(f(d)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def welch(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    t = (x.mean() - y.mean()) / np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    return float(t), float(df)


def cohen_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return float((x.mean() - y.mean()) / s) if s > 0 else 0.0


def bh_fdr(p, q=0.05):
    """Benjamini-Hochberg. Returns (reject mask, adjusted p) in the input order.

    Needed because the interaction screen tests every ordered pair in a pool at
    once -- thousands of simultaneous tests -- and nothing in this repo
    previously corrected for that.
    """
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = p[order] * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(adj, 0, 1)
    return out <= q, out


def _mad(x):
    x = np.asarray(x, float)
    m = 1.4826 * np.median(np.abs(x - np.median(x)))
    return m if m > 0 else (x.std() if x.std() > 0 else 1.0)


def perm_null_additive(M, n_perm=1000, seed=0, standardise=True):
    """Permutation null for interaction in replicated two-way cell matrices.

    M is (R, N, N): for each replicate r, cell (a, b) is the score with a in
    slot i and b in slot j. Under the null of NO interaction the matrix is
    additive, S = mu + alpha_a + beta_b + noise. Per replicate the additive fit
    (row and column means, off-diagonal cells only -- the diagonal is the
    repetition case and may carry a systematic effect of its own) is removed and
    the residual is the interaction estimate.

    STANDARDISATION. Calibration showed the pooled permutation null is correct
    when every token's cells are equally noisy and gives false family-wise
    positives when some are not: a noisy token's pairs have inflated |residual|
    against a null calibrated to the average. So residuals are divided by
    sqrt(0.5 * (s_a^2 + s_b^2)) with s_a, s_b robust (MAD) scales of row a and
    column b pooled over replicates, and the permutation runs on the
    standardised values. Effect sizes (obs) stay in score units; the test
    statistic (z) is unitless.

    Returns (obs, z, null, max_null):
      obs       (N, N)          mean raw residual across replicates
      z         (N, N)          mean standardised residual -- the statistic
      null      (n_perm, N, N)  permuted statistic
      max_null  (n_perm,)       max |statistic| over off-diagonal cells
    """
    rng = np.random.default_rng(seed)
    M = np.asarray(M, float)
    R, N, _ = M.shape
    off = ~np.eye(N, dtype=bool)
    resid = np.zeros_like(M)
    for r in range(R):
        X = M[r]
        mu = X[off].mean()
        alpha = np.array([X[a, off[a]].mean() for a in range(N)]) - mu
        beta = np.array([X[off[:, b], b].mean() for b in range(N)]) - mu
        resid[r] = X - mu - alpha[:, None] - beta[None, :]
    obs = resid.mean(0)
    if standardise:
        row_s = np.array([_mad(resid[:, a, off[a]].ravel()) for a in range(N)])
        col_s = np.array([_mad(resid[:, off[:, b], b].ravel()) for b in range(N)])
        scale = np.sqrt(0.5 * (row_s[:, None] ** 2 + col_s[None, :] ** 2))
        zres = resid / scale[None]
    else:
        zres = resid
    z = zres.mean(0)
    null = np.zeros((n_perm, N, N))
    n_off = int(off.sum())
    for k in range(n_perm):
        acc = np.zeros((N, N))
        for r in range(R):
            v = zres[r][off]
            perm = np.empty((N, N))
            perm[off] = v[rng.permutation(n_off)]
            # the diagonal is not part of the exchangeable set; draw it from the
            # same pooled residuals so the null covers the repetition cells too
            perm[~off] = v[rng.integers(0, n_off, N)]
            acc += perm
        null[k] = acc / R
    max_null = np.abs(null[:, off]).max(1)
    return obs, z, null, max_null


def interaction_test(M, n_perm=1000, q=0.05, seed=0, standardise=True):
    """The full screen pipeline on one (R, N, N) cell tensor, so the real data,
    the matched-null simulations and the calibration harness all run IDENTICAL
    code. Returns a dict with obs, z, p, padj, rej, fwer_p, fwer_thr."""
    obs, z, null, max_null = perm_null_additive(M, n_perm=n_perm, seed=seed,
                                                standardise=standardise)
    N = obs.shape[0]
    off = ~np.eye(N, dtype=bool)
    pooled = np.sort(np.abs(null[:, off].ravel()))
    absz = np.abs(z)
    p = 1.0 - np.searchsorted(pooled, absz, side="left") / len(pooled)
    p = np.clip(p + 1.0 / (len(pooled) + 1), 0, 1)
    rej, padj = bh_fdr(p.ravel(), q)
    max_obs = float(absz[off].max())
    return {"obs": obs, "z": z, "p": p, "padj": padj.reshape(N, N),
            "rej": rej.reshape(N, N), "fwer_p": float((max_null >= max_obs).mean()),
            "fwer_thr": float(np.percentile(max_null, 95)), "max_obs": max_obs,
            "null_sd": float(null[:, off].std()),
            "diag_null_means": null[:, ~off].mean(1)}


def additive_fit(M):
    """Per-replicate additive parameters and per-row/col robust noise scales,
    for simulating matched null data. Returns (mu (R,), alpha (R,N), beta (R,N),
    row_s (N,), col_s (N,))."""
    M = np.asarray(M, float)
    R, N, _ = M.shape
    off = ~np.eye(N, dtype=bool)
    mu = np.zeros(R); alpha = np.zeros((R, N)); beta = np.zeros((R, N))
    resid = np.zeros_like(M)
    for r in range(R):
        X = M[r]
        mu[r] = X[off].mean()
        alpha[r] = np.array([X[a, off[a]].mean() for a in range(N)]) - mu[r]
        beta[r] = np.array([X[off[:, b], b].mean() for b in range(N)]) - mu[r]
        resid[r] = X - mu[r] - alpha[r][:, None] - beta[r][None, :]
    row_s = np.array([_mad(resid[:, a, off[a]].ravel()) for a in range(N)])
    col_s = np.array([_mad(resid[:, off[:, b], b].ravel()) for b in range(N)])
    return mu, alpha, beta, row_s, col_s


def simulate_additive(mu, alpha, beta, row_s, col_s, rng, inject=None):
    """Draw one no-interaction dataset from an additive fit, with the observed
    heteroscedastic noise structure. inject=(a, b, effect) adds a true
    interaction to one cell in every replicate."""
    R, N = alpha.shape
    scale = np.sqrt(0.5 * (row_s[:, None] ** 2 + col_s[None, :] ** 2))
    M = np.zeros((R, N, N))
    for r in range(R):
        M[r] = mu[r] + alpha[r][:, None] + beta[r][None, :] + rng.normal(0, 1, (N, N)) * scale
    if inject is not None:
        a, b, eff = inject
        M[:, a, b] += eff
    return M
