"""Does the interaction screen's null machinery do what it claims?

Synthetic replicated cell matrices with KNOWN structure go through exactly the
pipeline the screen uses (stats.interaction_test). Cases:

  additive, homoscedastic    -- no interaction. BH should reject ~nothing.
  additive, heteroscedastic  -- no interaction, but 1/8 of tokens are 3x
                                noisier. Without standardisation the pooled
                                null is calibrated to the AVERAGE noise, so the
                                noisy tokens' pairs become false positives.
                                Run both ways to show the fix works.
  injected interaction       -- one true effect of known size; should be found.

This decides whether a positive result from the screen can be trusted.
"""
from __future__ import annotations
import argparse
import numpy as np

from .stats import interaction_test


def make(N, R, rng, row_scale=None, inject=None, sd=0.3):
    alpha = rng.normal(0, 1.0, N); beta = rng.normal(0, 1.0, N)
    scale = np.ones(N) if row_scale is None else row_scale
    cell = sd * np.sqrt(0.5 * (scale[:, None] ** 2 + scale[None, :] ** 2))
    M = np.stack([-4.0 + alpha[:, None] + beta[None, :] + rng.normal(0, 1, (N, N)) * cell
                  for _ in range(R)])
    if inject is not None:
        a, b, eff = inject
        M[:, a, b] += eff
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--r", type=int, default=14)
    ap.add_argument("--n-perm", type=int, default=300)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--q", type=float, default=0.05)
    a = ap.parse_args()
    N, R = a.n, a.r
    off = ~np.eye(N, dtype=bool)

    print("=" * 74)
    print(f"NULL CALIBRATION  N={N} R={R} perms={a.n_perm} q={a.q}")
    print("=" * 74)
    for std in (False, True):
        print(f"\n  standardise={std}")
        for case in ("homoscedastic", "heteroscedastic"):
            counts, fw = [], []
            for k in range(a.reps):
                rng = np.random.default_rng(100 + k)
                rs = None
                if case == "heteroscedastic":
                    rs = np.ones(N); rs[rng.choice(N, N // 8, replace=False)] = 3.0
                T = interaction_test(make(N, R, rng, row_scale=rs), a.n_perm, a.q, k,
                                     standardise=std)
                counts.append(int(T["rej"][off].sum())); fw.append(T["fwer_p"])
            flag = "  <-- false family-wise positives" if min(fw) < 0.05 else ""
            print(f"    {case:>16}: BH false rejections {counts}  "
                  f"family-wise p {np.round(fw, 3).tolist()}{flag}")

    print()
    print("INJECTED INTERACTION recovery, standardised (effect in units of cell noise)")
    print(f"{'effect':>8} | {'recovered':>9} | {'other rejections':>18} | {'fwer p':>7}")
    print("-" * 52)
    for eff in (0.3, 0.6, 0.9, 1.5):
        hit, other, fw = 0, [], []
        for k in range(a.reps):
            rng = np.random.default_rng(200 + k)
            rs = np.ones(N); rs[rng.choice(N, N // 8, replace=False)] = 3.0
            T = interaction_test(make(N, R, rng, row_scale=rs, inject=(3, 17, -eff * 0.3)),
                                 a.n_perm, a.q, k)
            hit += int(T["rej"][3, 17]); other.append(int(T["rej"][off].sum()) - int(T["rej"][3, 17]))
            fw.append(T["fwer_p"])
        print(f"{eff:8.2f} | {hit}/{a.reps:<7d} | {other!s:>18} | {np.mean(fw):7.3f}")


if __name__ == "__main__":
    main()
