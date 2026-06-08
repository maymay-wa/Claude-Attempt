"""One-off exploratory data analysis to inform target transforms & outlier handling.

Prints distribution stats for the affinity matrix, per-protein scale variation,
skewness, outlier counts, and whether a log transform improves symmetry.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew, kurtosis

from data import load_data


def main() -> None:
    d = load_data()
    a = d.affinity  # [N, P]
    flat = a.reshape(-1)

    print("=== Global affinity distribution ===")
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
    pcs = np.percentile(flat, qs)
    for q, v in zip(qs, pcs):
        print(f"  p{q:>5}: {v:.4f}")
    print(f"  mean={flat.mean():.4f}  std={flat.std():.4f}")
    print(f"  skew={skew(flat):.4f}  excess_kurtosis={kurtosis(flat):.4f}")
    print(f"  exact zeros: {(flat==0).sum()} ({100*(flat==0).mean():.2f}%)")
    print(f"  negatives:   {(flat<0).sum()}")

    # IQR outlier rule
    q1, q3 = np.percentile(flat, [25, 75])
    iqr = q3 - q1
    hi = q3 + 1.5 * iqr
    print(f"  IQR upper fence={hi:.4f}  values above: {(flat>hi).sum()} ({100*(flat>hi).mean():.3f}%)")

    print("\n=== log1p transform ===")
    lg = np.log1p(flat)
    print(f"  skew={skew(lg):.4f}  excess_kurtosis={kurtosis(lg):.4f}  (closer to 0 = more symmetric)")

    print("\n=== Per-protein scale variation (column-wise) ===")
    col_mean = a.mean(axis=0)
    col_std = a.std(axis=0)
    print(f"  protein mean affinity:  min={col_mean.min():.3f}  max={col_mean.max():.3f}  "
          f"ratio={col_mean.max()/max(col_mean.min(),1e-9):.1f}x")
    print(f"  protein std affinity:   min={col_std.min():.3f}  max={col_std.max():.3f}")
    print("  -> large mean ratio across proteins => absolute scale differs a lot per protein")

    print("\n=== Duplicate probes ===")
    uniq = len(set(d.dna_seqs))
    print(f"  {uniq}/{d.n_dna} unique DNA probes")


if __name__ == "__main__":
    main()
