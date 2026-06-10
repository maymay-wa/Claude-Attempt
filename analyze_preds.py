"""Diagnostic: where does the per-protein Pearson live in a fold's predictions?

Reads a preds_fold*.npz (saved by train.py: prot_idx, dna_idx, y_true raw
affinity, y_pred model output) and reports the distribution of per-protein
Pearson r across held-out proteins -- to tell whether a low mean is a tail of
near-zero "unpredictable" TFs (a ceiling) or a model-wide weakness.

Usage:
    python analyze_preds.py runs/preds_fold0.npz
"""

from __future__ import annotations

import sys

import numpy as np

from metrics import per_protein_correlations


def main(path: str) -> None:
    d = np.load(path)
    prot_idx, y_true, y_pred = d["prot_idx"], d["y_true"], d["y_pred"]
    corr = per_protein_correlations(prot_idx, y_true, y_pred)
    r = corr["pearson"]
    rho = corr["spearman"]
    ids = corr["protein_ids"]

    # per-protein affinity spread (a near-constant target -> unpredictable ceiling)
    spread = np.array([y_true[prot_idx == p].std() for p in ids])

    print(f"file: {path}")
    print(f"held-out proteins: {len(ids)}")
    print(f"Pearson  mean={r.mean():.4f}  std={r.std():.4f}  "
          f"min={r.min():.4f}  median={np.median(r):.4f}  max={r.max():.4f}")
    print(f"Spearman mean={rho.mean():.4f}")

    # histogram of per-protein Pearson
    edges = np.array([-1, 0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])
    hist, _ = np.histogram(r, bins=edges)
    print("\nper-protein Pearson histogram:")
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        bar = "#" * c
        print(f"  [{lo:+.2f},{hi:+.2f}) {c:3d}  {bar}")

    # how much of the gap to the mean is the bottom tail?
    for thr in (0.3, 0.5):
        m = r < thr
        print(f"\nproteins with r<{thr}: {m.sum()}  "
              f"(their mean r={r[m].mean() if m.any() else float('nan'):.3f}, "
              f"mean affinity-std={spread[m].mean() if m.any() else float('nan'):.4f})")
    # correlation between target spread and predictability
    if len(ids) > 2:
        c_sp = np.corrcoef(spread, r)[0, 1]
        print(f"\ncorr(per-protein target std, per-protein r) = {c_sp:.3f}  "
              f"(high -> low-variance TFs are the unpredictable ones)")

    # worst offenders
    order = np.argsort(r)
    print("\nworst 8 proteins (id, r, rho, aff_std):")
    for i in order[:8]:
        print(f"  prot {int(ids[i]):4d}  r={r[i]:+.3f}  rho={rho[i]:+.3f}  aff_std={spread[i]:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/preds_fold0.npz")
