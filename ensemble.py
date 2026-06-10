"""Seed-ensemble the held-out predictions from several training runs.

The brief's last trick: "Ensemble + RC averaging at inference. A few seeds
averaged is a cheap, consistent bump." Reverse-complement (RC) averaging is
already done inside training (config `tta_rc: true`), so this script handles the
other half -- averaging the per-protein predictions across runs trained with
different seeds, then reporting the per-protein Pearson/Spearman lift over the
single best run.

Each run dir holds preds_fold*.npz (saved by train.py) with parallel
prot_idx / dna_idx / y_true / y_pred arrays. We align rows by (prot_idx, dna_idx)
and average y_pred across runs.

    # train a few seeds into separate dirs:
    python train.py --config config_homeodomain.yaml --seed 0 --out-dir runs_hd_s0
    python train.py --config config_homeodomain.yaml --seed 1 --out-dir runs_hd_s1
    python train.py --config config_homeodomain.yaml --seed 2 --out-dir runs_hd_s2
    python ensemble.py runs_hd_s0 runs_hd_s1 runs_hd_s2
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from metrics import per_protein_correlations


def _load_run(run_dir: str) -> dict[tuple[int, int], tuple]:
    """Concatenate a run's per-fold preds into a {(prot,dna): (y_true, y_pred)} map."""
    out: dict[tuple[int, int], tuple] = {}
    files = sorted(glob.glob(os.path.join(run_dir, "preds_fold*.npz")))
    if not files:
        raise FileNotFoundError(f"no preds_fold*.npz in {run_dir!r}")
    for f in files:
        z = np.load(f)
        for p, d, yt, yp in zip(z["prot_idx"], z["dna_idx"], z["y_true"], z["y_pred"]):
            out[(int(p), int(d))] = (float(yt), float(yp))
    return out


def _corr_summary(maps: list[dict], use: list[int]) -> dict:
    """Per-protein correlation of the mean prediction over the runs in `use`."""
    keys = sorted(set.intersection(*[set(m) for m in (maps[i] for i in use)]))
    prot = np.array([k[0] for k in keys])
    y_true = np.array([maps[use[0]][k][0] for k in keys])
    y_pred = np.mean([[maps[i][k][1] for k in keys] for i in use], axis=0)
    return per_protein_correlations(prot, y_true, y_pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="run dirs (one per seed) to ensemble")
    args = ap.parse_args()

    maps = [_load_run(d) for d in args.run_dirs]
    print(f"loaded {len(maps)} runs, {[len(m) for m in maps]} predictions each\n")

    print("single runs (per-protein Pearson / Spearman):")
    singles = []
    for d, i in zip(args.run_dirs, range(len(maps))):
        c = _corr_summary(maps, [i])
        singles.append(c["pearson_mean"])
        print(f"  {os.path.basename(d):20s} pearson={c['pearson_mean']:.4f}  spearman={c['spearman_mean']:.4f}")

    ens = _corr_summary(maps, list(range(len(maps))))
    best = max(singles)
    print(f"\nensemble of {len(maps)} seeds:")
    print(f"  pearson={ens['pearson_mean']:.4f}  spearman={ens['spearman_mean']:.4f}")
    print(f"  lift over best single run: {ens['pearson_mean'] - best:+.4f} Pearson")


if __name__ == "__main__":
    main()
