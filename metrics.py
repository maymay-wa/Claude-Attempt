"""Per-protein correlation metrics for binding affinity prediction.

The task is generalization to unseen proteins, so the natural metric is, for
each held-out protein, the correlation between predicted and true affinity
across all DNA probes -- then averaged over proteins. Correlation is invariant
to the global target standardization used during training.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def per_protein_correlations(
    prot_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Pearson r and Spearman rho per protein, plus mean/std across proteins.

    Returns dict with arrays 'protein_ids', 'pearson', 'spearman' and scalar
    summaries 'pearson_mean','pearson_std','spearman_mean','spearman_std'.
    """
    ids = np.unique(prot_idx)
    pear, spear = [], []
    for pid in ids:
        m = prot_idx == pid
        yt, yp = y_true[m], y_pred[m]
        # constant prediction -> undefined correlation; treat as 0
        if yt.std() == 0 or yp.std() == 0:
            pear.append(0.0)
            spear.append(0.0)
            continue
        pear.append(pearsonr(yt, yp)[0])
        spear.append(spearmanr(yt, yp)[0])
    pear = np.array(pear)
    spear = np.array(spear)
    return {
        "protein_ids": ids,
        "pearson": pear,
        "spearman": spear,
        "pearson_mean": float(pear.mean()),
        "pearson_std": float(pear.std()),
        "spearman_mean": float(spear.mean()),
        "spearman_std": float(spear.std()),
    }
