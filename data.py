"""Data loading and preparation for protein-DNA binding affinity prediction.

Three input files (one record per line, aligned by index). Counts (P proteins,
N DNA probes, L bp) are read from the files, not hardcoded:
  - training_DBPs_small.txt   : P protein amino-acid sequences
  - training_seqs_small.txt   : N DNA probes, L bp, alphabet {A,C,G,T}
  - training_data_small.txt   : N x P affinity matrix (space-separated),
                                row i = DNA probe i, column j = protein j

This module parses the files into a long-format table of
(dna_idx, prot_idx, affinity) triples, provides one-hot encoding with
reverse-complement augmentation, a protein-grouped K-fold splitter (so that
held-out proteins are never seen in training), and a global target standardizer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import Dataset

# DNA alphabet -> index. One-hot dimension order is fixed and shared everywhere.
BASES = "ACGT"
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
# Reverse-complement base index map: A<->T, C<->G  (A=0,C=1,G=2,T=3)
COMPLEMENT_IDX = np.array([3, 2, 1, 0], dtype=np.int64)
_COMPLEMENT_T = torch.from_numpy(COMPLEMENT_IDX)  # reused by reverse_complement_onehot


def read_lines(path: str) -> list[str]:
    """Read non-empty, stripped lines from a text file."""
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


@dataclass
class BindingData:
    """Parsed dataset held in memory.

    Attributes:
        proteins:  list[str] of length P (amino-acid sequences)
        dna_seqs:  list[str] of length N (36 bp DNA strings)
        affinity:  float32 array [N, P] of binding affinities
        dna_onehot: float32 array [N, 4, L] one-hot encoding of dna_seqs
    """

    proteins: list[str]
    dna_seqs: list[str]
    affinity: np.ndarray  # [N, P]
    dna_onehot: np.ndarray  # [N, 4, L]

    @property
    def n_dna(self) -> int:
        return len(self.dna_seqs)

    @property
    def n_prot(self) -> int:
        return len(self.proteins)


def one_hot_encode(seqs: list[str]) -> np.ndarray:
    """One-hot encode a list of equal-length DNA strings -> [N, 4, L] float32."""
    n = len(seqs)
    length = len(seqs[0])
    out = np.zeros((n, 4, length), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, base in enumerate(s):
            idx = BASE_TO_IDX.get(base)
            if idx is not None:  # unknown char (e.g. N) stays all-zero
                out[i, idx, j] = 1.0
    return out


def reverse_complement_onehot(onehot: torch.Tensor) -> torch.Tensor:
    """Reverse-complement a one-hot batch [B, 4, L].

    Reverses the length axis and swaps complementary channels. Binding is
    approximately strand-symmetric, so this is a label-preserving augmentation.
    """
    comp = _COMPLEMENT_T.to(onehot.device)
    return onehot[:, comp, :].flip(dims=[-1])


def load_data(
    dbp_path: str = "training_DBPs_small.txt",
    seq_path: str = "training_seqs_small.txt",
    data_path: str = "training_data_small.txt",
) -> BindingData:
    """Load and align the three files; validates that shapes line up."""
    proteins = read_lines(dbp_path)
    dna_seqs = read_lines(seq_path)
    affinity = np.loadtxt(data_path, dtype=np.float32)  # [N, P]

    if affinity.ndim != 2:
        raise ValueError(f"affinity matrix must be 2D, got shape {affinity.shape}")
    n, p = affinity.shape
    if n != len(dna_seqs):
        raise ValueError(f"rows ({n}) != #DNA seqs ({len(dna_seqs)})")
    if p != len(proteins):
        raise ValueError(f"cols ({p}) != #proteins ({len(proteins)})")

    lengths = {len(s) for s in dna_seqs}
    if len(lengths) != 1:
        raise ValueError(f"DNA probes must be equal length, found {sorted(lengths)}")

    dna_onehot = one_hot_encode(dna_seqs)
    return BindingData(proteins, dna_seqs, affinity, dna_onehot)


class TargetTransform:
    """Fit-on-train-only affinity transform for the regression target.

    Steps applied in order (each optional/configurable):
      1. clip   — winsorize to [q, 1-q] global train quantiles (tames the thin
                  upper tail; ~0.6% of values sit above the IQR fence).
      2. log1p  — compress the right skew (skew 0.62 -> ~0 after log1p).
      3. standardize — z-score, either GLOBAL or PER-PROTEIN.

    Per-protein standardization is the recommended default: protein mean affinity
    spans ~94x across this dataset, so a global z-score lets high-binding proteins
    dominate the loss. Z-scoring within each protein makes every protein contribute
    its *specificity pattern* equally — exactly what the per-protein correlation
    metric rewards, and the only thing knowable for an unseen protein.

    Stats are fit on TRAINING proteins only (no leakage). Proteins unseen at fit
    time fall back to the global train mean/std, so transform() is always defined
    (used only for monitoring val loss; the selection metric is correlation, which
    is invariant to this affine map).
    """

    def __init__(self, mode: str = "per_protein", log1p: bool = False,
                 clip_quantile: float | None = None) -> None:
        if mode not in ("per_protein", "global"):
            raise ValueError(f"mode must be 'per_protein' or 'global', got {mode!r}")
        if clip_quantile is not None and not (0.0 < clip_quantile < 0.5):
            raise ValueError(f"clip_quantile is a per-side tail fraction in (0, 0.5), got {clip_quantile}")
        self.mode = mode
        self.log1p = log1p
        self.clip_quantile = clip_quantile
        self.clip_lo_: float | None = None
        self.clip_hi_: float | None = None
        self.gmean_: float = 0.0
        self.gstd_: float = 1.0
        self.pmean_: dict[int, float] = {}
        self.pstd_: dict[int, float] = {}

    def _pre(self, values: np.ndarray) -> np.ndarray:
        v = values.astype(np.float64)
        if self.clip_lo_ is not None:
            v = np.clip(v, self.clip_lo_, self.clip_hi_)
        if self.log1p:
            v = np.log1p(v)
        return v

    def fit(self, prot_idx: np.ndarray, values: np.ndarray) -> "TargetTransform":
        if self.clip_quantile:
            q = self.clip_quantile
            self.clip_lo_ = float(np.quantile(values, q))
            self.clip_hi_ = float(np.quantile(values, 1.0 - q))
        v = self._pre(values)
        self.gmean_, self.gstd_ = float(v.mean()), float(v.std()) or 1.0
        if self.mode == "per_protein":
            for pid in np.unique(prot_idx):
                m = prot_idx == pid
                self.pmean_[int(pid)] = float(v[m].mean())
                self.pstd_[int(pid)] = float(v[m].std()) or 1.0
        return self

    def transform(self, prot_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
        v = self._pre(values)
        if self.mode == "global":
            return ((v - self.gmean_) / self.gstd_).astype(np.float32)
        mean = np.array([self.pmean_.get(int(p), self.gmean_) for p in prot_idx])
        std = np.array([self.pstd_.get(int(p), self.gstd_) for p in prot_idx])
        return ((v - mean) / std).astype(np.float32)


class PairDataset(Dataset):
    """Yields (dna_idx, prot_idx, target) index triples for a set of pairs.

    Only lightweight index arrays are carried here. The DNA one-hot tensor and
    the ESM embeddings live on the compute device and are gathered by index in
    the training loop, so each batch ships just a few hundred ints/floats instead
    of stacking and transferring full one-hot tensors (much faster on MPS/CUDA).
    """

    def __init__(
        self,
        dna_idx: np.ndarray,
        prot_idx: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.dna_idx = torch.from_numpy(dna_idx.astype(np.int64))
        self.prot_idx = torch.from_numpy(prot_idx.astype(np.int64))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self) -> int:
        return len(self.dna_idx)

    def __getitem__(self, i: int):
        return self.dna_idx[i], self.prot_idx[i], self.targets[i]


def build_long_format(data: BindingData):
    """Flatten the [N, P] matrix into parallel (dna_idx, prot_idx, affinity) arrays.

    Returns int64 dna_idx [N*P], int64 prot_idx [N*P], float32 affinity [N*P].
    Ordering is row-major: all proteins for dna 0, then dna 1, etc.
    """
    n, p = data.affinity.shape
    dna_idx = np.repeat(np.arange(n), p)
    prot_idx = np.tile(np.arange(p), n)
    aff = data.affinity.reshape(-1)
    return dna_idx, prot_idx, aff


def protein_group_folds(n_prot: int, n_splits: int = 5, seed: int = 0):
    """Yield (train_prot_ids, val_prot_ids) for leave-proteins-out CV.

    Uses GroupKFold over protein ids so that proteins in a validation fold never
    appear in the corresponding training fold. Proteins are shuffled first so
    folds are not biased by file order.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_prot)
    gkf = GroupKFold(n_splits=n_splits)
    # dummy X/y over proteins; groups = protein id ensures whole-protein folds
    dummy = np.zeros((n_prot, 1))
    for train_pos, val_pos in gkf.split(dummy, groups=perm):
        yield perm[train_pos], perm[val_pos]


if __name__ == "__main__":
    # Sanity checks (Phase 1 of verification plan).
    here = os.path.dirname(os.path.abspath(__file__))
    d = load_data(
        os.path.join(here, "training_DBPs_small.txt"),
        os.path.join(here, "training_seqs_small.txt"),
        os.path.join(here, "training_data_small.txt"),
    )
    print(f"proteins={d.n_prot}  dna={d.n_dna}  affinity={d.affinity.shape}")
    print(f"dna_onehot={d.dna_onehot.shape}  dtype={d.dna_onehot.dtype}")
    n_dna, n_prot = d.affinity.shape
    seq_len = d.dna_onehot.shape[2]
    assert d.dna_onehot.shape == (n_dna, 4, seq_len), d.dna_onehot.shape
    # one-hot columns should sum to at most 1 (unknown bases stay all-zero)
    col_sums = d.dna_onehot.sum(axis=1)
    assert (col_sums <= 1.0 + 1e-6).all(), "one-hot columns must sum to <= 1"

    dna_idx, prot_idx, aff = build_long_format(d)
    assert dna_idx.shape == (n_dna * n_prot,)
    # check alignment: long-format value matches matrix
    k = 1234
    assert aff[k] == d.affinity[dna_idx[k], prot_idx[k]]

    # reverse-complement of a palindrome-free seq should differ; double RC == identity
    sample = torch.from_numpy(d.dna_onehot[:8])
    rc = reverse_complement_onehot(sample)
    rcrc = reverse_complement_onehot(rc)
    assert torch.allclose(rcrc, sample), "double reverse-complement must be identity"

    folds = list(protein_group_folds(d.n_prot, n_splits=5))
    val_union = np.concatenate([v for _, v in folds])
    assert sorted(val_union) == list(range(d.n_prot)), "every protein held out once"
    for tr, va in folds:
        assert set(tr).isdisjoint(set(va)), "train/val proteins must be disjoint"
    print(f"folds={len(folds)}  held-out sizes={[len(v) for _, v in folds]}")

    # TargetTransform: per-protein z-scoring must NOT collapse the target.
    for mode in ("per_protein", "global"):
        tfm = TargetTransform(mode=mode, log1p=True, clip_quantile=0.001).fit(prot_idx, aff)
        zt = tfm.transform(prot_idx, aff)
        assert zt.std() > 0.5, f"{mode}: transform collapsed variance (std={zt.std():.4f})"
        assert np.isfinite(zt).all(), f"{mode}: non-finite transformed targets"
        if mode == "per_protein":
            # each protein column should be ~zero-mean, ~unit-std after transform
            for pid in np.unique(prot_idx)[:5]:
                m = prot_idx == pid
                assert abs(zt[m].mean()) < 1e-3 and abs(zt[m].std() - 1) < 1e-2
    assert tfm.clip_lo_ < tfm.clip_hi_, "clip bounds must be ordered lo<hi"
    print("data.py sanity checks passed (incl. TargetTransform).")
