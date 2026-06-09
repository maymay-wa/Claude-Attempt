"""Leave-proteins-out cross-validation training for the binding predictor.

For each fold: train on all (DNA, protein) pairs whose protein is in the
training group, validate on pairs whose protein is held out (never seen during
training). Early stopping tracks the mean per-held-out-protein Pearson r.
Best-epoch validation predictions are cached for evaluate.py.

Usage:
    python train.py                       # full CV from config.yaml
    python train.py --folds 1 --epochs 2  # quick smoke test
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (
    PairDataset,
    TargetTransform,
    build_long_format,
    load_data,
    protein_group_folds,
    reverse_complement_onehot,
)
from metrics import per_protein_correlations
from model import build_model
from utils import load_config, pick_device


def pairs_for_proteins(dna_idx, prot_idx, aff, prot_ids: np.ndarray):
    """Subset the long-format arrays to rows whose protein is in prot_ids."""
    mask = np.isin(prot_idx, prot_ids)
    return dna_idx[mask], prot_idx[mask], aff[mask]


def run_epoch(model, loader, dna_onehot, esm_emb, device, rc_prob, optimizer=None, loss_fn=None):
    """One pass over loader. If optimizer is None, runs in eval mode.

    `dna_onehot` and `esm_emb` are device-resident lookup tables; the loader only
    supplies indices, so per-batch host->device transfer is just the index/target
    tensors. Returns (mean_loss, prot_idx, y_true, y_pred); the y arrays are only
    populated in eval mode (optimizer is None).
    """
    train_mode = optimizer is not None
    model.train(train_mode)
    total, n = 0.0, 0
    all_pidx, all_true, all_pred = [], [], []

    for didx, pidx, target in loader:
        dna = dna_onehot[didx.to(device)]  # gather one-hots on-device
        target = target.to(device)
        if train_mode and rc_prob > 0:
            # reverse-complement augmentation on a random subset of the batch
            flip = torch.rand(dna.shape[0], device=device) < rc_prob
            if flip.any():
                dna[flip] = reverse_complement_onehot(dna[flip])
        esm_vec = esm_emb[pidx.to(device)]

        with torch.set_grad_enabled(train_mode):
            pred = model(dna, esm_vec)
            loss = loss_fn(pred, target)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        else:
            all_pidx.append(pidx.numpy())
            all_true.append(target.cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())

        bs = dna.shape[0]
        total += loss.item() * bs
        n += bs

    if train_mode:
        return total / n, None, None, None
    return (
        total / n,
        np.concatenate(all_pidx),
        np.concatenate(all_true),
        np.concatenate(all_pred),
    )


def train_one_fold(fold, train_ids, val_ids, data, dna_onehot, esm_emb, cfg, device, out_dir):
    dna_idx, prot_idx, aff = build_long_format(data)
    tr = pairs_for_proteins(dna_idx, prot_idx, aff, train_ids)
    va = pairs_for_proteins(dna_idx, prot_idx, aff, val_ids)

    tcfg = cfg["target"]
    tfm = TargetTransform(
        mode=tcfg["mode"], log1p=tcfg["log1p"], clip_quantile=tcfg.get("clip_quantile")
    ).fit(tr[1], tr[2])
    tr_t = tfm.transform(tr[1], tr[2])
    va_t = tfm.transform(va[1], va[2])

    train_ds = PairDataset(tr[0], tr[1], tr_t)
    val_ds = PairDataset(va[0], va[1], va_t)
    pin = device.type == "cuda"
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, pin_memory=pin)

    seq_len = data.dna_onehot.shape[2]
    model = build_model(cfg, esm_dim=esm_emb.shape[1], seq_len=seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    loss_fn = nn.HuberLoss(delta=1.0)

    best_pearson = -np.inf
    best_state = None
    best_preds = None
    patience = cfg["early_stop_patience"]
    bad_epochs = 0

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, *_ = run_epoch(model, train_dl, dna_onehot, esm_emb, device, cfg["rc_prob"], optimizer, loss_fn)
        val_loss, vpidx, vtrue, vpred = run_epoch(model, val_dl, dna_onehot, esm_emb, device, 0.0, None, loss_fn)
        corr = per_protein_correlations(vpidx, vtrue, vpred)
        pe, sp = corr["pearson_mean"], corr["spearman_mean"]
        scheduler.step(pe)
        print(
            f"  fold {fold} epoch {epoch:02d}  train_loss={tr_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_pearson={pe:.4f}  val_spearman={sp:.4f}"
        )

        if pe > best_pearson:
            best_pearson = pe
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            # Store RAW true affinity + model-space prediction. Per-protein metrics
            # are correlation-based (invariant to the target transform), so no
            # inverse transform is needed and this stays uniform across modes.
            best_preds = {
                "prot_idx": vpidx,
                "dna_idx": va[0],  # aligned: val_ds preserves order (shuffle=False)
                "y_true": va[2],   # raw affinity
                "y_pred": vpred,   # model output (transformed space)
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  fold {fold}: early stop at epoch {epoch}")
                break

    # persist best checkpoint + predictions for this fold
    torch.save(best_state, os.path.join(out_dir, f"model_fold{fold}.pt"))
    np.savez(os.path.join(out_dir, f"preds_fold{fold}.npz"), **best_preds)
    print(f"  fold {fold} BEST val_pearson={best_pearson:.4f}")
    return best_preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--folds", type=int, default=None, help="limit #folds (smoke test)")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (smoke test)")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(os.path.join(here, args.config))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = pick_device()

    data = load_data(
        os.path.join(here, cfg["dbp_path"]),
        os.path.join(here, cfg["seq_path"]),
        os.path.join(here, cfg["data_path"]),
    )
    ckpt = torch.load(os.path.join(here, cfg["esm_cache"]), map_location="cpu")
    esm_emb = ckpt["embeddings"].to(device)
    embedder = ckpt.get("embedder", ckpt.get("model", "?"))
    if esm_emb.shape[0] != data.n_prot:
        raise ValueError(
            f"protein-embedding cache {cfg['esm_cache']!r} has {esm_emb.shape[0]} "
            f"rows but the data has {data.n_prot} proteins. The cache is stale for "
            f"the current {cfg['dbp_path']!r}; rebuild it with:\n"
            f"    python embed_proteins.py --embedder {embedder} --dbp {cfg['dbp_path']}"
        )
    # Both lookup tables are tiny; keep them device-resident so the per-batch
    # transfer is only the index/target tensors (see run_epoch).
    dna_onehot = torch.from_numpy(data.dna_onehot).to(device)
    print(f"device={device}  protein_emb={tuple(esm_emb.shape)} ({embedder})  pairs={data.n_dna*data.n_prot:,}")

    out_dir = os.path.join(here, "runs")
    os.makedirs(out_dir, exist_ok=True)

    folds = list(protein_group_folds(data.n_prot, n_splits=cfg["n_splits"], seed=cfg["seed"]))
    if args.folds is not None:
        folds = folds[: args.folds]

    fold_means = []
    for fold, (train_ids, val_ids) in enumerate(folds):
        print(f"\n=== Fold {fold}: train {len(train_ids)} proteins, hold out {len(val_ids)} ===")
        preds = train_one_fold(fold, train_ids, val_ids, data, dna_onehot, esm_emb, cfg, device, out_dir)
        corr = per_protein_correlations(preds["prot_idx"], preds["y_true"], preds["y_pred"])
        fold_means.append((corr["pearson_mean"], corr["spearman_mean"]))

    fm = np.array(fold_means)
    print("\n=== Cross-validation summary (leave-proteins-out) ===")
    print(f"Pearson : {fm[:,0].mean():.4f} +/- {fm[:,0].std():.4f}")
    print(f"Spearman: {fm[:,1].mean():.4f} +/- {fm[:,1].std():.4f}")
    print(f"Per-fold predictions saved in {out_dir}/preds_fold*.npz")


if __name__ == "__main__":
    main()
