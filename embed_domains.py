"""Cache ESM-2 embeddings of the *trimmed DNA-binding domain* of each protein.

The companion to embed_proteins.py, but it first runs `extract_dbd` (see
homeodomain.py) to cut each protein down to its ~60-residue DBD, then embeds
that. Same per-residue cache layout as `embed_proteins.py --per-residue`, so the
trained model is identical and a full-protein-vs-domain comparison is a clean
A/B: only the protein input region changes.

    python embed_domains.py --embedder esm2_t33_650M_UR50D   # per-residue (default)

Output: cache/<embedder>_homeodomain_perres.pt with
  {"embeddings": [P, R, D] fp16, "lengths": [P], "per_residue": True,
   "starts": [P], "methods": [P] (0=homeodomain anchor, 1=K/R window)}
R = max domain length (~60), so this cache is ~14x smaller than the full one.
"""

from __future__ import annotations

import argparse
import os

import torch

from data import read_lines
from embed_proteins import EMBEDDERS
from homeodomain import HD_LEN, extract_all
from utils import pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="esm2_t33_650M_UR50D",
                    choices=[k for k in EMBEDDERS if k.startswith("esm")])
    ap.add_argument("--dbp", default="training_DBPs_small.txt")
    ap.add_argument("--length", type=int, default=HD_LEN, help="domain window length")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    proteins = read_lines(os.path.join(here, args.dbp))
    dbds = extract_all(proteins, length=args.length)
    domains = [d.seq for d in dbds]
    n_hd = sum(d.method == "homeodomain" for d in dbds)

    device = pick_device()
    print(f"device={device}  embedder={args.embedder}  n_proteins={len(proteins)}")
    print(f"  DBD extraction: {n_hd} homeodomain-anchored, {len(dbds) - n_hd} K/R-window; "
          f"mean {sum(len(s) for s in domains) / len(domains):.0f} residues "
          f"(vs {sum(len(s) for s in proteins) / len(proteins):.0f} full)")

    # Per-residue ESM over the trimmed domains (reuses the embed_proteins factory).
    emb, lengths = EMBEDDERS[args.embedder](
        domains, device, args.batch_size, pool="mean", per_residue=True
    )
    assert emb.shape[0] == len(proteins) == lengths.shape[0]
    assert torch.isfinite(emb.float()).all(), "embeddings contain NaN/Inf"

    starts = torch.tensor([d.start for d in dbds], dtype=torch.long)
    methods = torch.tensor([0 if d.method == "homeodomain" else 1 for d in dbds], dtype=torch.long)

    os.makedirs(os.path.join(here, "cache"), exist_ok=True)
    out = args.out or os.path.join(here, "cache", f"{args.embedder}_homeodomain_perres.pt")
    torch.save({
        "embedder": args.embedder, "per_residue": True, "domain": "dbd",
        "embeddings": emb, "lengths": lengths, "starts": starts, "methods": methods,
    }, out)
    print(f"saved domain embeddings {tuple(emb.shape)} (lengths "
          f"{int(lengths.min())}..{int(lengths.max())}) -> {out}")


if __name__ == "__main__":
    main()
