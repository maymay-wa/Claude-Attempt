"""Precompute and cache fixed-size protein embeddings (the protein "encoder").

Protein embeddings are computed ONCE and cached to `cache/<name>.pt` as a
`[P, D]` tensor, then loaded by train.py. This decouples the (heavy, frozen)
protein representation from training, and makes the protein encoder swappable:
build a different cache and point `esm_cache` in config.yaml at it.

Registered embedders (`--embedder`):
  esm2_t33_650M_UR50D   frozen ESM-2 650M, mean-pooled  (1280-d, best quality)
  esm2_t30_150M_UR50D   frozen ESM-2 150M, mean-pooled  (640-d, good MPS default)
  esm2_t12_35M_UR50D    frozen ESM-2 35M, mean-pooled   (480-d, fastest)
  kmer3                 3-mer amino-acid composition    (8000-d, no model; baseline)

Add a new one with @register_embedder("name"); it must return a [P, D] tensor.

Run once, e.g.:
    python embed_proteins.py --embedder esm2_t30_150M_UR50D
    python embed_proteins.py --embedder kmer3            # cheap protein-encoder baseline
"""

from __future__ import annotations

import argparse
import itertools
import os

import torch

from data import read_lines

EMBEDDERS: dict[str, callable] = {}


def register_embedder(name):
    def deco(fn):
        EMBEDDERS[name] = fn
        return fn
    return deco


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --- ESM-2 family (frozen, mean-pooled over residues) ----------------------
_ESM_LOADERS = {
    "esm2_t33_650M_UR50D": 33,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t12_35M_UR50D": 12,
}


def _make_esm_embedder(model_name: str, repr_layer: int):
    @torch.no_grad()
    def _embed(proteins: list[str], device: torch.device, batch_size: int = 4) -> torch.Tensor:
        import esm

        loader = getattr(esm.pretrained, model_name)
        model, alphabet = loader()
        model = model.eval().to(device)
        batch_converter = alphabet.get_batch_converter()
        embs: list[torch.Tensor] = []
        for start in range(0, len(proteins), batch_size):
            chunk = proteins[start : start + batch_size]
            labeled = [(str(start + k), s) for k, s in enumerate(chunk)]
            _, _, tokens = batch_converter(labeled)
            out = model(tokens.to(device), repr_layers=[repr_layer], return_contacts=False)
            reps = out["representations"][repr_layer]
            for k, seq in enumerate(chunk):
                embs.append(reps[k, 1 : len(seq) + 1].mean(dim=0).float().cpu())
            print(f"  embedded {min(start + batch_size, len(proteins))}/{len(proteins)}")
        return torch.stack(embs, dim=0)

    return _embed


for _name, _layer in _ESM_LOADERS.items():
    register_embedder(_name)(_make_esm_embedder(_name, _layer))


# --- k-mer composition baseline (no model; demonstrates encoder swapping) --
@register_embedder("kmer3")
def _kmer3(proteins: list[str], device: torch.device, batch_size: int = 4) -> torch.Tensor:
    """Normalized 3-mer amino-acid composition; a cheap protein-encoder baseline."""
    aas = "ACDEFGHIKLMNPQRSTVWY"
    kmers = ["".join(t) for t in itertools.product(aas, repeat=3)]
    index = {km: i for i, km in enumerate(kmers)}
    out = torch.zeros(len(proteins), len(kmers))
    for i, seq in enumerate(proteins):
        for j in range(len(seq) - 2):
            idx = index.get(seq[j : j + 3])
            if idx is not None:
                out[i, idx] += 1.0
        total = out[i].sum()
        if total > 0:
            out[i] /= total
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="esm2_t30_150M_UR50D", choices=list(EMBEDDERS))
    ap.add_argument("--dbp", default="training_DBPs_small.txt")
    ap.add_argument("--out", default=None, help="output .pt path (auto if omitted)")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    proteins = read_lines(os.path.join(here, args.dbp))
    device = pick_device()
    print(f"device={device}  embedder={args.embedder}  n_proteins={len(proteins)}")

    emb = EMBEDDERS[args.embedder](proteins, device, args.batch_size)
    assert emb.shape[0] == len(proteins)
    assert torch.isfinite(emb).all(), "embeddings contain NaN/Inf"

    os.makedirs(os.path.join(here, "cache"), exist_ok=True)
    out = args.out or os.path.join(here, "cache", f"{args.embedder}.pt")
    torch.save({"embedder": args.embedder, "embeddings": emb}, out)
    print(f"saved embeddings {tuple(emb.shape)} -> {out}")


if __name__ == "__main__":
    main()
