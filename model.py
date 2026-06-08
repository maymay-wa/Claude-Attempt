"""Assembles a binding model from swappable encoders (see encoders.py).

  dna_onehot [B,4,L] --> dna_encoder    --> dna_emb  [B, Dd]
  esm_vec    [B,D]   --> protein_encoder--> prot_emb [B, Dp]
  (prot_emb, dna_emb) --> interaction --> affinity [B]

The three components are chosen by name in config.yaml, so you can swap the DNA
encoder (cnn/rnn/...), the protein head (mlp/linear/identity), or the fusion
(concat_hadamard/concat/bilinear) without changing training code.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from encoders import build_dna_encoder, build_interaction, build_protein_encoder


class BindingModel(nn.Module):
    def __init__(self, dna_encoder: nn.Module, protein_encoder: nn.Module, interaction: nn.Module):
        super().__init__()
        self.dna_encoder = dna_encoder
        self.protein_encoder = protein_encoder
        self.interaction = interaction

    def forward(self, dna_onehot: torch.Tensor, esm_vec: torch.Tensor) -> torch.Tensor:
        d = self.dna_encoder(dna_onehot)
        p = self.protein_encoder(esm_vec)
        return self.interaction(p, d)


def build_model(cfg: dict, esm_dim: int, seq_len: int) -> BindingModel:
    """Construct a BindingModel from the `dna_encoder`/`protein_encoder`/`interaction`
    spec dicts in the config."""
    dna = build_dna_encoder(cfg["dna_encoder"], seq_len=seq_len)
    prot = build_protein_encoder(cfg["protein_encoder"], in_dim=esm_dim)
    inter = build_interaction(cfg["interaction"], prot_dim=prot.out_dim, dna_dim=dna.out_dim)
    return BindingModel(dna, prot, inter)


if __name__ == "__main__":
    # Forward/backward shape test across every registered encoder/interaction combo.
    from encoders import DNA_ENCODERS, INTERACTIONS, PROTEIN_ENCODERS

    B, L, D = 16, 36, 640
    dna = torch.rand(B, 4, L)
    esm = torch.randn(B, D)
    print(f"DNA encoders:     {sorted(DNA_ENCODERS)}")
    print(f"Protein encoders: {sorted(PROTEIN_ENCODERS)}")
    print(f"Interactions:     {sorted(INTERACTIONS)}")
    for dna_name in DNA_ENCODERS:
        for inter_name in INTERACTIONS:
            cfg = {
                "dna_encoder": {"name": dna_name},
                "protein_encoder": {"name": "mlp"},
                "interaction": {"name": inter_name},
            }
            model = build_model(cfg, esm_dim=D, seq_len=L)
            out = model(dna, esm)
            assert out.shape == (B,), (dna_name, inter_name, out.shape)
            out.sum().backward()
            n = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  dna={dna_name:<5} interaction={inter_name:<16} ok  params={n:,}")
    print("model.py: all encoder/interaction combinations pass forward/backward.")
