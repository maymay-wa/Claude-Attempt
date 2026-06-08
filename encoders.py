"""Swappable building blocks for the binding model, selected by name from config.

Three registries let you drop in alternative components without touching the
training code:

  DNA_ENCODERS       one-hot DNA [B,4,L] -> [B, out_dim]
  PROTEIN_ENCODERS   cached protein vector [B,D] -> [B, out_dim]
  INTERACTIONS       (prot_emb, dna_emb) -> affinity scalar [B]

Add a new encoder by subclassing the relevant base and decorating it with
@register_*("name"); then set that name in config.yaml. `build_*` reads a spec
dict {"name": ..., <kwargs>} and constructs the module.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# registries
# ---------------------------------------------------------------------------
DNA_ENCODERS: dict[str, type] = {}
PROTEIN_ENCODERS: dict[str, type] = {}
INTERACTIONS: dict[str, type] = {}


def _register(reg):
    def deco(name):
        def inner(cls):
            reg[name] = cls
            return cls
        return inner
    return deco


register_dna_encoder = _register(DNA_ENCODERS)
register_protein_encoder = _register(PROTEIN_ENCODERS)
register_interaction = _register(INTERACTIONS)


def _build(reg, spec: dict, **injected):
    """Instantiate reg[spec['name']] with the remaining spec keys + injected kwargs."""
    spec = dict(spec)
    name = spec.pop("name")
    if name not in reg:
        raise KeyError(f"unknown component {name!r}; available: {sorted(reg)}")
    return reg[name](**injected, **spec)


def build_dna_encoder(spec: dict, seq_len: int) -> "BaseDNAEncoder":
    return _build(DNA_ENCODERS, spec, seq_len=seq_len)


def build_protein_encoder(spec: dict, in_dim: int) -> "BaseProteinEncoder":
    return _build(PROTEIN_ENCODERS, spec, in_dim=in_dim)


def build_interaction(spec: dict, prot_dim: int, dna_dim: int) -> nn.Module:
    return _build(INTERACTIONS, spec, prot_dim=prot_dim, dna_dim=dna_dim)


# ---------------------------------------------------------------------------
# DNA encoders: one-hot [B, 4, L] -> [B, out_dim]
# ---------------------------------------------------------------------------
class BaseDNAEncoder(nn.Module):
    out_dim: int


@register_dna_encoder("cnn")
class CNNEncoder(BaseDNAEncoder):
    """DeepBind/DeepSEA-style conv stack with global max+avg pooling."""

    def __init__(self, seq_len: int, channels: int = 128, out_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.conv = nn.Sequential(
            nn.Conv1d(4, channels, kernel_size=8, padding="same"),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding="same"),
            nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding="same"),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        self.project = nn.Sequential(nn.Linear(2 * channels, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        pooled = torch.cat([h.amax(dim=-1), h.mean(dim=-1)], dim=-1)
        return self.project(pooled)


@register_dna_encoder("rnn")
class BiLSTMEncoder(BaseDNAEncoder):
    """Bidirectional LSTM over the DNA sequence; mean-pools hidden states."""

    def __init__(self, seq_len: int, hidden: int = 128, num_layers: int = 1,
                 out_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.out_dim = out_dim
        self.lstm = nn.LSTM(
            input_size=4, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.project = nn.Sequential(nn.Linear(2 * hidden, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x.transpose(1, 2))  # [B, L, 2*hidden]
        return self.project(h.mean(dim=1))


# ---------------------------------------------------------------------------
# Protein encoders: cached vector [B, D] -> [B, out_dim]
# ---------------------------------------------------------------------------
class BaseProteinEncoder(nn.Module):
    out_dim: int


@register_protein_encoder("mlp")
class MLPProteinEncoder(BaseProteinEncoder):
    def __init__(self, in_dim: int, out_dim: int = 256, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_protein_encoder("linear")
class LinearProteinEncoder(BaseProteinEncoder):
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_protein_encoder("identity")
class IdentityProteinEncoder(BaseProteinEncoder):
    """Passthrough; lets the interaction head consume raw embeddings directly."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ---------------------------------------------------------------------------
# Interactions: (prot_emb, dna_emb) -> affinity scalar [B]
# ---------------------------------------------------------------------------
@register_interaction("concat_hadamard")
class ConcatHadamardHead(nn.Module):
    """Project both towers to a shared space; fuse concat[p, d, p*d] -> MLP -> 1."""

    def __init__(self, prot_dim: int, dna_dim: int, emb_dim: int = 256,
                 hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.p_proj = nn.Linear(prot_dim, emb_dim)
        self.d_proj = nn.Linear(dna_dim, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(3 * emb_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        p, d = self.p_proj(p), self.d_proj(d)
        return self.head(torch.cat([p, d, p * d], dim=-1)).squeeze(-1)


@register_interaction("concat")
class ConcatHead(nn.Module):
    """Plain concatenation of the two towers -> MLP -> 1 (no explicit product)."""

    def __init__(self, prot_dim: int, dna_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(prot_dim + dna_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([p, d], dim=-1)).squeeze(-1)


@register_interaction("bilinear")
class BilinearHead(nn.Module):
    """Low-rank bilinear interaction score: p^T W d via shared projection."""

    def __init__(self, prot_dim: int, dna_dim: int, rank: int = 256, dropout: float = 0.3):
        super().__init__()
        self.p_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(prot_dim, rank))
        self.d_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(dna_dim, rank))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, p: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return (self.p_proj(p) * self.d_proj(d)).sum(dim=-1) + self.bias
