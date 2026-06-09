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

import inspect

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
    """Instantiate reg[spec['name']] with injected kwargs + the spec keys that the
    target constructor actually accepts.

    Spec keys the chosen component doesn't take are dropped, so a single config
    block (e.g. the `interaction` block) can carry the union of options' params
    and you can swap components by changing `name` alone. The flip side: a typo'd
    config key is silently ignored rather than raising.
    """
    spec = dict(spec)
    name = spec.pop("name")
    if name not in reg:
        raise KeyError(f"unknown component {name!r}; available: {sorted(reg)}")
    cls = reg[name]
    accepted = inspect.signature(cls).parameters
    kwargs = {k: v for k, v in spec.items() if k in accepted}
    return cls(**injected, **kwargs)


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


@register_dna_encoder("cnn_attn")
class CNNAttnEncoder(BaseDNAEncoder):
    """Same conv stack as `cnn`, but the uniform average-pool branch is replaced
    by a learned multi-head *attentive pool* over sequence positions.

    A 1x1 conv scores every position per head; softmax over the length axis turns
    those scores into attention weights, and each head returns its weighted sum of
    conv features. The global max-pool branch is kept identical to `cnn`, so the
    only thing that changes versus `cnn` is avg-pool -> attentive-pool. That makes
    the A/B comparison a clean test of whether *learned* position weighting beats
    uniform averaging -- i.e. whether a full transformer (self-attention across
    positions) is likely worth the extra engineering.

    pooled = concat[ global_max(h) , head_1, ..., head_H ]  -> Linear -> LayerNorm
    """

    def __init__(self, seq_len: int, channels: int = 128, out_dim: int = 256,
                 dropout: float = 0.2, attn_heads: int = 4):
        super().__init__()
        self.out_dim = out_dim
        self.attn_heads = attn_heads
        self.conv = nn.Sequential(
            nn.Conv1d(4, channels, kernel_size=8, padding="same"),
            nn.BatchNorm1d(channels), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding="same"),
            nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding="same"),
            nn.BatchNorm1d(channels), nn.ReLU(),
        )
        # per-position, per-head attention logits over the length axis
        self.attn = nn.Conv1d(channels, attn_heads, kernel_size=1)
        # max branch (1*channels) + attentive branch (attn_heads*channels)
        self.project = nn.Sequential(
            nn.Linear((1 + attn_heads) * channels, out_dim), nn.LayerNorm(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)                                   # [B, C, L]
        w = torch.softmax(self.attn(h), dim=-1)            # [B, H, L] (over positions)
        attn_pooled = torch.bmm(w, h.transpose(1, 2))      # [B, H, C]
        attn_pooled = attn_pooled.reshape(h.shape[0], -1)  # [B, H*C]
        pooled = torch.cat([h.amax(dim=-1), attn_pooled], dim=-1)
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
