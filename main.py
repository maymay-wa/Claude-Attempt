"""Submission entry point: score DNA probes for one DNA-binding protein (DBP).

Usage (exactly as specified in the project brief):

    python main.py <ofile> <DBP> <DNA>

  <ofile>  output file path; one predicted score per line, in the same order as
           the probes in <DNA> (textual file, one number per line).
  <DBP>    protein name, e.g. "DBP1" .. "DBP64". The protein sequence is looked
           up by position in the test-DBP file (line 1 = DBP1, ... line 64 =
           DBP64); a bare integer (1-based) is also accepted.
  <DNA>    DNA-probe test file: one probe (e.g. 36 bp, alphabet {A,C,G,T}) per
           line. Scores are produced for every probe in order.

How it works
------------
The trained two-tower model (frozen ESM-2 protein tower + CNN/cross-attention DNA
tower) generalises to *unseen* proteins, so it scores arbitrary (protein, probe)
pairs. For the named DBP we:
  1. fetch its frozen per-residue ESM-2 embedding (from the cached test embeddings
     if available -- see embed_proteins.py -- else compute it on the fly), then
  2. one-hot encode the probes and score every probe through the model, averaging
     a probe with its reverse complement (binding is ~strand-symmetric), and
  3. average an ensemble of fold checkpoints (each member per-protein z-scored
     first, so no single member's scale dominates -- the grade is Pearson Corr,
     which is invariant to that affine map).

Prediction time is printed to stderr (the brief measures prediction time).

Examples
--------
    python main.py DBP1.txt DBP1 test_seqs.txt
    python main.py out/DBP5.txt DBP5 test_seqs.txt --run-dir runs_caffeine
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import torch

from data import one_hot_encode, read_lines, reverse_complement_onehot
from model import build_model
from utils import load_config, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_dbp_index(name: str) -> int:
    """Map a DBP name to a 0-based row index in the DBP file.

    Accepts "DBP1"/"dbp_12"/"12" etc. The brief assigns names by file position
    (line 1 == DBP1), so "DBP<k>" -> index k-1. A bare integer is treated as a
    1-based line number for convenience.
    """
    m = re.search(r"(\d+)", str(name))
    if not m:
        raise ValueError(f"could not parse a protein number from DBP name {name!r}")
    num = int(m.group(1))
    if num < 1:
        raise ValueError(f"DBP number must be >= 1, got {num} (from {name!r})")
    return num - 1


def load_protein_residue_emb(
    dbp_index: int,
    proteins: list[str],
    esm_cache: str,
    embedder: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (reps [1, R, D] float, mask [1, R] bool, esm_dim) for one protein.

    Prefers the precomputed per-residue cache (index-aligned with the DBP file);
    falls back to computing the frozen embedding on the fly with ESM-2 so the
    script still works on a DBP/cache mismatch.
    """
    cache_path = esm_cache if os.path.isabs(esm_cache) else os.path.join(HERE, esm_cache)
    if os.path.exists(cache_path):
        # mmap: only the one protein slice we index is read off disk (the cache is
        # ~90 MB), so a per-DBP main.py call stays cheap.
        ckpt = torch.load(cache_path, map_location="cpu", mmap=True)
        if not ckpt.get("per_residue", False):
            raise ValueError(
                f"{cache_path!r} is a pooled cache; the cross-attention model needs "
                f"per-residue embeddings. Rebuild with:\n"
                f"    python embed_proteins.py --embedder {ckpt.get('embedder', embedder)} "
                f"--per-residue --dbp <dbp_file> --out {cache_path}"
            )
        emb_all = ckpt["embeddings"]  # [P, R, D] fp16
        if dbp_index < emb_all.shape[0]:
            length = int(ckpt["lengths"][dbp_index])
            # Guard against a cache built from a *different* DBP file: the cached
            # residue count must equal this protein's length, else fall through to
            # re-embedding the actual sequence (index-alignment can't be trusted).
            if length == len(proteins[dbp_index]):
                reps = emb_all[dbp_index].float().unsqueeze(0).to(device)  # [1, R, D]
                R = reps.shape[1]
                mask = (torch.arange(R, device=device) < length).unsqueeze(0)  # [1, R]
                return reps, mask, reps.shape[-1]
            print(
                f"[main] cached length {length} != protein {dbp_index} length "
                f"{len(proteins[dbp_index])}; cache is misaligned, re-embedding.",
                file=sys.stderr,
            )
        else:
            print(
                f"[main] DBP index {dbp_index} >= cached proteins {emb_all.shape[0]}; "
                f"computing embedding on the fly.",
                file=sys.stderr,
            )

    # ---- on-the-fly fallback: embed this single protein with frozen ESM-2 ----
    import esm

    seq = proteins[dbp_index]
    print(f"[main] embedding protein {dbp_index} ({len(seq)} aa) with {embedder} ...",
          file=sys.stderr)
    repr_layer = {
        "esm2_t36_3B_UR50D": 36, "esm2_t33_650M_UR50D": 33,
        "esm2_t30_150M_UR50D": 30, "esm2_t12_35M_UR50D": 12,
    }[embedder]
    loader = getattr(esm.pretrained, embedder)
    esm_model, alphabet = loader()
    esm_model = esm_model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    with torch.no_grad():
        _, _, tokens = batch_converter([("query", seq)])
        out = esm_model(tokens.to(device), repr_layers=[repr_layer], return_contacts=False)
        reps_seq = out["representations"][repr_layer][0, 1 : len(seq) + 1].float()  # [L, D]
    reps = reps_seq.unsqueeze(0).to(device)  # [1, L, D]
    mask = torch.ones(1, reps.shape[1], dtype=torch.bool, device=device)
    return reps, mask, reps.shape[-1]


def encode_protein(model, reps: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Run the protein tower once -> token/pooled embedding for the single protein."""
    enc = model.protein_encoder
    try:
        return enc(reps, mask)            # attn_pool_tokens takes a residue mask
    except TypeError:
        return enc(reps)                  # pooled protein heads take no mask


@torch.no_grad()
def score_probes(
    model,
    dna_onehot: torch.Tensor,
    p_emb: torch.Tensor,
    device: torch.device,
    block: int = 4096,
    tta_rc: bool = True,
) -> np.ndarray:
    """Score every probe for one protein -> [N] float32 (raw model output)."""
    model.eval()
    inter = model.interaction
    N = dna_onehot.shape[0]
    preds = np.empty(N, dtype=np.float32)

    def pair_predict(d_emb):
        if hasattr(inter, "score_grid"):
            return inter.score_grid(p_emb, d_emb).squeeze(1)  # [Bn,1] -> [Bn]
        # pooled fallback: one protein row broadcast over the probe block
        d2 = d_emb
        p2 = p_emb.expand(d2.shape[0], *p_emb.shape[1:])
        return inter(p2, d2)

    for s in range(0, N, block):
        dna = dna_onehot[s : s + block]
        d_emb = model.dna_encoder(dna)
        pred = pair_predict(d_emb)
        if tta_rc:
            d_rc = model.dna_encoder(reverse_complement_onehot(dna))
            pred = 0.5 * (pred + pair_predict(d_rc))
        preds[s : s + dna.shape[0]] = pred.float().cpu().numpy()
    return preds


def zscore(x: np.ndarray) -> np.ndarray:
    """Per-vector z-score; constant vectors map to zeros (correlation-invariant)."""
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-8 else np.zeros_like(x)


def resolve_checkpoints(args) -> list[str]:
    if args.checkpoints:
        cks = args.checkpoints
    else:
        run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(HERE, args.run_dir)
        cks = sorted(glob.glob(os.path.join(run_dir, "model_fold*.pt")))
    if not cks:
        raise FileNotFoundError(
            f"no model checkpoints found (run-dir={args.run_dir!r}). Train first, e.g.:\n"
            f"    python train.py --config {args.config} --out-dir {args.run_dir}"
        )
    return cks


def main() -> None:
    ap = argparse.ArgumentParser(description="Score DNA probes for one DBP (binding predictor).")
    ap.add_argument("ofile", help="output file: one score per line, in probe order")
    ap.add_argument("dbp", help="DBP name, e.g. DBP1 (1-based position in the DBP file)")
    ap.add_argument("dna", help="DNA-probe test file: one probe per line")
    ap.add_argument("--dbp-file", default="test_DBPs.txt",
                    help="file the DBP name indexes into (default: test_DBPs.txt)")
    ap.add_argument("--config", default="config.yaml", help="model config (component selection)")
    ap.add_argument("--run-dir", default="runs_caffeine",
                    help="dir holding model_fold*.pt to ensemble (if --checkpoints unset)")
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="explicit checkpoint paths to ensemble (overrides --run-dir)")
    ap.add_argument("--esm-cache", default="cache/test_esm2_t33_650M_UR50D_perres.pt",
                    help="per-residue test-protein embedding cache (index-aligned with --dbp-file)")
    ap.add_argument("--device", default=None, help="cpu | mps | cuda (default: auto)")
    ap.add_argument("--no-tta-rc", action="store_true", help="disable reverse-complement averaging")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    cfg = load_config(args.config if os.path.isabs(args.config) else os.path.join(HERE, args.config))

    dbp_file = args.dbp_file if os.path.isabs(args.dbp_file) else os.path.join(HERE, args.dbp_file)
    proteins = read_lines(dbp_file)
    dbp_index = parse_dbp_index(args.dbp)
    if dbp_index >= len(proteins):
        raise IndexError(f"{args.dbp!r} -> index {dbp_index} but {dbp_file!r} has {len(proteins)} proteins")

    probes = read_lines(args.dna)
    if not probes:
        raise ValueError(f"no probes read from {args.dna!r}")

    t0 = time.time()  # prediction timer (the brief measures prediction time)

    # protein side (frozen) -- cache embedder name for the on-the-fly fallback
    embedder = "esm2_t33_650M_UR50D"
    cache_path = args.esm_cache if os.path.isabs(args.esm_cache) else os.path.join(HERE, args.esm_cache)
    if os.path.exists(cache_path):
        embedder = torch.load(cache_path, map_location="cpu", mmap=True).get("embedder", embedder)
    reps, mask, esm_dim = load_protein_residue_emb(dbp_index, proteins, args.esm_cache, embedder, device)

    # DNA side
    dna_onehot = torch.from_numpy(one_hot_encode(probes)).to(device)
    seq_len = dna_onehot.shape[2]

    # ensemble of fold checkpoints
    checkpoints = resolve_checkpoints(args)
    tta_rc = not args.no_tta_rc
    member_scores = []
    for ck in checkpoints:
        model = build_model(cfg, esm_dim=esm_dim, seq_len=seq_len).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        p_emb = encode_protein(model, reps, mask)
        s = score_probes(model, dna_onehot, p_emb, device, tta_rc=tta_rc)
        member_scores.append(zscore(s))  # standardise before averaging (Pearson-invariant)
    scores = np.mean(member_scores, axis=0) if len(member_scores) > 1 else member_scores[0]

    elapsed = time.time() - t0

    out_path = args.ofile
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(f"{v:.6f}" for v in scores) + "\n")

    print(
        f"[main] {args.dbp} (idx {dbp_index}): scored {len(scores)} probes with "
        f"{len(checkpoints)} model(s) on {device} in {elapsed:.2f}s -> {out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
