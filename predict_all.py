"""Score every test DBP and assemble the submission zip.

The brief's submission is "64 scored PBM files (e.g. DBP1.txt) zipped in one zip
file". Calling main.py 64 times reloads the model and the embedding cache on each
call; this script loads them ONCE and scores all proteins against all probes in a
single batched pass, so it is far faster (and reports the per-DBP prediction time
used by the run-time grade).

    python predict_all.py                       # -> submission/DBP1.txt .. DBP64.txt + submission.zip
    python predict_all.py --run-dir runs_caffeine --dna test_seqs.txt

Output files are written in probe order, one score per line -- identical format to
main.py, so a per-DBP file here equals `python main.py <ofile> DBP<i> <dna>`.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
import zipfile

import numpy as np
import torch

from data import one_hot_encode, read_lines, reverse_complement_onehot
from main import encode_protein, zscore
from model import build_model
from utils import load_config, pick_device

HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def score_all(model, dna_onehot, p_emb_all, device, block=4096, tta_rc=True) -> np.ndarray:
    """Score every probe against every protein -> [N, P] (raw model output).

    p_emb_all is the protein tower output for ALL proteins ([P, K, Dp] for the
    cross-attention head). Encodes each probe block once and reuses it across all
    proteins via the interaction's batched grid.
    """
    model.eval()
    inter = model.interaction
    N, P = dna_onehot.shape[0], p_emb_all.shape[0]
    preds = np.empty((N, P), dtype=np.float32)

    def grid(d_emb):
        if hasattr(inter, "score_grid"):
            return inter.score_grid(p_emb_all, d_emb)           # [Bn, P]
        out = torch.empty(d_emb.shape[0], P, device=device)
        for j in range(P):
            pj = p_emb_all[j].unsqueeze(0).expand(d_emb.shape[0], *p_emb_all.shape[1:])
            out[:, j] = inter(pj, d_emb)
        return out

    for s in range(0, N, block):
        dna = dna_onehot[s : s + block]
        d_emb = model.dna_encoder(dna)
        pred = grid(d_emb)
        if tta_rc:
            pred = 0.5 * (pred + grid(model.dna_encoder(reverse_complement_onehot(dna))))
        preds[s : s + dna.shape[0]] = pred.float().cpu().numpy()
    return preds


def main() -> None:
    ap = argparse.ArgumentParser(description="Score all test DBPs and build the submission zip.")
    ap.add_argument("--dna", default="test_seqs.txt", help="DNA-probe test file (one probe/line)")
    ap.add_argument("--dbp-file", default="test_DBPs.txt", help="test DBP sequences (one/line)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-dir", default="runs_caffeine", help="dir with model_fold*.pt to ensemble")
    ap.add_argument("--checkpoints", nargs="*", default=None, help="explicit checkpoints (overrides --run-dir)")
    ap.add_argument("--esm-cache", default="cache/test_esm2_t33_650M_UR50D_perres.pt",
                    help="per-residue test embeddings (index-aligned with --dbp-file)")
    ap.add_argument("--out-dir", default="submission", help="dir for DBP*.txt")
    ap.add_argument("--zip", default="submission.zip", help="output zip of all score files")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-tta-rc", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    cfg = load_config(args.config if os.path.isabs(args.config) else os.path.join(HERE, args.config))

    proteins = read_lines(args.dbp_file if os.path.isabs(args.dbp_file) else os.path.join(HERE, args.dbp_file))
    probes = read_lines(args.dna if os.path.isabs(args.dna) else os.path.join(HERE, args.dna))
    P, N = len(proteins), len(probes)

    cache_path = args.esm_cache if os.path.isabs(args.esm_cache) else os.path.join(HERE, args.esm_cache)
    ckpt = torch.load(cache_path, map_location="cpu")
    if not ckpt.get("per_residue", False):
        raise ValueError(f"{cache_path!r} is pooled; need a per-residue cache (embed with --per-residue).")
    reps_all = ckpt["embeddings"].float().to(device)            # [P, R, D]
    if reps_all.shape[0] != P:
        raise ValueError(f"cache has {reps_all.shape[0]} proteins but {args.dbp_file} has {P}")
    R = reps_all.shape[1]
    mask_all = (torch.arange(R, device=device)[None, :] < ckpt["lengths"].to(device)[:, None])  # [P,R]
    esm_dim = reps_all.shape[-1]

    dna_onehot = torch.from_numpy(one_hot_encode(probes)).to(device)
    seq_len = dna_onehot.shape[2]

    if args.checkpoints:
        checkpoints = args.checkpoints
    else:
        run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(HERE, args.run_dir)
        checkpoints = sorted(glob.glob(os.path.join(run_dir, "model_fold*.pt")))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints in {args.run_dir!r}")

    print(f"device={device}  proteins={P}  probes={N}  models={len(checkpoints)}")
    t0 = time.time()
    # accumulate per-member, per-protein z-scored predictions, then average members
    acc = np.zeros((N, P), dtype=np.float64)
    for ck in checkpoints:
        model = build_model(cfg, esm_dim=esm_dim, seq_len=seq_len).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        p_emb_all = encode_protein(model, reps_all, mask_all)   # [P, K, Dp]
        preds = score_all(model, dna_onehot, p_emb_all, device, tta_rc=not args.no_tta_rc)  # [N,P]
        acc += np.stack([zscore(preds[:, j]) for j in range(P)], axis=1)
    scores = acc / len(checkpoints)                              # [N, P]
    elapsed = time.time() - t0

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(HERE, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for j in range(P):
        p = os.path.join(out_dir, f"DBP{j + 1}.txt")
        with open(p, "w") as fh:
            fh.write("\n".join(f"{v:.6f}" for v in scores[:, j]) + "\n")
        paths.append(p)

    zip_path = args.zip if os.path.isabs(args.zip) else os.path.join(HERE, args.zip)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, arcname=os.path.basename(p))

    runtime_score = max(min(1.0, 2.0 - elapsed / 600.0), 0.0)  # brief's 10% run-time term
    print(f"wrote {P} score files -> {out_dir}/  and  {zip_path}")
    print(f"prediction time: {elapsed:.1f}s total for {P} DBPs ({elapsed / P:.3f}s/DBP)  "
          f"-> run-time score max(min(1, 2-{elapsed:.0f}/600), 0) = {runtime_score:.3f}")


if __name__ == "__main__":
    main()
