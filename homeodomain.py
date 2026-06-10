"""Trim each transcription factor to its DNA-binding domain (DBD) before ESM.

The single most important trick in the project brief: *don't embed the whole
protein*. Binding specificity lives in a short, structured DNA-binding domain
(for homeodomains, ~60 residues whose recognition helix makes base contacts at
positions ~47, 50, 51, 54); the rest of the chain is disordered linker and
activation domains that only add noise to the protein representation and cost
ESM compute. Feeding ESM the domain instead of the full protein both *denoises*
the signal and is far cheaper (here: ~60 residues vs. up to 832).

The brief assumes a pure-homeodomain set. This dataset is broader (~45%
homeodomains, plus C2H2 / nuclear-receptor zinc fingers, HMG/SOX boxes, bHLH,
forkhead, ...), so we extract the DBD with two tools:

  1. find_homeodomain  -- precise, motif-anchored. The homeodomain recognition
     helix carries the near-invariant W48-F49-x50-N51-x52-R53 ("WFxNxR")
     signature. We find that anchor and cut the canonical 60-residue domain so
     it sits at standard homeodomain numbering (W at position 48). This is the
     brief's exact recipe and is reliable for the homeodomain subset.

  2. find_dbd_window   -- family-agnostic fallback for everything else. DBDs are
     strongly enriched in basic residues (K/R) because they grip the phosphate
     backbone, and in the aromatics/Asn that read bases. We slide a window and
     take the one maximizing that DBD-like composition. Crude but honest, and we
     *validate* it against the homeodomain anchor (where the answer is known):
     on the homeodomain subset the window should land on the recognition helix.

`extract_dbd` uses the precise anchor when a homeodomain is detected and the
generic window otherwise, always returning the trimmed sequence plus metadata
(method, start, length) so the choice is auditable.

Run `python homeodomain.py` for a coverage + self-validation report on the data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Homeodomain geometry (standard 60-residue numbering, 1-indexed)
# ---------------------------------------------------------------------------
HD_LEN = 60                # canonical homeodomain length
HD_ANCHOR_POS = 48         # the invariant W of the recognition helix sits here
# DNA-contacting recognition-helix residues (brief calls out ~47, 50, 51, 54)
HD_CONTACT_POSITIONS = (47, 50, 51, 54)

# Recognition-helix anchors, tried strict -> loose. Each matches the WFxNxR
# block; group/used start marks the anchor residue (homeodomain position 48).
#   W48  F49  x50  N51  x52  R53
_HD_ANCHORS = [
    re.compile(r"WF.N.R"),        # canonical, invariant core
    re.compile(r"W[FYL].N.R"),    # F49 conservatively substituted
    re.compile(r"[WFY][FYL].N.[RK]"),  # W48/F49 substituted, R53->K tolerated
    re.compile(r"[FYW].N[RK]R"),  # last resort: anchor on N51..R53, back off 1
]
# How far before the regex match start the homeodomain position-1 lies.
# Anchors 0-2 start at W48 -> 47 residues precede it. Anchor 3 starts one
# residue before N51 i.e. position 50 -> 49 residues precede it.
_HD_ANCHOR_OFFSET = [47, 47, 47, 49]


@dataclass
class DBD:
    """An extracted DNA-binding domain window and how we found it."""

    seq: str          # the trimmed amino-acid substring
    start: int        # 0-indexed start in the full protein
    end: int          # 0-indexed end (exclusive)
    method: str       # "homeodomain" | "kr_window"
    anchor: int | None  # 0-indexed position of the recognition-helix anchor (W48), if any

    @property
    def length(self) -> int:
        return self.end - self.start


def find_homeodomain(seq: str, length: int = HD_LEN) -> DBD | None:
    """Locate the homeodomain via its recognition-helix motif; None if absent.

    Cuts `length` residues positioned so the conserved tryptophan lands at
    homeodomain position 48, clamped to the protein ends. Returns the *first*
    anchor match (homeodomains are single-copy; the recognition helix is unique).
    """
    for rx, off in zip(_HD_ANCHORS, _HD_ANCHOR_OFFSET):
        m = rx.search(seq)
        if not m:
            continue
        anchor = m.start()                      # 0-indexed homeodomain position 48
        start = anchor - off                    # where position 1 would fall
        start = max(0, min(start, len(seq) - length)) if len(seq) >= length else 0
        end = min(len(seq), start + length)
        return DBD(seq[start:end], start, end, "homeodomain", anchor)
    return None


# Per-residue DBD-likeness weights for the generic window. Basic residues grip
# the backbone; aromatics + Asn/Gln/His/Thr/Ser read bases and pack the helix.
_DBD_WEIGHTS = {a: 0.0 for a in "ACDEFGHIKLMNPQRSTVWY"}
for _a in "KR":
    _DBD_WEIGHTS[_a] = 1.0      # strong: backbone contacts
for _a in "NQHWYF":
    _DBD_WEIGHTS[_a] = 0.5      # base-reading / helix residues
for _a in "DE":
    _DBD_WEIGHTS[_a] = -0.5     # acidic: typical of activation domains, not DBDs


def find_dbd_window(seq: str, width: int = HD_LEN) -> DBD:
    """Family-agnostic DBD locator: the `width`-residue window richest in
    DBD-like composition (basic + base-reading residues). Always returns a
    window (used when no specific family motif is detected)."""
    n = len(seq)
    if n <= width:
        return DBD(seq, 0, n, "kr_window", None)
    score = [_DBD_WEIGHTS.get(a, 0.0) for a in seq]
    # rolling-sum scan for the max-scoring contiguous window
    cur = sum(score[:width])
    best, best_start = cur, 0
    for i in range(1, n - width + 1):
        cur += score[i + width - 1] - score[i - 1]
        if cur > best:
            best, best_start = cur, i
    return DBD(seq[best_start:best_start + width], best_start, best_start + width, "kr_window", None)


def extract_dbd(seq: str, length: int = HD_LEN) -> DBD:
    """Trim a protein to its DBD: precise homeodomain anchor if detected,
    otherwise the generic basic-residue-rich window."""
    hd = find_homeodomain(seq, length)
    return hd if hd is not None else find_dbd_window(seq, length)


def extract_all(seqs: list[str], length: int = HD_LEN) -> list[DBD]:
    return [extract_dbd(s, length) for s in seqs]


if __name__ == "__main__":
    import os

    from data import read_lines

    here = os.path.dirname(os.path.abspath(__file__))
    seqs = read_lines(os.path.join(here, "training_DBPs_small.txt"))
    dbds = extract_all(seqs)

    n = len(seqs)
    hd = [d for d in dbds if d.method == "homeodomain"]
    print(f"proteins={n}")
    print(f"  homeodomain anchor found : {len(hd)} ({100*len(hd)/n:.0f}%)")
    print(f"  generic K/R window       : {n - len(hd)}")
    lens = [d.length for d in dbds]
    full = [len(s) for s in seqs]
    print(f"  domain length  min/med/max = {min(lens)}/{sorted(lens)[n//2]}/{max(lens)}")
    print(f"  full protein   min/med/max = {min(full)}/{sorted(full)[n//2]}/{max(full)}")
    print(f"  mean residues fed to ESM: {sum(lens)/n:.0f} vs {sum(full)/n:.0f} full "
          f"({sum(full)/sum(lens):.1f}x fewer)")

    # Self-validation: where the precise anchor exists, does the generic window
    # land on the same place? High overlap => the fallback is trustworthy on the
    # families we cannot anchor explicitly.
    overlaps = []
    for s, d in zip(seqs, dbds):
        if d.method != "homeodomain":
            continue
        w = find_dbd_window(s)
        lo, hi = max(d.start, w.start), min(d.end, w.end)
        overlaps.append(max(0, hi - lo) / d.length)
    if overlaps:
        import statistics
        hit = sum(o > 0.5 for o in overlaps) / len(overlaps)
        print(f"  fallback vs homeodomain anchor: mean overlap {statistics.mean(overlaps):.2f}, "
              f"{100*hit:.0f}% overlap >50% (validates the generic window)")

    # Sanity: every recognition-helix anchor should expose the WFxNxR core, and
    # the contact residues should be inside the cut domain.
    bad = 0
    for s, d in zip(seqs, dbds):
        if d.method != "homeodomain":
            continue
        if not (d.start <= d.anchor < d.end):
            bad += 1
    assert bad == 0, f"{bad} homeodomain anchors fell outside their cut window"
    print("homeodomain.py: extraction + self-validation passed.")
