# Protein–DNA Binding Affinity Predictor

Predicts binding affinity between DNA-binding proteins (TFs) and short DNA
probes, designed to **generalize to proteins never seen during training**.

A two-tower model: a learned CNN over one-hot DNA (DeepBind/DeepSEA style) and a
**frozen ESM-2** protein encoder, fused via a multiplicative-interaction MLP and
trained as regression on continuous affinity. Generalization is measured with
**leave-proteins-out** cross-validation.

## Data
- `training_DBPs_small.txt` — 50 protein amino-acid sequences
- `training_seqs_small.txt` — 5,000 DNA probes (36 bp, {A,C,G,T})
- `training_data_small.txt` — 5,000 × 50 affinity matrix (row=DNA, col=protein)

## Pipeline
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python explore_data.py               # 0. EDA: distribution / outliers / per-protein scale
python data.py                       # 1. sanity-check parsing, folds, target transform
python embed_proteins.py             # 2. cache frozen ESM-2 embeddings (run once)
python model.py                      # 3. forward/backward over ALL encoder combos
python train.py --folds 1 --epochs 2 # 4. smoke test (1 fold, 2 epochs)
python train.py                      # 5. full leave-proteins-out CV
python evaluate.py                   # 6. aggregate metrics + kNN/mean baselines
```

## Modularity — swapping components
Models are assembled from three registries in `encoders.py`, chosen by `name` in
`config.yaml`. Drop in a new component by subclassing the relevant base and
decorating it (`@register_dna_encoder("...")`), then set that name in config.
No training-code changes needed.

| Slot | Config key | Built-in options |
|---|---|---|
| DNA encoder | `dna_encoder.name` | `cnn`, `rnn` |
| Protein head | `protein_encoder.name` | `mlp`, `linear`, `identity` |
| Interaction | `interaction.name` | `concat_hadamard`, `concat`, `bilinear` |
| Protein embedding | `esm_cache` | `esm2_t33_650M`, `esm2_t30_150M`, `esm2_t12_35M`, `kmer3` |

Swap the **protein encoder** by building a different cache and pointing at it:
```bash
python embed_proteins.py --embedder esm2_t33_650M_UR50D   # higher quality (1280-d)
python embed_proteins.py --embedder kmer3                 # cheap no-ESM baseline
# then set  esm_cache: cache/<name>.pt  in config.yaml
```

## Data handling (see `explore_data.py`)
The `target` block in `config.yaml` controls the regression target transform,
fit on **training proteins only** (no leakage):
- `mode: per_protein` (default) — z-score within each protein. Protein mean
  affinity spans ~94× here, so a global z-score would let strong binders dominate;
  per-protein z-scoring learns *specificity*, which is what the metric rewards and
  the only thing knowable for an unseen protein.
- `log1p: true` — compress the right skew (skew 0.62 → ~0).
- `clip_quantile: 0.001` — winsorize to the [0.1%, 99.9%] train quantiles (tames
  the thin upper tail); `null` to disable.

## Files
| File | Role |
|---|---|
| `explore_data.py` | EDA: distribution, skew, outliers, per-protein scale |
| `data.py` | parsing, one-hot + reverse-complement, protein-grouped folds, `TargetTransform` |
| `embed_proteins.py` | protein-embedder registry (ESM-2 / kmer) → cached `[P, D]` |
| `encoders.py` | swappable DNA / protein / interaction registries |
| `model.py` | `BindingModel` + `build_model` factory |
| `metrics.py` | per-protein Pearson/Spearman |
| `train.py` | leave-proteins-out CV loop, early stopping, per-fold predictions |
| `evaluate.py` | aggregate metrics + per-protein-mean and kNN-in-ESM baselines |
| `config.yaml` | all hyperparameters + component selection |

## Reading the results
Success = mean held-out-protein **Spearman ρ meaningfully above the kNN-in-ESM
baseline**. With only 50 proteins expect high variance; the full dataset (more
proteins) is what makes unseen-protein generalization work.
