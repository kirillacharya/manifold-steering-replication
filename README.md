# Manifold Steering Replication

Replication of the main figure from **"Manifold Steering Reveals the Shared
Geometry of Neural Network Representation and Behavior"**
([Wurgaft et al., 2026](https://arxiv.org/abs/2605.05115)) on
**Gemma-2-2B**, plus an extension to two-dimensional concept structure
(a day × hour torus). Built as the final project for Stanford **CS 221M**
(Mechanistic Interpretability); the 2-page report is in
[`report/main.pdf`](report/main.pdf).

<p align="center">
  <img src="figures/weekdays_activation.png" width="44%">
  <img src="figures/weekdays_probs.png" width="49%">
</p>

**The claim.** Concept domains like weekdays trace smooth, cyclic manifolds in
both *activation space* (layer-24 hidden states) and *behavior space*
(Hellinger-embedded output distributions). Steering **along** the activation
manifold produces smooth, ordered behavioral transitions
(Tue → Wed → Thu → Fri); ordinary **linear** steering cuts off-manifold and
"teleports" probability mass.

## Quickstart — regenerate every figure without a GPU

Per-prompt activations and steering probabilities are committed as small
checkpoints, so all figures rebuild in seconds on any machine:

```bash
git clone <this repo> && cd manifold-steering-replication
uv sync                                      # or: pip install -e .

uv run python scripts/plot_weekdays_separate.py   # behavior + activation manifolds
uv run python scripts/plot_months_separate.py
uv run python scripts/plot_probs_separate.py      # probability trajectories
uv run python scripts/render_torus_3d_L24.py      # day×hour torus embedding
uv run python scripts/plot_subspace_hours.py      # torus steering grids
```

## Full pipeline — collect activations from scratch

Requires a GPU (CUDA or Apple MPS) and access to the gated
[`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it) weights
(`huggingface-cli login` or `export HF_TOKEN=...`). Everything is
deterministic — no sampling, no training — and every script takes `--seed`
for exact reproducibility.

```bash
# 1. Weekday + month activations and per-prompt output probabilities (layer 24)
uv run python scripts/run_weekdays.py --layer 24 --seed 0
uv run python scripts/run_months.py   --layer 24 --seed 0

# 2. Steering interventions (linear vs. manifold) for both domains
uv run python scripts/run_steering_probs.py --seed 0

# 3. Torus extension: 168 day×hour prompts, subspaces, product steering
uv run python scripts/run_gemma_torus.py --layer 24 --acts-only \
    --acts-cache figures/gemma_torus/acts_2b_L24.pt --seed 0
uv run python scripts/run_hierarchical_steering.py --seed 0

# 4. Then the plotting scripts from the quickstart above
```

On a SLURM cluster: `sbatch slurm/steering.sbatch` (edit the partition/account
header for your site), or `bash slurm/run_all.sh` on any GPU box.

## How it works

| Step | Script | What it does |
|---|---|---|
| Prompts → activations | `run_weekdays.py`, `run_months.py` | 42/72 arithmetic prompts ("Q: What day is two days after Monday? A:"), grouped by answer concept; records last-token layer-24 hidden state + output distribution per prompt |
| Manifold fitting | (inside plot/steering scripts) | Per-concept centroids → PCA (r=6 weekdays, r=11 months) → cubic spline through centroids in concept order; behavior manifold via Hellinger map p→√p first |
| Steering | `run_steering_probs.py` | 30 steps from source to target; a forward hook swaps the layer-24 last-token hidden state with the path point; linear path lerps raw centroids, manifold path sweeps the spline and decodes via PCA⁻¹ |
| Torus extension | `run_gemma_torus.py`, `run_hierarchical_steering.py` | 168 "It is HH:00 on {Day}" prompts; day/hour 2D subspaces by OLS on [sin θ, cos θ] labels (QR-orthogonalized); product-coordinate steering moves day and hour independently |
| Figures | `plot_*.py`, `render_torus_3d_L24.py` | All read from `checkpoints/` and `figures/*/` caches — no model needed |

**Practical note on layer choice:** patching layer 12 fails — keys/values from
the original prompt override the patch over the 14 remaining blocks. Layer 24
(92% depth, 2 blocks remaining) replicates cleanly.

## Repo map

```
scripts/        # collection + plotting (one experiment per file)
checkpoints/    # committed per-prompt activations & steering probs (.npz)
figures/        # generated figures + torus activation caches (.pt)
slurm/          # sbatch wrappers + generic run_all.sh
report/         # LaTeX source + built report.pdf (2-page walkthrough)
```

## Reproducibility notes

- The pipeline is fully deterministic (forward passes, PCA, splines); `--seed`
  guards any future stochastic op. Reruns from the committed checkpoints
  reproduce the report figures exactly (up to font antialiasing).
- Model: `google/gemma-2-2b-it`, bfloat16 on GPU / float32 on CPU, 26 layers,
  d=2304. Figures in the report were produced at layer 24.

## Citation

```bibtex
@article{wurgaft2026manifold,
  title={Manifold Steering Reveals the Shared Geometry of Neural Network
         Representation and Behavior},
  author={Wurgaft, Daniel and Rager, Can and Kowal, Matthew and others},
  journal={arXiv preprint arXiv:2605.05115},
  year={2026}
}
```

MIT licensed. Questions → kacharya@stanford.edu
