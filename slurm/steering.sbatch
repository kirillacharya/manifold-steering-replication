#!/bin/bash
#SBATCH --job-name=manifold-steering
#SBATCH --partition=gpu          # adjust for your cluster (mc3: check `sinfo`)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x-%j.out

# Full collection + steering pipeline on a SLURM cluster.
# Prereqs: `uv sync` run once on the login node; HF_TOKEN set for gated Gemma.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

uv run python scripts/run_weekdays.py --layer 24 --seed "${SEED:-0}"
uv run python scripts/run_months.py   --layer 24 --seed "${SEED:-0}"
uv run python scripts/run_steering_probs.py --seed "${SEED:-0}"
uv run python scripts/run_gemma_torus.py --layer 24 --acts-only --seed "${SEED:-0}" \
    --acts-cache figures/gemma_torus/acts_2b_L24.pt
uv run python scripts/run_hierarchical_steering.py --seed "${SEED:-0}"

# Figures (CPU-only, fast)
uv run python scripts/plot_weekdays_separate.py
uv run python scripts/plot_months_separate.py
uv run python scripts/plot_probs_separate.py
uv run python scripts/render_torus_3d_L24.py
uv run python scripts/plot_subspace_hours.py
echo "All done."
