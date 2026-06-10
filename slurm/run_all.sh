#!/bin/bash
# Generic (non-SLURM) full pipeline for any CUDA/MPS box. Same steps as
# steering.sbatch; device is auto-detected by each script.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-0}"
uv run python scripts/run_weekdays.py --layer 24 --seed "$SEED"
uv run python scripts/run_months.py   --layer 24 --seed "$SEED"
uv run python scripts/run_steering_probs.py --seed "$SEED"
uv run python scripts/run_gemma_torus.py --layer 24 --acts-only --seed "$SEED" \
    --acts-cache figures/gemma_torus/acts_2b_L24.pt
uv run python scripts/run_hierarchical_steering.py --seed "$SEED"

uv run python scripts/plot_weekdays_separate.py
uv run python scripts/plot_months_separate.py
uv run python scripts/plot_probs_separate.py
uv run python scripts/render_torus_3d_L24.py
uv run python scripts/plot_subspace_hours.py
echo "All done."
