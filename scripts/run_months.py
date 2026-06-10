"""Manifold steering replication — months domain, Gemma 2 2B.

Replicates Figure 4 from:
  "Manifold Steering Reveals the Shared Geometry of Neural Network
   Representation and Behavior" (Wurgaft et al., 2026)

for the months domain using google/gemma-2-2b-it.

Pipeline:
  1. Build prompts  → group by correct answer concept (January–December)
  2. Collect hidden states at --layer + output probs for concept tokens
  3. Compute per-concept centroids (activation centroid, probability centroid)
  4. PCA to k=11 dims  (12 concepts → rank ≤ 11)
  5. Fit cubic spline through ordered PCA centroids
  6. Linear steering  : lerp between raw activation centroids
     Manifold steering: sweep t in spline space, decode back to hidden space
  7. For each path point, replace activation at --layer and record output probs
  8. Plot probability trajectories (Figure 4 style) → figures/

MODEL  : google/gemma-2-2b-it
LAYER  : set via --layer (default 24; hidden_states index, 26 layers total)
DOMAIN : months  (12 ordered concepts: January … December)
PAIR   : January → July  (indices 0 → 6, spans 5 intermediate months)
OUTPUT     : figures/months_L{LAYER}_January_July_steering.png
CHECKPOINT : checkpoints/months_L{LAYER}_acts_probs.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────── config ──────────────────────────────────

MODEL_NAME = "google/gemma-2-2b-it"
LAYER      = 24    # default; overridden by --layer
N_PCA      = 11    # PCA dims: 12 concepts → rank ≤ 11
N_STEPS    = 30    # interpolation steps along each steering path

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
NUMBERS       = ["one", "two", "three", "four", "five", "six"]
NUMBER_TO_INT = {w: i + 1 for i, w in enumerate(NUMBERS)}

TEMPLATE = "Q: What month is {number} months after {entity}?\nA:"

# Steering pair: January (0) → July (6), spans Feb, Mar, Apr, May, Jun
START_CONCEPT = "January"
END_CONCEPT   = "July"
START_IDX     = MONTHS.index(START_CONCEPT)
END_IDX       = MONTHS.index(END_CONCEPT)

# Base prompt for intervention: answer is the midpoint concept (April, idx 3)
BASE_PROMPT = "Q: What month is three months after January?\nA:"

# Colors: 12 months — cycling through a qualitative palette
MONTH_COLORS = [
    "#e6194b", "#f58231", "#ffe119", "#bfef45",
    "#3cb44b", "#42d4f4", "#4363d8", "#911eb4",
    "#f032e6", "#a9a9a9", "#9A6324", "#000075",
]

# ─────────────────────────────────── prompts ─────────────────────────────────

def build_prompts() -> dict[str, list[str]]:
    """Build all (entity, number) prompts grouped by correct answer concept.

    12 months × 6 number words = 72 prompts.
    Answer is cyclic: (entity_idx + number) % 12.
    Each answer concept gets exactly 6 prompts.
    """
    prompts_by_concept: dict[str, list[str]] = {m: [] for m in MONTHS}
    for entity in MONTHS:
        for number in NUMBERS:
            result_idx = (MONTHS.index(entity) + NUMBER_TO_INT[number]) % 12
            result = MONTHS[result_idx]
            prompt = TEMPLATE.format(number=number, entity=entity)
            prompts_by_concept[result].append(prompt)
    return prompts_by_concept

# ─────────────────────────────────── model ───────────────────────────────────

def load_model(device: torch.device) -> tuple:
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    print(f"  Loading {MODEL_NAME}  (dtype={dtype}, device={device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=str(device) if device.type != "mps" else None,
    )
    if device.type == "mps":
        model = model.to(device)
    model.eval()
    return tokenizer, model


def get_concept_token_ids(tokenizer) -> dict[str, int]:
    """Return token ID for each month (with leading space, last subword)."""
    ids = {}
    for month in MONTHS:
        toks = tokenizer.encode(" " + month, add_special_tokens=False)
        ids[month] = toks[-1]
    return ids

# ─────────────────────────────────── collection ──────────────────────────────

def collect(
    model,
    tokenizer,
    prompts_by_concept: dict[str, list[str]],
    concept_ids: dict[str, int],
    device: torch.device,
) -> tuple[dict[str, list], dict[str, list]]:
    acts_by_concept:  dict[str, list] = {m: [] for m in MONTHS}
    probs_by_concept: dict[str, list] = {m: [] for m in MONTHS}

    id_list = [concept_ids[m] for m in MONTHS]

    with torch.no_grad():
        for concept in MONTHS:
            for prompt in tqdm(
                prompts_by_concept[concept],
                desc=f"  Collecting {concept}",
                leave=False,
            ):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                out = model(**inputs, output_hidden_states=True)

                hs  = out.hidden_states[LAYER]
                act = hs[0, -1, :].float().cpu().numpy()

                logits    = out.logits[0, -1, :].float().cpu()
                probs_all = torch.softmax(logits, dim=-1).numpy()

                concept_probs = np.array([probs_all[tid] for tid in id_list])
                other_prob    = max(0.0, 1.0 - concept_probs.sum())
                prob_vec      = np.append(concept_probs, other_prob)

                acts_by_concept[concept].append(act)
                probs_by_concept[concept].append(prob_vec)

    return acts_by_concept, probs_by_concept

# ─────────────────────────────────── checkpointing ──────────────────────────

def checkpoint_path() -> Path:
    return Path(f"checkpoints/months_L{LAYER}_acts_probs.npz")


def save_checkpoint(
    acts_by_concept:  dict[str, list],
    probs_by_concept: dict[str, list],
) -> None:
    cp = checkpoint_path()
    cp.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for month in MONTHS:
        arrays[f"acts_{month}"]  = np.array(acts_by_concept[month])
        arrays[f"probs_{month}"] = np.array(probs_by_concept[month])
    np.savez(cp, **arrays)
    print(f"  Checkpoint saved → {cp}")


def load_checkpoint() -> tuple[dict[str, list], dict[str, list]]:
    cp = checkpoint_path()
    data = np.load(cp)
    acts_by_concept:  dict[str, list] = {}
    probs_by_concept: dict[str, list] = {}
    for month in MONTHS:
        acts_by_concept[month]  = list(data[f"acts_{month}"])
        probs_by_concept[month] = list(data[f"probs_{month}"])
    print(f"  Checkpoint loaded ← {cp}")
    return acts_by_concept, probs_by_concept

# ─────────────────────────────────── centroids ───────────────────────────────

def compute_centroids(
    acts_by_concept:  dict[str, list],
    probs_by_concept: dict[str, list],
) -> tuple[np.ndarray, np.ndarray]:
    act_centroids  = np.array([np.mean(acts_by_concept[m],  axis=0) for m in MONTHS])
    prob_centroids = np.array([np.mean(probs_by_concept[m], axis=0) for m in MONTHS])
    return act_centroids, prob_centroids

# ─────────────────────────────────── manifold fitting ────────────────────────

def fit_activation_manifold(
    act_centroids: np.ndarray,
) -> tuple[PCA, np.ndarray, CubicSpline]:
    pca = PCA(n_components=N_PCA)
    pca_centroids = pca.fit_transform(act_centroids)       # (12, N_PCA)

    t_knots = np.arange(len(MONTHS), dtype=float)          # [0, 1, ..., 11]
    spline  = CubicSpline(t_knots, pca_centroids)

    evr = pca.explained_variance_ratio_
    print(f"  PCA EVR: {[f'{v:.1%}' for v in evr]}")
    print(f"  Total explained: {evr.sum():.1%}")

    return pca, pca_centroids, spline

# ─────────────────────────────────── steering paths ──────────────────────────

def linear_path(act_centroids: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, N_STEPS)
    h0, h1 = act_centroids[start_idx], act_centroids[end_idx]
    return np.array([(1 - t) * h0 + t * h1 for t in ts])


def manifold_path(pca: PCA, spline: CubicSpline, start_idx: int, end_idx: int) -> np.ndarray:
    t_intrinsic = np.linspace(float(start_idx), float(end_idx), N_STEPS)
    pca_points  = spline(t_intrinsic)
    act_points  = pca.inverse_transform(pca_points)
    return act_points

# ─────────────────────────────────── intervention ────────────────────────────

def steer_and_collect(
    model,
    tokenizer,
    base_prompt: str,
    path_acts: np.ndarray,
    concept_ids: dict[str, int],
    device: torch.device,
) -> np.ndarray:
    id_list   = [concept_ids[m] for m in MONTHS]
    inputs    = tokenizer(base_prompt, return_tensors="pt").to(device)
    all_probs = []
    dtype     = next(model.parameters()).dtype

    with torch.no_grad():
        for act_vec in tqdm(path_acts, desc="    Steering", leave=False):
            replacement = torch.tensor(act_vec, dtype=dtype, device=device)

            def _hook(module, inp, output, _rep=replacement):
                h = output[0].clone() if isinstance(output, tuple) else output.clone()
                h[:, -1, :] = _rep
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h

            handle = model.model.layers[LAYER - 1].register_forward_hook(_hook)
            out    = model(**inputs)
            handle.remove()

            logits        = out.logits[0, -1, :].float().cpu()
            probs_all     = torch.softmax(logits, dim=-1).numpy()
            concept_probs = np.array([probs_all[tid] for tid in id_list])
            other_prob    = max(0.0, 1.0 - concept_probs.sum())
            all_probs.append(np.append(concept_probs, other_prob))

    return np.array(all_probs)  # (N_STEPS, 13)

# ─────────────────────────────────── plotting ────────────────────────────────

def plot_trajectories(
    probs_linear:   np.ndarray,
    probs_manifold: np.ndarray,
    start: str,
    end:   str,
    out_path: str,
) -> None:
    ts = np.linspace(0.0, 1.0, N_STEPS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, mode, probs in [
        (axes[0], "Linear Steering",   probs_linear),
        (axes[1], "Manifold Steering", probs_manifold),
    ]:
        for i, month in enumerate(MONTHS):
            ax.plot(ts, probs[:, i], color=MONTH_COLORS[i], label=month[:3], lw=2)
        ax.plot(ts, probs[:, -1], color="gray", label="other", lw=1,
                ls="--", alpha=0.6)
        ax.axvline(0.0, color="black", lw=0.8, ls=":")
        ax.axvline(1.0, color="black", lw=0.8, ls=":")
        ax.set_title(mode, fontsize=13)
        ax.set_xlabel("Steering progress  t", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="upper center", ncol=7,
                  bbox_to_anchor=(0.5, -0.18))

    axes[0].set_ylabel("Output probability", fontsize=11)
    fig.suptitle(
        f"Months: {start} → {end}  |  Gemma 2 2B  |  Layer {LAYER}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")


def print_prob_centroids(prob_centroids: np.ndarray) -> None:
    print("\nProbability centroids (months):")
    header = f"{'Centroid':<12}" + "".join(f"{m[:3]:>6}" for m in MONTHS) + f"{'other':>7}"
    print(header)
    for i, month in enumerate(MONTHS):
        row = f"{month:<12}"
        for j in range(len(MONTHS)):
            val = prob_centroids[i, j]
            marker = "*" if j == i else " "
            row += f"{val:>5.2f}{marker}"
        row += f"{prob_centroids[i, -1]:>7.3f}"
        print(row)

# ─────────────────────────────────── main ────────────────────────────────────

def _set_seed(seed: int) -> None:
    """Fix all RNG seeds. The pipeline is deterministic (no sampling, no
    training); this guards exact reproducibility of any future stochastic op."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main() -> None:
    global LAYER

    parser = argparse.ArgumentParser(description="Months manifold steering")
    parser.add_argument("--layer",     type=int, default=24,
                        help="Hidden state layer index (default: 24)")
    parser.add_argument("--recollect", action="store_true",
                        help="Force re-collection even if checkpoint exists")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for exact reproducibility.")
    args = parser.parse_args()
    _set_seed(args.seed)
    LAYER = args.layer

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available()          else
        torch.device("cpu")
    )
    print(f"Device: {device}  |  Layer: {LAYER}")

    cp = checkpoint_path()
    if cp.exists() and not args.recollect:
        acts_by_concept, probs_by_concept = load_checkpoint()
    else:
        tokenizer, model = load_model(device)
        concept_ids      = get_concept_token_ids(tokenizer)

        print("\nToken IDs:")
        for m, tid in concept_ids.items():
            print(f"  {m:<12}: {tid}  ({tokenizer.decode([tid])})")

        print("\nBuilding prompts ...")
        prompts_by_concept = build_prompts()
        total = sum(len(v) for v in prompts_by_concept.values())
        print(f"  {total} prompts total, {total // len(MONTHS)} per concept")

        print("\nCollecting activations and probabilities ...")
        acts_by_concept, probs_by_concept = collect(
            model, tokenizer, prompts_by_concept, concept_ids, device)
        save_checkpoint(acts_by_concept, probs_by_concept)

    # ── Centroids ──
    act_centroids, prob_centroids = compute_centroids(acts_by_concept, probs_by_concept)
    print_prob_centroids(prob_centroids)

    # ── Manifold fitting ──
    print("\nFitting activation manifold ...")
    pca, pca_centroids, spline = fit_activation_manifold(act_centroids)

    # ── Steering paths ──
    print(f"\nSteering: {START_CONCEPT} → {END_CONCEPT}  (indices {START_IDX} → {END_IDX})")
    path_lin = linear_path(act_centroids, START_IDX, END_IDX)
    path_mfd = manifold_path(pca, spline, START_IDX, END_IDX)

    # ── Load model for intervention (may already be loaded) ──
    if cp.exists() and not args.recollect:
        tokenizer, model = load_model(device)
        concept_ids = get_concept_token_ids(tokenizer)

    print(f"\nBase prompt: {BASE_PROMPT!r}")
    print("Running linear steering ...")
    probs_linear   = steer_and_collect(model, tokenizer, BASE_PROMPT,
                                       path_lin, concept_ids, device)
    print("Running manifold steering ...")
    probs_manifold = steer_and_collect(model, tokenizer, BASE_PROMPT,
                                       path_mfd, concept_ids, device)

    # ── Plot ──
    suffix   = f"L{LAYER}_{START_CONCEPT}_{END_CONCEPT}"
    out_path = f"figures/months_{suffix}_steering.png"
    print(f"\nPlotting → {out_path}")
    plot_trajectories(probs_linear, probs_manifold, START_CONCEPT, END_CONCEPT, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
