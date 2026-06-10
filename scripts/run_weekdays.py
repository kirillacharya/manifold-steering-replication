"""Manifold steering replication — weekdays domain, Gemma 2 2B.

Replicates Figure 4 from:
  "Manifold Steering Reveals the Shared Geometry of Neural Network
   Representation and Behavior" (Goodfire, 2024)

for the weekdays domain using google/gemma-2-2b-it.

Pipeline:
  1. Build prompts  → group by correct answer concept (Monday–Sunday)
  2. Collect hidden states at --layer + output probs for concept tokens
  3. Compute per-concept centroids (activation centroid, probability centroid)
  4. PCA (scikit-learn) to k=6 dims  (7 concepts → rank ≤ 6)
  5. Fit cubic spline through ordered PCA centroids
  6. Linear steering  : lerp between raw activation centroids
     Manifold steering: sweep t in spline space, decode back to hidden space
  7. For each path point, replace activation at --layer and record output probs
  8. Plot probability trajectories (Figure 4 style) → figures/

MODEL  : google/gemma-2-2b-it
LAYER  : set via --layer (default 12; hidden_states index, 26 layers total)
DOMAIN : weekdays  (7 ordered concepts: Monday … Sunday)
PAIR   : Tuesday → Friday  (indices 1 → 4, spans 3 intermediate concepts)
OUTPUT     : figures/weekdays_L{LAYER}_Tuesday_Friday_steering.png
CHECKPOINT : checkpoints/weekdays_L{LAYER}_acts_probs.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────── config ──────────────────────────────────

MODEL_NAME = "google/gemma-2-2b-it"
LAYER      = 12     # default; overridden by --layer in main()
N_PCA      = 6      # PCA dims: 7 concepts → rank ≤ 6
N_STEPS    = 30     # interpolation steps along each steering path

DAYS         = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUMBERS      = ["one", "two", "three", "four", "five", "six"]
NUMBER_TO_INT = {w: i + 1 for i, w in enumerate(NUMBERS)}

# Prompt template from the original paper (causalab/tasks/natural_domains_arithmetic/config.py)
TEMPLATE = "Q: What day is {number} days after {entity}?\nA:"

# ─────────────────────────────────── prompts ─────────────────────────────────

def build_prompts() -> dict[str, list[str]]:
    """Build all (entity, number) prompts and group by correct answer concept.

    For each of the 7 entities × 6 number words → 42 prompts total.
    Answer is cyclic: (entity_idx + number) % 7.
    Each answer concept gets ~6 prompts.
    """
    prompts_by_concept: dict[str, list[str]] = {d: [] for d in DAYS}
    for entity in DAYS:
        for number in NUMBERS:
            result_idx = (DAYS.index(entity) + NUMBER_TO_INT[number]) % 7
            result = DAYS[result_idx]
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
    """Return token ID for each weekday concept (with leading space).

    Weekday names are single tokens in Gemma's tokenizer when preceded by a
    space.  We take the last token of the encoding to handle rare edge cases.
    """
    ids = {}
    for day in DAYS:
        toks = tokenizer.encode(" " + day, add_special_tokens=False)
        ids[day] = toks[-1]
    return ids

# ─────────────────────────────────── collection ──────────────────────────────

def collect(
    model,
    tokenizer,
    prompts_by_concept: dict[str, list[str]],
    concept_ids: dict[str, int],
    device: torch.device,
) -> tuple[dict[str, list], dict[str, list]]:
    """Run each prompt through the model; record layer-12 hidden state and
    output probability distribution over concepts + "other".

    Returns
    -------
    acts_by_concept  : dict[day -> list of (d_model,) float32 numpy arrays]
    probs_by_concept : dict[day -> list of (n_concepts+1,) float32 numpy arrays]
                       last entry is the "other" class probability
    """
    acts_by_concept:  dict[str, list] = {d: [] for d in DAYS}
    probs_by_concept: dict[str, list] = {d: [] for d in DAYS}

    id_list = [concept_ids[d] for d in DAYS]

    with torch.no_grad():
        for concept in DAYS:
            for prompt in tqdm(
                prompts_by_concept[concept],
                desc=f"  Collecting {concept}",
                leave=False,
            ):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                out = model(**inputs, output_hidden_states=True)

                # Hidden state at layer 12, last token position
                hs = out.hidden_states[LAYER]          # (1, seq_len, d_model)
                act = hs[0, -1, :].float().cpu().numpy()  # (d_model,)

                # Output probability distribution
                logits = out.logits[0, -1, :].float().cpu()  # (vocab_size,)
                probs_all = torch.softmax(logits, dim=-1).numpy()

                concept_probs = np.array([probs_all[tid] for tid in id_list])
                other_prob    = max(0.0, 1.0 - concept_probs.sum())
                prob_vec      = np.append(concept_probs, other_prob)

                acts_by_concept[concept].append(act)
                probs_by_concept[concept].append(prob_vec)

    return acts_by_concept, probs_by_concept

# ─────────────────────────────────── checkpointing ──────────────────────────

def checkpoint_path() -> Path:
    """Layer-specific checkpoint so different --layer runs don't collide."""
    return Path(f"checkpoints/weekdays_L{LAYER}_acts_probs.npz")


def save_checkpoint(
    acts_by_concept:  dict[str, list],
    probs_by_concept: dict[str, list],
) -> None:
    """Save per-prompt activations and probabilities to disk."""
    cp = checkpoint_path()
    cp.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for day in DAYS:
        arrays[f"acts_{day}"]  = np.array(acts_by_concept[day])
        arrays[f"probs_{day}"] = np.array(probs_by_concept[day])
    np.savez(cp, **arrays)
    print(f"  Checkpoint saved → {cp}")


def load_checkpoint() -> tuple[dict[str, list], dict[str, list]]:
    """Load per-prompt activations and probabilities from disk."""
    cp = checkpoint_path()
    data = np.load(cp)
    acts_by_concept:  dict[str, list] = {}
    probs_by_concept: dict[str, list] = {}
    for day in DAYS:
        acts_by_concept[day]  = list(data[f"acts_{day}"])
        probs_by_concept[day] = list(data[f"probs_{day}"])
    print(f"  Checkpoint loaded ← {cp}")
    return acts_by_concept, probs_by_concept


# ─────────────────────────────────── centroids ───────────────────────────────

def compute_centroids(
    acts_by_concept:  dict[str, list],
    probs_by_concept: dict[str, list],
) -> tuple[np.ndarray, np.ndarray]:
    """Average activations and probability vectors per concept.

    Returns
    -------
    act_centroids  : (n_concepts, d_model)
    prob_centroids : (n_concepts, n_concepts+1)
    """
    act_centroids  = np.array([np.mean(acts_by_concept[d],  axis=0) for d in DAYS])
    prob_centroids = np.array([np.mean(probs_by_concept[d], axis=0) for d in DAYS])
    return act_centroids, prob_centroids

# ─────────────────────────────────── manifold fitting ────────────────────────

def fit_activation_manifold(
    act_centroids: np.ndarray,
) -> tuple[PCA, np.ndarray, CubicSpline]:
    """Reduce centroids with PCA, then fit a cubic spline through them.

    The spline maps a scalar intrinsic coordinate t ∈ {0,…,6} (concept index)
    to a point in PCA space.  Manifold steering interpolates t between two
    endpoint concepts and decodes back to full hidden-state space.

    Returns
    -------
    pca           : fitted sklearn PCA object
    pca_centroids : (n_concepts, N_PCA) — centroids in PCA space
    spline        : scipy CubicSpline  t → pca_point
    """
    pca = PCA(n_components=N_PCA)
    pca_centroids = pca.fit_transform(act_centroids)       # (7, N_PCA)

    t_knots = np.arange(len(DAYS), dtype=float)            # [0, 1, 2, 3, 4, 5, 6]
    spline  = CubicSpline(t_knots, pca_centroids)          # t → (N_PCA,)

    return pca, pca_centroids, spline

# ─────────────────────────────────── steering paths ──────────────────────────

def linear_path(
    act_centroids: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    """Direct linear interpolation between raw activation centroids.

    π_linear(t) = (1−t)·h₀ + t·h₁   for t ∈ [0,1]

    Returns : (N_STEPS, d_model)
    """
    ts = np.linspace(0.0, 1.0, N_STEPS)
    h0, h1 = act_centroids[start_idx], act_centroids[end_idx]
    return np.array([(1 - t) * h0 + t * h1 for t in ts])


def manifold_path(
    pca: PCA,
    spline: CubicSpline,
    start_idx: int,
    end_idx: int,
) -> np.ndarray:
    """Interpolate in intrinsic spline coordinates, decode to hidden space.

    π_manifold(t) = pca⁻¹(spline((1−t)·u₀ + t·u₁))   for t ∈ [0,1]
    where u₀, u₁ are the intrinsic (concept-index) coordinates of the endpoints.

    Returns : (N_STEPS, d_model)
    """
    t_intrinsic = np.linspace(float(start_idx), float(end_idx), N_STEPS)
    pca_points  = spline(t_intrinsic)          # (N_STEPS, N_PCA)
    act_points  = pca.inverse_transform(pca_points)  # (N_STEPS, d_model)
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
    """For each activation vector in path_acts, replace layer-12 at the last
    token position and record output probabilities over concepts + "other".

    The hook is attached to model.model.layers[LAYER-1] (decoder block 11,
    whose output corresponds to hidden_states[LAYER]).

    Returns : (N_STEPS, n_concepts+1)
    """
    id_list   = [concept_ids[d] for d in DAYS]
    inputs    = tokenizer(base_prompt, return_tensors="pt").to(device)
    all_probs = []
    dtype     = next(model.parameters()).dtype

    with torch.no_grad():
        for act_vec in tqdm(path_acts, desc="    Steering", leave=False):
            replacement = torch.tensor(
                act_vec, dtype=dtype, device=device
            )  # (d_model,)

            def _hook(module, inp, output, _rep=replacement):
                h = output[0].clone() if isinstance(output, tuple) else output.clone()
                h[:, -1, :] = _rep          # replace last token position
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h

            handle = model.model.layers[LAYER - 1].register_forward_hook(_hook)
            out    = model(**inputs)
            handle.remove()

            logits    = out.logits[0, -1, :].float().cpu()
            probs_all = torch.softmax(logits, dim=-1).numpy()

            concept_probs = np.array([probs_all[tid] for tid in id_list])
            other_prob    = max(0.0, 1.0 - concept_probs.sum())
            all_probs.append(np.append(concept_probs, other_prob))

    return np.array(all_probs)  # (N_STEPS, n_concepts+1)

# ─────────────────────────────────── plotting ────────────────────────────────

# One color per weekday; "other" is always gray
DAY_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
    "#f58231", "#911eb4", "#42d4f4",
]


def plot_trajectories(
    probs_linear:   np.ndarray,
    probs_manifold: np.ndarray,
    start: str,
    end:   str,
    out_path: str,
) -> None:
    """Recreate Figure 4-style side-by-side probability trajectory plot.

    Left panel  : linear steering
    Right panel : manifold steering
    x-axis : steering progress t ∈ [0, 1]
    y-axis : output probability
    """
    ts  = np.linspace(0.0, 1.0, N_STEPS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, mode, probs in [
        (axes[0], "Linear Steering",   probs_linear),
        (axes[1], "Manifold Steering", probs_manifold),
    ]:
        for i, day in enumerate(DAYS):
            ax.plot(ts, probs[:, i], color=DAY_COLORS[i], label=day, lw=2)
        ax.plot(ts, probs[:, -1], color="gray", label="other", lw=1,
                ls="--", alpha=0.6)
        ax.axvline(0.0, color="black", lw=0.8, ls=":")
        ax.axvline(1.0, color="black", lw=0.8, ls=":")
        ax.set_title(mode, fontsize=13)
        ax.set_xlabel("Steering progress  t", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="upper center", ncol=4,
                  bbox_to_anchor=(0.5, -0.18))

    axes[0].set_ylabel("Output probability", fontsize=11)
    fig.suptitle(
        f"Weekdays: {start} → {end}  |  Gemma 2 2B  |  Layer {LAYER}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")

def plot_3d_manifolds(
    act_centroids:  np.ndarray,   # (7, 2304)
    prob_centroids: np.ndarray,   # (7, 8)
    pca_full:       PCA,          # N_PCA-component PCA used for the spline
    spline:         CubicSpline,  # t -> (N_PCA,) spline in PCA space
    path_lin:       np.ndarray,   # (N_STEPS, 2304)
    path_mfd:       np.ndarray,   # (N_STEPS, 2304)
    probs_linear:   np.ndarray,   # (N_STEPS, 8)
    probs_manifold: np.ndarray,   # (N_STEPS, 8)
    start_idx:      int,
    end_idx:        int,
    out_path:       str,
) -> None:
    """Two-panel 3D figure matching Figure 4's manifold visualisations.

    Left  — Activation Space: concept centroids + spline manifold + steering paths
             projected into the top-3 PCA components of the activation centroids.
    Right — Behavior Space: probability centroids in Hellinger space (√p)
             + behavior manifold spline + steering output trajectories,
             projected into top-3 PCA components of the Hellinger centroids.

    Each panel has:
      • gray floor plane with shadow projections of all curves
      • colored diamond markers per weekday concept
      • steel-blue manifold curve through all 7 ordered concepts
      • dashed gray line = linear steering path / trajectory
      • solid black line = manifold steering path / trajectory
    """
    # ── Activation space ──
    pca3_act = PCA(n_components=3)
    act_3d   = pca3_act.fit_transform(act_centroids)      # (7, 3)

    # Full manifold curve: densely sample spline → decode to R^2304 → project
    t_dense    = np.linspace(0.0, float(len(DAYS) - 1), 300)
    pca6_curve = spline(t_dense)                          # (300, N_PCA)
    act_curve  = pca_full.inverse_transform(pca6_curve)   # (300, 2304)
    act_curve3 = pca3_act.transform(act_curve)            # (300, 3)

    lin_act3 = pca3_act.transform(path_lin)               # (N_STEPS, 3)
    mfd_act3 = pca3_act.transform(path_mfd)               # (N_STEPS, 3)

    # ── Behavior space (Hellinger: h = √p) ──
    hell_centroids = np.sqrt(np.clip(prob_centroids, 0.0, None))  # (7, 8)
    pca3_beh = PCA(n_components=3)
    beh_3d   = pca3_beh.fit_transform(hell_centroids)     # (7, 3)

    # Behavior manifold: spline through ordered Hellinger centroids in 3D beh space
    t_knots   = np.arange(len(DAYS), dtype=float)
    beh_spline = CubicSpline(t_knots, beh_3d)
    beh_curve3 = beh_spline(t_dense)                      # (300, 3)

    # Project steering outputs
    hell_lin = np.sqrt(np.clip(probs_linear,   0.0, None))  # (N_STEPS, 8)
    hell_mfd = np.sqrt(np.clip(probs_manifold, 0.0, None))  # (N_STEPS, 8)
    lin_beh3 = pca3_beh.transform(hell_lin)               # (N_STEPS, 3)
    mfd_beh3 = pca3_beh.transform(hell_mfd)               # (N_STEPS, 3)

    # ── Figure layout ──
    fig = plt.figure(figsize=(15, 6.5))

    panels = [
        ("Activation Space", act_3d,  act_curve3,  lin_act3, mfd_act3),
        ("Behavior Space",   beh_3d,  beh_curve3,  lin_beh3, mfd_beh3),
    ]

    for col, (title, pts3, curve3, lin3, mfd3) in enumerate(panels):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")

        # ── floor plane ──
        z_vals   = np.concatenate([pts3[:, 2], curve3[:, 2], lin3[:, 2], mfd3[:, 2]])
        z_floor  = z_vals.min() - 0.12 * (z_vals.max() - z_vals.min())

        x_all = np.concatenate([pts3[:, 0], curve3[:, 0]])
        y_all = np.concatenate([pts3[:, 1], curve3[:, 1]])
        pad_x = 0.1 * (x_all.max() - x_all.min())
        pad_y = 0.1 * (y_all.max() - y_all.min())
        xx, yy = np.meshgrid(
            [x_all.min() - pad_x, x_all.max() + pad_x],
            [y_all.min() - pad_y, y_all.max() + pad_y],
        )
        ax.plot_surface(xx, yy, np.full_like(xx, z_floor),
                        alpha=0.12, color="gray", zorder=0, linewidth=0)

        # ── manifold curve (full 7-concept arc) ──
        ax.plot(curve3[:, 0], curve3[:, 1], curve3[:, 2],
                color="steelblue", lw=1.8, alpha=0.65, label="Manifold curve", zorder=3)
        ax.plot(curve3[:, 0], curve3[:, 1], np.full(len(curve3), z_floor),
                color="steelblue", lw=0.7, alpha=0.20, zorder=1)

        # ── linear steering path / trajectory ──
        ax.plot(lin3[:, 0], lin3[:, 1], lin3[:, 2],
                color="#888888", lw=2.0, ls="--", label="Linear steering", zorder=4)
        ax.plot(lin3[:, 0], lin3[:, 1], np.full(len(lin3), z_floor),
                color="#888888", lw=0.7, alpha=0.20, ls="--", zorder=1)

        # ── manifold steering path / trajectory ──
        ax.plot(mfd3[:, 0], mfd3[:, 1], mfd3[:, 2],
                color="black", lw=2.2, label="Manifold steering", zorder=5)
        ax.plot(mfd3[:, 0], mfd3[:, 1], np.full(len(mfd3), z_floor),
                color="black", lw=0.7, alpha=0.20, zorder=1)

        # ── concept markers + labels ──
        for i, day in enumerate(DAYS):
            ax.scatter(*pts3[i], color=DAY_COLORS[i], s=90, marker="D",
                       edgecolors="k", lw=0.5, zorder=6)
            ax.scatter(pts3[i, 0], pts3[i, 1], z_floor,
                       color=DAY_COLORS[i], s=35, marker="D", alpha=0.30, zorder=1)
            ax.text(pts3[i, 0], pts3[i, 1], pts3[i, 2],
                    f" {day[:3]}", fontsize=7, zorder=7, va="bottom")

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("PC 1", fontsize=8, labelpad=2)
        ax.set_ylabel("PC 2", fontsize=8, labelpad=2)
        ax.set_zlabel("PC 3", fontsize=8, labelpad=2)
        ax.tick_params(labelsize=6, pad=1)
        ax.view_init(elev=22, azim=-55)

    # Shared legend from first axis
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        f"3D Manifold — Weekdays {DAYS[start_idx]} → {DAYS[end_idx]}"
        f"  |  Gemma 2 2B  |  Layer {LAYER}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.07, 1, 0.97])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_path}")


def plot_3d_manifolds_plotly(
    act_centroids:  np.ndarray,
    prob_centroids: np.ndarray,
    pca_full:       PCA,
    spline:         CubicSpline,
    path_lin:       np.ndarray,
    path_mfd:       np.ndarray,
    probs_linear:   np.ndarray,
    probs_manifold: np.ndarray,
    start_idx:      int,
    end_idx:        int,
    out_path:       str,
) -> None:
    """Interactive Plotly version of the 3D manifold figure.

    Produces a self-contained HTML file with two side-by-side 3D subplots
    (Activation Space | Behavior Space), fully interactive (rotate, zoom,
    hover to see concept names and coordinates).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ── Activation space projections ──
    pca3_act = PCA(n_components=3)
    act_3d   = pca3_act.fit_transform(act_centroids)

    t_dense    = np.linspace(0.0, float(len(DAYS) - 1), 300)
    pca6_curve = spline(t_dense)
    act_curve  = pca_full.inverse_transform(pca6_curve)
    act_curve3 = pca3_act.transform(act_curve)

    lin_act3 = pca3_act.transform(path_lin)
    mfd_act3 = pca3_act.transform(path_mfd)

    # ── Behavior space projections (Hellinger: h = √p) ──
    hell_centroids = np.sqrt(np.clip(prob_centroids, 0.0, None))
    pca3_beh = PCA(n_components=3)
    beh_3d   = pca3_beh.fit_transform(hell_centroids)

    t_knots    = np.arange(len(DAYS), dtype=float)
    beh_spline = CubicSpline(t_knots, beh_3d)
    beh_curve3 = beh_spline(t_dense)

    hell_lin = np.sqrt(np.clip(probs_linear,   0.0, None))
    hell_mfd = np.sqrt(np.clip(probs_manifold, 0.0, None))
    lin_beh3 = pca3_beh.transform(hell_lin)
    mfd_beh3 = pca3_beh.transform(hell_mfd)

    # ── Build figure ──
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Activation Space", "Behavior Space"],
        horizontal_spacing=0.02,
    )

    panels = [
        ("scene",  act_3d, act_curve3, lin_act3, mfd_act3),
        ("scene2", beh_3d, beh_curve3, lin_beh3, mfd_beh3),
    ]

    for col_idx, (scene, pts3, curve3, lin3, mfd3) in enumerate(panels, start=1):
        show_legend = (col_idx == 1)   # only first panel contributes to legend

        z_floor = (
            min(pts3[:, 2].min(), curve3[:, 2].min(),
                lin3[:, 2].min(), mfd3[:, 2].min())
            - 0.12 * (pts3[:, 2].max() - pts3[:, 2].min())
        )

        # Floor plane via Surface (2×2 grid at z_floor)
        x_all = np.concatenate([pts3[:, 0], curve3[:, 0]])
        y_all = np.concatenate([pts3[:, 1], curve3[:, 1]])
        pad   = 0.12
        x_pad = pad * (x_all.max() - x_all.min())
        y_pad = pad * (y_all.max() - y_all.min())
        xs = np.array([[x_all.min() - x_pad, x_all.max() + x_pad]] * 2)
        ys = np.array([[y_all.min() - y_pad] * 2, [y_all.max() + y_pad] * 2])
        zs = np.full_like(xs, z_floor)
        fig.add_trace(
            go.Surface(
                x=xs, y=ys, z=zs,
                colorscale=[[0, "lightgray"], [1, "lightgray"]],
                opacity=0.25, showscale=False,
                hoverinfo="skip",
                name="Floor",
                showlegend=False,
            ),
            row=1, col=col_idx,
        )

        # Manifold curve
        fig.add_trace(
            go.Scatter3d(
                x=curve3[:, 0], y=curve3[:, 1], z=curve3[:, 2],
                mode="lines",
                line=dict(color="steelblue", width=4),
                name="Manifold curve",
                showlegend=show_legend,
                legendgroup="manifold_curve",
                hovertemplate="Manifold curve<extra></extra>",
            ),
            row=1, col=col_idx,
        )
        # Shadow on floor
        fig.add_trace(
            go.Scatter3d(
                x=curve3[:, 0], y=curve3[:, 1], z=np.full(len(curve3), z_floor),
                mode="lines",
                line=dict(color="steelblue", width=1.5),
                opacity=0.22, showlegend=False, hoverinfo="skip",
                legendgroup="manifold_curve_shadow",
            ),
            row=1, col=col_idx,
        )

        # Linear steering
        fig.add_trace(
            go.Scatter3d(
                x=lin3[:, 0], y=lin3[:, 1], z=lin3[:, 2],
                mode="lines",
                line=dict(color="#888888", width=4, dash="dash"),
                name="Linear steering",
                showlegend=show_legend,
                legendgroup="linear",
                hovertemplate="Linear  t=%{customdata:.2f}<extra></extra>",
                customdata=np.linspace(0, 1, len(lin3)),
            ),
            row=1, col=col_idx,
        )
        fig.add_trace(
            go.Scatter3d(
                x=lin3[:, 0], y=lin3[:, 1], z=np.full(len(lin3), z_floor),
                mode="lines",
                line=dict(color="#888888", width=1.5, dash="dash"),
                opacity=0.22, showlegend=False, hoverinfo="skip",
                legendgroup="linear_shadow",
            ),
            row=1, col=col_idx,
        )

        # Manifold steering
        fig.add_trace(
            go.Scatter3d(
                x=mfd3[:, 0], y=mfd3[:, 1], z=mfd3[:, 2],
                mode="lines",
                line=dict(color="black", width=4),
                name="Manifold steering",
                showlegend=show_legend,
                legendgroup="manifold_steer",
                hovertemplate="Manifold  t=%{customdata:.2f}<extra></extra>",
                customdata=np.linspace(0, 1, len(mfd3)),
            ),
            row=1, col=col_idx,
        )
        fig.add_trace(
            go.Scatter3d(
                x=mfd3[:, 0], y=mfd3[:, 1], z=np.full(len(mfd3), z_floor),
                mode="lines",
                line=dict(color="black", width=1.5),
                opacity=0.22, showlegend=False, hoverinfo="skip",
                legendgroup="manifold_steer_shadow",
            ),
            row=1, col=col_idx,
        )

        # Concept markers
        for i, day in enumerate(DAYS):
            fig.add_trace(
                go.Scatter3d(
                    x=[pts3[i, 0]], y=[pts3[i, 1]], z=[pts3[i, 2]],
                    mode="markers+text",
                    marker=dict(
                        symbol="diamond",
                        size=8,
                        color=DAY_COLORS[i],
                        line=dict(color="black", width=1),
                    ),
                    text=[day],
                    textposition="top center",
                    textfont=dict(size=10),
                    name=day,
                    showlegend=False,
                    legendgroup=f"day_{i}",
                    hovertemplate=f"{day}<extra></extra>",
                ),
                row=1, col=col_idx,
            )
            # Shadow
            fig.add_trace(
                go.Scatter3d(
                    x=[pts3[i, 0]], y=[pts3[i, 1]], z=[z_floor],
                    mode="markers",
                    marker=dict(symbol="diamond", size=5,
                                color=DAY_COLORS[i], opacity=0.3),
                    showlegend=False, hoverinfo="skip",
                    legendgroup=f"day_{i}_shadow",
                ),
                row=1, col=col_idx,
            )

        # Scene axes labels
        camera = dict(eye=dict(x=1.5, y=-1.5, z=1.0))
        axis_style = dict(
            showbackground=True,
            backgroundcolor="white",
            gridcolor="lightgray",
            showline=False,
            tickfont=dict(size=9),
        )
        fig.update_scenes(
            xaxis=dict(title="PC 1", **axis_style),
            yaxis=dict(title="PC 2", **axis_style),
            zaxis=dict(title="PC 3", **axis_style),
            camera=camera,
            selector=dict(type="scene"),
        )

    fig.update_layout(
        title=dict(
            text=(
                f"3D Manifold — Weekdays {DAYS[start_idx]} → {DAYS[end_idx]}"
                f"  |  Gemma 2 2B  |  Layer {LAYER}"
            ),
            x=0.5,
            font=dict(size=16),
        ),
        legend=dict(
            x=0.5, y=-0.02, xanchor="center", yanchor="top",
            orientation="h", font=dict(size=12),
        ),
        width=1400, height=650,
        margin=dict(l=0, r=0, t=60, b=60),
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"  Saved → {out_path}")


def plot_slider_plotly(
    act_centroids:  np.ndarray,
    prob_centroids: np.ndarray,
    pca_full:       PCA,
    spline:         CubicSpline,
    path_lin:       np.ndarray,
    path_mfd:       np.ndarray,
    probs_linear:   np.ndarray,
    probs_manifold: np.ndarray,
    start_idx:      int,
    end_idx:        int,
    out_path:       str,
) -> None:
    """Goodfire-style interactive HTML with custom slider.

    Layout:
      BEHAVIOR SPACE — stacked probability line charts (manifold / linear)
                       + 3D behavior manifold below
      ACTIVATION SPACE — native HTML range slider + 3D activation manifold

    The slider updates both 3D marker positions and the vertical cursor on
    the probability charts via Plotly.restyle (instant, no Plotly frames).
    3D scenes have no axis boxes — just a wireframe floor, colored diamond
    concept markers with colored labels, and steering path curves.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import json as _json

    # ── projections ──
    pca3_act = PCA(n_components=3)
    act_3d   = pca3_act.fit_transform(act_centroids)

    t_dense    = np.linspace(0.0, float(len(DAYS) - 1), 300)
    act_curve3 = pca3_act.transform(pca_full.inverse_transform(spline(t_dense)))
    lin_act3   = pca3_act.transform(path_lin)
    mfd_act3   = pca3_act.transform(path_mfd)

    hell_c   = np.sqrt(np.clip(prob_centroids, 0.0, None))
    pca3_beh = PCA(n_components=3)
    beh_3d   = pca3_beh.fit_transform(hell_c)

    beh_sp     = CubicSpline(np.arange(len(DAYS), dtype=float), beh_3d)
    beh_curve3 = beh_sp(t_dense)
    lin_beh3   = pca3_beh.transform(np.sqrt(np.clip(probs_linear,   0.0, None)))
    mfd_beh3   = pca3_beh.transform(np.sqrt(np.clip(probs_manifold, 0.0, None)))

    ts = np.linspace(0.0, 1.0, N_STEPS)

    # ── helper: one clean 3D figure ──
    def _make_3d(pts3, curve3, lin3, mfd3):
        all_x = np.concatenate([pts3[:,0], curve3[:,0]])
        all_y = np.concatenate([pts3[:,1], curve3[:,1]])
        all_z = np.concatenate([pts3[:,2], curve3[:,2], lin3[:,2], mfd3[:,2]])
        z_fl  = all_z.min() - 0.18 * (all_z.max() - all_z.min())
        pad   = 0.18
        x0 = all_x.min() - pad * (all_x.max() - all_x.min())
        x1 = all_x.max() + pad * (all_x.max() - all_x.min())
        y0 = all_y.min() - pad * (all_y.max() - all_y.min())
        y1 = all_y.max() + pad * (all_y.max() - all_y.min())

        # wireframe floor — all grid lines as one None-separated trace
        n_grid = 18
        xg = np.linspace(x0, x1, n_grid)
        yg = np.linspace(y0, y1, n_grid)
        fx, fy, fz = [], [], []
        for xv in xg:
            fx += [xv, xv, None]; fy += [y0, y1, None]; fz += [z_fl, z_fl, None]
        for yv in yg:
            fx += [x0, x1, None]; fy += [yv, yv, None]; fz += [z_fl, z_fl, None]

        traces = [
            # floor
            go.Scatter3d(x=fx, y=fy, z=fz, mode="lines",
                         line=dict(color="rgba(140,140,140,0.32)", width=1),
                         showlegend=False, hoverinfo="skip"),
            # shadow markers (all 7 in one trace)
            go.Scatter3d(x=pts3[:,0], y=pts3[:,1],
                         z=np.full(len(pts3), z_fl),
                         mode="markers",
                         marker=dict(symbol="diamond", size=4,
                                     color=DAY_COLORS, opacity=0.22),
                         showlegend=False, hoverinfo="skip"),
            # manifold curve — solid light gray
            go.Scatter3d(x=curve3[:,0].tolist(), y=curve3[:,1].tolist(), z=curve3[:,2].tolist(),
                         mode="lines", line=dict(color="#b8b8b8", width=3),
                         showlegend=False, hoverinfo="skip"),
            # linear path — gray dotted
            go.Scatter3d(x=lin3[:,0].tolist(), y=lin3[:,1].tolist(), z=lin3[:,2].tolist(),
                         mode="lines", line=dict(color="#999", width=2, dash="dot"),
                         showlegend=False, hoverinfo="skip"),
            # manifold steering path — dark dotted
            go.Scatter3d(x=mfd3[:,0].tolist(), y=mfd3[:,1].tolist(), z=mfd3[:,2].tolist(),
                         mode="lines", line=dict(color="#333", width=2, dash="dot"),
                         showlegend=False, hoverinfo="skip"),
        ]
        # concept markers — one trace each so text renders per-marker
        for i, day in enumerate(DAYS):
            traces.append(go.Scatter3d(
                x=[pts3[i,0]], y=[pts3[i,1]], z=[pts3[i,2]],
                mode="markers+text",
                text=[day], textposition="top center",
                textfont=dict(size=12, color=DAY_COLORS[i],
                              family="Arial Black, Arial Bold, Arial"),
                marker=dict(symbol="diamond", size=11, color=DAY_COLORS[i],
                            line=dict(color="rgba(0,0,0,0.35)", width=1)),
                showlegend=False,
                hovertemplate=f"{day}<extra></extra>",
            ))

        n_static = len(traces)   # = 5 + 7 = 12

        # moving markers (initial: step 0)
        traces.append(go.Scatter3d(
            x=[mfd3[0,0]], y=[mfd3[0,1]], z=[mfd3[0,2]],
            mode="markers",
            marker=dict(size=11, color="#222",
                        line=dict(color="white", width=2.5)),
            showlegend=False, hoverinfo="skip",
        ))
        traces.append(go.Scatter3d(
            x=[lin3[0,0]], y=[lin3[0,1]], z=[lin3[0,2]],
            mode="markers",
            marker=dict(size=11, color="#888",
                        line=dict(color="white", width=2.5)),
            showlegend=False, hoverinfo="skip",
        ))

        # Axis style: keep the 3D renderer active but style subtly.
        def _ax(label):
            return dict(
                title=dict(text=label, font=dict(size=9, color="#bbb")),
                showgrid=True,
                gridcolor="rgba(200,200,200,0.4)",
                gridwidth=1,
                zeroline=False,
                showticklabels=True,
                tickfont=dict(size=8, color="#aaa"),
                showaxeslabels=True,
                showspikes=False,
                showbackground=True,
                backgroundcolor="rgba(235,238,242,0.5)",
            )
        fig = go.Figure(data=traces, layout=go.Layout(
            scene=dict(
                xaxis=_ax("PC 1"),
                yaxis=_ax("PC 2"),
                zaxis=_ax("PC 3"),
                bgcolor="white",
                camera=dict(eye=dict(x=1.4, y=-1.9, z=0.65)),
            ),
            paper_bgcolor="white",
            margin=dict(l=0, r=0, t=0, b=0),
            height=480,
            showlegend=False,
        ))
        return fig, n_static

    act_fig, act_ns = _make_3d(act_3d, act_curve3, lin_act3, mfd_act3)
    beh_fig, beh_ns = _make_3d(beh_3d, beh_curve3, lin_beh3, mfd_beh3)

    # ── probability line charts (stacked: manifold / linear) ──
    def _make_prob():
        fig = make_subplots(
            rows=2, cols=1,
            vertical_spacing=0.06,
            subplot_titles=["Geometry-Aware Steering", "Linear Steering"],
            shared_xaxes=True,
        )
        t_ax = np.linspace(0.0, 1.0, N_STEPS)
        per_row = len(DAYS) + 2   # n_days + other + vline = 9

        # Explicitly convert to Python lists — avoids numpy dtype issues in to_json()
        t_list = t_ax.tolist()
        for row, probs in [(1, probs_manifold), (2, probs_linear)]:
            for i, day in enumerate(DAYS):
                fig.add_trace(go.Scatter(
                    x=t_list, y=probs[:, i].tolist(),
                    mode="lines", line=dict(color=DAY_COLORS[i], width=2),
                    name=day, showlegend=False,
                    hovertemplate=f"{day}: %{{y:.2f}}<extra></extra>",
                ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=t_list, y=probs[:, -1].tolist(),
                mode="lines",
                line=dict(color="#bbb", width=1.5, dash="dash"),
                name="non-day tokens",
                showlegend=(row == 1),
                hoverinfo="skip",
            ), row=row, col=1)
            # vertical cursor (dynamic)
            fig.add_trace(go.Scatter(
                x=[0.0, 0.0], y=[0.0, 1.05],
                mode="lines",
                line=dict(color="#333", width=1.5, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ), row=row, col=1)

        vl1 = per_row - 1         # = 8 (vline in row 1)
        vl2 = 2 * per_row - 1     # = 17 (vline in row 2)

        ax = dict(showgrid=True, gridcolor="#efefef", gridwidth=1,
                  zeroline=False, showline=False,
                  tickfont=dict(size=10, color="#777"))
        fig.update_xaxes(
            range=[0, 1],
            tickvals=[0, 0.25, 0.5, 0.75],
            ticktext=["0", "0.25", "0.5", "0.75"],
            **ax,
        )
        fig.update_yaxes(range=[0, 1.08], tickvals=[0, 0.5, 1.0], **ax)
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(l=42, r=20, t=32, b=8),
            height=260,
            showlegend=True,
            legend=dict(x=0.98, xanchor="right", y=0.98, yanchor="top",
                        font=dict(size=10, color="#888"),
                        bgcolor="rgba(0,0,0,0)"),
            font=dict(family="Arial, sans-serif", color="#444"),
        )
        for ann in fig.layout.annotations:
            ann.update(font=dict(size=12, color="#555"))
        return fig, vl1, vl2

    prob_fig, vl1, vl2 = _make_prob()

    # ── serialize to JSON for embedding ──
    act_json  = act_fig.to_json()
    beh_json  = beh_fig.to_json()
    prob_json = prob_fig.to_json()
    dyn_json  = _json.dumps({
        "mfd_act3": mfd_act3.tolist(),
        "lin_act3": lin_act3.tolist(),
        "mfd_beh3": mfd_beh3.tolist(),
        "lin_beh3": lin_beh3.tolist(),
        "ts":       ts.tolist(),
    })

    # ── HTML template ──
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Manifold Steering — {DAYS[start_idx]} → {DAYS[end_idx]}  |  Layer {LAYER}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: Arial, sans-serif;
  background: #f0f0f0;
  padding: 28px 32px;
  color: #333;
}}
.section-label {{
  font-size: 17px;
  font-weight: 900;
  letter-spacing: 2px;
  margin: 28px 0 10px;
  color: #222;
}}
.section-label .sub {{
  font-size: 13px;
  font-weight: 400;
  letter-spacing: 0;
  color: #aaa;
  margin-left: 8px;
}}
.card {{
  background: white;
  border-radius: 14px;
  border: 1px solid #e2e2e2;
  padding: 20px 20px 12px;
  margin-bottom: 20px;
  max-width: 920px;
}}
.slider-row {{
  padding: 6px 4px 4px;
}}
input[type=range] {{
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 4px;
  border-radius: 2px;
  background: #d5d5d5;
  outline: none;
  cursor: pointer;
}}
input[type=range]::-webkit-slider-thumb {{
  -webkit-appearance: none;
  appearance: none;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: #4a4a4a;
  cursor: pointer;
  border: 3px solid white;
  box-shadow: 0 1px 5px rgba(0,0,0,0.25);
}}
input[type=range]::-moz-range-thumb {{
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: #4a4a4a;
  cursor: pointer;
  border: 3px solid white;
  box-shadow: 0 1px 5px rgba(0,0,0,0.25);
  border: none;
}}
.slider-hint {{
  text-align: center;
  font-size: 13px;
  color: #c0c0c0;
  margin: 6px 0 10px;
}}
</style>
</head>
<body>

<div class="section-label">
  BEHAVIOR SPACE
  <span class="sub">(output token probabilities)</span>
</div>
<div class="card">
  <div id="prob-plot"></div>
  <div id="beh-plot"></div>
</div>

<div class="section-label">ACTIVATION SPACE</div>
<div class="card">
  <div class="slider-row">
    <input type="range" id="steer-slider" min="0" max="{N_STEPS - 1}" value="0">
  </div>
  <p class="slider-hint">Drag slider to transition between days</p>
  <div id="act-plot"></div>
</div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script>
const ACT  = {act_json};
const BEH  = {beh_json};
const PROB = {prob_json};
const DYN  = {dyn_json};
const ACT_NS = {act_ns};
const BEH_NS = {beh_ns};
const VL1    = {vl1};
const VL2    = {vl2};
const CFG    = {{displayModeBar: false, responsive: true}};

Plotly.newPlot('prob-plot', PROB.data, PROB.layout, CFG);
Plotly.newPlot('act-plot',  ACT.data,  ACT.layout,  CFG);
Plotly.newPlot('beh-plot',  BEH.data,  BEH.layout,  CFG);

document.getElementById('steer-slider').addEventListener('input', function () {{
  const i = parseInt(this.value);
  const t = DYN.ts[i];

  Plotly.restyle('act-plot', {{
    x: [[DYN.mfd_act3[i][0]], [DYN.lin_act3[i][0]]],
    y: [[DYN.mfd_act3[i][1]], [DYN.lin_act3[i][1]]],
    z: [[DYN.mfd_act3[i][2]], [DYN.lin_act3[i][2]]],
  }}, [ACT_NS, ACT_NS + 1]);

  Plotly.restyle('beh-plot', {{
    x: [[DYN.mfd_beh3[i][0]], [DYN.lin_beh3[i][0]]],
    y: [[DYN.mfd_beh3[i][1]], [DYN.lin_beh3[i][1]]],
    z: [[DYN.mfd_beh3[i][2]], [DYN.lin_beh3[i][2]]],
  }}, [BEH_NS, BEH_NS + 1]);

  Plotly.restyle('prob-plot', {{
    x: [[t, t], [t, t]],
  }}, [VL1, VL2]);
}});
</script>
</body>
</html>"""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Saved → {out_path}")



# ─────────────────────────────────── main ────────────────────────────────────

def _set_seed(seed: int) -> None:
    """Fix all RNG seeds. The pipeline is deterministic (no sampling, no
    training); this guards exact reproducibility of any future stochastic op."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main() -> None:
    parser = argparse.ArgumentParser(description="Weekday manifold steering")
    parser.add_argument("--start",  default="Tuesday", choices=DAYS)
    parser.add_argument("--end",    default="Friday",  choices=DAYS)
    parser.add_argument(
        "--base-prompt",
        default=None,
        help="Prompt used as the base input during steering interventions. "
             "Default: a prompt whose answer is the midpoint concept.",
    )
    parser.add_argument("--layer",  type=int, default=12,
                        help="hidden_states index to patch (1–26, default 12).")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--recollect",
        action="store_true",
        help="Ignore existing checkpoint and re-run collection from scratch.",
    )
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for exact reproducibility.")
    args = parser.parse_args()
    _set_seed(args.seed)

    # ── override global LAYER before anything else uses it ──
    global LAYER
    LAYER = args.layer

    # ── device ──
    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── prompts (always needed for base-prompt selection) ──
    print("\n[2/7] Building prompts ...")
    prompts_by_concept = build_prompts()
    for d in DAYS:
        print(f"  {d}: {len(prompts_by_concept[d])} prompts")

    # ── collect (or load from checkpoint) ──
    if not args.recollect and checkpoint_path().exists():
        print("\n[1/7] Loading model ... skipped (checkpoint found)")
        print("\n[3/7] Collecting activations and output probabilities ...")
        acts_by_concept, probs_by_concept = load_checkpoint()
        concept_ids = None   # not needed past this point
    else:
        print("\n[1/7] Loading model ...")
        tokenizer, model = load_model(device)
        concept_ids = get_concept_token_ids(tokenizer)
        print("  Concept token IDs:", {d: concept_ids[d] for d in DAYS})

        print("\n[3/7] Collecting activations and output probabilities ...")
        acts_by_concept, probs_by_concept = collect(
            model, tokenizer, prompts_by_concept, concept_ids, device
        )
        save_checkpoint(acts_by_concept, probs_by_concept)

        # Keep model and concept_ids alive for the steering phase below

    # ── centroids ──
    print("\n[4/7] Computing centroids ...")
    act_centroids, prob_centroids = compute_centroids(acts_by_concept, probs_by_concept)
    print(f"  Activation centroids : {act_centroids.shape}")
    print(f"  Probability centroids: {prob_centroids.shape}")

    # ── manifold fitting ──
    print("\n[5/7] Fitting activation manifold (PCA → cubic spline) ...")
    pca, pca_centroids, spline = fit_activation_manifold(act_centroids)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA explained variance ({N_PCA} components): {var_explained:.3f}")

    start_idx = DAYS.index(args.start)
    end_idx   = DAYS.index(args.end)
    print(f"  Steering pair: {args.start} (t={start_idx}) → {args.end} (t={end_idx})")

    # Base prompt: use a neutral prompt (not one of the endpoints)
    if args.base_prompt:
        base_prompt = args.base_prompt
    else:
        # Pick a prompt whose answer is the midpoint concept
        mid_idx = (start_idx + end_idx) // 2
        mid_day = DAYS[mid_idx]
        base_prompt = prompts_by_concept[mid_day][0]
    print(f"  Base prompt: {base_prompt!r}")

    # ── steering paths ──
    print("\n[6/7] Computing steering paths ...")
    path_lin = linear_path(act_centroids, start_idx, end_idx)
    path_mfd = manifold_path(pca, spline, start_idx, end_idx)
    print(f"  Linear path shape  : {path_lin.shape}")
    print(f"  Manifold path shape: {path_mfd.shape}")

    # ── interventions (model must be loaded) ──
    print("\n[7/7] Running steering interventions ...")
    if concept_ids is None:
        # Loaded from checkpoint — need model now for the steering phase
        print("  Loading model for steering ...")
        tokenizer, model = load_model(device)
        concept_ids = get_concept_token_ids(tokenizer)
    print("  Linear:")
    probs_linear   = steer_and_collect(model, tokenizer, base_prompt, path_lin, concept_ids, device)
    print("  Manifold:")
    probs_manifold = steer_and_collect(model, tokenizer, base_prompt, path_mfd, concept_ids, device)

    # ── plot probability trajectories ──
    out_path = f"figures/weekdays_L{LAYER}_{args.start}_{args.end}_steering.png"
    plot_trajectories(probs_linear, probs_manifold, args.start, args.end, out_path)

    # ── 3D manifold figure (static PNG) ──
    out_3d = f"figures/weekdays_L{LAYER}_{args.start}_{args.end}_3d.png"
    plot_3d_manifolds(
        act_centroids  = act_centroids,
        prob_centroids = prob_centroids,
        pca_full       = pca,
        spline         = spline,
        path_lin       = path_lin,
        path_mfd       = path_mfd,
        probs_linear   = probs_linear,
        probs_manifold = probs_manifold,
        start_idx      = start_idx,
        end_idx        = end_idx,
        out_path       = out_3d,
    )

    # ── 3D manifold figure (interactive Plotly HTML, no slider) ──
    out_3d_html = f"figures/weekdays_L{LAYER}_{args.start}_{args.end}_3d.html"
    plot_3d_manifolds_plotly(
        act_centroids  = act_centroids,
        prob_centroids = prob_centroids,
        pca_full       = pca,
        spline         = spline,
        path_lin       = path_lin,
        path_mfd       = path_mfd,
        probs_linear   = probs_linear,
        probs_manifold = probs_manifold,
        start_idx      = start_idx,
        end_idx        = end_idx,
        out_path       = out_3d_html,
    )

    # ── 3D manifold figure with steering-progress slider ──
    out_slider = f"figures/weekdays_L{LAYER}_{args.start}_{args.end}_slider.html"
    plot_slider_plotly(
        act_centroids  = act_centroids,
        prob_centroids = prob_centroids,
        pca_full       = pca,
        spline         = spline,
        path_lin       = path_lin,
        path_mfd       = path_mfd,
        probs_linear   = probs_linear,
        probs_manifold = probs_manifold,
        start_idx      = start_idx,
        end_idx        = end_idx,
        out_path       = out_slider,
    )


if __name__ == "__main__":
    main()
