"""Create a Figure-4-style composite panel for Weekdays and Months domains.

Loads checkpoints for both domains, runs linear and manifold steering at
layer 24, and plots a 2×2 grid:

    Columns : Weekdays (Tue→Fri)  |  Months (Jan→Jul)
    Row 1   : Manifold Steering
    Row 2   : Linear Steering

Output: figures/figure4_composite.png
"""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME = "google/gemma-2-2b-it"
LAYER      = 24
N_STEPS    = 30

# Weekdays
DAYS    = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
NUMBERS = ["one","two","three","four","five","six"]
NUMBER_TO_INT = {w: i+1 for i, w in enumerate(NUMBERS)}
WD_TEMPLATE   = "Q: What day is {number} days after {entity}?\nA:"
WD_START, WD_END = 1, 4   # Tuesday → Friday
WD_BASE_PROMPT   = "Q: What day is two days after Monday?\nA:"
WD_N_PCA         = 6

DAY_COLORS = [
    "#e6194b","#3cb44b","#ffe119","#4363d8",
    "#f58231","#911eb4","#42d4f4",
]

# Months
MONTHS = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December",
]
MO_TEMPLATE  = "Q: What month is {number} months after {entity}?\nA:"
MO_START, MO_END = 0, 6   # January → July
MO_BASE_PROMPT   = "Q: What month is three months after January?\nA:"
MO_N_PCA         = 11

MONTH_COLORS = [
    "#e6194b","#f58231","#ffe119","#bfef45",
    "#3cb44b","#42d4f4","#4363d8","#911eb4",
    "#f032e6","#a9a9a9","#9A6324","#000075",
]

# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(device):
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    print(f"  Loading {MODEL_NAME} ...")
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype,
                device_map=str(device) if device.type != "mps" else None)
    if device.type == "mps":
        model = model.to(device)
    model.eval()
    return tok, model

# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_wd_checkpoint():
    cp   = Path(f"checkpoints/weekdays_L{LAYER}_acts_probs.npz")
    data = np.load(cp)
    acts  = np.array([data[f"acts_{d}"]  for d in DAYS])   # (7, n, d)
    probs = np.array([data[f"probs_{d}"] for d in DAYS])   # (7, n, 8)
    centroids_act  = acts.mean(axis=1)   # (7, d)
    centroids_prob = probs.mean(axis=1)  # (7, 8)
    print(f"  Weekdays checkpoint loaded ← {cp}")
    return centroids_act, centroids_prob

def load_mo_checkpoint():
    cp   = Path(f"checkpoints/months_L{LAYER}_acts_probs.npz")
    data = np.load(cp)
    acts  = np.array([data[f"acts_{m}"]  for m in MONTHS])
    probs = np.array([data[f"probs_{m}"] for m in MONTHS])
    centroids_act  = acts.mean(axis=1)
    centroids_prob = probs.mean(axis=1)
    print(f"  Months checkpoint loaded ← {cp}")
    return centroids_act, centroids_prob

# ── Manifold fitting ──────────────────────────────────────────────────────────

def fit_manifold(act_centroids, n_pca):
    pca  = PCA(n_components=n_pca)
    pca_c = pca.fit_transform(act_centroids)
    t_k  = np.arange(len(act_centroids), dtype=float)
    spl  = CubicSpline(t_k, pca_c)
    return pca, spl

# ── Steering paths ────────────────────────────────────────────────────────────

def make_linear_path(centroids, i0, i1):
    ts = np.linspace(0., 1., N_STEPS)
    h0, h1 = centroids[i0], centroids[i1]
    return np.array([(1-t)*h0 + t*h1 for t in ts])

def make_manifold_path(pca, spl, i0, i1):
    t_int = np.linspace(float(i0), float(i1), N_STEPS)
    return pca.inverse_transform(spl(t_int))

# ── Intervention ──────────────────────────────────────────────────────────────

def steer(model, tok, base_prompt, path_acts, id_list, device):
    inputs = tok(base_prompt, return_tensors="pt").to(device)
    dtype  = next(model.parameters()).dtype
    all_p  = []
    n_concepts = len(id_list)

    with torch.no_grad():
        for vec in tqdm(path_acts, desc="    steer", leave=False):
            rep = torch.tensor(vec, dtype=dtype, device=device)

            def _hook(m, i, o, _r=rep):
                h = o[0].clone() if isinstance(o, tuple) else o.clone()
                h[:,-1,:] = _r
                return (h,)+o[1:] if isinstance(o, tuple) else h

            handle = model.model.layers[LAYER-1].register_forward_hook(_hook)
            out    = model(**inputs)
            handle.remove()

            logits = out.logits[0,-1,:].float().cpu()
            pa     = torch.softmax(logits, dim=-1).numpy()
            cp     = np.array([pa[tid] for tid in id_list])
            all_p.append(np.append(cp, max(0., 1.-cp.sum())))

    return np.array(all_p)   # (N_STEPS, n_concepts+1)

# ── Plotting ──────────────────────────────────────────────────────────────────

def draw_panel(ax, ts, probs, concepts, colors, title=None, ylabel=None,
               show_legend=False, legend_ncol=4):
    for i, (name, col) in enumerate(zip(concepts, colors)):
        lbl = name[:3] if len(name) > 3 else name
        ax.plot(ts, probs[:, i], color=col, lw=2.0, label=lbl)
    ax.plot(ts, probs[:,-1], color="gray", lw=1.0, ls="--",
            alpha=0.55, label="other")
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.axvline(1, color="black", lw=0.6, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Steering progress  t", fontsize=9)
    ax.grid(True, alpha=0.18)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if show_legend:
        ax.legend(fontsize=6.5, loc="upper center", ncol=legend_ncol,
                  bbox_to_anchor=(0.5, -0.32), framealpha=0.85)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse, random
    parser = argparse.ArgumentParser(description="Steering probs for weekdays+months (Figure 4)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for exact reproducibility.")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available()         else
        torch.device("cpu")
    )
    print(f"Device: {device}  |  Layer: {LAYER}")

    # Load checkpoints
    wd_act, wd_prob = load_wd_checkpoint()
    mo_act, mo_prob = load_mo_checkpoint()

    # Fit manifolds
    wd_pca, wd_spl = fit_manifold(wd_act, WD_N_PCA)
    mo_pca, mo_spl = fit_manifold(mo_act, MO_N_PCA)

    # Build paths
    wd_lin = make_linear_path(wd_act, WD_START, WD_END)
    wd_mfd = make_manifold_path(wd_pca, wd_spl, WD_START, WD_END)
    mo_lin = make_linear_path(mo_act, MO_START, MO_END)
    mo_mfd = make_manifold_path(mo_pca, mo_spl, MO_START, MO_END)

    # Load model
    tok, model = load_model(device)

    wd_ids = {d: tok.encode(" "+d, add_special_tokens=False)[-1] for d in DAYS}
    mo_ids = {m: tok.encode(" "+m, add_special_tokens=False)[-1] for m in MONTHS}
    wd_id_list = [wd_ids[d] for d in DAYS]
    mo_id_list = [mo_ids[m] for m in MONTHS]

    ts = np.linspace(0., 1., N_STEPS)

    print("\nWeekdays — linear ...")
    wd_p_lin = steer(model, tok, WD_BASE_PROMPT, wd_lin, wd_id_list, device)
    print("Weekdays — manifold ...")
    wd_p_mfd = steer(model, tok, WD_BASE_PROMPT, wd_mfd, wd_id_list, device)
    print("Months — linear ...")
    mo_p_lin = steer(model, tok, MO_BASE_PROMPT, mo_lin, mo_id_list, device)
    print("Months — manifold ...")
    mo_p_mfd = steer(model, tok, MO_BASE_PROMPT, mo_mfd, mo_id_list, device)

    # Save steering probs for reuse
    npz_out = Path(f"checkpoints/steering_probs_L{LAYER}.npz")
    np.savez(npz_out,
             wd_p_lin=wd_p_lin, wd_p_mfd=wd_p_mfd,
             mo_p_lin=mo_p_lin, mo_p_mfd=mo_p_mfd)
    print(f"  Steering probs saved → {npz_out}")

    # ── Figure: 2 rows × 2 cols ───────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.52,
        wspace=0.10,
        top=0.88, bottom=0.18,
        left=0.08, right=0.97,
    )

    ax_mfd_wd = fig.add_subplot(gs[0, 0])
    ax_mfd_mo = fig.add_subplot(gs[0, 1], sharey=ax_mfd_wd)
    ax_lin_wd = fig.add_subplot(gs[1, 0])
    ax_lin_mo = fig.add_subplot(gs[1, 1], sharey=ax_lin_wd)

    # Row labels as text on the left
    for ax, label in [(ax_mfd_wd, "Manifold\nSteering"), (ax_lin_wd, "Linear\nSteering")]:
        ax.set_ylabel(label, fontsize=10, fontweight="bold", labelpad=8)

    # Column headers
    ax_mfd_wd.set_title("Weekdays  (Tuesday → Friday)",  fontsize=11, fontweight="bold", pad=6)
    ax_mfd_mo.set_title("Months  (January → July)", fontsize=11, fontweight="bold", pad=6)

    draw_panel(ax_mfd_wd, ts, wd_p_mfd, DAYS,   DAY_COLORS,
               show_legend=True, legend_ncol=4)
    draw_panel(ax_mfd_mo, ts, mo_p_mfd, MONTHS, MONTH_COLORS,
               show_legend=True, legend_ncol=6)
    draw_panel(ax_lin_wd, ts, wd_p_lin, DAYS,   DAY_COLORS,
               show_legend=False)
    draw_panel(ax_lin_mo, ts, mo_p_lin, MONTHS, MONTH_COLORS,
               show_legend=False)

    # Hide y tick labels on right column
    plt.setp(ax_mfd_mo.get_yticklabels(), visible=False)
    plt.setp(ax_lin_mo.get_yticklabels(), visible=False)

    fig.suptitle(
        "Figure 4 Replication — Manifold vs. Linear Steering  "
        "|  Gemma 2 2B  |  Layer 24",
        fontsize=12, fontweight="bold", y=0.96,
    )

    out = Path("figures/figure4_composite.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
