"""Hierarchical / torus manifold steering — Gemma 2 2B, layer 24.

Steering pair: (Tuesday, 09:00) → (Friday, 15:00)

Phase 0  Sanity-check: does Gemma reliably answer day / hour probes?
Phase 1  Collect last-token residual activations at layer 24 for all
         7 × 24 = 168 prompts  "It is {HH}:00 on {Day}."
Phase 2  OLS regression → V_day (2×d), V_hour (2×d)  (same as torus benchmark)
Phase 3  Day centroids (7 × 2) and hour centroids (24 × 2) in subspace coords.
         Fit cubic splines through ordered centroids.
Phase 4  Build steering paths
         a) Linear  : lerp(h[Tue,09], h[Fri,15], t)
         b) Torus   : h_base + day_spline(t·Δday) @ V_day
                             + hour_spline(t·Δhour) @ V_hour
Phase 5  For each path point, patch layer-24 residual stream in two probes:
           Day probe  → weekday token probabilities (Mon–Sun)
           Hour probe → hour token probabilities (09–15, or time-bin fallback)
Phase 6  Plot + metrics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# ─────────────────────────────────── constants ───────────────────────────────

MODEL_NAME = "google/gemma-2-2b-it"
LAYER      = 24          # residual-stream layer to patch (hook on layers[LAYER-1])
N_STEPS    = 50          # steering path discretisation

DAYS  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
HOURS = list(range(24))

# Steering endpoints
SRC_DAY, SRC_HOUR = 1, 9    # Tuesday 09:00
TGT_DAY, TGT_HOUR = 4, 15   # Friday  15:00

# Probed hours (tight window around the journey for readability)
PROBED_HOURS = list(range(7, 18))   # 07–17, covers src→tgt with margin

RESULTS_DIR = Path("figures/hierarchical_steering")
ACTS_CACHE  = RESULTS_DIR / "acts_L24.pt"
SANITY_FILE = RESULTS_DIR / "sanity_check.json"


# ─────────────────────────────────── prompts ─────────────────────────────────

def activation_prompt(day: str, hour: int) -> str:
    return f"It is {hour:02d}:00 on {day}."


# Probing prompts – fixed; we patch the activation to steer content.
DAY_PROBE  = "The appointment is at 09:00 on Tuesday. The day is"
HOUR_PROBE = "The appointment is at 09:00 on Tuesday. The time is"


# ─────────────────────────────────── model ───────────────────────────────────

def load_model(device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    print(f"  Loading {MODEL_NAME}  (dtype={dtype}) ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=str(device) if device.type != "mps" else None,
    )
    if device.type == "mps":
        mdl = mdl.to(device)
    mdl.eval()
    return tok, mdl


# ─────────────────────────────────── token helpers ───────────────────────────

def get_day_token_ids(tok) -> dict[str, int]:
    ids = {}
    for day in DAYS:
        toks = tok.encode(" " + day, add_special_tokens=False)
        ids[day] = toks[-1]
    return ids


def get_hour_token_ids(tok, hours: list[int]) -> dict[int, int] | None:
    """Try to get single-token IDs for hour integers.

    Returns None if any hour maps to multiple tokens (triggers fallback).
    """
    ids = {}
    for h in hours:
        # try both " 9" and " 09" spellings
        for spelling in (f" {h}", f" {h:02d}"):
            toks = tok.encode(spelling, add_special_tokens=False)
            if len(toks) == 1:
                ids[h] = toks[0]
                break
        else:
            print(f"  Hour {h:02d} is multi-token – will use time-bin fallback.")
            return None
    return ids


TIME_BINS = {
    "morning":   list(range(6,  12)),
    "afternoon": list(range(12, 17)),
    "evening":   list(range(17, 21)),
    "night":     list(range(21, 24)) + list(range(0, 6)),
}
TIME_BIN_NAMES = ["morning", "afternoon", "evening", "night"]
TIME_BIN_COLORS = ["#f9c74f", "#f8961e", "#f3722c", "#277da1"]


def get_timebin_token_ids(tok) -> dict[str, int]:
    ids = {}
    for name in TIME_BIN_NAMES:
        toks = tok.encode(" " + name, add_special_tokens=False)
        ids[name] = toks[-1]
    return ids


# ─────────────────────────────────── Phase 0: sanity check ───────────────────

def sanity_check(tok, mdl, device: torch.device) -> dict:
    """Run day and hour probes on (Tuesday, 09:00); report top-5 tokens."""
    print("\n── Phase 0: Sanity check ─────────────────────────────────────────")
    results = {}

    day_ids  = get_day_token_ids(tok)
    hour_ids = get_hour_token_ids(tok, PROBED_HOURS)

    for label, probe in [("day", DAY_PROBE), ("hour", HOUR_PROBE)]:
        inputs = tok(probe, return_tensors="pt").to(device)
        with torch.no_grad():
            out    = mdl(**inputs)
            logits = out.logits[0, -1, :].float().cpu()
        probs  = torch.softmax(logits, dim=-1)
        top5   = probs.topk(5)
        top5_tokens = [(tok.decode([idx.item()]), round(p.item(), 4))
                       for idx, p in zip(top5.indices, top5.values)]
        print(f"  {label.upper()} probe top-5: {top5_tokens}")
        results[label] = top5_tokens

    # Check if day tokens are reliable
    day_probe_inputs = tok(DAY_PROBE, return_tensors="pt").to(device)
    with torch.no_grad():
        out    = mdl(**day_probe_inputs)
        logits = out.logits[0, -1, :].float().cpu()
    probs   = torch.softmax(logits, dim=-1).numpy()
    day_probs = {d: float(probs[tid]) for d, tid in day_ids.items()}
    top_day   = max(day_probs, key=day_probs.get)
    print(f"  Day probs: { {d: f'{p:.3f}' for d,p in day_probs.items()} }")
    print(f"  → Top day: {top_day}  (expected: Tuesday)")
    results["day_probe_works"] = (top_day == "Tuesday")

    # Check hour tokens
    if hour_ids is not None:
        hour_probe_inputs = tok(HOUR_PROBE, return_tensors="pt").to(device)
        with torch.no_grad():
            out    = mdl(**hour_probe_inputs)
            logits = out.logits[0, -1, :].float().cpu()
        probs = torch.softmax(logits, dim=-1).numpy()
        hour_probs = {h: float(probs[tid]) for h, tid in hour_ids.items()}
        top_hour   = max(hour_probs, key=hour_probs.get)
        print(f"  Hour probs (07–17): { {h: f'{p:.3f}' for h,p in hour_probs.items()} }")
        print(f"  → Top hour: {top_hour}  (expected: 9)")
        results["hour_probe_works"] = (top_hour == 9)
        results["use_time_bins"]    = False
    else:
        results["hour_probe_works"] = False
        results["use_time_bins"]    = True
        print("  → Falling back to time bins: morning / afternoon / evening / night")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SANITY_FILE.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Sanity results saved → {SANITY_FILE}")
    return results


# ─────────────────────────────────── Phase 1: activations ────────────────────

def collect_activations(tok, mdl, device: torch.device) -> torch.Tensor:
    """Collect last-token hidden state at layer LAYER for all 168 prompts.

    Returns: (168, d_model) float32 on CPU.
    """
    if ACTS_CACHE.exists():
        print(f"  Loading cached activations ← {ACTS_CACHE}")
        return torch.load(ACTS_CACHE, weights_only=False)

    print("\n── Phase 1: Collecting activations ──────────────────────────────")
    all_acts = []
    for day in tqdm(DAYS, desc="  Days"):
        for hour in HOURS:
            prompt = activation_prompt(day, hour)
            inputs = tok(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out = mdl(**inputs, output_hidden_states=True)
            hs  = out.hidden_states[LAYER]        # (1, seq, d)
            act = hs[0, -1, :].float().cpu()
            all_acts.append(act)

    acts = torch.stack(all_acts, dim=0)           # (168, d)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(acts, ACTS_CACHE)
    print(f"  Saved → {ACTS_CACHE}  shape={acts.shape}")
    return acts


# ─────────────────────────────────── Phase 2: subspaces ──────────────────────

def find_subspace(acts: torch.Tensor, angles: np.ndarray) -> torch.Tensor:
    """(2, d) orthonormal subspace most correlated with circular angle."""
    labels = torch.tensor(
        np.stack([np.sin(angles), np.cos(angles)], axis=1), dtype=torch.float32)
    acts_c = acts - acts.mean(0)
    dirs = []
    for i in range(2):
        li = labels[:, i]
        li = (li - li.mean()) / (li.std() + 1e-8)
        d  = acts_c.T @ li / len(li)
        d  = d / (d.norm() + 1e-10)
        dirs.append(d)
    Q, _ = torch.linalg.qr(torch.stack(dirs, dim=1))
    return Q[:, :2].T       # (2, d)


def build_subspaces(acts: torch.Tensor,
                    day_idx: np.ndarray,
                    hour_idx: np.ndarray):
    """Return V_day (2,d), V_hour (2,d), h_base (d,)."""
    print("\n── Phase 2: Building subspaces ──────────────────────────────────")
    theta_day  = 2 * np.pi * day_idx  / 7
    theta_hour = 2 * np.pi * hour_idx / 24

    V_day  = find_subspace(acts, theta_day)
    V_hour = find_subspace(acts, theta_hour)

    overlap = (V_day @ V_hour.T).abs().max().item()
    print(f"  Day–hour subspace overlap (max |cos|): {overlap:.4f}")

    # h_base: mean activation with day+hour subspace components removed
    proj_d = acts @ V_day.T    # (168, 2)
    proj_h = acts @ V_hour.T   # (168, 2)
    recon  = (proj_d @ V_day) + (proj_h @ V_hour)
    h_base = (acts - recon).mean(0)   # (d,)
    return V_day, V_hour, h_base


# ─────────────────────────────────── Phase 3: centroids + splines ────────────

def build_centroids_and_splines(acts: torch.Tensor,
                                V_day: torch.Tensor,
                                V_hour: torch.Tensor,
                                day_idx: np.ndarray,
                                hour_idx: np.ndarray):
    """
    Returns
    -------
    day_centroids  : (7, 2)  — mean V_day projection per day
    hour_centroids : (24, 2) — mean V_hour projection per hour
    day_spline     : CubicSpline  float → (2,)  parameterized by day index
    hour_spline    : CubicSpline  float → (2,)  parameterized by hour index
    """
    print("\n── Phase 3: Centroids and splines ───────────────────────────────")
    proj_d = (acts @ V_day.T).numpy()    # (168, 2)
    proj_h = (acts @ V_hour.T).numpy()   # (168, 2)

    # Day centroids: average V_day projections across all 24 hours
    day_centroids = np.array([
        proj_d[day_idx == d].mean(0) for d in range(7)
    ])                                    # (7, 2)

    # Hour centroids: average V_hour projections across all 7 days
    hour_centroids = np.array([
        proj_h[hour_idx == h].mean(0) for h in range(24)
    ])                                    # (24, 2)

    print(f"  Day centroids  shape: {day_centroids.shape}")
    print(f"  Hour centroids shape: {hour_centroids.shape}")

    # Splines
    day_spline  = CubicSpline(np.arange(7,  dtype=float), day_centroids)
    hour_spline = CubicSpline(np.arange(24, dtype=float), hour_centroids)

    return day_centroids, hour_centroids, day_spline, hour_spline


# ─────────────────────────────────── Phase 4: steering paths ─────────────────

def make_paths(acts: torch.Tensor,
               V_day: torch.Tensor,
               V_hour: torch.Tensor,
               h_base: torch.Tensor,
               day_spline: CubicSpline,
               hour_spline: CubicSpline,
               day_idx: np.ndarray,
               hour_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    path_linear : (N_STEPS, d)  — direct lerp between endpoint activations
    path_torus  : (N_STEPS, d)  — spline-factored torus path
    """
    print("\n── Phase 4: Building steering paths ─────────────────────────────")
    acts_np = acts.numpy()

    # Find the single activation vector for each endpoint in the 7×24 grid
    src_mask = (day_idx == SRC_DAY) & (hour_idx == SRC_HOUR)
    tgt_mask = (day_idx == TGT_DAY) & (hour_idx == TGT_HOUR)
    h_src = acts_np[src_mask][0]   # (d,)
    h_tgt = acts_np[tgt_mask][0]   # (d,)

    ts = np.linspace(0.0, 1.0, N_STEPS)

    # ── Linear ──
    path_linear = np.array([(1 - t) * h_src + t * h_tgt for t in ts])

    # ── Torus ──
    # Interpolate day coordinate from SRC_DAY to TGT_DAY along day spline
    # Interpolate hour coordinate from SRC_HOUR to TGT_HOUR along hour spline
    day_ts  = np.linspace(float(SRC_DAY),  float(TGT_DAY),  N_STEPS)
    hour_ts = np.linspace(float(SRC_HOUR), float(TGT_HOUR), N_STEPS)

    day_coords  = day_spline(day_ts)    # (N_STEPS, 2)
    hour_coords = hour_spline(hour_ts)  # (N_STEPS, 2)

    V_day_np  = V_day.numpy()    # (2, d)
    V_hour_np = V_hour.numpy()   # (2, d)
    h_base_np = h_base.numpy()   # (d,)

    path_torus = (
        h_base_np[None, :]                        # broadcast base
        + day_coords  @ V_day_np                  # (N_STEPS, d)
        + hour_coords @ V_hour_np                 # (N_STEPS, d)
    )

    src_name = f"{DAYS[SRC_DAY]} {SRC_HOUR:02d}:00"
    tgt_name = f"{DAYS[TGT_DAY]} {TGT_HOUR:02d}:00"
    print(f"  Linear path:  {src_name} → {tgt_name}")
    print(f"  Torus  path:  day spline [{SRC_DAY}→{TGT_DAY}]  ×  "
          f"hour spline [{SRC_HOUR}→{TGT_HOUR}]")

    return path_linear, path_torus


# ─────────────────────────────────── Phase 5: steer + probe ──────────────────

def patch_and_probe(mdl, tok, probe: str, path: np.ndarray,
                    concept_ids: list[int], device: torch.device) -> np.ndarray:
    """Patch layer-LAYER activation and record output probs for concept_ids.

    Returns: (N_STEPS, len(concept_ids)+1)  last column = 'other'
    """
    inputs = tok(probe, return_tensors="pt").to(device)
    dtype  = next(mdl.parameters()).dtype
    all_probs = []

    with torch.no_grad():
        for act_vec in tqdm(path, desc=f"    Probing '{probe[:30]}…'", leave=False):
            replacement = torch.tensor(act_vec, dtype=dtype, device=device)

            def _hook(module, inp, output, _rep=replacement):
                h = output[0].clone() if isinstance(output, tuple) else output.clone()
                h[:, -1, :] = _rep
                return (h,) + output[1:] if isinstance(output, tuple) else h

            handle = mdl.model.layers[LAYER - 1].register_forward_hook(_hook)
            out    = mdl(**inputs)
            handle.remove()

            logits    = out.logits[0, -1, :].float().cpu()
            probs_all = torch.softmax(logits, dim=-1).numpy()

            cp        = np.array([probs_all[tid] for tid in concept_ids])
            other     = max(0.0, 1.0 - cp.sum())
            all_probs.append(np.append(cp, other))

    return np.array(all_probs)


def run_steering(mdl, tok, path_linear, path_torus,
                 day_ids: dict, hour_ids: dict | None,
                 use_time_bins: bool, device: torch.device):
    """Run both paths through both probes; return prob arrays."""
    print("\n── Phase 5: Steering + probing ──────────────────────────────────")

    # Day concept IDs (ordered Monday–Sunday)
    day_concept_ids = [day_ids[d] for d in DAYS]

    print("  Day probe — linear path")
    day_lin  = patch_and_probe(mdl, tok, DAY_PROBE,  path_linear, day_concept_ids, device)
    print("  Day probe — torus path")
    day_tor  = patch_and_probe(mdl, tok, DAY_PROBE,  path_torus,  day_concept_ids, device)

    if use_time_bins:
        bin_ids = get_timebin_token_ids(tok)
        hour_concept_ids = [bin_ids[b] for b in TIME_BIN_NAMES]
        print("  Hour probe (time bins) — linear path")
        hour_lin = patch_and_probe(mdl, tok, HOUR_PROBE, path_linear, hour_concept_ids, device)
        print("  Hour probe (time bins) — torus path")
        hour_tor = patch_and_probe(mdl, tok, HOUR_PROBE, path_torus,  hour_concept_ids, device)
        hour_labels = TIME_BIN_NAMES
        hour_colors = TIME_BIN_COLORS
    else:
        hour_concept_ids = [hour_ids[h] for h in PROBED_HOURS]
        print("  Hour probe — linear path")
        hour_lin = patch_and_probe(mdl, tok, HOUR_PROBE, path_linear, hour_concept_ids, device)
        print("  Hour probe — torus path")
        hour_tor = patch_and_probe(mdl, tok, HOUR_PROBE, path_torus,  hour_concept_ids, device)
        hour_labels = [f"{h:02d}h" for h in PROBED_HOURS]
        hour_colors = plt.cm.plasma(np.linspace(0, 1, len(PROBED_HOURS))).tolist()

    return day_lin, day_tor, hour_lin, hour_tor, hour_labels, hour_colors


# ─────────────────────────────────── Phase 6: plots ──────────────────────────

DAY_COLORS = [
    "#e6194b","#3cb44b","#ffe119","#4363d8",
    "#f58231","#911eb4","#42d4f4",
]


def _prob_panel(ax, ts, probs, labels, colors, title):
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(ts, probs[:, i], color=color, label=label, lw=2)
    ax.plot(ts, probs[:, -1], color="gray", lw=1, ls="--", alpha=0.5, label="other")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Steering progress  t", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7, loc="upper center", ncol=5,
              bbox_to_anchor=(0.5, -0.22))


def plot_results(day_lin, day_tor, hour_lin, hour_tor,
                 hour_labels, hour_colors, use_time_bins: bool):
    ts = np.linspace(0.0, 1.0, N_STEPS)
    src_name = f"{DAYS[SRC_DAY]} {SRC_HOUR:02d}:00"
    tgt_name = f"{DAYS[TGT_DAY]} {TGT_HOUR:02d}:00"
    title_base = f"Gemma 2 2B  |  Layer {LAYER}  |  {src_name} → {tgt_name}"

    # ── Plot 1: Weekday probabilities (2×2 grid) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey="row")
    fig.suptitle(f"Hierarchical Steering — {title_base}", fontsize=12, fontweight="bold")

    _prob_panel(axes[0, 0], ts, day_lin,  DAYS, DAY_COLORS, "Day probs — Linear Steering")
    _prob_panel(axes[0, 1], ts, day_tor,  DAYS, DAY_COLORS, "Day probs — Torus Steering")
    _prob_panel(axes[1, 0], ts, hour_lin, hour_labels, hour_colors,
                f"Hour probs — Linear Steering  ({'bins' if use_time_bins else 'exact'})")
    _prob_panel(axes[1, 1], ts, hour_tor, hour_labels, hour_colors,
                f"Hour probs — Torus Steering  ({'bins' if use_time_bins else 'exact'})")

    axes[0, 0].set_ylabel("P(token)", fontsize=10)
    axes[1, 0].set_ylabel("P(token)", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = RESULTS_DIR / "hierarchical_steering_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")

    # ── Plot 2: Side-by-side day only (Figure-4 style) ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    fig.suptitle(f"Day probabilities  |  {title_base}", fontsize=12, fontweight="bold")
    _prob_panel(axes[0], ts, day_lin, DAYS, DAY_COLORS, "Linear Steering")
    _prob_panel(axes[1], ts, day_tor, DAYS, DAY_COLORS, "Torus Steering")
    axes[0].set_ylabel("Output probability", fontsize=11)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    out2 = RESULTS_DIR / "hierarchical_day_steering.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out2}")

    # ── Plot 3: Side-by-side hour only ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    fig.suptitle(f"Hour probabilities  |  {title_base}", fontsize=12, fontweight="bold")
    _prob_panel(axes[0], ts, hour_lin, hour_labels, hour_colors, "Linear Steering")
    _prob_panel(axes[1], ts, hour_tor, hour_labels, hour_colors, "Torus Steering")
    axes[0].set_ylabel("Output probability", fontsize=11)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    out3 = RESULTS_DIR / "hierarchical_hour_steering.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out3}")


# ─────────────────────────────────── metrics ─────────────────────────────────

def print_metrics(day_lin, day_tor, hour_lin, hour_tor,
                  hour_labels, use_time_bins: bool):
    ts = np.linspace(0.0, 1.0, N_STEPS)
    print("\n── Metrics ──────────────────────────────────────────────────────")

    # Target probabilities at endpoint (t=1)
    tgt_day_name  = DAYS[TGT_DAY]
    tgt_day_idx   = TGT_DAY
    print(f"  Target day  ({tgt_day_name}) prob at t=1:")
    print(f"    Linear : {day_lin[-1, tgt_day_idx]:.3f}")
    print(f"    Torus  : {day_tor[-1, tgt_day_idx]:.3f}")

    if not use_time_bins:
        tgt_hour = TGT_HOUR
        tgt_hour_idx = PROBED_HOURS.index(tgt_hour) if tgt_hour in PROBED_HOURS else -1
        if tgt_hour_idx >= 0:
            print(f"  Target hour ({tgt_hour:02d}h) prob at t=1:")
            print(f"    Linear : {hour_lin[-1, tgt_hour_idx]:.3f}")
            print(f"    Torus  : {hour_tor[-1, tgt_hour_idx]:.3f}")

    # Intermediate peak order for days (expect Wed, Thu between Tue and Fri)
    for label, probs in [("Linear", day_lin), ("Torus", day_tor)]:
        peak_order = [DAYS[np.argmax(probs[i, :7])] for i in range(N_STEPS)]
        # Find unique ordered peaks (ignoring consecutive duplicates)
        seen, ordered = [], []
        for d in peak_order:
            if not ordered or d != ordered[-1]:
                ordered.append(d)
        print(f"  Day peak sequence ({label}): {' → '.join(ordered)}")

    # Monotonicity of expected hour coordinate
    if not use_time_bins:
        probed = np.array(PROBED_HOURS)
        for label, probs in [("Linear", hour_lin), ("Torus", hour_tor)]:
            exp_hour = (probs[:, :-1] * probed[None, :]).sum(1) / (probs[:, :-1].sum(1) + 1e-8)
            mono = np.all(np.diff(exp_hour) >= 0)
            print(f"  Expected hour monotone ({label}): {mono}"
                  f"  range [{exp_hour[0]:.1f} → {exp_hour[-1]:.1f}]")


# ─────────────────────────────────── main ────────────────────────────────────

def _set_seed(seed: int) -> None:
    """Fix all RNG seeds. The pipeline is deterministic (no sampling, no
    training); this guards exact reproducibility of any future stochastic op."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",      default="auto")
    parser.add_argument("--skip-sanity", action="store_true")
    parser.add_argument("--use-bins",    action="store_true",
                        help="Force time-bin fallback for hours")
    parser.add_argument("--plots-only",  action="store_true",
                        help="Skip model loading; reuse cached acts + steering_probs.npz")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for exact reproducibility.")
    args = parser.parse_args()
    _set_seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():           device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else:                                   device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Determine whether the model is needed ──────────────────────────────
    acts_cached   = ACTS_CACHE.exists()
    probes_cached = (RESULTS_DIR / "steering_probs.npz").exists()
    sanity_cached = SANITY_FILE.exists()

    need_model = (
        not args.plots_only
        and (not acts_cached or not probes_cached
             or (not args.skip_sanity and not sanity_cached))
    )

    if need_model:
        print(f"Device: {device}")
        tok, mdl = load_model(device)
    else:
        print("Caches found — skipping model load.")
        tok = mdl = None

    # ── Phase 0: Sanity ──
    if need_model and not args.skip_sanity:
        sanity = sanity_check(tok, mdl, device)
        use_time_bins = args.use_bins or sanity.get("use_time_bins", False)
    elif sanity_cached:
        import json as _json
        use_time_bins = args.use_bins or _json.loads(SANITY_FILE.read_text()).get("use_time_bins", False)
    else:
        use_time_bins = args.use_bins or True   # safe default

    if need_model:
        day_ids  = get_day_token_ids(tok)
        hour_ids = None if use_time_bins else get_hour_token_ids(tok, PROBED_HOURS)
        if hour_ids is None:
            use_time_bins = True
    else:
        day_ids = hour_ids = None

    # ── Phase 1: Activations ──
    if need_model or not acts_cached:
        acts = collect_activations(tok, mdl, device)   # (168, d)
    else:
        print(f"\n── Phase 1: Loading cached activations ← {ACTS_CACHE}")
        acts = torch.load(ACTS_CACHE, weights_only=False)

    # Build index arrays  (row i = day day_idx[i], hour hour_idx[i])
    day_idx  = np.array([d for d in range(7) for _ in range(24)])
    hour_idx = np.array([h for _ in range(7) for h in range(24)])

    # ── Phase 2: Subspaces ──
    V_day, V_hour, h_base = build_subspaces(acts, day_idx, hour_idx)

    # ── Phase 3: Centroids + splines ──
    day_centroids, hour_centroids, day_spline, hour_spline = \
        build_centroids_and_splines(acts, V_day, V_hour, day_idx, hour_idx)

    # ── Phase 4: Paths ──
    path_linear, path_torus = make_paths(
        acts, V_day, V_hour, h_base,
        day_spline, hour_spline,
        day_idx, hour_idx,
    )

    # ── Phase 5: Steer ──
    if probes_cached and not need_model:
        print("\n── Phase 5: Loading cached steering probs ───────────────────────")
        _npz = np.load(RESULTS_DIR / "steering_probs.npz")
        day_lin, day_tor = _npz["day_lin"], _npz["day_tor"]
        hour_lin, hour_tor = _npz["hour_lin"], _npz["hour_tor"]
        n_hour = hour_lin.shape[1] - 1
        use_time_bins_plot = (n_hour == 4)
        if use_time_bins_plot:
            hour_labels = TIME_BIN_NAMES
            hour_colors = TIME_BIN_COLORS
        else:
            hour_labels = [f"{h:02d}h" for h in PROBED_HOURS]
            hour_colors = plt.cm.plasma(np.linspace(0, 1, n_hour)).tolist()
        use_time_bins = use_time_bins_plot
    else:
        day_lin, day_tor, hour_lin, hour_tor, hour_labels, hour_colors = run_steering(
            mdl, tok, path_linear, path_torus,
            day_ids, hour_ids, use_time_bins, device,
        )
        np.savez(
            RESULTS_DIR / "steering_probs.npz",
            day_lin=day_lin, day_tor=day_tor,
            hour_lin=hour_lin, hour_tor=hour_tor,
        )
        print(f"\n  Raw probs saved → {RESULTS_DIR}/steering_probs.npz")

    # ── Phase 6: Plot + metrics ──
    print("\n── Phase 6: Plotting ────────────────────────────────────────────")
    plot_results(day_lin, day_tor, hour_lin, hour_tor,
                 hour_labels, hour_colors, use_time_bins)
    print_metrics(day_lin, day_tor, hour_lin, hour_tor,
                  hour_labels, use_time_bins)
    print("\nDone.")


if __name__ == "__main__":
    main()
