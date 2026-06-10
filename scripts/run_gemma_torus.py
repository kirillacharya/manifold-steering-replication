"""Torus analysis on real Gemma 2 2B activations — Day-of-Week × Hour-of-Day.

PROMPTS: "It's {HH}:00 on {Day}" for all 7 days × 24 hours = 168 prompts.
Each prompt encodes both a day (parent) and an hour (child).

HIERARCHY (following synthetic torus experiment):
  Parent = day-of-week circle:  θ_day  = 2π * day_idx / 7
  Child  = hour-of-day circle:  φ_hour = 2π * hour_idx / 24

ANALYSIS:
  1. Harvest activations from layer 12 of Gemma 2 2B (mid-layer, 26 total).
  2. Find day subspace via OLS regression against [sin θ, cos θ].
  3. Find hour subspace via OLS regression against [sin φ, cos φ].
  4. Project activations onto both subspaces — check torus geometry.
  5. Load GemmaScope 2B SAE (16k), compute restricted R² for day and hour.
  6. Produce Plotly HTML with 4 tabs (same structure as run_torus.py).

MODEL: google/gemma-2-2b-it  (~4.5 GB bfloat16 — Mac-friendly)
SAE:   google/gemma-scope-2b-pt-res, layer 12, 16k width
OUTPUT: figures/gemma_torus/viz_2b.html
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import plotly.graph_objects as go
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
# SAE analysis (steps 3-6) is optional and needs the nonlinear_features
# package; activation harvesting (--acts-only) has no such dependency.
try:
    from nonlinear_features.jumprelu_sae import JumpReLUSAE
    from nonlinear_features.evaluate_real import (
        select_atoms_by_label_correlation,
        compute_concept_direction,
        _expand_labels,
    )
    HAS_SAE = True
except ImportError:
    HAS_SAE = False


DAYS  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
HOURS = list(range(24))

DAY_COLORS = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4"
]
DAY_NAMES_SHORT = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompts(n_days: int = 7, n_hours: int = 24):
    """n_days × n_hours prompts.  Returns (prompts, day_idx, hour_idx).

    Quick mode: n_days=3, n_hours=8 → 24 prompts for fast end-to-end check.
    Full mode:  n_days=7, n_hours=24 → 168 prompts.
    """
    days  = DAYS[:n_days]
    hours = list(range(0, 24, max(1, 24 // n_hours)))[:n_hours]
    prompts, day_idxs, hour_idxs = [], [], []
    for di, day in enumerate(days):
        for hi in hours:
            prompts.append(f"It's {hi:02d}:00 on {day}")
            day_idxs.append(di)
            hour_idxs.append(hi)
    return prompts, np.array(day_idxs), np.array(hour_idxs)


# ---------------------------------------------------------------------------
# Activation harvesting
# ---------------------------------------------------------------------------

def harvest(model_name: str, layer: int, device: str,
            batch_size: int = 16) -> torch.Tensor:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    prompts, day_idx, hour_idx = build_prompts()

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    print(f"Loading {model_name} (dtype={dtype}, device={device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype,
        device_map=device if device != "mps" else None,
    )
    if device == "mps":
        model = model.to(device)
    model.eval()

    all_acts = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Harvesting"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt",
                           padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs  = out.hidden_states[layer]          # (B, seq, d)
        mask = inputs["attention_mask"]
        last = mask.sum(dim=1) - 1
        bidx = torch.arange(hs.size(0), device=device)
        all_acts.append(hs[bidx, last].float().cpu())

    return torch.cat(all_acts, dim=0)           # (168, d)


# ---------------------------------------------------------------------------
# Subspace discovery via regression
# ---------------------------------------------------------------------------

def find_concept_subspace(acts: torch.Tensor, angles: np.ndarray) -> torch.Tensor:
    """Return (2, d) orthonormal subspace most correlated with circular angle.

    Regresses activations against [sin(angle), cos(angle)], then QR-orthogonalises.
    """
    labels = torch.tensor(
        np.stack([np.sin(angles), np.cos(angles)], axis=1),
        dtype=torch.float32,
    )
    acts_c = acts - acts.mean(0)
    dirs = []
    for i in range(2):
        li = labels[:, i]
        li = (li - li.mean()) / (li.std() + 1e-8)
        d = acts_c.T @ li / len(li)
        d = d / (d.norm() + 1e-10)
        dirs.append(d)
    D = torch.stack(dirs, dim=1)    # (d, 2)
    Q, _ = torch.linalg.qr(D)
    return Q[:, :2].T               # (2, d)


# ---------------------------------------------------------------------------
# Restricted R² (same probe as synthetic experiment)
# ---------------------------------------------------------------------------

def restricted_r2(codes: torch.Tensor, target: torch.Tensor,
                  labels_t: torch.Tensor, n_atoms: int = 8) -> float:
    n = codes.shape[0]
    if n < 5 or target.abs().sum() < 1e-8:
        return float("nan")
    selected = select_atoms_by_label_correlation(codes, labels_t, n_atoms)
    if not selected:
        return 0.0
    X = torch.cat([codes[:, selected], torch.ones(n, 1)], dim=1).cpu()
    Y = target.cpu()
    total_var = (Y - Y.mean(0)).pow(2).sum().item()
    if total_var < 1e-10:
        return 1.0
    W = torch.linalg.lstsq(X, Y, driver="gelsd").solution
    resid = (Y - X @ W).pow(2).sum().item()
    return float(1 - resid / total_var)


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

def _pca3(x_np):
    pca = PCA(n_components=3)
    proj = pca.fit_transform(x_np)
    return proj, pca.explained_variance_ratio_


def fig_pca_day(acts: torch.Tensor, day_idx: np.ndarray) -> go.Figure:
    proj, var = _pca3(acts.numpy())
    traces = []
    for d in range(7):
        mask = day_idx == d
        traces.append(go.Scatter3d(
            x=proj[mask,0], y=proj[mask,1], z=proj[mask,2],
            mode="markers",
            marker=dict(size=5, opacity=0.8, color=DAY_COLORS[d]),
            name=DAY_NAMES_SHORT[d],
        ))
    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=(
            "<b>Gemma Activation PCA — coloured by Day of Week</b><br>"
            f"<sup>PC1={var[0]:.1%}  PC2={var[1]:.1%}  PC3={var[2]:.1%}  "
            f"(top-3: {sum(var[:3]):.1%})</sup>"
        ), x=0.5),
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
                   camera=dict(eye=dict(x=1.5,y=1.5,z=0.8))),
        margin=dict(l=0,r=0,t=80,b=0), height=680,
        paper_bgcolor="#111", font=dict(color="#ddd"),
        legend=dict(bgcolor="rgba(30,30,30,0.8)", font=dict(size=11)),
    )
    return fig


def fig_pca_hour(acts: torch.Tensor, hour_idx: np.ndarray) -> go.Figure:
    proj, var = _pca3(acts.numpy())
    fig = go.Figure(go.Scatter3d(
        x=proj[:,0], y=proj[:,1], z=proj[:,2],
        mode="markers",
        marker=dict(size=5, opacity=0.8,
                    color=hour_idx.astype(float),
                    colorscale="HSV", cmin=0, cmax=23,
                    colorbar=dict(title="Hour", thickness=12,
                                  tickvals=[0,6,12,18,23],
                                  ticktext=["0h","6h","12h","18h","23h"])),
    ))
    fig.update_layout(
        title=dict(text=(
            "<b>Gemma Activation PCA — coloured by Hour of Day</b><br>"
            f"<sup>PC1={var[0]:.1%}  PC2={var[1]:.1%}  PC3={var[2]:.1%}</sup>"
        ), x=0.5),
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3",
                   camera=dict(eye=dict(x=1.5,y=1.5,z=0.8))),
        margin=dict(l=0,r=0,t=80,b=0), height=680,
        paper_bgcolor="#111", font=dict(color="#ddd"),
    )
    return fig


def fig_known_subspaces(acts: torch.Tensor,
                         V_day: torch.Tensor, V_hour: torch.Tensor,
                         day_idx: np.ndarray, hour_idx: np.ndarray) -> go.Figure:
    proj_d = (acts @ V_day.T).numpy()    # (168, 2)
    proj_h = (acts @ V_hour.T).numpy()   # (168, 2)

    # Day panel
    traces = []
    for d in range(7):
        mask = day_idx == d
        traces.append(go.Scatter(
            x=proj_d[mask,0], y=proj_d[mask,1],
            mode="markers",
            marker=dict(size=8, opacity=0.8, color=DAY_COLORS[d]),
            name=DAY_NAMES_SHORT[d],
            legendgroup="day", legendgrouptitle_text="Day",
            xaxis="x1", yaxis="y1",
        ))

    # Hour panel
    traces.append(go.Scatter(
        x=proj_h[:,0], y=proj_h[:,1],
        mode="markers",
        marker=dict(size=8, opacity=0.8,
                    color=hour_idx.astype(float),
                    colorscale="HSV", cmin=0, cmax=23,
                    colorbar=dict(title="Hour", thickness=12,
                                  tickvals=[0,6,12,18,23],
                                  ticktext=["0h","6h","12h","18h","23h"],
                                  x=1.02)),
        name="hour",
        legendgroup="hour", legendgrouptitle_text="Hour",
        xaxis="x2", yaxis="y2",
    ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=dict(text=(
            "<b>Known-Subspace Projections (Gemma)</b><br>"
            "<sup>Left: regression subspace for day | "
            "Right: regression subspace for hour</sup>"
        ), x=0.5),
        xaxis =dict(title="V_day dim-1",  domain=[0.0, 0.44],
                    scaleanchor="y1", scaleratio=1),
        yaxis =dict(title="V_day dim-2",  scaleanchor="x1", scaleratio=1),
        xaxis2=dict(title="V_hour dim-1", domain=[0.56, 1.0],
                    scaleanchor="y2", scaleratio=1),
        yaxis2=dict(title="V_hour dim-2", anchor="x2",
                    scaleanchor="x2", scaleratio=1),
        paper_bgcolor="#111", plot_bgcolor="#1a1a1a",
        font=dict(color="#ddd"),
        height=600, margin=dict(l=60,r=80,t=80,b=60),
        legend=dict(bgcolor="rgba(30,30,30,0.8)", tracegroupgap=10),
    )
    return fig


def fig_r2_bar(day_r2: float, hour_r2: float) -> go.Figure:
    fig = go.Figure([
        go.Bar(name="Day-of-week R² (parent)",
               x=["Day of Week"], y=[day_r2],
               marker_color="#4CAF50", width=0.4,
               text=[f"{day_r2:.3f}"], textposition="outside"),
        go.Bar(name="Hour-of-day R² (child)",
               x=["Hour of Day"], y=[hour_r2],
               marker_color="#2196F3", width=0.4,
               text=[f"{hour_r2:.3f}"], textposition="outside"),
    ])
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        title=dict(text=(
            "<b>GemmaScope SAE — Restricted R² for Day and Hour Subspaces</b><br>"
            "<sup>Day = parent manifold (always encoded) | "
            "Hour = child manifold (conditionally encoded)</sup>"
        ), x=0.5),
        yaxis=dict(title="Restricted R²", range=[0, 1.15]),
        xaxis=dict(title="Concept"),
        paper_bgcolor="#111", plot_bgcolor="#1a1a1a",
        font=dict(color="#ddd"),
        barmode="group",
        legend=dict(bgcolor="rgba(30,30,30,0.8)"),
        height=480, margin=dict(l=60,r=20,t=80,b=60),
    )
    return fig


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

def build_html(tabs: list[tuple[str, str]], title: str) -> str:
    buttons = [
        f'<button class="tab-btn" onclick="showTab(\'{n}\')" id="btn-{n}">{n}</button>'
        for n, _ in tabs
    ]
    divs = [
        f'<div class="tab-pane" id="tab-{n}" style="display:none">{h}</div>'
        for n, h in tabs
    ]
    first = tabs[0][0]
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body{{font-family:system-ui,sans-serif;margin:0;background:#0f0f0f;color:#eee}}
  h1{{text-align:center;padding:16px 0 4px;font-size:1.2em;color:#ccc}}
  .tab-bar{{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;
            padding:4px 8px 12px;border-bottom:1px solid #333}}
  .tab-btn{{padding:6px 18px;border:1px solid #444;border-radius:5px;
            background:#1e1e1e;color:#bbb;cursor:pointer;font-size:.9em}}
  .tab-btn:hover{{background:#2a2a2a}}
  .tab-btn.active{{background:#3a6ea8;color:#fff;border-color:#3a6ea8}}
  .tab-pane{{padding:0 12px 12px}}
</style></head><body>
<h1>{title}</h1>
<div class="tab-bar">{''.join(buttons)}</div>
{''.join(divs)}
<script>
function showTab(id){{
  document.querySelectorAll('.tab-pane').forEach(d=>d.style.display='none');
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('tab-'+id).style.display='block';
  document.getElementById('btn-'+id).classList.add('active');
}}
showTab('{first}');document.getElementById('btn-{first}').classList.add('active');
</script></body></html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="google/gemma-2-2b-it")
    parser.add_argument("--layer",        type=int, default=12)
    parser.add_argument("--device",       default="auto")
    parser.add_argument("--batch-size",   type=int, default=16)
    parser.add_argument("--sae-local-path", default=None,
                        help="Path to GemmaScope params.safetensors (if already downloaded)")
    parser.add_argument("--sae-repo",      default="google/gemma-scope-2b-pt-res")
    parser.add_argument("--sae-width",     default="16k")
    parser.add_argument("--sae-l0",        default="medium")
    parser.add_argument("--sae-subfolder", default=None,
                        help="Explicit subfolder within the HF repo, e.g. "
                             "'layer_12/width_16k/average_l0_71'. "
                             "Overrides --sae-width and --sae-l0.")
    parser.add_argument("--n-atoms",      type=int, default=8)
    parser.add_argument("--quick",        action="store_true",
                        help="3 days × 8 hours = 24 prompts — fast end-to-end smoke test")
    parser.add_argument("--acts-cache",   default="figures/gemma_torus/acts_2b.pt",
                        help="Cache path for harvested activations (skip harvesting if exists)")
    parser.add_argument("--output",       default="figures/gemma_torus/viz_2b.html")
    parser.add_argument("--acts-only",    action="store_true",
                        help="Stop after harvesting activations (no SAE needed). "
                             "Sufficient for the torus figure.")
    parser.add_argument("--seed",         type=int, default=0,
                        help="Random seed (pipeline is deterministic; for reproducibility)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():           device = "cuda"
        elif torch.backends.mps.is_available(): device = "mps"
        else:                                   device = "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_days, n_hours = (3, 8) if args.quick else (7, 24)
    prompts, day_idx, hour_idx = build_prompts(n_days, n_hours)
    print(f"Prompts: {len(prompts)}  ({n_days} days × {n_hours} hours)"
          + ("  [QUICK MODE]" if args.quick else ""))
    print(f"  Example: \"{prompts[0]}\"  →  day={day_idx[0]}, hour={hour_idx[0]}")

    # ---- 1. Activations ----
    cache_path = args.acts_cache
    if args.quick and cache_path == "figures/gemma_torus/acts_2b.pt":
        cache_path = "figures/gemma_torus/acts_2b_quick.pt"
    cache = Path(cache_path)
    if cache.exists():
        print(f"\nLoading cached activations from {cache}")
        acts = torch.load(cache, weights_only=False)
    else:
        print(f"\nHarvesting activations (layer {args.layer}) ...")
        acts = harvest(args.model, args.layer, device, args.batch_size)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(acts, cache)
        print(f"  Saved to {cache}")
    print(f"  Activations: {acts.shape}")

    # ---- 2. Find concept subspaces ----
    print("\nFinding day and hour subspaces via regression ...")
    theta_day  = 2 * np.pi * day_idx  / 7
    theta_hour = 2 * np.pi * hour_idx / 24
    V_day  = find_concept_subspace(acts, theta_day)    # (2, d)
    V_hour = find_concept_subspace(acts, theta_hour)   # (2, d)

    # Check orthogonality between the two subspaces
    overlap = (V_day @ V_hour.T).abs().max().item()
    print(f"  Day–hour subspace overlap (max |cos|): {overlap:.4f}")

    if args.acts_only:
        print("\n--acts-only: done (activations cached, subspaces verified).")
        return

    # ---- 3. Load SAE ----
    if not HAS_SAE:
        sys.exit("SAE analysis requires the nonlinear_features package; "
                 "rerun with --acts-only for the torus figure pipeline.")
    print("\nLoading GemmaScope SAE ...")
    if args.sae_local_path:
        sae = JumpReLUSAE.from_pretrained(args.sae_local_path, device=device)
    else:
        sae = JumpReLUSAE.from_huggingface(
            repo_id=args.sae_repo, layer=args.layer,
            site="resid_post", width=args.sae_width, l0=args.sae_l0,
            subfolder=args.sae_subfolder,
            device=device,
        )
    print(f"  SAE: d_in={sae.d_in}, d_sae={sae.d_sae}")

    # ---- 4. Encode ----
    acts_dev = acts.to(device)
    with torch.no_grad():
        codes = sae.encode(acts_dev).cpu()    # (168, d_sae)

    # ---- 5. Restricted R² ----
    print("\nComputing restricted R² ...")
    # Targets: projections onto concept subspaces
    target_day  = (acts @ V_day.T)     # (168, 2)
    target_hour = (acts @ V_hour.T)    # (168, 2)

    labels_day  = torch.tensor(
        np.stack([np.sin(theta_day),  np.cos(theta_day)],  axis=1), dtype=torch.float32)
    labels_hour = torch.tensor(
        np.stack([np.sin(theta_hour), np.cos(theta_hour)], axis=1), dtype=torch.float32)

    r2_day  = restricted_r2(codes, target_day,  labels_day,  args.n_atoms)
    r2_hour = restricted_r2(codes, target_hour, labels_hour, args.n_atoms)
    print(f"  Day  R² = {r2_day:.4f}")
    print(f"  Hour R² = {r2_hour:.4f}")

    # ---- 6. Build HTML ----
    print("\nBuilding Plotly figures ...")
    I = lambda f: f.to_html(full_html=False, include_plotlyjs=False)
    tabs = [
        ("PCA — day",        I(fig_pca_day(acts, day_idx))),
        ("PCA — hour",       I(fig_pca_hour(acts, hour_idx))),
        ("Known subspaces",  I(fig_known_subspaces(acts, V_day, V_hour,
                                                    day_idx, hour_idx))),
        ("R² (SAE)",         I(fig_r2_bar(r2_day, r2_hour))),
    ]

    out = Path(args.output)
    out.write_text(build_html(tabs,
        "Gemma 2 2B — Torus Hierarchy: Day-of-Week × Hour-of-Day (Real Data)"))
    print(f"\nSaved → {out}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
