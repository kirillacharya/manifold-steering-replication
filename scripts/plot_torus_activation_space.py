"""Plot 2D activation space for the hierarchical torus steering experiment.

Projects all 168 (day × hour) activations onto V_day[0] × V_hour[0] — the
leading directions of the day and hour subspaces — revealing the torus grid
structure of concept representations.

No model inference needed: loads acts_L24.pt, recomputes subspaces, projects.

Outputs:
  figures/hierarchical_steering/torus_activation_space.png
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline

RESULTS_DIR = Path("figures/hierarchical_steering")
ACTS_CACHE  = RESULTS_DIR / "acts_L24.pt"

DAYS  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAYS_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HOURS = list(range(24))

SRC_DAY,  SRC_HOUR  = 1,  9   # Tuesday 09:00
TGT_DAY,  TGT_HOUR  = 4,  15  # Friday  15:00
N_STEPS = 80

DAY_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
    "#f58231", "#911eb4", "#42d4f4",
]


# ── subspace helpers (identical to plot_subspace_hours.py) ───────────────────

def find_subspace(acts: torch.Tensor, angles: np.ndarray) -> torch.Tensor:
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
    return Q[:, :2].T   # (2, d)


def main():
    print(f"Loading acts from {ACTS_CACHE}")
    acts = torch.load(ACTS_CACHE, weights_only=False)   # (168, d)
    print(f"  shape: {acts.shape}")

    day_idx  = np.array([d for d in range(7)  for _ in range(24)])
    hour_idx = np.array([h for _ in range(7)  for h in range(24)])

    theta_day  = 2 * np.pi * day_idx  / 7
    theta_hour = 2 * np.pi * hour_idx / 24

    V_day  = find_subspace(acts, theta_day)    # (2, d)
    V_hour = find_subspace(acts, theta_hour)   # (2, d)

    # Project all 168 activations
    acts_np    = acts.numpy().astype(np.float64)
    V_day_np   = V_day.numpy().astype(np.float64)
    V_hour_np  = V_hour.numpy().astype(np.float64)

    proj_day  = acts_np @ V_day_np.T    # (168, 2)
    proj_hour = acts_np @ V_hour_np.T   # (168, 2)

    # Use leading direction of each subspace as axes
    x_all = proj_day[:, 0]    # V_day[0]  → x-axis
    y_all = proj_hour[:, 0]   # V_hour[0] → y-axis

    # ── Centroids ──
    day_centroids_x  = np.array([x_all[day_idx == d].mean() for d in range(7)])
    day_centroids_y  = np.array([y_all[day_idx == d].mean() for d in range(7)])
    hour_centroids_x = np.array([x_all[hour_idx == h].mean() for h in range(24)])
    hour_centroids_y = np.array([y_all[hour_idx == h].mean() for h in range(24)])

    # Per-(day, hour) centroid — one point per grid cell
    centroid_x = np.array([
        x_all[(day_idx == d) & (hour_idx == h)][0]
        for d in range(7) for h in range(24)
    ])
    centroid_y = np.array([
        y_all[(day_idx == d) & (hour_idx == h)][0]
        for d in range(7) for h in range(24)
    ])
    centroid_day  = np.array([d for d in range(7) for h in range(24)])
    centroid_hour = np.array([h for d in range(7) for h in range(24)])

    # ── Day and hour splines through centroid means (for manifold curve) ──
    day_spline_x  = CubicSpline(np.arange(7,  dtype=float), day_centroids_x)
    day_spline_y  = CubicSpline(np.arange(7,  dtype=float), day_centroids_y)
    hour_spline_x = CubicSpline(np.arange(24, dtype=float), hour_centroids_x)
    hour_spline_y = CubicSpline(np.arange(24, dtype=float), hour_centroids_y)

    # ── Steering paths in full activation space ──
    # Reconstruct h_base = mean with both subspaces removed
    proj_d = acts @ V_day.T
    proj_h = acts @ V_hour.T
    recon  = (proj_d @ V_day) + (proj_h @ V_hour)
    h_base = (acts - recon).mean(0)   # (d,)
    h_base_np = h_base.numpy().astype(np.float64)

    # Full-space splines (same as plot_subspace_hours.py)
    proj_d_np = proj_d.numpy().astype(np.float64)  # (168, 2)
    proj_h_np = proj_h.numpy().astype(np.float64)
    day_cents_full  = np.array([proj_d_np[day_idx == d].mean(0) for d in range(7)])
    hour_cents_full = np.array([proj_h_np[hour_idx == h].mean(0) for h in range(24)])
    day_spline_full  = CubicSpline(np.arange(7,  dtype=float), day_cents_full)
    hour_spline_full = CubicSpline(np.arange(24, dtype=float), hour_cents_full)

    ts = np.linspace(0.0, 1.0, N_STEPS)

    src_act = acts_np[(day_idx == SRC_DAY) & (hour_idx == SRC_HOUR)][0]
    tgt_act = acts_np[(day_idx == TGT_DAY) & (hour_idx == TGT_HOUR)][0]

    path_linear = np.array([(1 - t) * src_act + t * tgt_act for t in ts])

    day_ts  = np.linspace(float(SRC_DAY),  float(TGT_DAY),  N_STEPS)
    hour_ts = np.linspace(float(SRC_HOUR), float(TGT_HOUR), N_STEPS)
    path_torus = (
        h_base_np[None, :]
        + day_spline_full(day_ts)  @ V_day_np
        + hour_spline_full(hour_ts) @ V_hour_np
    )

    # Project steering paths onto the 2D display axes
    lin_x = path_linear @ V_day_np[0]
    lin_y = path_linear @ V_hour_np[0]
    tor_x = path_torus  @ V_day_np[0]
    tor_y = path_torus  @ V_hour_np[0]

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    # Background scatter: all 168 individual activations (faint, colored by day)
    for d in range(7):
        mask = day_idx == d
        ax.scatter(x_all[mask], y_all[mask],
                   color=DAY_COLORS[d], alpha=0.18, s=18, zorder=1)

    # Draw hour grid lines: connect same-hour points across days
    for h in range(0, 24, 3):   # every 3 hours to avoid clutter
        xs = [x_all[(day_idx == d) & (hour_idx == h)][0] for d in range(7)]
        ys = [y_all[(day_idx == d) & (hour_idx == h)][0] for d in range(7)]
        ax.plot(xs, ys, color="gray", lw=0.6, alpha=0.35, zorder=2)
        # Label on leftmost point (Monday)
        ax.text(xs[0] - 0.003, ys[0], f"{h:02d}h",
                fontsize=5.5, color="gray", va="center", ha="right", alpha=0.7)

    # Draw day grid lines: connect same-day points across hours
    for d in range(7):
        xs = [x_all[(day_idx == d) & (hour_idx == h)][0] for h in range(24)]
        ys = [y_all[(day_idx == d) & (hour_idx == h)][0] for h in range(24)]
        ax.plot(xs, ys, color=DAY_COLORS[d], lw=1.0, alpha=0.4, zorder=2)

    # Day centroids (large colored diamonds + label)
    for d in range(7):
        ax.scatter(day_centroids_x[d], day_centroids_y[d],
                   color=DAY_COLORS[d], s=160, marker="D", zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.text(day_centroids_x[d], day_centroids_y[d] + 0.004,
                DAYS_SHORT[d], fontsize=8, fontweight="bold",
                color=DAY_COLORS[d], ha="center", va="bottom", zorder=6,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # Source and target markers
    src_x = x_all[(day_idx == SRC_DAY)  & (hour_idx == SRC_HOUR)][0]
    src_y = y_all[(day_idx == SRC_DAY)  & (hour_idx == SRC_HOUR)][0]
    tgt_x = x_all[(day_idx == TGT_DAY)  & (hour_idx == TGT_HOUR)][0]
    tgt_y = y_all[(day_idx == TGT_DAY)  & (hour_idx == TGT_HOUR)][0]

    ax.scatter(src_x, src_y, s=200, marker="o", color="white",
               edgecolors="black", linewidths=2.0, zorder=8)
    ax.scatter(tgt_x, tgt_y, s=200, marker="*", color="gold",
               edgecolors="black", linewidths=1.0, zorder=8)

    # Steering paths
    ax.plot(lin_x, lin_y, color="crimson", lw=2.2, ls="--",
            zorder=7, label="Linear steering")
    ax.plot(tor_x, tor_y, color="black",   lw=2.2, ls="-",
            zorder=7, label="Torus steering")

    # Arrows on paths to show direction
    for path_x, path_y, col in [(lin_x, lin_y, "crimson"),
                                  (tor_x, tor_y, "black")]:
        mid = N_STEPS // 2
        dx  = path_x[mid + 1] - path_x[mid - 1]
        dy  = path_y[mid + 1] - path_y[mid - 1]
        ax.annotate("", xy=(path_x[mid] + dx * 0.5, path_y[mid] + dy * 0.5),
                    xytext=(path_x[mid] - dx * 0.5, path_y[mid] - dy * 0.5),
                    arrowprops=dict(arrowstyle="->", color=col, lw=2.0),
                    zorder=9)

    ax.set_xlabel("$V_\\mathrm{day}[0]$ projection  (day subspace)", fontsize=12)
    ax.set_ylabel("$V_\\mathrm{hour}[0]$ projection  (hour subspace)", fontsize=12)
    ax.set_title(
        "Torus Activation Space  |  Gemma 2 2B  |  Layer 24  |  "
        "Tuesday 09:00 → Friday 15:00",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.15)

    legend_els = [
        Line2D([0], [0], color="crimson", lw=2, ls="--", label="Linear steering"),
        Line2D([0], [0], color="black",   lw=2, ls="-",  label="Torus steering"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="black", ms=9, lw=0,
               label=f"Source  (Tue 09:00)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="black", ms=11, lw=0,
               label=f"Target  (Fri 15:00)"),
    ]
    ax.legend(handles=legend_els, fontsize=10, loc="lower right",
              framealpha=0.9)

    plt.tight_layout()
    out = RESULTS_DIR / "torus_activation_space.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
