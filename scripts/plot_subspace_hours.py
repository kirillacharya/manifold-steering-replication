"""Plot hour representation along steering paths via V_hour subspace projection.

No model inference needed — loads acts_L24.pt, recomputes subspaces and paths
in numpy, then for each step finds the nearest hour centroid.

This gives exact hour labels (9, 10, 11, ..., 15) avoiding the Gemma
multi-token problem for two-digit hours.

Outputs:
  figures/hierarchical_steering/subspace_hour_steering.png
  figures/hierarchical_steering/subspace_combined.png
"""

from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

RESULTS_DIR = Path("figures/hierarchical_steering")
ACTS_CACHE  = RESULTS_DIR / "acts_L24.pt"
NPZ         = RESULTS_DIR / "steering_probs.npz"

DAYS  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
HOURS = list(range(24))
N_STEPS = 50
SRC_DAY, SRC_HOUR = 1, 9    # Tuesday 09:00
TGT_DAY, TGT_HOUR = 4, 15   # Friday  15:00

DAY_COLORS = [
    "#e6194b","#3cb44b","#ffe119","#4363d8",
    "#f58231","#911eb4","#42d4f4",
]


# ── helpers ──────────────────────────────────────────────────────────────────

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
    return Q[:, :2].T       # (2, d)


def normalize_concept(arr: np.ndarray) -> np.ndarray:
    concept = arr[:, :-1]
    total   = np.maximum(concept.sum(axis=1, keepdims=True), 1e-12)
    return concept / total


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading acts from {ACTS_CACHE}")
    acts = torch.load(ACTS_CACHE, weights_only=False)   # (168, d)
    print(f"  shape: {acts.shape}")

    day_idx  = np.array([d for d in range(7) for _ in range(24)])
    hour_idx = np.array([h for _ in range(7) for h in range(24)])

    theta_day  = 2 * np.pi * day_idx  / 7
    theta_hour = 2 * np.pi * hour_idx / 24

    V_day  = find_subspace(acts, theta_day)    # (2, d)
    V_hour = find_subspace(acts, theta_hour)   # (2, d)

    # h_base: mean with both subspaces removed
    proj_d = acts @ V_day.T
    proj_h = acts @ V_hour.T
    recon  = (proj_d @ V_day) + (proj_h @ V_hour)
    h_base = (acts - recon).mean(0)            # (d,)

    # ── Centroids ──
    proj_d_np = proj_d.numpy()  # (168, 2)
    proj_h_np = proj_h.numpy()  # (168, 2)

    day_centroids  = np.array([proj_d_np[day_idx == d].mean(0) for d in range(7)])
    hour_centroids = np.array([proj_h_np[hour_idx == h].mean(0) for h in range(24)])

    day_spline  = CubicSpline(np.arange(7,  dtype=float), day_centroids)
    hour_spline = CubicSpline(np.arange(24, dtype=float), hour_centroids)

    # Cast everything to float64 to avoid float32 overflow in 2304-dim matmul
    acts_np    = acts.numpy().astype(np.float64)
    V_day_np   = V_day.numpy().astype(np.float64)
    V_hour_np  = V_hour.numpy().astype(np.float64)
    h_base_np  = h_base.numpy().astype(np.float64)
    hour_centroids = hour_centroids.astype(np.float64)
    day_centroids  = day_centroids.astype(np.float64)

    # ── Steering paths ──
    src_mask = (day_idx == SRC_DAY) & (hour_idx == SRC_HOUR)
    tgt_mask = (day_idx == TGT_DAY) & (hour_idx == TGT_HOUR)
    h_src = acts_np[src_mask][0]
    h_tgt = acts_np[tgt_mask][0]

    ts = np.linspace(0.0, 1.0, N_STEPS)

    path_linear = np.array([(1 - t) * h_src + t * h_tgt for t in ts])

    day_ts  = np.linspace(float(SRC_DAY),  float(TGT_DAY),  N_STEPS)
    hour_ts = np.linspace(float(SRC_HOUR), float(TGT_HOUR), N_STEPS)
    day_coords  = day_spline(day_ts)
    hour_coords = hour_spline(hour_ts)

    path_torus = (
        h_base_np[None, :]
        + day_coords  @ V_day_np
        + hour_coords @ V_hour_np
    )

    # ── Project onto V_hour → nearest centroid ──
    def proj_to_hour(path: np.ndarray) -> np.ndarray:
        """For each path step, return nearest hour centroid index (0-23)."""
        coords = path @ V_hour_np.T    # (N, 2)
        # Euclidean distance to each of 24 centroids
        dists  = np.linalg.norm(coords[:, None, :] - hour_centroids[None, :, :], axis=2)  # (N, 24)
        return np.argmin(dists, axis=1)   # (N,)

    def proj_to_day(path: np.ndarray) -> np.ndarray:
        coords = path @ V_day_np.T
        dists  = np.linalg.norm(coords[:, None, :] - day_centroids[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

    # Calibrate sigma to half the mean nearest-neighbour centroid distance
    hour_cdists = np.linalg.norm(
        hour_centroids[:, None, :] - hour_centroids[None, :, :], axis=2)
    np.fill_diagonal(hour_cdists, np.inf)
    sigma_hour = hour_cdists.min(axis=1).mean() / 2    # ≈ 4.09 for L24

    day_cdists = np.linalg.norm(
        day_centroids[:, None, :] - day_centroids[None, :, :], axis=2)
    np.fill_diagonal(day_cdists, np.inf)
    sigma_day  = day_cdists.min(axis=1).mean() / 2

    print(f"  sigma_hour={sigma_hour:.3f}  sigma_day={sigma_day:.3f}")

    def soft_hour_prob(path: np.ndarray) -> np.ndarray:
        """Soft-max over centroid distances → (N, 24) probability distribution.

        Residualizes the day subspace before projecting onto V_hour, to remove
        ~9% day-hour cross-contamination.
        """
        # Remove day subspace component: path_pure = path - (path·Vday.T) Vday
        day_coeff  = path @ V_day_np.T                  # (N, 2)
        day_reconst = day_coeff @ V_day_np               # (N, d)
        path_pure  = path - day_reconst                  # (N, d)
        coords = path_pure @ V_hour_np.T                 # (N, 2)
        dists  = np.linalg.norm(coords[:, None, :] - hour_centroids[None, :, :], axis=2)
        logits = -dists / sigma_hour
        logits -= logits.max(axis=1, keepdims=True)
        exp    = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)   # (N, 24)

    def soft_day_prob(path: np.ndarray) -> np.ndarray:
        coords = path @ V_day_np.T
        dists  = np.linalg.norm(coords[:, None, :] - day_centroids[None, :, :], axis=2)
        logits = -dists / sigma_day
        logits -= logits.max(axis=1, keepdims=True)
        exp    = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)   # (N, 7)

    # ── Expected hour coordinate (single curve per path) ─────────────────────
    all_hours      = np.arange(24, dtype=float)
    hour_probs_lin = soft_hour_prob(path_linear)    # (N, 24)
    hour_probs_tor = soft_hour_prob(path_torus)     # (N, 24)
    exp_hour_lin   = (hour_probs_lin * all_hours[None, :]).sum(1)   # (N,)
    exp_hour_tor   = (hour_probs_tor * all_hours[None, :]).sum(1)   # (N,)

    day_probs_lin  = soft_day_prob(path_linear)    # (N, 7)
    day_probs_tor  = soft_day_prob(path_torus)

    src_name  = f"{DAYS[SRC_DAY]} {SRC_HOUR:02d}:00"
    tgt_name  = f"{DAYS[TGT_DAY]} {TGT_HOUR:02d}:00"
    title_base = f"Gemma 2 2B  |  Layer 24  |  {src_name} → {tgt_name}"

    def day_prob_panel(ax, ts, probs, labels, colors, title):
        for i, (lab, col) in enumerate(zip(labels, colors)):
            ax.plot(ts, probs[:, i], color=col, label=lab, lw=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Steering progress  t", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc="upper center", ncol=4,
                  bbox_to_anchor=(0.5, -0.25))

    # ── Plot 1: Expected hour (linear vs torus overlaid) ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ts, exp_hour_lin, "b--", lw=2.5, label="Linear steering")
    ax.plot(ts, exp_hour_tor, "g-",  lw=2.5, label="Torus steering")
    ax.axhline(SRC_HOUR, color="gray", ls=":", lw=1.5, alpha=0.7, label=f"Source: {SRC_HOUR:02d}h")
    ax.axhline(TGT_HOUR, color="red",  ls=":", lw=1.5, alpha=0.7, label=f"Target: {TGT_HOUR:02d}h")
    ax.set_yticks(list(range(7, 18)))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(7, 18)])
    ax.set_xlabel("Steering progress  t", fontsize=12)
    ax.set_ylabel("Expected hour (subspace)", fontsize=12)
    ax.set_title(f"Hour steering: expected time in V_hour  |  {title_base}", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out1 = RESULTS_DIR / "subspace_hour_steering.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out1}")

    # ── Plot 2: Day subspace probs ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    fig.suptitle(f"Day probabilities (subspace centroids)  |  {title_base}",
                 fontsize=11, fontweight="bold")
    day_prob_panel(axes[0], ts, day_probs_lin, DAYS, DAY_COLORS, "Linear Steering")
    day_prob_panel(axes[1], ts, day_probs_tor, DAYS, DAY_COLORS, "Torus Steering")
    axes[0].set_ylabel("P(day | day subspace)", fontsize=11)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    out2 = RESULTS_DIR / "subspace_day_steering.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out2}")

    # ── Plot 3: Combined 2×2 ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"Hierarchical Steering (subspace centroids)  —  {title_base}",
                 fontsize=12, fontweight="bold")

    # Top row: day probs side-by-side
    ax00 = fig.add_subplot(2, 2, 1)
    ax01 = fig.add_subplot(2, 2, 2, sharey=ax00)
    day_prob_panel(ax00, ts, day_probs_lin, DAYS, DAY_COLORS, "Day probs — Linear Steering")
    day_prob_panel(ax01, ts, day_probs_tor, DAYS, DAY_COLORS, "Day probs — Torus Steering")
    ax00.set_ylabel("P(day)", fontsize=10)

    # Bottom row: expected hour (spans both columns)
    ax_bot = fig.add_subplot(2, 1, 2)
    ax_bot.plot(ts, exp_hour_lin, "b--", lw=2.5, label="Linear steering")
    ax_bot.plot(ts, exp_hour_tor, "g-",  lw=2.5, label="Torus steering")
    ax_bot.axhline(SRC_HOUR, color="gray", ls=":", lw=1.5, alpha=0.7, label=f"Source {SRC_HOUR:02d}:00")
    ax_bot.axhline(TGT_HOUR, color="red",  ls=":", lw=1.5, alpha=0.7, label=f"Target {TGT_HOUR:02d}:00")
    ax_bot.set_yticks(list(range(7, 18)))
    ax_bot.set_yticklabels([f"{h:02d}:00" for h in range(7, 18)])
    ax_bot.set_xlabel("Steering progress  t", fontsize=11)
    ax_bot.set_ylabel("Expected hour (subspace)", fontsize=11)
    ax_bot.set_title("Hour steering: expected time in V_hour  (both paths overlaid)", fontsize=11)
    ax_bot.set_xlim(0, 1)
    ax_bot.legend(fontsize=10, ncol=4)
    ax_bot.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out3 = RESULTS_DIR / "subspace_combined.png"
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out3}")

    # ── Plot 4: Figure-6(c)-style 2D grid snapshots ──────────────────────────
    # At each of N_SNAP steering steps, show joint P(day, hour) as a small grid.
    # Joint probability: independent assumption (day ⊥ hour given V orthogonality).
    N_SNAP    = 6
    snap_idx  = np.round(np.linspace(0, N_STEPS - 1, N_SNAP)).astype(int)
    snap_ts   = ts[snap_idx]

    hour_window = list(range(7, 18))   # 11 hours shown in grid rows
    n_h = len(hour_window)
    n_d = 7

    day_labels_short  = ["M", "Tu", "W", "Th", "F", "Sa", "Su"]

    def joint_grid(day_p, hour_p):
        """Outer-product joint distribution over (day, hour_window)."""
        hp = hour_p[hour_window]           # (n_h,)
        return np.outer(hp, day_p)         # (n_h, n_d)  row=hour, col=day

    # Source/target positions in the grid
    src_col = SRC_DAY                                   # Tuesday = 1
    src_row = hour_window.index(SRC_HOUR)               # 09h
    tgt_col = TGT_DAY                                   # Friday  = 4
    tgt_row = hour_window.index(TGT_HOUR)               # 15h

    fig, axes = plt.subplots(
        2, N_SNAP,
        figsize=(N_SNAP * 1.6 + 0.5, 4.5),
        gridspec_kw={"hspace": 0.35, "wspace": 0.12},
    )
    fig.suptitle(
        f"Day × Hour probability grid  |  {title_base}",
        fontsize=12, fontweight="bold",
    )

    vmax = 0.0
    grids_lin = []
    grids_tor = []
    for si in snap_idx:
        g_lin = joint_grid(day_probs_lin[si], hour_probs_lin[si])
        g_tor = joint_grid(day_probs_tor[si], hour_probs_tor[si])
        grids_lin.append(g_lin)
        grids_tor.append(g_tor)
        vmax = max(vmax, g_lin.max(), g_tor.max())

    row_labels = ["Torus", "Linear"]
    for row, (grids, row_label) in enumerate(zip([grids_tor, grids_lin], row_labels)):
        for col, (g, t_val) in enumerate(zip(grids, snap_ts)):
            ax = axes[row, col]
            ax.imshow(g, aspect="auto", cmap="RdBu_r", vmin=0, vmax=vmax,
                      origin="upper", interpolation="nearest")

            # Mark source (white circle) and target (red star)
            ax.plot(src_col, src_row, "o", color="white", ms=7, mew=1.5)
            ax.plot(tgt_col, tgt_row, "*", color="gold",  ms=9, mew=0.8)

            ax.set_xticks(range(n_d))
            ax.set_xticklabels(day_labels_short, fontsize=6)
            if col == 0:
                ax.set_yticks(range(0, n_h, 2))
                ax.set_yticklabels([f"{hour_window[i]:02d}h"
                                    for i in range(0, n_h, 2)], fontsize=6)
                ax.set_ylabel(row_label, fontsize=9, fontweight="bold")
            else:
                ax.set_yticks([])

            if row == 0:
                ax.set_title(f"t={t_val:.2f}", fontsize=8)
            ax.tick_params(length=2)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap="RdBu_r",
                                norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:, -1], fraction=0.08, pad=0.04,
                 label="P(day, hour)")

    # Legend for markers
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor="gray", ms=7, label=f"Source ({DAYS[SRC_DAY][:3]} {SRC_HOUR:02d}h)"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="gold",
               markeredgecolor="gray", ms=9, label=f"Target ({DAYS[TGT_DAY][:3]} {TGT_HOUR:02d}h)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=2,
               fontsize=8, bbox_to_anchor=(0.45, -0.02))

    out_grid = RESULTS_DIR / "subspace_2d_grid.png"
    plt.savefig(out_grid, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_grid}")

    # ── Plot 5 (old Plot 4): Heatmap — all hours side-by-side ────────────────
    # Show P(each hour 0-23) as colour along t, for both paths
    window_all = list(range(6, 19))   # 06–18h for readability
    heat_lin = hour_probs_lin[:, window_all].T   # (13, N)
    heat_tor = hour_probs_tor[:, window_all].T

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    fig.suptitle(f"Hour probability heatmap (06–18h)  |  {title_base}",
                 fontsize=11, fontweight="bold")
    for ax, heat, path_label in zip(axes, [heat_lin, heat_tor],
                                    ["Linear Steering", "Torus Steering"]):
        im = ax.imshow(heat, aspect="auto", origin="lower",
                       extent=[0, 1, 5.5, 18.5],
                       cmap="viridis", vmin=0, vmax=heat.max())
        ax.axhline(SRC_HOUR, color="white", lw=1.5, ls="--", alpha=0.8)
        ax.axhline(TGT_HOUR, color="red",   lw=1.5, ls="--", alpha=0.8)
        ax.set_yticks(window_all)
        ax.set_yticklabels([f"{h:02d}:00" for h in window_all])
        ax.set_xlabel("Steering progress  t", fontsize=10)
        ax.set_title(path_label, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(hour)")
    axes[0].set_ylabel("Hour", fontsize=11)
    # Legend for reference lines
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[
        Line2D([0],[0], color="white", lw=1.5, ls="--", label=f"Source {SRC_HOUR:02d}:00"),
        Line2D([0],[0], color="red",   lw=1.5, ls="--", label=f"Target {TGT_HOUR:02d}:00"),
    ], fontsize=9, loc="upper right")
    plt.tight_layout()
    out4 = RESULTS_DIR / "subspace_hour_heatmap.png"  # noqa: E501
    plt.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out4}")

    # ── Plot 6: Per-hour probability lines (07–17h) ──────────────────────────
    # One colored line per hour, side-by-side Linear vs Torus.
    hour_line_window = list(range(7, 18))   # 11 hours
    cmap_h = plt.cm.plasma
    hour_colors = [cmap_h(i / (len(hour_line_window) - 1))
                   for i in range(len(hour_line_window))]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    fig.suptitle(
        f"Per-hour probabilities (07–17h)  |  {title_base}",
        fontsize=11, fontweight="bold",
    )

    for ax, h_probs, path_label in zip(
        axes,
        [hour_probs_lin, hour_probs_tor],
        ["Linear Steering", "Torus Steering"],
    ):
        for i, (h, col) in enumerate(zip(hour_line_window, hour_colors)):
            lw  = 2.5 if h in (SRC_HOUR, TGT_HOUR) else 1.4
            ls  = "-"  if h == TGT_HOUR else ("--" if h == SRC_HOUR else "-")
            ax.plot(ts, h_probs[:, h], color=col, lw=lw, ls=ls,
                    label=f"{h:02d}:00")
        ax.axvline(0, color="gray", ls=":", lw=1.0, alpha=0.5)
        ax.axvline(1, color="gray", ls=":", lw=1.0, alpha=0.5)
        ax.set_title(path_label, fontsize=11)
        ax.set_xlabel("Steering progress  t", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("P(hour | V_hour subspace)", fontsize=11)
    # Legend on the right panel
    axes[1].legend(
        fontsize=7, loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, -0.20),
        title="Hour", title_fontsize=8,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_hlines = RESULTS_DIR / "subspace_hour_lines.png"
    plt.savefig(out_hlines, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_hlines}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    print(f"\n── Metrics ──────────────────────────────────────────────────────")
    for label, dp, exp_h in [("Linear", day_probs_lin, exp_hour_lin),
                              ("Torus",  day_probs_tor, exp_hour_tor)]:
        src_day_p = dp[0,  SRC_DAY]
        tgt_day_p = dp[-1, TGT_DAY]
        peak_seq  = [DAYS[np.argmax(dp[i])] for i in range(N_STEPS)]
        ordered   = []
        for d in peak_seq:
            if not ordered or d != ordered[-1]:
                ordered.append(d)
        print(f"  {label}:")
        print(f"    Source: day {DAYS[SRC_DAY]}={src_day_p:.3f}  hour={exp_h[0]:.2f}h")
        print(f"    Target: day {DAYS[TGT_DAY]}={tgt_day_p:.3f}  hour={exp_h[-1]:.2f}h  (target={TGT_HOUR}h)")
        print(f"    Day peak sequence: {' → '.join(ordered)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
