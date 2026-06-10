"""Render the day×hour torus figure (poster Fig. "torus_repr").

Loads figures/gemma_torus/acts_2b_L24.pt, maps day/hour angles onto a
mathematical torus surface, saves figures/gemma_torus/torus_3d_real.png
(and a PCA view, torus_3d_pca.png). No model inference needed.
"""

from pathlib import Path
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))

DAYS  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
DAY_COLORS = [
    "#e6194b","#3cb44b","#ffe119","#4363d8",
    "#f58231","#911eb4","#42d4f4",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def find_concept_subspace(acts: torch.Tensor, angles: np.ndarray) -> torch.Tensor:
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


def torus_embed(theta: np.ndarray, phi: np.ndarray,
                R: float = 2.0, r: float = 1.0):
    """Map (day angle θ, hour angle φ) → 3D torus coords."""
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = (R + r * np.cos(phi)) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z


def angle_from_proj(proj: np.ndarray) -> np.ndarray:
    """atan2 angle from 2D subspace projection, normalised to [0, 2π)."""
    a = np.arctan2(proj[:, 1], proj[:, 0])
    return a % (2 * np.pi)


# ── Part II: real Gemma ───────────────────────────────────────────────────────

def make_gemma_3d():
    cache = Path("figures/gemma_torus/acts_2b_L24.pt")
    print(f"Loading {cache} ...")
    acts = torch.load(cache, weights_only=False)   # (168, 2304)
    print(f"  shape: {acts.shape}")

    day_idx  = np.array([d for d in range(7) for _ in range(24)])
    hour_idx = np.array([h for _ in range(7) for h in range(24)])

    theta_day  = 2 * np.pi * day_idx  / 7
    theta_hour = 2 * np.pi * hour_idx / 24

    V_day  = find_concept_subspace(acts, theta_day)   # (2, d)
    V_hour = find_concept_subspace(acts, theta_hour)  # (2, d)

    proj_d = (acts @ V_day.T).numpy()    # (168, 2)
    proj_h = (acts @ V_hour.T).numpy()   # (168, 2)

    day_angle  = angle_from_proj(proj_d)
    hour_angle = angle_from_proj(proj_h)

    # ── Figure A: mathematical torus embedding ────────────────────────────────
    tx, ty, tz = torus_embed(day_angle, hour_angle, R=2.0, r=0.9)

    fig = plt.figure(figsize=(14, 5.5))
    fig.suptitle(
        "Gemma 2 2B layer-24 activations embedded on Day$\\times$Hour torus",
        fontsize=12, fontweight="bold",
    )

    for col_idx, (cvals, cmap, clabel, ctitle) in enumerate([
        (day_idx.astype(float),  None,       "Day",   "Coloured by day-of-week"),
        (hour_idx.astype(float), "plasma",   "Hour",  "Coloured by hour-of-day"),
    ]):
        ax = fig.add_subplot(1, 2, col_idx + 1, projection="3d")

        if cmap is None:
            # Discrete day colors
            for d, col in enumerate(DAY_COLORS):
                mask = day_idx == d
                ax.scatter(tx[mask], ty[mask], tz[mask],
                           c=col, s=35, alpha=0.85, label=DAYS[d][:3])
            ax.legend(fontsize=7, loc="upper left", ncol=2,
                      bbox_to_anchor=(-0.05, 1.0))
        else:
            sc = ax.scatter(tx, ty, tz, c=cvals, cmap=cmap,
                            s=35, alpha=0.85, vmin=0, vmax=23)
            plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=clabel)

        # Draw torus wireframe for reference
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, 2 * np.pi, 40)
        U, V = np.meshgrid(u, v)
        Wx = (2.0 + 0.9 * np.cos(V)) * np.cos(U)
        Wy = (2.0 + 0.9 * np.cos(V)) * np.sin(U)
        Wz = 0.9 * np.sin(V)
        ax.plot_wireframe(Wx, Wy, Wz, color="lightgray", alpha=0.12,
                          linewidth=0.4, rstride=4, cstride=4)

        ax.set_title(ctitle, fontsize=10)
        ax.set_xlabel("x", fontsize=8); ax.set_ylabel("y", fontsize=8)
        ax.set_zlabel("z", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=28, azim=55)

    plt.tight_layout()
    out_torus = Path("figures/gemma_torus/torus_3d_real.png")
    plt.savefig(out_torus, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_torus}")

    # ── Figure B: 3D PCA (same style as synthetic) ────────────────────────────
    pca = PCA(n_components=3)
    Z = pca.fit_transform(acts.numpy())
    var = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(13, 5.5))
    fig.suptitle(
        f"Gemma 2 2B layer-24 — 3D PCA  "
        f"(PC1={100*var[0]:.1f}%, PC2={100*var[1]:.1f}%, PC3={100*var[2]:.1f}%)",
        fontsize=12, fontweight="bold",
    )

    for col_idx, (cvals, cmap, clabel, ctitle) in enumerate([
        (day_idx.astype(float),  None,      "Day",   "Coloured by day-of-week"),
        (hour_idx.astype(float), "plasma",  "Hour",  "Coloured by hour-of-day"),
    ]):
        ax = fig.add_subplot(1, 2, col_idx + 1, projection="3d")
        if cmap is None:
            for d, col in enumerate(DAY_COLORS):
                mask = day_idx == d
                ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2],
                           c=col, s=35, alpha=0.85, label=DAYS[d][:3])
            ax.legend(fontsize=7, loc="upper left", ncol=2,
                      bbox_to_anchor=(-0.05, 1.0))
        else:
            sc = ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=cvals, cmap=cmap,
                            s=35, alpha=0.85, vmin=0, vmax=23)
            plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label=clabel)

        ax.set_title(ctitle, fontsize=10)
        ax.set_xlabel("PC1", fontsize=8); ax.set_ylabel("PC2", fontsize=8)
        ax.set_zlabel("PC3", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=28, azim=55)

    plt.tight_layout()
    out_pca = Path("figures/gemma_torus/torus_3d_pca.png")
    plt.savefig(out_pca, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out_pca}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    make_gemma_3d()
    print("\nDone.")
