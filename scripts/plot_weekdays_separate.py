"""Two separate figures: weekdays_behavior.png and weekdays_activation.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA

LAYER   = 24
N_STEPS = 30

DAYS       = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
DAY_COLORS = ["#e6194b","#f58231","#ffe119","#bfef45","#3cb44b","#42d4f4","#4363d8"]
DAY_START, DAY_END, DAY_N_PCA = 1, 4, 6

cp_w   = np.load(f"checkpoints/weekdays_L{LAYER}_acts_probs.npz")
act_cw = np.array([cp_w[f"acts_{d}"].mean(0) for d in DAYS])
prob_cw= np.array([cp_w[f"probs_{d}"].mean(0) for d in DAYS])

pca_w  = PCA(n_components=DAY_N_PCA)
pca_cw = pca_w.fit_transform(act_cw)
spl_w  = CubicSpline(np.arange(len(DAYS), dtype=float), pca_cw)

ts       = np.linspace(0., 1., N_STEPS)
t_path_w = np.linspace(float(DAY_START), float(DAY_END), N_STEPS)
path_lin_w = np.array([(1-t)*act_cw[DAY_START] + t*act_cw[DAY_END] for t in ts])
path_mfd_w = pca_w.inverse_transform(spl_w(t_path_w))

t_dense_w  = np.linspace(0., float(len(DAYS)-1), 300)

# Activation space
pca3_act_w = PCA(n_components=3).fit(act_cw)
act_3d_w   = pca3_act_w.transform(act_cw)
act_curve_w= pca3_act_w.transform(pca_w.inverse_transform(spl_w(t_dense_w)))
lin_act3_w = pca3_act_w.transform(path_lin_w)
mfd_act3_w = pca3_act_w.transform(path_mfd_w)

# Behavior space
hell_cw    = np.sqrt(np.clip(prob_cw, 0., None))
pca3_beh_w = PCA(n_components=3).fit(hell_cw)
beh_3d_w   = pca3_beh_w.transform(hell_cw)
beh_curve_w= CubicSpline(np.arange(len(DAYS), dtype=float),
                          pca3_beh_w.transform(hell_cw))(t_dense_w)
hell_lin_w = np.sqrt(np.clip(
    np.array([(1-t)*prob_cw[DAY_START]+t*prob_cw[DAY_END] for t in ts]), 0., None))
hell_mfd_w = np.sqrt(np.clip(
    np.array([[np.interp(t_path_w[i], np.arange(len(DAYS)), prob_cw[:,j])
               for j in range(prob_cw.shape[1])] for i in range(N_STEPS)]), 0., None))
lin_beh3_w = pca3_beh_w.transform(hell_lin_w)
mfd_beh3_w = pca3_beh_w.transform(hell_mfd_w)


def draw_panel(ax, pts3, curve3, lin3, mfd3, elev=22, azim=-55):
    z_vals  = np.concatenate([pts3[:,2], curve3[:,2], lin3[:,2], mfd3[:,2]])
    z_floor = z_vals.min() - 0.12*(z_vals.max()-z_vals.min())

    x_all = np.concatenate([pts3[:,0], curve3[:,0]])
    y_all = np.concatenate([pts3[:,1], curve3[:,1]])
    px = 0.1*(x_all.max()-x_all.min()); py = 0.1*(y_all.max()-y_all.min())
    xx, yy = np.meshgrid([x_all.min()-px, x_all.max()+px],
                          [y_all.min()-py, y_all.max()+py])
    ax.plot_surface(xx, yy, np.full_like(xx, z_floor),
                    alpha=0.12, color="gray", linewidth=0, zorder=0)

    ax.plot(curve3[:,0], curve3[:,1], curve3[:,2],
            color="steelblue", lw=3.5, alpha=0.65, zorder=3)
    ax.plot(curve3[:,0], curve3[:,1], np.full(len(curve3), z_floor),
            color="steelblue", lw=1.2, alpha=0.20, zorder=1)

    ax.plot(lin3[:,0], lin3[:,1], lin3[:,2],
            color="#888888", lw=4.5, ls="--", zorder=4)
    ax.plot(lin3[:,0], lin3[:,1], np.full(len(lin3), z_floor),
            color="#888888", lw=1.2, ls="--", alpha=0.25, zorder=1)

    ax.plot(mfd3[:,0], mfd3[:,1], mfd3[:,2],
            color="black", lw=3.5, zorder=5)
    ax.plot(mfd3[:,0], mfd3[:,1], np.full(len(mfd3), z_floor),
            color="black", lw=1.0, alpha=0.18, zorder=1)

    for i, (name, col) in enumerate(zip(DAYS, DAY_COLORS)):
        ax.scatter(*pts3[i], color=col, s=100, marker="D",
                   edgecolors="k", lw=0.7, zorder=6)
        ax.scatter(pts3[i,0], pts3[i,1], z_floor,
                   color=col, s=30, marker="D", alpha=0.25, zorder=1)
        ax.text(pts3[i,0], pts3[i,1], pts3[i,2],
                f" {name[:3]}", fontsize=9, fontweight="bold", zorder=7, va="bottom")

    ax.set_xlabel("PC 1", fontsize=13, labelpad=5, fontweight="bold")
    ax.set_ylabel("PC 2", fontsize=13, labelpad=5, fontweight="bold")
    ax.set_zlabel("PC 3", fontsize=13, labelpad=8, fontweight="bold")
    ax.zaxis.label.set_clip_on(False)
    ax.tick_params(labelsize=10, pad=2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1.0, 1.0, 0.6])


out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)

# ── Figure 1: Behavior Space ──────────────────────────────────────────────────
fig1 = plt.figure(figsize=(9, 7))
ax1  = fig1.add_axes([0.02, 0.02, 0.68, 0.94], projection="3d")
draw_panel(ax1, beh_3d_w, beh_curve_w, lin_beh3_w, mfd_beh3_w)
fig1.savefig(out_dir / "weekdays_behavior.png", dpi=150,
             bbox_inches="tight", pad_inches=0.35,
             bbox_extra_artists=[ax1.zaxis.label])
print("Saved → figures/weekdays_behavior.png")
plt.close(fig1)

# ── Figure 2: Activation Space ────────────────────────────────────────────────
fig2 = plt.figure(figsize=(9, 7))
ax2  = fig2.add_axes([0.02, 0.02, 0.68, 0.94], projection="3d")
draw_panel(ax2, act_3d_w, act_curve_w, lin_act3_w, mfd_act3_w)
fig2.savefig(out_dir / "weekdays_activation.png", dpi=150,
             bbox_inches="tight", pad_inches=0.35,
             bbox_extra_artists=[ax2.zaxis.label])
print("Saved → figures/weekdays_activation.png")
plt.close(fig2)
