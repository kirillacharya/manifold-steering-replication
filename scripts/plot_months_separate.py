"""Two separate figures: months_behavior.png and months_activation.png"""
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

MONTHS       = ["January","February","March","April","May","June",
                "July","August","September","October","November","December"]
MONTH_COLORS = ["#e6194b","#f58231","#ffe119","#bfef45","#3cb44b","#42d4f4",
                "#4363d8","#911eb4","#f032e6","#a9a9a9","#9A6324","#000075"]
MO_START, MO_END, MO_N_PCA = 0, 6, 11

cp_m    = np.load(f"checkpoints/months_L{LAYER}_acts_probs.npz")
act_cm  = np.array([cp_m[f"acts_{m}"].mean(0) for m in MONTHS])
prob_cm = np.array([cp_m[f"probs_{m}"].mean(0) for m in MONTHS])

pca_m   = PCA(n_components=MO_N_PCA)
pca_cm  = pca_m.fit_transform(act_cm)
spl_m   = CubicSpline(np.arange(len(MONTHS), dtype=float), pca_cm)

ts         = np.linspace(0., 1., N_STEPS)
t_path_m   = np.linspace(float(MO_START), float(MO_END), N_STEPS)
path_lin_m = np.array([(1-t)*act_cm[MO_START] + t*act_cm[MO_END] for t in ts])
path_mfd_m = pca_m.inverse_transform(spl_m(t_path_m))
t_dense_m  = np.linspace(0., float(len(MONTHS)-1), 300)

# Activation space
pca3_act_m  = PCA(n_components=3).fit(act_cm)
act_3d_m    = pca3_act_m.transform(act_cm)
act_curve_m = pca3_act_m.transform(pca_m.inverse_transform(spl_m(t_dense_m)))
lin_act3_m  = pca3_act_m.transform(path_lin_m)
mfd_act3_m  = pca3_act_m.transform(path_mfd_m)

# Behavior space
hell_cm     = np.sqrt(np.clip(prob_cm, 0., None))
pca3_beh_m  = PCA(n_components=3).fit(hell_cm)
beh_3d_m    = pca3_beh_m.transform(hell_cm)
beh_curve_m = CubicSpline(np.arange(len(MONTHS), dtype=float),
                           pca3_beh_m.transform(hell_cm))(t_dense_m)
hell_lin_m  = np.sqrt(np.clip(
    np.array([(1-t)*prob_cm[MO_START]+t*prob_cm[MO_END] for t in ts]), 0., None))
hell_mfd_m  = np.sqrt(np.clip(
    np.array([[np.interp(t_path_m[i], np.arange(len(MONTHS)), prob_cm[:,j])
               for j in range(prob_cm.shape[1])] for i in range(N_STEPS)]), 0., None))
lin_beh3_m  = pca3_beh_m.transform(hell_lin_m)
mfd_beh3_m  = pca3_beh_m.transform(hell_mfd_m)


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

    for i, (name, col) in enumerate(zip(MONTHS, MONTH_COLORS)):
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
draw_panel(ax1, beh_3d_m, beh_curve_m, lin_beh3_m, mfd_beh3_m)
fig1.savefig(out_dir / "months_behavior.png", dpi=150,
             bbox_inches="tight", pad_inches=0.35,
             bbox_extra_artists=[ax1.zaxis.label])
print("Saved → figures/months_behavior.png")
plt.close(fig1)

# ── Figure 2: Activation Space ────────────────────────────────────────────────
fig2 = plt.figure(figsize=(9, 7))
ax2  = fig2.add_axes([0.02, 0.02, 0.68, 0.94], projection="3d")
draw_panel(ax2, act_3d_m, act_curve_m, lin_act3_m, mfd_act3_m)
fig2.savefig(out_dir / "months_activation.png", dpi=150,
             bbox_inches="tight", pad_inches=0.35,
             bbox_extra_artists=[ax2.zaxis.label])
print("Saved → figures/months_activation.png")
plt.close(fig2)
