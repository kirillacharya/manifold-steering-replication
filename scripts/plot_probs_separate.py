"""Two separate figures: weekdays_probs.png and months_probs.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

LAYER   = 24
N_STEPS = 30
ts      = np.linspace(0., 1., N_STEPS)

DAYS       = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
DAY_COLORS = ["#e6194b","#f58231","#ffe119","#bfef45","#3cb44b","#42d4f4","#4363d8"]

MONTHS       = ["January","February","March","April","May","June",
                "July","August","September","October","November","December"]
MONTH_COLORS = ["#e6194b","#f58231","#ffe119","#bfef45","#3cb44b","#42d4f4",
                "#4363d8","#911eb4","#f032e6","#a9a9a9","#9A6324","#000075"]

_sp         = np.load(f"checkpoints/steering_probs_L{LAYER}.npz")
lin_probs_w = _sp["wd_p_lin"]
mfd_probs_w = _sp["wd_p_mfd"]
lin_probs_m = _sp["mo_p_lin"]
mfd_probs_m = _sp["mo_p_mfd"]


def draw_prob_panel(ax, probs, concepts, colors, ylabel=None, show_xticks=True):
    for i, (name, col) in enumerate(zip(concepts, colors)):
        ax.plot(ts, probs[:, i], color=col, lw=3.0, label=name)
    ax.plot(ts, probs[:, -1], color="gray", lw=1.8, ls="--", alpha=0.5, label="Other")
    ax.axvline(0, color="black", lw=1.2, ls=":")
    ax.axvline(1, color="black", lw=1.2, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.15)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
    if show_xticks:
        ax.set_xlabel("Steering parameter $t$", fontsize=13, fontweight="bold")
    else:
        plt.setp(ax.get_xticklabels(), visible=False)


def make_probs_figure(lin_probs, mfd_probs, concepts, colors, ncol, outpath):
    fig, (ax_mfd, ax_lin) = plt.subplots(2, 1, figsize=(9, 6),
                                          gridspec_kw={"hspace": 0.35})

    draw_prob_panel(ax_mfd, mfd_probs, concepts, colors,
                    ylabel="P(token)", show_xticks=False)
    draw_prob_panel(ax_lin, lin_probs, concepts, colors,
                    ylabel="P(token)", show_xticks=True)

    # Row labels
    ax_mfd.set_title("Manifold Steering", fontsize=13, fontweight="bold", pad=6)
    ax_lin.set_title("Linear Steering",   fontsize=13, fontweight="bold", pad=6)

    # Shared legend below
    handles, labels = ax_lin.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               fontsize=11, bbox_to_anchor=(0.5, -0.08),
               framealpha=0.95, handlelength=1.8, handletextpad=0.5,
               borderpad=0.7, columnspacing=0.8, edgecolor="black",
               prop={"size": 11, "weight": "bold"})

    fig.savefig(outpath, dpi=150, bbox_inches="tight", pad_inches=0.25)
    print(f"Saved → {outpath}")
    plt.close(fig)


out_dir = Path("figures")
out_dir.mkdir(parents=True, exist_ok=True)

make_probs_figure(lin_probs_w, mfd_probs_w, DAYS,   DAY_COLORS,
                  ncol=8, outpath=out_dir / "weekdays_probs.png")
make_probs_figure(lin_probs_m, mfd_probs_m, MONTHS, MONTH_COLORS,
                  ncol=7, outpath=out_dir / "months_probs.png")
