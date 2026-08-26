# Symmetric vs asymmetric antipodal pair: best decoders for both (closed form) and the
# minimal MSE each achieves, split into the x1 and x3 contributions.
# Asymmetric = |f3| / |f1| = RATIO (long f3, short f1); s = |f1| x1 - |f3| x3.
# Run from the repo root: python -m figures.asym_compare
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import BLUE, MUTED, ORANGE, P, arms, posterior_mean, x1_posterior, x3_posterior
from figures.posterior_sym import frame

OUT = Path("figures/asym_compare")
RATIO, M, SEED = 4.25, 2 ** 19, 7
CASES = (("symmetric", 1.0, 1.0), ("asymmetric", RATIO ** 0.5, RATIO ** -0.5))  # (name, |f3| = a, |f1| = b)


def bill(a, b):
    g = torch.Generator().manual_seed(SEED)
    x1 = (torch.rand(M, generator=g) < P) * torch.rand(M, generator=g)
    x3 = (torch.rand(M, generator=g) < P) * torch.rand(M, generator=g)
    s = b * x1 - a * x3
    e1 = torch.nan_to_num(posterior_mean(s, b, -a, P), nan=0.0)
    e3 = torch.nan_to_num(posterior_mean(s, -a, b, P), nan=0.0)
    e1[s == 0], e3[s == 0] = 0.0, 0.0
    return ((x1 - e1) ** 2).mean().item(), ((x3 - e3) ** 2).mean().item()


def curves(ax, a, b):
    limits = []
    for s in arms(a, b):
        for f, color in ((x1_posterior, BLUE), (x3_posterior, ORANGE)):
            y = f(s, a, b)
            ax.plot(s, y, color=color, lw=2.5, zorder=3)
            limits.append((round(float(y[np.abs(s).argmin()]), 4), color))
    for v in {v for v, _ in limits}:  # one-sided limits at s = 0: open circles, black where the curves coincide
        colors = {c for w, c in limits if w == v}
        ax.plot([0], [v], "o", ms=6.5, mfc="white", mec=colors.pop() if len(colors) == 1 else "black", mew=1.6, zorder=4)
    ax.plot([0], [0], "o", ms=6.5, color="black", zorder=5)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), layout="constrained", width_ratios=(1.5, 1.5, 1))
    bills = {}
    for ax, (name, a, b) in zip(axes, CASES):
        curves(ax, a, b)
        frame(ax, marks=False, lo=-a, hi=b, xticks=np.round(np.arange(-np.floor(a * 2) / 2, b + 1e-9, 0.5), 2))
        ax.set_title(name)
        bills[name] = bill(a, b)
    axes[0].set_ylabel("predicted feature")
    axes[0].legend(handles=[plt.Line2D([], [], color=BLUE, lw=2.5, label="$\\hat{x}_1(s)$"),
                            plt.Line2D([], [], color=ORANGE, lw=2.5, label="$\\hat{x}_3(s)$")],
                   loc="upper center", frameon=False)
    axb = axes[2]
    for i, (name, _, _) in enumerate(CASES):
        m1, m3 = bills[name]
        axb.bar(i, m1, 0.55, color=BLUE, label="$x_1$" if i == 0 else None)
        axb.bar(i, m3, 0.55, bottom=m1, color=ORANGE, label="$x_3$" if i == 0 else None)
        axb.text(i, m1 + m3 + 0.0003, f"{m1 + m3:.4f}", ha="center", va="bottom")
        print(f"{name}: x1 {m1:.4f} + x3 {m3:.4f} = {m1 + m3:.4f}")
    axb.set_xticks(range(len(CASES)), [c[0] for c in CASES])
    axb.set_xlim(-0.6, len(CASES) - 0.4)
    axb.set_ylim(0, 1.6 * max(sum(v) for v in bills.values()))
    axb.set_ylabel("MSE of the best decoder")
    axb.legend(loc="upper right", frameon=False)
    axb.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=200)
