# What each decoder class makes of the symmetric and the asymmetric pair: the best fit from
# the class (quadratic = 1 bilinear MLP, degree 16 = 4 bilinear MLPs, tied ReLU) drawn over
# the unconstrained closed-form decoders, and its MSE per feature against the unconstrained floor.
# Run from the repo root: python -m figures.class_compare
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import BLUE, MUTED, ORANGE, P, arms, class_fit, x1_posterior, x3_posterior
from figures.asym_compare import CASES, M, SEED, bill
from figures.posterior_sym import frame

OUT = Path("figures")
KINDS = (("quadratic", "single bilinear MLP (quadratic decoder)"),
         ("deg16", "four bilinear MLPs (degree-16 decoder)"),
         ("relu_tied", "tied ReLU model"))


def fit(kind, a, b):
    g = torch.Generator().manual_seed(SEED)
    x1 = (torch.rand(M, generator=g) < P) * torch.rand(M, generator=g)
    x3 = (torch.rand(M, generator=g) < P) * torch.rand(M, generator=g)
    s = b * x1 - a * x3
    grid = torch.linspace(-a + 1e-4, b - 1e-4, 400)
    if kind == "relu_tied":  # class_fit ties the slopes as (x1: a, x3: b); here f1 is the short one, so mirror
        F, mse = class_fit(kind, -s, torch.stack([x3, x1], 1), a, b, -grid.flip(0))
        return grid, F.flip(0).flip(1), mse.flip(0)
    F, mse = class_fit(kind, s, torch.stack([x1, x3], 1), a, b, grid)
    return grid, F, mse


plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
for kind, title in KINDS:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), layout="constrained", width_ratios=(1.5, 1.5, 1))
    mses, floors = {}, {}
    for ax, (name, a, b) in zip(axes, CASES):
        for t in arms(a, b):
            ax.plot(t, x1_posterior(t, a, b), color=MUTED, lw=1.4, zorder=2)
            ax.plot(t, x3_posterior(t, a, b), color=MUTED, lw=1.4, zorder=2)
        grid, F, mse = fit(kind, a, b)
        ax.plot(grid, F[:, 0], color=BLUE, lw=2.5, zorder=3)
        ax.plot(grid, F[:, 1], color=ORANGE, lw=2.5, zorder=3)
        frame(ax, marks=False, lo=-a, hi=b, xticks=np.round(np.arange(-np.floor(a * 2) / 2, b + 1e-9, 0.5), 2))
        ax.set_ylim(min(-0.04, F.min().item() - 0.05), 1.04)
        ax.set_title(name)
        mses[name], floors[name] = (mse[0].item(), mse[1].item()), sum(bill(a, b))
    axes[0].set_ylabel("predicted feature")
    axes[0].legend(handles=[plt.Line2D([], [], color=BLUE, lw=2.5, label="$\\hat{x}_1'(s)$ from the class"),
                            plt.Line2D([], [], color=ORANGE, lw=2.5, label="$\\hat{x}_3'(s)$ from the class"),
                            plt.Line2D([], [], color=MUTED, lw=1.4, label="unconstrained decoder")],
                   loc="upper center", frameon=False, fontsize=10)
    axb = axes[2]
    for i, (name, _, _) in enumerate(CASES):
        m1, m3 = mses[name]
        axb.bar(i, m1, 0.55, color=BLUE, label="$x_1$" if i == 0 else None)
        axb.bar(i, m3, 0.55, bottom=m1, color=ORANGE, label="$x_3$" if i == 0 else None)
        axb.text(i, m1 + m3 + 0.0003, f"{m1 + m3:.4f}", ha="center", va="bottom")
        axb.plot([i - 0.34, i + 0.34], [floors[name]] * 2, color="black", lw=1.4, ls="--",
                 label="unconstrained floor" if i == 0 else None)
        print(f"{kind} {name}: x1 {m1:.4f} + x3 {m3:.4f} = {m1 + m3:.4f}  (floor {floors[name]:.4f})")
    axb.set_xticks(range(len(CASES)), [c[0] for c in CASES])
    axb.set_xlim(-0.6, len(CASES) - 0.4)
    axb.set_ylim(0, 1.6 * max(sum(v) for v in mses.values()))
    axb.set_ylabel("MSE of the best decoder in the class")
    axb.legend(loc="upper right", frameon=False, fontsize=10)
    axb.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.savefig(OUT / f"class_{kind}.pdf")
    fig.savefig(OUT / f"class_{kind}.png", dpi=200)
