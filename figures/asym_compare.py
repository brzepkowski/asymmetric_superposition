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
    """The MSE the best possible decoder (the closed-form
    posterior) achieves on the pair with lever lengths |f3| = a, |f1| = b, evaluated by
    Monte Carlo on M samples, returned as the (x1, x3) contributions separately. The fixed
    seed gives every call identical draws, so the cases compare without sampling noise.
    At s = 0 the prediction is overridden to 0: a sampled reading of exactly 0 almost
    always means neither feature was active (the appendix's point mass).
    """
    g = torch.Generator().manual_seed(SEED)
    x1 = (torch.rand(M, generator=g) < P) * torch.rand(M, generator=g)
    x3 = (torch.rand(M, generator=g) < P) * torch.rand(M, generator=g)
    s = b * x1 - a * x3
    e1 = torch.nan_to_num(posterior_mean(s, b, -a, P), nan=0.0)
    e3 = torch.nan_to_num(posterior_mean(s, -a, b, P), nan=0.0)
    e1[s == 0], e3[s == 0] = 0.0, 0.0
    return ((x1 - e1) ** 2).mean().item(), ((x3 - e3) ** 2).mean().item()


def curves(ax, a, b):
    # `limits` collects the data for drawing the open circles at s = 0 — the markers of the jump
    # discontinuity that both posterior curves have there. Mechanically, inside the double loop below
    # each iteration handles one curve on one arm (2 curves x 2 arms = 4 entries).
    # y[np.abs(s).argmin()] picks the curve's value at the grid point nearest s = 0 — since
    # arms() ends its grids 1e-6 short of zero, that's numerically the one-sided limit. Each
    # entry appended to limits is a pair (limit value rounded to 4 decimals, curve color)
    limits = []
    for s in arms(a, b):
        # the numpy posteriors, since the grids are numpy (bill() scores torch samples, so it uses posterior_mean)
        for f, color in ((x1_posterior, BLUE), (x3_posterior, ORANGE)):
            y = f(s, a, b)
            ax.plot(s, y, color=color, lw=2.5, zorder=3)
            limits.append((round(float(y[np.abs(s).argmin()]), 4), color))
    # one open circle per distinct limit value; its edge takes the curve's color, or black
    # when both curves share the same limit
    for value in {value for value, _ in limits}:
        curve_colors = {color for limit, color in limits if limit == value}
        edge_color = curve_colors.pop() if len(curve_colors) == 1 else "black"
        ax.plot([0], [value], "o", ms=6.5, mfc="white", mec=edge_color, mew=1.6, zorder=4)
    ax.plot([0], [0], "o", ms=6.5, color="black", zorder=5)


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
    fig, axd = plt.subplot_mosaic([["symmetric", "symmetric", "asymmetric", "asymmetric"],
                                   [".", "bars", "bars", "."]],
                                  figsize=(9, 6.6), layout="constrained")
    bills = {}
    for name, a, b in CASES:
        ax = axd[name]
        curves(ax, a, b)
        frame(ax, marks=False, lo=-a, hi=b, xticks=np.round(np.arange(-np.floor(a * 2) / 2, b + 1e-9, 0.5), 2))
        ax.set_title(name)
        bills[name] = bill(a, b)
    axd["symmetric"].set_ylabel("predicted feature")
    axd["symmetric"].legend(handles=[plt.Line2D([], [], color=BLUE, lw=2.5, label="$\\hat{x}_1(s)$"),
                                     plt.Line2D([], [], color=ORANGE, lw=2.5, label="$\\hat{x}_3(s)$")],
                            loc="upper center", frameon=False)
    axb = axd["bars"]
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
