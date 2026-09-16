# Why the co-active case has its density: the pair (x1, x3) is uniform over the unit square, the
# reading s = a x1 - b x3 is constant along parallel lines, and the length of a line's crossing of
# the square, normalized, is the density rho_c(s). For the symmetric pair (a = b = 1) the crossings
# are the diagonals and the density is the triangle 1 - |s|; for the asymmetric pair (b / a = 4.25,
# as in asym_compare.py) it is a trapezoid, flat where the crossings span the square's full width.
# Run from the repo root: python -m figures.appendix_square
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import BLUE_RAMP, INK, ORANGE

RATIO = 4.25
FIGS = (("figures/appendix_square", 1.0, 1.0, (-0.75, -0.25, 0.0, 0.5, 0.75),
         "$s = x_1 - x_3$", "$\\rho_c(s) = 1 - |s|$", "diagonal"),
        ("figures/appendix_square_asym", RATIO ** -0.5, RATIO ** 0.5, (-1.8, -0.8, 0.0, 0.25, 0.45),
         "$s = a x_1 - b x_3$", "$\\rho_c(s)$", "crossing"))
COLORS = [BLUE_RAMP[0], BLUE_RAMP[1], ORANGE, BLUE_RAMP[2], BLUE_RAMP[3]]

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
for out, a, b, S, xlabel, ylabel, word in FIGS:
    rho = lambda s: np.minimum(np.minimum((s + b) / a, (a - s) / a), 1) / b
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
    ax = axes[0]
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, color=INK, lw=1.2))
    for s, c in zip(S, COLORS):  # the segment a x1 - b x3 = s inside the square, over the admissible x3
        x3 = np.array([max(0.0, -s / b), min(1.0, (a - s) / b)])
        ax.plot((s + b * x3) / a, x3, color=c, lw=2.6)
        ax.annotate(f"$s = {s:g}$", (((s + b * x3) / a).mean(), x3.mean()), xytext=(6, -6), textcoords="offset points", color=c, fontsize=10)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_3$")
    ax.set_title("both active: the pair $(x_1, x_3)$ is uniform on the unit square", fontsize=11)
    ax = axes[1]
    ss = np.linspace(-b, a, 401)
    ax.plot(ss, rho(ss), color=INK, lw=2.2)
    for s, c in zip(S, COLORS):
        ax.plot([s, s], [0, rho(s)], color=c, lw=2.6)
        ax.plot([s], [rho(s)], "o", color=c, ms=7)
    ax.set_xlim(-b - 0.025 * (a + b), a + 0.025 * (a + b))
    ax.set_ylim(0, 1.08 / b)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"the length of the {word}, normalized: the density of the reading", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(Path(out).with_suffix(".pdf"))
    fig.savefig(Path(out).with_suffix(".png"), dpi=200)
