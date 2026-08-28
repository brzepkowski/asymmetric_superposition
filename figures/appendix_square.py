# Why the co-active case has a triangular density: the pair (x1, x3) is uniform over the unit square,
# the reading s = x1 - x3 is constant along each diagonal of the square, and the length of that
# diagonal, longest at s = 0 and shrinking linearly to a corner at s = +-1, is the density rho_c(s).
# Run from the repo root: python -m figures.appendix_square
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import BLUE_RAMP, INK, ORANGE

OUT = Path("figures/appendix_square")
S = (-0.75, -0.25, 0.0, 0.5, 0.75)
COLORS = [BLUE_RAMP[0], BLUE_RAMP[1], ORANGE, BLUE_RAMP[2], BLUE_RAMP[3]]

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
ax = axes[0]
ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, color=INK, lw=1.2))
for s, c in zip(S, COLORS):  # the segment x1 - x3 = s inside the square, from x3 = max(0, -s) to x3 = min(1, 1 - s)
    x3 = np.array([max(0.0, -s), min(1.0, 1 - s)])
    ax.plot(x3 + s, x3, color=c, lw=2.6)
    ax.annotate(f"$s = {s:g}$", ((x3 + s).mean(), x3.mean()), xytext=(6, -6), textcoords="offset points", color=c, fontsize=10)
ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
ax.set_aspect("equal")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_3$")
ax.set_title("both active: the pair $(x_1, x_3)$ is uniform on the unit square", fontsize=11)
ax = axes[1]
ss = np.linspace(-1, 1, 201)
ax.plot(ss, 1 - np.abs(ss), color=INK, lw=2.2)
for s, c in zip(S, COLORS):
    ax.plot([s, s], [0, 1 - abs(s)], color=c, lw=2.6)
    ax.plot([s], [1 - abs(s)], "o", color=c, ms=7)
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(0, 1.08)
ax.set_xlabel("$s = x_1 - x_3$")
ax.set_ylabel("$\\rho_c(s) = 1 - |s|$")
ax.set_title("the length of the diagonal, normalized: the density of the reading", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
