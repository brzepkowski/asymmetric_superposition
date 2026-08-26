# The geometry found by the free search (search_geometry.py): two opened pairs, longs pinned to the
# axes, shorts opened by EPS degrees (counter-clockwise positive) at length ratio R, with the
# parallelogram each pair of co-active features can land in; an inset zooms on the origin, where
# the shorts live. Run from the repo root: python -m figures.optimal_geometry
import math
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from common import BLUE, INK, MUTED, ORANGE, PBLUE, parallelogram

OUT = Path("figures/optimal_geometry")
EPS, R = (-20.7, 20.4), (13.5, 13.0)  # opening angles and length ratios of pairs (f1, f3) and (f2, f4)
ZOOM = 0.12  # half-width of the inset

t1, t2 = math.radians(EPS[0]), math.radians(90 + EPS[1])
F = {1: (math.cos(t1) / R[0], math.sin(t1) / R[0]), 2: (math.cos(t2) / R[1], math.sin(t2) / R[1]),
     3: (-1.0, 0.0), 4: (0.0, -1.0)}
PAIRS = ((1, 3), (2, 4))


def draw(ax, lim, fontsize):
    for i, j in combinations(F, 2):
        same = (i, j) in PAIRS
        ax.add_patch(Polygon(parallelogram(F[i], F[j]), closed=True, facecolor=ORANGE if same else BLUE,
                             alpha=0.55 if same else 0.15, edgecolor="none", zorder=2 if same else 1))
    ax.axhline(0, color=MUTED, lw=0.6, alpha=0.6, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.6, alpha=0.6, zorder=0)
    for i, (x, y) in F.items():
        ax.annotate("", xy=(x, y), xytext=(0, 0), zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=PBLUE, lw=2.4, shrinkA=0, shrinkB=0))
        n = math.hypot(x, y)
        ax.text(x + 0.1 * lim * x / n, y + 0.1 * lim * y / n, f"$f_{i}$", color=PBLUE, fontsize=fontsize,
                ha="center", va="center", clip_on=True)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")


plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
fig, ax = plt.subplots(figsize=(6.2, 5.8), layout="constrained")
draw(ax, 1.3, 15)
ax.set_ylim(-1.62, 1.3)  # room for the legend under the geometry
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE, alpha=0.55, label="co-active features of one pair"),
                   plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, alpha=0.15, label="co-active features of different pairs")],
          loc="lower right", frameon=False, fontsize=11)
axins = ax.inset_axes([0.59, 0.59, 0.38, 0.38])
draw(axins, ZOOM, 12)
axins.set_xticks([-0.1, 0, 0.1])
axins.set_yticks([-0.1, 0, 0.1])
axins.tick_params(labelsize=8)
ax.indicate_inset_zoom(axins, edgecolor=INK, lw=1.0)
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
