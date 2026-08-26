# The geometry found by the free search (search_geometry.py): two opened pairs, longs pinned to the
# axes, shorts opened by EPS degrees at length ratio R, with the parallelogram each pair of
# co-active features can land in. Run from the repo root: python -m figures.optimal_geometry
import math
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from common import BLUE, MUTED, ORANGE, PBLUE, parallelogram

OUT = Path("figures/optimal_geometry")
EPS, R = (11.0, 9.4), (2.35, 2.82)  # opening angles and length ratios of pairs (f1, f3) and (f2, f4)

t1, t2 = math.radians(EPS[0]), math.radians(90 + EPS[1])
F = {1: (math.cos(t1) / R[0], math.sin(t1) / R[0]), 2: (math.cos(t2) / R[1], math.sin(t2) / R[1]),
     3: (-1.0, 0.0), 4: (0.0, -1.0)}
PAIRS = ((1, 3), (2, 4))

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
fig, ax = plt.subplots(figsize=(5.4, 5.0), layout="constrained")
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
    ax.text(x + 0.12 * x / n, y + 0.12 * y / n, f"$f_{i}$", color=PBLUE, fontsize=15, ha="center", va="center")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor=ORANGE, alpha=0.55, label="co-active features of one pair"),
                   plt.Rectangle((0, 0), 1, 1, facecolor=BLUE, alpha=0.15, label="co-active features of different pairs")],
          loc="upper right", frameon=False, fontsize=11)
ax.text(-1.3, -1.52, f"$(f_1, f_3)$: opened by ${EPS[0]}^\\circ$, $|f_3| / |f_1| = {R[0]}$\n"
                     f"$(f_2, f_4)$: opened by ${EPS[1]}^\\circ$, $|f_4| / |f_2| = {R[1]}$", fontsize=11, va="bottom")
ax.set_xlim(-1.35, 1.3)
ax.set_ylim(-1.58, 1.3)
ax.set_aspect("equal")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
