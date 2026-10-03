# The MSE interlude of the presentation: points on a vertical line and the single output that
# minimizes the summed squared distances to them, their average. Run from the repo root:
# python -m figures.mse_average
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import BLUE, MUTED, ORANGE

OUT = Path("figures/mse_average")

rng = np.random.default_rng(53)
pts = rng.uniform(-1, 1, 7)
mean = pts.mean()

fig, ax = plt.subplots(figsize=(1.8, 2.7), layout="constrained")
ax.axvline(0, color=MUTED, lw=1.2, zorder=1)
ax.scatter(np.zeros_like(pts), pts, s=45, color=BLUE, zorder=2)
ax.scatter([0], [mean], s=260, color=ORANGE, marker="*", zorder=3)
ax.text(0.22, mean, "average", color=ORANGE, fontsize=12, ha="left", va="center")
ax.set_xlim(-0.35, 1.3)
ax.set_ylim(-1.15, 1.15)
ax.axis("off")
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
