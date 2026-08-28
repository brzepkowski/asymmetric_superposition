# The appendix figure behind the closed-form decoder of the symmetric pair: for each of the four cases
# (neither, feature 1 alone, feature 3 alone, both), how often it produces a reading s (left) and what
# x1 is in that case (right); the closed form is the average of the right panel weighted by the left.
# Run from the repo root: python -m figures.appendix_cases
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import AQUA, BLUE, INK, MUTED, ORANGE, P

OUT = Path("figures/appendix_cases")
s_neg, s_pos = np.linspace(-1, 0, 200), np.linspace(0, 1, 200)
s = np.concatenate([s_neg, s_pos])
w1, w3, w13 = P * (1 - P) * np.ones_like(s_pos), P * (1 - P) * np.ones_like(s_neg), P * P * (1 - np.abs(s))
x1_alone, x1_both = s_pos, (1 + s) / 2
post = np.where(s < 0, P * (1 + s) ** 2 / (2 * (1 + P * s)), (2 * (1 - P) * s + P * (1 - s ** 2)) / (2 * (1 - P * s)))

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
ax = axes[0]
ax.plot(s_pos, w1, color=BLUE, lw=2.2, label="feature 1 alone: $p(1-p) \\cdot 1$ on $(0, 1]$")
ax.plot(s_neg, w3, color=ORANGE, lw=2.2, label="feature 3 alone: $p(1-p) \\cdot 1$ on $[-1, 0)$")
ax.plot(s, w13, color=AQUA, lw=2.2, label="both: $p^2 \\cdot (1 - |s|)$")
ax.plot([0, 0], [0, 0.27], color=INK, lw=2.2)
ax.plot([0], [0.27], "^", color=INK, ms=8, clip_on=False, label="neither: $(1-p)^2 \\cdot \\delta(s)$")
ax.set_ylim(0, 0.3)
ax.set_ylabel("$P(c)\\, \\rho_c(s)$")
ax.set_title("weight of each case", fontsize=12)
ax = axes[1]
ax.plot(s_pos, x1_alone, color=BLUE, lw=2.2, label="feature 1 alone: $s$")
ax.plot(s_neg, np.zeros_like(s_neg), color=ORANGE, lw=2.2, label="feature 3 alone: $0$")
ax.plot(s, x1_both, color=AQUA, lw=2.2, label="both: $(1+s)/2$")
ax.plot(s, post, color=INK, lw=2.8, ls="--", label="weighted average $\\hat{x}_1(s) = \\mathbb{E}[x_1 \\mid s]$")
ax.plot([0], [0], "o", color=INK, ms=6, zorder=5)
ax.set_ylim(-0.03, 1.03)
ax.set_ylabel("$\\mathbb{E}[x_1 \\mid s, c]$")
ax.set_title("best guess of each case", fontsize=12)
for ax in axes:
    ax.set_xlabel("$s = x_1 - x_3$")
    ax.set_xlim(-1.02, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
axes[0].legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(0.53, 1.0))
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
