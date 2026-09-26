# Should a pair be opened as little as possible? One pair opened by EPS degrees (f3 tilted off the
# antipode of f1, as in opened_cut.py), the other closed, swept over the angle and scored by binned
# decoders of several resolutions, fitted on one sample and scored on another.
# Run from the repo root: python -m figures.opening_sweep
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import BLUE_RAMP, P, binned_fit, sample_x

OUT = Path("figures/opening_sweep")
M, M_EVAL, SEED = 2 ** 20, 2 ** 19, 31  # bins fitted on M samples, scored on M_EVAL fresh ones
EPS = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7.5, 10, 12.5, 15, 20, 25, 30, 35, 40]
N_BINS = (24, 48, 96)


def W_of(eps):
    """The full four-feature encoder matrix at opening angle `eps` (degrees), shape (2, 4):
    two rows (the two coordinates of the plane) and four columns, one per embedding —
      f1 = (1, 0),  f2 = (0, 1),  f3 = (-cos(eps), sin(eps)),  f4 = (0, -1)
    — so the tensor literal's inner lists are the x- and the y-coordinates of all four.
    Only f3's column depends on eps: W_of(0) is the perfect antipodal cross, and increasing
    eps tilts f3 off the antipode of f1 while the (f2, f4) pair stays closed; W_of(25)
    reproduces the geometry of opened_cut.py.
    """
    er = math.radians(eps)
    return torch.tensor([[1.0, 0.0, -math.cos(er), 0.0], [0.0, 1.0, math.sin(er), -1.0]])


x = sample_x(M, P, torch.Generator().manual_seed(SEED))
x_eval = sample_x(M_EVAL, P, torch.Generator().manual_seed(SEED + 1))
err = lambda pred: (x_eval - pred).pow(2).mean().item()

# recap: binned_fit returns a function (the cell-mean predictor), so the second bracket pair
# calls it right away — fitted on the readings of x, queried on the fresh readings of x_eval
binned = {n: np.array([err(binned_fit(x @ W_of(e).T, x, n)(x_eval @ W_of(e).T)) for e in EPS]) for n in N_BINS}
for n, c in binned.items():
    print(f"{n:4d} bins: closed {c[0]:.4f}, minimum {c.min():.4f} at {EPS[c.argmin()]} deg, {EPS[-1]} deg {c[-1]:.4f}")

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
for (n, c), color in zip(binned.items(), BLUE_RAMP):
    ax.plot(EPS, c, color=color, lw=1.8, label=f"{n} bins per axis")
    ax.plot(EPS[c.argmin()], c.min(), "v", color=color, ms=7)
ax.set_ylabel("MSE")
ax.set_title("binned decoders", fontsize=12)
ax.set_xlabel("opening angle $\\varepsilon$ (degrees)")
ax.set_xlim(-1, EPS[-1] + 1)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=10, loc="upper right")
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
