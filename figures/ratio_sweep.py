# Does the short embedding of an opened pair keep helping as it shrinks? Both pairs opened as in the
# search optimum (each short tilted by EPS toward the long embedding of the other pair), the length
# ratio r = |long| / |short| swept at fixed angle and scored by binned decoders of several resolutions,
# fitted on one sample and scored on another. Run from the repo root: python -m figures.ratio_sweep
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import BLUE_RAMP, MUTED, P, binned_fit, sample_x

OUT = Path("figures/ratio_sweep")
M, M_EVAL, SEED = 2 ** 20, 2 ** 19, 31  # bins fitted on M samples, scored on M_EVAL fresh ones
EPS, RATIO = (20.7, 20.4), 13.0  # the search optimum's opening angles of (f1, f3) and (f2, f4), and its length ratio
RS = [float(r) for r in np.geomspace(1, 100, 41)]
N_BINS = (24, 48, 96)


def W_of(r):
    e1, e2 = map(math.radians, EPS)
    return torch.tensor([[math.cos(e1) / r, -math.sin(e2) / r, -1.0, 0.0], [-math.sin(e1) / r, math.cos(e2) / r, 0.0, -1.0]])


x = sample_x(M, P, torch.Generator().manual_seed(SEED))
x_eval = sample_x(M_EVAL, P, torch.Generator().manual_seed(SEED + 1))
err = lambda pred: (x_eval - pred).pow(2).mean().item()

binned = {n: np.array([err(binned_fit(x @ W_of(r).T, x, n)(x_eval @ W_of(r).T)) for r in RS]) for n in N_BINS}
for n, c in binned.items():
    print(f"{n:4d} bins: minimum {c.min():.4f} at r = {RS[c.argmin()]:.1f}")

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
for (n, c), color in zip(binned.items(), BLUE_RAMP):
    ax.plot(RS, c, color=color, lw=1.8, label=f"{n} bins per axis")
    ax.plot(RS[int(c.argmin())], c.min(), "v", color=color, ms=7)
ax.axvline(RATIO, color=MUTED, lw=1, ls="--", zorder=0)
ax.set_xscale("log")
ax.set_xlabel("length ratio $|f_3| / |f_1| = |f_4| / |f_2|$")
ax.set_xticks([1, 2, 4, 13, 30, 100], ["1", "2", "4", "13", "30", "100"])
ax.set_ylabel("MSE")
ax.set_title("binned decoders", fontsize=12)
ax.set_ylim(0, 0.012)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=10, loc="upper left")
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
