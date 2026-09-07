# Does the short embedding of an opened pair keep helping as it shrinks? Both pairs opened as in the
# search optimum (each short tilted by EPS toward the long embedding of the other pair), the length
# ratio r = |long| / |short| swept at fixed angle and scored by binned decoders of several resolutions,
# fitted on one sample and scored on another. Top: the MSE averaged over the four features, for several
# resolutions; middle: the same sweep split into the MSE of each feature, at the finest resolution; bottom:
# the MSE summed over the two features of each pair, as in the closed-pair sweep.
# Run from the repo root: python -m figures.ratio_sweep
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import BLUE, BLUE_RAMP, MUTED, ORANGE, P, binned_fit, sample_x

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
err = lambda pred: (x_eval - pred).pow(2).mean(0).numpy()  # MSE per feature

per_feature = {n: np.array([err(binned_fit(x @ W_of(r).T, x, n)(x_eval @ W_of(r).T)) for r in RS]) for n in N_BINS}
binned = {n: e.mean(1) for n, e in per_feature.items()}
for n, c in binned.items():
    print(f"{n:4d} bins: minimum {c.min():.4f} at r = {RS[c.argmin()]:.1f}")
for i, c in enumerate(per_feature[N_BINS[-1]].T):
    print(f"{N_BINS[-1]} bins, x{i + 1}: r = 1 -> {c[0]:.4f}, r = {RATIO:.0f} -> {c[np.abs(np.array(RS) - RATIO).argmin()]:.4f}, r = 100 -> {c[-1]:.4f}")
pairs = {"$x_1 + x_3$": per_feature[N_BINS[-1]][:, [0, 2]].sum(1), "$x_2 + x_4$": per_feature[N_BINS[-1]][:, [1, 3]].sum(1)}
for name, c in pairs.items():
    print(f"{N_BINS[-1]} bins, {name}: r = 1 -> {c[0]:.4f}, minimum {c.min():.4f} at r = {RS[c.argmin()]:.1f}, r = 100 -> {c[-1]:.4f}")

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
fig, axes = plt.subplots(3, 1, figsize=(6.5, 10.8), layout="constrained", sharey=True)
ax = axes[0]
for (n, c), color in zip(binned.items(), BLUE_RAMP):
    ax.plot(RS, c, color=color, lw=1.8, label=f"{n} bins per axis")
    ax.plot(RS[int(c.argmin())], c.min(), "v", color=color, ms=7)
ax.set_title("MSE averaged over the four features", fontsize=12)
ax = axes[1]
ax.plot(RS, binned[N_BINS[-1]], color="black", lw=2.5, label="average of the four")
for i, color, ls in ((0, BLUE, "-"), (1, BLUE, "--"), (2, ORANGE, "-"), (3, ORANGE, "--")):
    ax.plot(RS, per_feature[N_BINS[-1]][:, i], color=color, lw=1.8, ls=ls, label=f"$x_{i + 1}$ ({'short, tilted' if i < 2 else 'long'})")
ax.set_title(f"MSE of each feature, {N_BINS[-1]} bins per axis", fontsize=12)
ax = axes[2]
for (name, c), color, ls in zip(pairs.items(), (BLUE, ORANGE), ("-", "--")):
    ax.plot(RS, c, color=color, lw=1.8, ls=ls, label=name)
    ax.plot(RS[int(c.argmin())], c.min(), "v", color=color, ms=7)
ax.set_title(f"MSE summed over each pair, {N_BINS[-1]} bins per axis", fontsize=12)
for ax in axes:
    ax.axvline(RATIO, color=MUTED, lw=1, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 13, 30, 100], ["1", "2", "4", "13", "30", "100"])
    ax.set_ylim(0, 0.012)
    ax.set_ylabel("MSE")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
axes[-1].set_xlabel("length ratio $|f_3| / |f_1| = |f_4| / |f_2|$")
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
