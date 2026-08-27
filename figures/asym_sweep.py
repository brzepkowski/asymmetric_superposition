# Does more asymmetry keep helping? The MSE of the best decoder of one antipodal pair as a function of
# the length ratio r = |f3| / |f1| (long f3, short f1; s = |f1| x1 - |f3| x3):
#   left  -- the closed-form decoder, total and split into the x1 and x3 contributions (its MSE by quadrature)
#   right -- the closed form against binned decoders of a few resolutions (bins over the segment)
# Run from the repo root: python -m figures.asym_sweep
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import quad

from common import BLUE, BLUE_RAMP, MUTED, ORANGE, P, posterior_mean
from figures.asym_compare import M, RATIO, SEED

OUT = Path("figures/asym_sweep")
RS = np.geomspace(1, 100, 81)
N_BINS = (50, 100, 400)

g = torch.Generator().manual_seed(SEED)
x1 = (torch.rand(M, generator=g) < P) * torch.rand(M, generator=g)
x3 = (torch.rand(M, generator=g) < P) * torch.rand(M, generator=g)
X = torch.stack([x1, x3], 1)


def closed(r):  # exact MSE of the closed-form decoder: E[x^2] - E[xhat(s)^2] over the density of s (s = 0 contributes nothing)
    a, b = r ** 0.5, r ** -0.5
    dens = lambda s: P * (1 - P) * ((0 < s < b) / b + (-a < s < 0) / a) + P * P * max(0.0, min(1.0, (s + a) / b) - max(0.0, s / b)) / a
    knots = sorted({-a, min(0.0, b - a), 0.0, b})
    mse = []
    for w, v in ((b, -a), (-a, b)):
        f = lambda s: posterior_mean(torch.tensor([s], dtype=torch.float64), w, v, P).item() ** 2 * dens(s)
        mse.append(P / 3 - sum(quad(f, lo, hi)[0] for lo, hi in zip(knots, knots[1:])))
    return tuple(mse)


def binned(r, n_bins):  # bins of equal width over the segment [-|f3|, |f1|]; exact-zero readings are a bin of their own
    a, b = r ** 0.5, r ** -0.5
    s = b * x1 - a * x3
    nz = s != 0
    idx = ((s[nz] + a) / (a + b) * n_bins).long().clamp(0, n_bins - 1)
    cnt = torch.zeros(n_bins).index_add_(0, idx, torch.ones(int(nz.sum())))
    mean = torch.zeros(n_bins, 2).index_add_(0, idx, X[nz]) / cnt.clamp(min=1)[:, None]
    pred = torch.zeros(M, 2)
    pred[nz] = mean[idx]
    return (X - pred).pow(2).mean(0).sum().item()


parts = np.array([closed(r) for r in RS])
total = parts.sum(1)
best = total.argmin()
print(f"closed form: r = 1 -> {total[0]:.4f};  minimum {total[best]:.4f} at r = {RS[best]:.2f};  r = {RATIO} -> {total[np.abs(RS - RATIO).argmin()]:.4f};  r = 100 -> {total[-1]:.4f}")
curves = {n: np.array([binned(r, n) for r in RS]) for n in N_BINS}
for n, c in curves.items():
    print(f"{n:4d} bins: minimum {c.min():.4f} at r = {RS[c.argmin()]:.2f};  back above the r = 1 value at r = {RS[np.argmax((c > c[0]) & (RS > 2))]:.1f}")

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
ax = axes[0]
ax.plot(RS, total, color="black", lw=2.5, label="$x_1 + x_3$ (sum)")
ax.plot(RS, parts[:, 0], color=BLUE, lw=1.8, label="$x_1$ (shortened)")
ax.plot(RS, parts[:, 1], color=ORANGE, lw=1.8, label="$x_3$ (lengthened)")
ax.plot(RS[best], total[best], "v", color="black", ms=7)
ax.set_title("closed-form decoder", fontsize=12)
ax = axes[1]
ax.plot(RS, total, color="black", lw=2.5, label="closed form, $x_1 + x_3$ (sum)")
for (n, c), color in zip(curves.items(), BLUE_RAMP):
    ax.plot(RS, c, color=color, lw=1.8, label=f"binned, {n} bins, $x_1 + x_3$ (sum)")
    ax.plot(RS[c.argmin()], c.min(), "v", color=color, ms=7)
ax.set_title("binned decoders (one per feature, on the same bins)", fontsize=12)
for ax in axes:
    ax.axvline(RATIO, color=MUTED, lw=1, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("length ratio $|f_3| / |f_1|$")
    ax.set_xticks([1, 2, 4.25, 10, 30, 100], ["1", "2", "4.25", "10", "30", "100"])
    ax.set_ylim(0, 0.027)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
axes[0].set_ylabel("MSE")
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
