# The free search of "Search for the best strategy": the long embedding of each pair pinned to an axis
# (f3 = (-1, 0), f4 = (0, -1)), its partner a free vector (angle, length), every geometry scored by the
# MSE of a binned decoder on the reading plane (BINS x BINS cells over the readings' range, the origin
# at a cell centre so that no cell edge runs along an embedding line). Random starts plus a few
# hand-picked ones, Nelder-Mead from the best of them, and the winners re-scored on a fresh sample.
# Prints the geometry that optimal_geometry.py draws and the floor of the closed symmetric cross.
# Run from the repo root: python -m figures.search_geometry
import math

import torch
from scipy.optimize import minimize

from common import P, sample_x

M, BINS, N_RAND, N_START = 2 ** 19, 96, 150, 3
LO, HI = math.log(0.05), math.log(1.5)  # range of the short/long length ratio in the random starts
HAND = {"closed, r = 1": (0.0, 0.0, 0.0, 0.0), "closed, r = 3": (0.0, math.log(1 / 3), 0.0, math.log(1 / 3)),
        "opened 10 deg, r = 3": (10 / 90, math.log(1 / 3), 10 / 90, math.log(1 / 3)),
        "opened 20 deg, r = 2": (20 / 90, math.log(0.5), 20 / 90, math.log(0.5))}


def kernel(q):  # q = (angle of f1 off the antipode of f3, in units of 90 deg; log |f1|; the same for f2, f4)
    t1, t2 = math.radians(q[0] * 90), math.radians(90 + q[2] * 90)
    r1, r2 = math.exp(q[1]), math.exp(q[3])
    return torch.tensor([[r1 * math.cos(t1), r2 * math.cos(t2), -1.0, 0.0],
                         [r1 * math.sin(t1), r2 * math.sin(t2), 0.0, -1.0]])


def binned_mse(z, x):
    h = (z.max(0).values - z.min(0).values) / BINS
    cell = torch.floor(z / h + 0.5).long()
    cell = cell - cell.min(0).values
    idx = cell[:, 0] * (cell[:, 1].max() + 1) + cell[:, 1]
    cnt = torch.zeros(idx.max() + 1).index_add_(0, idx, torch.ones(len(z)))
    mean = torch.zeros(len(cnt), x.shape[1]).index_add_(0, idx, x) / cnt.clamp(min=1)[:, None]
    return (x - mean[idx]).pow(2).mean().item()


def report(q):  # signed opening angles (counter-clockwise positive), long/short length ratios
    ang = lambda v: (v * 90 + 180) % 360 - 180
    return (f"(f1, f3): opened by {ang(q[0]):+.1f} deg, |f3| / |f1| = {math.exp(-q[1]):.2f};   "
            f"(f2, f4): opened by {ang(q[2]):+.1f} deg, |f4| / |f2| = {math.exp(-q[3]):.2f}")


x = sample_x(M, P, torch.Generator().manual_seed(11))
x_fresh = sample_x(M, P, torch.Generator().manual_seed(22))
obj = lambda q: binned_mse(x @ kernel(q).T, x)
fresh = lambda q: binned_mse(x_fresh @ kernel(q).T, x_fresh)
g = torch.Generator().manual_seed(3)
cand = []
for _ in range(N_RAND):
    u = torch.rand(4, generator=g)
    q = [4 * u[0].item() - 2, LO + u[1].item() * (HI - LO), 4 * u[2].item() - 2, LO + u[3].item() * (HI - LO)]
    cand.append((obj(q), q))
cand.sort(key=lambda t: t[0])
starts = [(f"random start {i + 1}", q) for i, (_, q) in enumerate(cand[:N_START])] + list(HAND.items())
best = None
for name, q0 in starts:
    res = minimize(obj, q0, method="Nelder-Mead", options=dict(xatol=5e-3, fatol=2e-6, maxfev=400))
    score = fresh(res.x)
    print(f"{name:20s} -> {res.fun:.5f} (search sample), {score:.5f} (fresh): {report(res.x)}")
    if best is None or score < best[0]:
        best = (score, res.x)

print(f"\nbest geometry: {report(best[1])}")
print(f"MSE floor per feature: {best[0]:.5f};  closed symmetric cross: {fresh([0.0, 0.0, 0.0, 0.0]):.5f}")
