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

from common import P, measure, sample_x, tilt

M, BINS, N_RAND, N_START = 2 ** 19, 96, 150, 3
LO, HI = math.log(0.05), math.log(1.5)  # range of the short/long length ratio in the random starts
HAND = {"closed, r = 1": (0.0, 0.0, 0.0, 0.0), "closed, r = 3": (0.0, math.log(1 / 3), 0.0, math.log(1 / 3)),
        "opened 10 deg, r = 3": (10 / 90, math.log(1 / 3), 10 / 90, math.log(1 / 3)),
        "opened 20 deg, r = 2": (20 / 90, math.log(0.5), 20 / 90, math.log(0.5))}


def kernel(q):
    """The encoder matrix of one search point. The long embeddings are pinned to the axes,
    f3 = (-1, 0) and f4 = (0, -1), and `q` describes their free partners with 4 numbers:
      - q[0]: the angle of f1 off the antipode of f3 (the +x axis), as a fraction of 90 deg —
        the code multiplies it by 90, so 0 is a closed pair and 1 a right angle
        (the fraction keeps all four coordinates at comparable scales, which suits Nelder-Mead);
      - q[1]: log |f1| — the log keeps the length positive and lets the search move
        multiplicatively (and since |f3| = 1, exp(q[1]) is the pair's short/long ratio);
      - q[2], q[3]: the same two numbers for f2, measured off the antipode of f4 (the +y axis).
    Returns the (2, 4) encoder matrix, columns f1, f2, f3, f4, rows the plane's coordinates.
    """
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


def report(q):
    """One line describing the geometry at the search point `q` (rebuilt via kernel, so it
    can be called on whatever point the optimizer is at).
    
    Returns a string of two parts:
      - the pairs as measure() finds them — "(f1, f3) opened by 20.7 deg, ratio 13.5; ..."
        — reported rather than assumed, because the search may drift from the nominal
        pairing (e.g. pair f1 with f4); "no antipodal pairs" when measure finds none;
      - the raw coordinates of the four embedding vectors.
    Printing is left to the callers.
    """
    W = kernel(q)
    pairs = "; ".join(f"(f{p['pair'][0] + 1}, f{p['pair'][1] + 1}) opened by {tilt(p['cos']):.1f} deg, ratio {p['ratio']:.1f}"
                      for p in measure(W)["pairs"]) or "no antipodal pairs"
    return pairs + "   " + " ".join(f"f{i + 1}=({W[0, i]:+.3f}, {W[1, i]:+.3f})" for i in range(4))


x = sample_x(M, P, torch.Generator().manual_seed(11))
x_fresh = sample_x(M, P, torch.Generator().manual_seed(22))
obj = lambda q: binned_mse(x @ kernel(q).T, x)
fresh = lambda q: binned_mse(x_fresh @ kernel(q).T, x_fresh)
g = torch.Generator().manual_seed(3)
cand = []
for _ in range(N_RAND):
    u = torch.rand(4, generator=g)
    # one random candidate, four uniform draws u in [0, 1) stretched onto the search box:
    #   - angles: 4u - 2 covers [-2, 2) in the fraction-of-90-deg units, i.e. +-180 deg off the
    #     antipode — the free embeddings can start pointing anywhere in the plane;
    #   - log-lengths: LO + u (HI - LO) makes the lengths log-uniform between 0.05 and 1.5, as
    #     much mass on "twenty times shorter than the pinned partner" as on "roughly equal".
    # A candidate is only screened by obj below; the best N_START become Nelder-Mead starts
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
