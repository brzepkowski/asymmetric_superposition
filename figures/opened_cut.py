# One pair slightly opened (f3 tilted by EPS degrees off the antipode of f1), the other closed.
#   opened_geometry -- the embeddings, the co-active parallelograms, and a vertical cut
#                      through the thin (f1, f3) strip
#   opened_decoded  -- the reading of feature 3 along that cut: the unconstrained decoder
#                      and the best fit from each polynomial class and the tied ReLU
#   opened_3d       -- each of those functions as a 3D surface over the hidden plane,
#                      with the cut and the strip drawn on top, viewed from the f1 side
#                      (f1 facing the reader, f2 right, f4 left)
# Run from the repo root: python -m figures.opened_cut
import math
from itertools import combinations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull

from common import AQUA, BLUE, BLUE_RAMP, INK, MUTED, ORANGE, P, PBLUE, oracle_xhat, parallelogram, poly_predictor, sample_x

OUT = Path("figures")
EPS, CUT_X, M, SEED = 25.0, 0.6, 2 ** 17, 31
CLASSES = ((2, "single bilinear MLP (quadratic)"), (4, "two bilinear MLPs (quartic)"),
           (8, "three bilinear MLPs (degree 8)"), (16, "four bilinear MLPs (degree 16)"))

er = math.radians(EPS)
F = {1: (1.0, 0.0), 2: (0.0, 1.0), 3: (-math.cos(er), math.sin(er)), 4: (0.0, -1.0)}
strip = (1 - CUT_X) * math.tan(er)  # vertical extent of the (f1, f3) parallelogram at x = CUT_X

plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})

fig, ax = plt.subplots(figsize=(5.4, 4.6), layout="constrained")
for i, j in combinations(F, 2):
    same = (i, j) == (1, 3)
    ax.add_patch(Polygon(parallelogram(F[i], F[j]), closed=True, facecolor=ORANGE if same else BLUE,
                         alpha=0.55 if same else 0.15, edgecolor="none", zorder=2 if same else 1))
ax.axhline(0, color=MUTED, lw=0.6, alpha=0.6, zorder=0)
ax.axvline(0, color=MUTED, lw=0.6, alpha=0.6, zorder=0)
for i, (x, y) in F.items():
    ax.annotate("", xy=(x, y), xytext=(0, 0), zorder=3,
                arrowprops=dict(arrowstyle="-|>", color=PBLUE, lw=2.4, shrinkA=0, shrinkB=0))
    ax.text(1.12 * x, 1.12 * y, f"$f_{i}$", color=PBLUE, fontsize=15, ha="center", va="center")
ax.plot([CUT_X, CUT_X], [-0.3, 1.1], ls="--", color="black", lw=1.4, zorder=4)
ax.plot([CUT_X, CUT_X], [0, strip], color=ORANGE, lw=7, solid_capstyle="butt", zorder=5)
ax.text(CUT_X + 0.05, 1.08, "the cut", ha="left", va="top")
ax.annotate("the strip", xy=(CUT_X, strip / 2), xytext=(0.95, 0.5), color=ORANGE,
            arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.2, 1.55)
ax.set_aspect("equal")
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(OUT / "opened_geometry.pdf")
fig.savefig(OUT / "opened_geometry.png", dpi=200)

W = torch.tensor([F[i] for i in (1, 2, 3, 4)]).T
x = sample_x(M, P, torch.Generator().manual_seed(SEED))
z = x @ W.T
ts = torch.linspace(-0.2, 0.95, 400)
cut = torch.stack([torch.full_like(ts, CUT_X), ts], 1)
# tied ReLU: x3_hat = relu(g * f3.z + b), scale g and bias b fitted on the whole plane
h, x3 = z @ W[:, 2], x[:, 2]
bs = torch.linspace(-2, 1, 301)
best = None
for g in torch.logspace(-1.5, 1.5, 61).tolist():
    m = (x3[:, None] - torch.relu(g * h[:, None] + bs)).pow(2).mean(0)
    if best is None or m.min() < best[0]:
        best = (m.min().item(), g, bs[m.argmin()].item())
_, g, b = best
fig, ax = plt.subplots(figsize=(6.0, 3.8), layout="constrained")
ax.axvspan(0, strip, color=ORANGE, alpha=0.25, lw=0)
ax.text(strip / 2, 0.4, "the strip", color=ORANGE, ha="center", va="top")
ax.plot(ts, oracle_xhat(W, cut)[:, 2], color=MUTED, lw=3.5, label="unconstrained decoder $\\hat{x}_3$")
for (deg, lbl), color in zip(CLASSES, BLUE_RAMP):
    ax.plot(ts, poly_predictor(z, x, deg)(cut)[:, 2], color=color, lw=2.2, label=lbl)
ax.plot(ts, torch.relu(g * (cut @ W[:, 2]) + b), color=AQUA, lw=2.2, label="tied ReLU")
ax.set_xlabel("position along the cut")
ax.set_ylabel("predicted feature 3")
ax.set_xlim(-0.2, 0.95)
ax.set_ylim(-0.1, 0.42)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
fig.savefig(OUT / "opened_decoded.pdf")
fig.savefig(OUT / "opened_decoded.png", dpi=200)

GRID = 200
verts = torch.tensor(list(product((0.0, 1.0), repeat=4))) @ W.T
lo, hi = verts.min(0).values, verts.max(0).values
G1, G2 = torch.meshgrid(*(torch.linspace(lo[d], hi[d], GRID) for d in range(2)), indexing="ij")
pts = torch.stack([G1.flatten(), G2.flatten()], 1)
hull = verts[ConvexHull(verts.numpy()).vertices]
inside = torch.from_numpy(MplPath(hull.numpy()).contains_points(pts.numpy()))

preds = [("unconstrained decoder $\\hat{x}_3$", lambda q: oracle_xhat(W, q)[:, 2])]
preds += [(lbl, (lambda f: lambda q: f(q)[:, 2])(poly_predictor(z, x, deg))) for deg, lbl in CLASSES]
preds += [("tied ReLU", lambda q: torch.relu(g * (q @ W[:, 2]) + b))]

fig = plt.figure(figsize=(15, 8.8), layout="constrained")
for n, (lbl, f) in enumerate(preds):
    Z = f(pts).clamp(-0.1, 1.1).masked_fill(~inside, float("nan")).view(GRID, GRID)
    ax = fig.add_subplot(2, 3, n + 1, projection="3d", computed_zorder=False)
    ax.plot_surface(G1.numpy(), G2.numpy(), np.ma.masked_invalid(Z.numpy()), cmap="viridis",
                    vmin=-0.1, vmax=1.0, lw=0, antialiased=False, rcount=GRID, ccount=GRID, zorder=1)
    yc = f(cut)
    on = (ts >= 0) & (ts <= strip)
    ax.plot(cut[:, 0], cut[:, 1], yc + 0.01, color="black", lw=1.4, zorder=3)
    ax.plot(cut[on, 0], cut[on, 1], yc[on] + 0.01, color=ORANGE, lw=3.2, zorder=4)
    for i, (fx, fy) in F.items():
        ax.plot([0, fx], [0, fy], [-0.1, -0.1], color=INK, lw=0.9, zorder=0)
        ax.text(1.15 * fx, 1.15 * fy, -0.1, f"$f_{i}$", fontsize=10, ha="center", va="center")
    ax.set_zlim(-0.1, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([0, 0.5, 1])
    ax.view_init(elev=30, azim=0)
    ax.set_title(lbl, fontsize=11)
fig.savefig(OUT / "opened_3d.pdf")
fig.savefig(OUT / "opened_3d.png", dpi=200)
