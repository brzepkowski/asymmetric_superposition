# The two representative models (gallery.py) against the best decoder of their class, over the
# reading plane u = W x (no encoder bias, so the origin is "nothing active"):
#   binned3d_<arch>   -- 3D surfaces of the binned decoder E[x_i | u], one panel per feature
#   class3d_<arch>    -- the same for the least-squares polynomial of the model's degree (2 / 16)
#   class_diff_<arch> -- top-down maps of (model - best of the class) per feature, embeddings drawn
#   cut_long          -- the longest embedding's line: binned decoder, best of the class, model
# and the MSE gap and surrogate score per model. Run from the repo root: python -m figures.class_check
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as Fn
from itertools import product

from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_erosion
from scipy.spatial import ConvexHull

from common import BLUE, INK, MUTED, P, all_runs, classify, load_model, poly_predictor, sample_x

OUT = Path("figures")
M, SEED, BINS, MIN_CNT, FINE, SHRINK = 2 ** 20, 7, 40, 20, 200, 0.97  # FINE: grid for the polynomial and model surfaces
ARCHES = (("bilinear1", 2, "single bilinear MLP"), ("bilinear4", 16, "four bilinear MLPs"))
AZIM = {"bilinear1": (50, -40, -40, 50), "bilinear4": (-40, 50, -40, 50)}  # per-feature view, rotated where the arm hides the surface


def pick(runs, arch):
    recs = runs[arch]
    modal = Counter(classify(r["m"]) for r in recs).most_common(1)[0][0]
    return min((r for r in recs if classify(r["m"]) == modal), key=lambda r: r["eval_mse"])


def decode(model, u):
    z = u + model.encoder.bias
    for layer in model.mlp_layers:
        z = layer(z)
    return z


def cell_index(u, lo, hi):
    return ((u - lo) / (hi - lo) * BINS).long().clamp(0, BINS - 1)


def flat_index(u, lo, hi):
    idx = cell_index(u, lo, hi)
    return idx[:, 0] * BINS + idx[:, 1]


def binned(u, x, lo, hi):
    flat = flat_index(u, lo, hi)
    cnt = torch.zeros(BINS * BINS).index_add_(0, flat, torch.ones(len(u)))
    mean = torch.zeros(BINS * BINS, 4).index_add_(0, flat, x) / cnt.clamp(min=1)[:, None]
    mean[cnt == 0] = float("nan")
    return mean, cnt, flat


def inside_zonotope(W, pts):  # the readings fill the zonotope W [0,1]^4, a convex polygon; trimmed by SHRINK at the rim
    verts = torch.tensor(list(product((0.0, 1.0), repeat=4))) @ W.T
    hull = verts[ConvexHull(verts.numpy()).vertices]
    hull = hull.mean(0) + SHRINK * (hull - hull.mean(0))
    return torch.from_numpy(MplPath(hull.numpy()).contains_points(pts.numpy()))


def lines(ax, W, own, z=None):
    for i in range(4):
        kw = dict(color=INK, lw=1.8 if i == own else 0.8, alpha=1.0 if i == own else 0.55)
        if z is None:
            ax.plot([0, W[0, i]], [0, W[1, i]], **kw)
            ax.annotate(f"$f_{i + 1}$", (W[0, i], W[1, i]), fontsize=9)
        else:
            ax.plot([0, W[0, i]], [0, W[1, i]], [z, z], **kw)
            ax.text(1.12 * W[0, i], 1.12 * W[1, i], z, f"$f_{i + 1}$", fontsize=9, ha="center", va="center")


def surfaces(Zs, U0, U1, W, title, path, azims):
    zlim = (min(np.nanmin(Z) for Z in Zs) - 0.05, max(np.nanmax(Z) for Z in Zs) + 0.05)
    fig = plt.figure(figsize=(16, 4.2), layout="constrained")
    for i, Z in enumerate(Zs):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.plot_surface(U0, U1, np.ma.masked_invalid(Z), cmap="viridis", vmin=zlim[0], vmax=zlim[1], lw=0, antialiased=False,
                        rcount=Z.shape[0], ccount=Z.shape[1])  # no downsampling: the default 50x50 draws a sawtooth on the fine grid
        lines(ax, W, i, z=zlim[0])
        ax.set_zlim(*zlim)
        ax.set_zticks([0, 0.5, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.view_init(elev=28, azim=azims[i])
        ax.set_title(f"$\\hat{{x}}_{i + 1}$", fontsize=13)
    fig.suptitle(title)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)


plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})
runs = all_runs()
x = sample_x(M, P, torch.Generator().manual_seed(SEED))
cuts = []
for arch, deg, name in ARCHES:
    rec = pick(runs, arch)
    model = load_model(rec["path"])
    W = model.w_enc.detach()
    Wn = W.numpy()
    u = x @ W.T
    lo, hi = u.min(0).values - 1e-3, u.max(0).values + 1e-3
    with torch.no_grad():
        y_model = decode(model, u)
        assert (y_model - model(x)).abs().max() < 1e-5
    fit = poly_predictor(u, x, deg)
    y_fit = fit(u).float()
    mean, cnt, flat = binned(u, x, lo, hi)
    mse = {k: (x - y).pow(2).mean().item() for k, y in (("binned", mean[flat]), ("fit", y_fit), ("model", y_model))}
    score = 1 - (y_model - y_fit).pow(2).sum().item() / (y_model - y_model.mean(0)).pow(2).sum().item()
    print(f"{name}, seed {rec['seed']}: MSE binned {mse['binned']:.4f}  best of class {mse['fit']:.4f}  model {mse['model']:.4f}  "
          f"gap {(mse['model'] - mse['fit']) / mse['model'] * 100:.1f}%  surrogate score {score:.4f}")

    def mesh(n):  # cell-centre grid of the reading plane; the polynomial and the model are drawn on the FINE one
        centers = [lo[d] + (torch.arange(n) + 0.5) / n * (hi[d] - lo[d]) for d in range(2)]
        U0, U1 = torch.meshgrid(centers[0], centers[1], indexing="ij")
        return U0, U1, torch.stack([U0.flatten(), U1.flatten()], 1)

    # support: coarse bins with enough samples, eroded by one bin so that no drawn point lies beyond the data
    support = torch.from_numpy(binary_erosion((cnt >= MIN_CNT).view(BINS, BINS).numpy())).flatten()
    U0, U1, _ = mesh(BINS)
    G_bin = mean.view(BINS, BINS, 4).clone()
    G_bin[~support.view(BINS, BINS)] = float("nan")
    F0, F1, fine = mesh(FINE)
    with torch.no_grad():
        G_model = decode(model, fine).view(FINE, FINE, 4)
    G_fit = fit(fine).float().view(FINE, FINE, 4)
    # fine support: inside the zonotope (straight edges) and where the interpolated sample density is high enough
    dens = Fn.interpolate(cnt.view(1, 1, BINS, BINS), size=(FINE, FINE), mode="bilinear")[0, 0]
    fine_support = (dens >= MIN_CNT) & inside_zonotope(W, fine).view(FINE, FINE)
    for G in (G_model, G_fit):
        G[~fine_support] = float("nan")
    label = f"{name}, seed {rec['seed']}"
    surfaces([G_bin[:, :, i].numpy() for i in range(4)], U0.numpy(), U1.numpy(), Wn, f"binned decoder, {label}", OUT / f"binned3d_{arch}", AZIM[arch])
    surfaces([G_fit[:, :, i].clamp(-0.1, 1.1).numpy() for i in range(4)], F0.numpy(), F1.numpy(), Wn,
             f"best decoder of the class (degree {deg}), {label}", OUT / f"class3d_{arch}", AZIM[arch])

    diff = (G_model - G_fit).numpy()
    vmax = np.nanpercentile(np.abs(diff), 99)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), layout="constrained")
    for i, ax in enumerate(axes):
        im = ax.imshow(diff[:, :, i].T, origin="lower", extent=[lo[0], hi[0], lo[1], hi[1]], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        lines(ax, Wn, i)
        ax.set_title(f"$\\hat{{x}}_{i + 1} - \\hat{{x}}_{i + 1}'$", fontsize=13)
        ax.set_xlabel("$u_1$")
        ax.set_aspect("equal")
    axes[0].set_ylabel("$u_2$")
    fig.colorbar(im, ax=axes[-1], shrink=0.85)
    fig.suptitle(f"model $-$ best decoder of the class, {label}")
    fig.savefig(OUT / f"class_diff_{arch}.pdf")
    fig.savefig(OUT / f"class_diff_{arch}.png", dpi=200)

    k = W.norm(dim=0).argmax().item()
    pair = next(p["pair"] for p in rec["m"]["pairs"] if k in p["pair"])
    partner = pair[0] if pair[1] == k else pair[1]
    ts = torch.linspace(-0.95 * W[:, partner].norm(), 0.95 * W[:, k].norm(), 600)
    cut = ts[:, None] * W[:, k] / W[:, k].norm()
    on = (cnt >= MIN_CNT)[flat_index(cut, lo, hi)]
    with torch.no_grad():
        yc_model = decode(model, cut)[:, k]
    cuts.append((label, k, ts[on], mean[flat_index(cut, lo, hi)][on, k], fit(cut).float()[on, k], yc_model[on]))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
for ax, (label, k, ts, yb, yf, ym) in zip(axes, cuts):
    ax.plot(ts, yb, color=MUTED, lw=2.5, label="binned decoder")
    ax.plot(ts, yf, color=INK, lw=1.2, label="best decoder of the class")
    ax.plot(ts, ym, color=BLUE, ls="--", lw=1.8, label="the trained model")
    ax.set_title(f"{label}: $\\hat{{x}}_{k + 1}$ along the line of $f_{k + 1}$", fontsize=12)
    ax.set_xlabel(f"reading along $f_{k + 1}$")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("predicted feature")
axes[0].legend(frameon=False, fontsize=10)
fig.savefig(OUT / "cut_long.pdf")
fig.savefig(OUT / "cut_long.png", dpi=200)
