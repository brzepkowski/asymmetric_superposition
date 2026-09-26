# The two representative models (the seeds picked by gallery.py), retrained on a fixed sample of 2^20
# inputs instead of 4096 (cached under checkpoints/big/), against the best decoder of their class, over
# the reading plane u = W x (no encoder bias, so the origin is "nothing active"):
#   binned3d_<arch>   -- 3D surfaces of the binned decoder E[x_i | u], one panel per feature
#   class3d_<arch>    -- the same for the least-squares polynomial of the model's degree (2 / 16)
#   class_diff_<arch> -- top-down maps of (model - best of the class) per feature, embeddings drawn
#   cut_long          -- the longest embedding's line: binned decoder, best of the class, model
# and the MSE gap per model. Run from the repo root: python -m figures.class_check
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

from common import BLUE, CKPT, INK, MUTED, P, all_runs, classify, load_model, measure, poly_predictor, sample_x, train

OUT = Path("figures")
BIG, BATCH, DEVICE = CKPT / "big", 2 ** 20, "cuda"  # the retraining: same seed (so the same initialization), 2^20 samples
M, SEED, BINS, MIN_CNT, FINE, SHRINK = 2 ** 20, 7, 40, 20, 200, 0.97  # FINE: grid for the polynomial and model surfaces
FIT_CNT = 60  # denser support for the polynomial and model surfaces: below it the degree-16 fit oscillates at the rim
ARCHES = (("bilinear1", 2, "single bilinear MLP"), ("bilinear4", 16, "four bilinear MLPs"))
AZIM = {"bilinear1": (50, -40, -40, 50), "bilinear4": (-40, 50, -40, 50)}  # per-feature view, rotated where the arm hides the surface


def pick(runs, arch):
    """The representative run of an architecture: among its runs with the most common strategy
    label, the one with the lowest eval MSE (the same choice gallery.py makes). Inputs:
      - `runs`: the catalogue from all_runs(), {arch name: list of run dicts}.
      - `arch`: the architecture name, e.g. "bilinear4".
    Returns one run dict of runs[arch] (keys "seed", "W", "m", "eval_mse", ...).
    """
    recs = runs[arch]
    modal = Counter(classify(r["m"]) for r in recs).most_common(1)[0][0]
    return min((r for r in recs if classify(r["m"]) == modal), key=lambda r: r["eval_mse"])


def decode(model, u):
    """The model's decoder alone, evaluated at arbitrary points of the reading plane. Inputs:
      - `model`: a SuperModel.
      - `u`: points of the plane u = W x, (m, 2) — the readings without the encoder bias.
    Adds the encoder bias back and runs the stack of bilinear MLPs; returns xhat, (m, 4).
    On the sample itself this equals model(x), which the assert below checks.
    """
    z = u + model.encoder.bias
    for layer in model.mlp_layers:
        z = layer(z)
    return z


def cell_index(u, lo, hi):
    """The 2D cell coordinates of points on the BINS x BINS grid covering the plane. Inputs:
      - `u`: the points, (m, 2).
      - `lo`, `hi`: the grid's range per axis, (2,) each.
    Returns (m, 2) integers in [0, BINS - 1]; the clamp folds a point on or past the upper
    edge into the last cell.
    """
    return ((u - lo) / (hi - lo) * BINS).long().clamp(0, BINS - 1)


def flat_index(u, lo, hi):
    """cell_index flattened row-major into one table position per point (same inputs).
    Returns (m,) integers indexing the BINS * BINS tables of binned().
    """
    idx = cell_index(u, lo, hi)
    return idx[:, 0] * BINS + idx[:, 1]


def binned(u, x, lo, hi):
    """The binned decoder's table over the reading plane. Inputs:
      - `u`: the sampled readings, (M, 2); `x`: the features behind them, (M, 4).
      - `lo`, `hi`: the grid's range per axis, (2,) each.
    Returns (mean, cnt, flat):
      - `mean`: (BINS * BINS, 4), each cell's mean of x — NaN for a cell with no samples,
        so empty cells leave holes in the surfaces instead of a fake 0 (unlike binned_fit),
      - `cnt`: (BINS * BINS,), each cell's sample count,
      - `flat`: (M,), each sample's table position, so mean[flat] is the per-sample prediction.
    """
    flat = flat_index(u, lo, hi)
    cnt = torch.zeros(BINS * BINS).index_add_(0, flat, torch.ones(len(u)))
    mean = torch.zeros(BINS * BINS, 4).index_add_(0, flat, x) / cnt.clamp(min=1)[:, None]
    mean[cnt == 0] = float("nan")
    return mean, cnt, flat


def inside_zonotope(W, pts):
    """The mask of the points lying inside the zonotope W [0,1]^4 — the convex polygon the
    readings can reach — shrunk by SHRINK toward its centre to trim the rim.

    Geometrically, the zonotope is what you get by "sweeping" the four embedding segments over
    one another: start at the origin (nothing active), and add any fraction of f1, plus any
    fraction of f2, and so on. Adding one segment to a point gives a segment; adding a segment
    to a segment gives a parallelogram (the co-active parallelograms of the geometry figures
    are exactly the two-feature slices of this); adding all four gives a convex polygon. That
    construction — a sum of line segments — is what the word zonotope means. In 2D it's always
    a centrally symmetric convex polygon with at most 2 x 4 = 8 edges, each edge parallel to
    one of the embeddings.

    Inputs:
      - `W`: the encoder weight, (2, 4).
      - `pts`: the points to test, (m, 2).
    Returns a bool tensor (m,), True for the points inside the shrunk polygon.
    """
    # product enumerates every 4-tuple of 0.0 or 1.0 — (0,0,0,0), (0,0,0,1), ..., (1,1,1,1) —
    # the 16 corners of the feature hypercube [0,1]^4
    verts = torch.tensor(list(product((0.0, 1.0), repeat=4))) @ W.T
    hull = verts[ConvexHull(verts.numpy()).vertices]
    hull = hull.mean(0) + SHRINK * (hull - hull.mean(0))
    return torch.from_numpy(MplPath(hull.numpy()).contains_points(pts.numpy()))


def lines(ax, W, own, z=None):
    """Draws the four embedding segments with their labels on one panel. Inputs:
      - `ax`: the target axes; `W`: the encoder weight, (2, 4).
      - `own`: the feature index the panel shows — its segment is drawn bold, the rest faint.
      - `z`: None on a 2D panel; on a 3D one, the height at which the segments are drawn.
    Returns nothing; draws onto ax.
    """
    for i in range(4):
        kw = dict(color=INK, lw=1.8 if i == own else 0.8, alpha=1.0 if i == own else 0.55)
        if z is None:
            ax.plot([0, W[0, i]], [0, W[1, i]], **kw)
            ax.annotate(f"$f_{i + 1}$", (W[0, i], W[1, i]), fontsize=9)
        else:
            ax.plot([0, W[0, i]], [0, W[1, i]], [z, z], **kw)
            ax.text(1.12 * W[0, i], 1.12 * W[1, i], z, f"$f_{i + 1}$", fontsize=9, ha="center", va="center")


def surfaces(Zs, U0, U1, W, title, path, azims):
    """One figure of four 3D surface panels, one per feature, saved as pdf and png. Inputs:
      - `Zs`: four (n, n) arrays, one surface per feature; NaN cells become holes.
      - `U0`, `U1`: the (n, n) meshgrid coordinates of the reading plane.
      - `W`: the encoder weight (2, 4), for the embedding segments on the floor.
      - `title`: the suptitle; `path`: the output basename (suffixes added here).
      - `azims`: four azimuth angles, one view per panel (from AZIM).
    Returns nothing; the z-range is shared across the panels so their heights compare.
    """
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
    model = load_model(train(arch, rec["seed"], root=BIG, batch=BATCH, device=DEVICE) / "model.pt")
    W = model.w_enc.detach()
    print(f"{name}, seed {rec['seed']}, {BATCH} samples: geometry {classify(measure(W))}")
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
    print(f"{name}, seed {rec['seed']}: MSE binned {mse['binned']:.4f}  best of class {mse['fit']:.4f}  model {mse['model']:.4f}  "
          f"gap {(mse['model'] - mse['fit']) / mse['model'] * 100:.1f}%")

    def mesh(n):
        """An n x n cell-centre grid over the current model's reading range [lo, hi].
        Returns (U0, U1, pts): the (n, n) meshgrids and the flattened points (n * n, 2).
        Called with BINS for the binned surface, FINE for the polynomial and model ones.
        """
        centers = [lo[d] + (torch.arange(n) + 0.5) / n * (hi[d] - lo[d]) for d in range(2)]
        U0, U1 = torch.meshgrid(centers[0], centers[1], indexing="ij")
        return U0, U1, torch.stack([U0.flatten(), U1.flatten()], 1)

    # --- binned3d_<arch>: the binned decoder's surfaces, on the coarse grid ---
    # support: coarse bins with enough samples, eroded by one bin so that no drawn point lies beyond the data
    support = torch.from_numpy(binary_erosion((cnt >= MIN_CNT).view(BINS, BINS).numpy())).flatten()
    U0, U1, _ = mesh(BINS)
    G_bin = mean.view(BINS, BINS, 4).clone()
    G_bin[~support.view(BINS, BINS)] = float("nan")
    # --- class3d_<arch> and class_diff_<arch>: the model and its best fit, on the fine grid ---
    F0, F1, fine = mesh(FINE)
    with torch.no_grad():
        G_model = decode(model, fine).view(FINE, FINE, 4)
    G_fit = fit(fine).float().view(FINE, FINE, 4)
    # fine support: inside the zonotope (straight edges) and where the interpolated sample density is high enough
    dens = Fn.interpolate(cnt.view(1, 1, BINS, BINS), size=(FINE, FINE), mode="bilinear")[0, 0]
    fine_support = (dens >= FIT_CNT) & inside_zonotope(W, fine).view(FINE, FINE)
    for G in (G_model, G_fit):
        G[~fine_support] = float("nan")
    label = f"{name}, seed {rec['seed']}"
    surfaces([G_bin[:, :, i].numpy() for i in range(4)], U0.numpy(), U1.numpy(), Wn, f"binned decoder, {label}", OUT / f"binned3d_{arch}", AZIM[arch])  # saves binned3d_<arch>
    surfaces([G_fit[:, :, i].clamp(-0.1, 1.1).numpy() for i in range(4)], F0.numpy(), F1.numpy(), Wn,
             f"best decoder of the class (degree {deg}), {label}", OUT / f"class3d_{arch}", AZIM[arch])  # saves class3d_<arch>

    # --- class_diff_<arch>: top-down maps of (model - best of the class) per feature ---
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

    # --- cut_long, this model's half: the curves along the longest embedding's line, collected
    # here and drawn into the shared figure after the loop ---
    k = W.norm(dim=0).argmax().item()
    # the partner is looked at only to take the range needed to construct `ts` (how far the
    # negative arm reaches); the line itself is just an extension of the longest embedding,
    # stored as column k of W — through the origin, in both directions
    pair = next(p["pair"] for p in measure(W)["pairs"] if k in p["pair"])
    partner = pair[0] if pair[1] == k else pair[1]
    ts = torch.linspace(-0.95 * W[:, partner].norm(), 0.95 * W[:, k].norm(), 600)
    cut = ts[:, None] * W[:, k] / W[:, k].norm()
    on = (cnt >= MIN_CNT)[flat_index(cut, lo, hi)]
    with torch.no_grad():
        yc_model = decode(model, cut)[:, k]
    cuts.append((label, k, ts[on], mean[flat_index(cut, lo, hi)][on, k], fit(cut).float()[on, k], yc_model[on]))

# --- cut_long: one panel per model, the curves collected in the loop above ---
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
