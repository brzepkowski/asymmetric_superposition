# The 4-feature pipeline rerun with 5 features in the same 2D bottleneck (exploration, not in the post):
#   python five_features.py search        the free geometry search of "Search for the best strategy"
#   python five_features.py train [seeds] the strategy-table runs, 20 seeds x 5 archs (checkpoints/n5/)
#   python five_features.py table         per-run geometries and the per-arch aggregate
#   python five_features.py plot          all 100 trained encoder geometries (figures/five_geometries)
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize

import common
from common import BLUE, MUTED, ORANGE, P, binned_fit, load_model, measure, sample_x, sparse_batch, tilt, train

common.N = 5
common.MLP_LAYERS = {a: [(5, 5)] * len(l) for a, l in common.MLP_LAYERS.items()}
ROOT = common.CKPT / "n5"
M, BINS, N_RAND, N_START = 2 ** 19, 96, 150, 3
LO, HI = math.log(0.05), math.log(1.5)  # range of the lengths in the random starts
PENTAGON = (0.8, 1.6, 2.4, 3.2, 0.0, 0.0, 0.0, 0.0, 0.0)


def kernel(q):  # q = angles of f2..f5 in units of 90 deg (f1 pinned to the +x axis) + log lengths of f1..f5
    ang = [0.0] + [qi * math.pi / 2 for qi in q[:4]]
    r = [math.exp(qi) for qi in q[4:]]
    return torch.tensor([[ri * math.cos(t) for ri, t in zip(r, ang)],
                         [ri * math.sin(t) for ri, t in zip(r, ang)]])


def binned_mse(z, x):
    return (x - binned_fit(z, x, BINS)(z)).pow(2).mean().item()


def report(q):
    W = kernel(q)
    m = measure(W)
    pairs = "; ".join(f"(f{p['pair'][0] + 1}, f{p['pair'][1] + 1}) opened {tilt(p['cos']):.1f} deg, ratio {p['ratio']:.1f}"
                      for p in m["pairs"]) or "no pairs"
    dead = " dead: " + ",".join(f"f{i + 1}" for i in m["dead"]) if m["dead"] else ""
    return pairs + dead + "   " + " ".join(f"f{i + 1}=({W[0, i]:+.3f}, {W[1, i]:+.3f})" for i in range(5))


mode = sys.argv[1]
if mode == "search":
    x = sample_x(M, P, torch.Generator().manual_seed(11))
    x_fresh = sample_x(M, P, torch.Generator().manual_seed(22))
    obj = lambda q: binned_mse(x @ kernel(q).T, x)
    fresh = lambda q: binned_mse(x_fresh @ kernel(q).T, x_fresh)
    g = torch.Generator().manual_seed(3)
    cand = []
    for _ in range(N_RAND):
        u = torch.rand(9, generator=g)
        q = [4 * v for v in u[:4].tolist()] + [LO + v * (HI - LO) for v in u[4:].tolist()]
        cand.append((obj(q), q))
    cand.sort(key=lambda t: t[0])
    starts = [(f"random start {i + 1}", q) for i, (_, q) in enumerate(cand[:N_START])] + [("pentagon", PENTAGON)]
    best = None
    for name, q0 in starts:
        res = minimize(obj, q0, method="Nelder-Mead", options=dict(xatol=5e-3, fatol=2e-6, maxfev=2000))
        score = fresh(res.x)
        print(f"{name:15s} -> {res.fun:.5f} (search sample), {score:.5f} (fresh): {report(res.x)}", flush=True)
        if best is None or score < best[0]:
            best = (score, res.x)
    print(f"\nbest geometry: {report(best[1])}")
    print(f"MSE floor per feature: {best[0]:.5f};  pentagon (unoptimized): {fresh(PENTAGON):.5f}")
elif mode == "train":
    for seed in [int(s) for s in sys.argv[2:]] or common.SEEDS:
        for arch in common.ARCHES:
            train(arch, seed, root=ROOT)
elif mode == "table":
    torch.manual_seed(9999)
    X = sparse_batch(65536, 5, P)
    for arch in common.ARCHES:
        counts = {}
        for path in sorted((ROOT / arch).glob("seed*/model.pt"), key=lambda p: int(p.parent.name[4:])):
            model = load_model(path)
            with torch.no_grad():
                mse = torch.nn.functional.mse_loss(model(X), X).item()
            W = model.w_enc.detach()
            m = measure(W)
            label = f"{len(m['dead'])} dead, {len(m['pairs'])} pairs"
            counts[label] = counts.get(label, 0) + 1
            norms = " ".join(f"{v:.2f}" for v in sorted(W.norm(dim=0).tolist(), reverse=True))
            pairs = "; ".join(f"(f{p['pair'][0] + 1},f{p['pair'][1] + 1}) {tilt(p['cos']):.0f} deg, r {p['ratio']:.1f}"
                              for p in m["pairs"]) or "no pairs"
            print(f"{arch:9s} seed {int(path.parent.name[4:]):2d}  mse {mse:.4f}  norms {norms}  {pairs}")
        print(f"{arch}: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + "\n")
elif mode == "plot":
    TITLES = {"bilinear1": "1 bilinear MLP", "bilinear2": "2 bilinear MLPs", "bilinear3": "3 bilinear MLPs",
              "bilinear4": "4 bilinear MLPs", "relu_tied": "tied ReLU"}
    torch.manual_seed(9999)
    X = sparse_batch(65536, 5, P)
    fig, axes = plt.subplots(10, 10, figsize=(16, 17), layout="constrained")
    for r, arch in enumerate(common.ARCHES):
        for k, path in enumerate(sorted((ROOT / arch).glob("seed*/model.pt"), key=lambda p: int(p.parent.name[4:]))):
            model = load_model(path)
            with torch.no_grad():
                mse = torch.nn.functional.mse_loss(model(X), X).item()
            W = model.w_enc.detach()
            ax = axes[2 * r + k // 10, k % 10]
            color = {i: MUTED for i in range(5)}
            for pair, c in zip(measure(W)["pairs"], (BLUE, ORANGE)):
                color[pair["pair"][0]] = color[pair["pair"][1]] = c
            lim = 1.3 * W.abs().max().item()
            for i in range(5):
                ax.annotate("", xy=(W[0, i].item(), W[1, i].item()), xytext=(0, 0),
                            arrowprops=dict(arrowstyle="-|>", color=color[i], lw=1.5, shrinkA=0, shrinkB=0))
                ax.text(1.16 * W[0, i].item(), 1.16 * W[1, i].item(), f"$f_{i + 1}$", color=color[i],
                        fontsize=7, ha="center", va="center")
            ax.axhline(0, color=MUTED, lw=0.4, alpha=0.5, zorder=0)
            ax.axvline(0, color=MUTED, lw=0.4, alpha=0.5, zorder=0)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"seed {k}, mse {mse:.4f}", fontsize=8)
        axes[2 * r, 0].set_ylabel(TITLES[arch], fontsize=12)
    fig.savefig("figures/five_geometries.pdf")
    fig.savefig("figures/five_geometries.png", dpi=200)
