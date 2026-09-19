# Representative trained encoder geometries, 1 vs 4 bilinear MLPs: the modal strategy of each
# architecture, its best-eval seed. Also prints the strategy table of the write-up (20 runs per row).
# Run from the repo root: python -m figures.gallery
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import BLUE, MUTED, ORANGE, all_runs, classify

OUT = Path("figures/gallery")
PBLUE = "#3A82B5"
TABLE = (("symmetric pairs", ("neither",)), ("asymmetric pairs", ("asymmetry only",)),
         ("asymmetric + opened pairs", ("asymmetry + tilt",)),
         ("other", ("tilt only", "sacrificed feature", "other geometry")))

plt.rcParams.update({"font.size": 12, "axes.titlesize": 14})
runs = all_runs()
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4), layout="constrained")
for ax, (arch, title) in zip(axes, (("bilinear1", "1 bilinear layer"), ("bilinear4", "4 bilinear layers"))):
    recs = runs[arch]
    modal = Counter(classify(r["m"]) for r in recs).most_common(1)[0][0]
    rec = min((r for r in recs if classify(r["m"]) == modal), key=lambda r: r["eval_mse"])
    W, m = rec["W"], rec["m"]
    color = {i: MUTED for i in range(4)}
    for pair, c in zip(m["pairs"], (BLUE, ORANGE)):
        color[pair["pair"][0]] = color[pair["pair"][1]] = c
    lim = 1.3 * W.abs().max().item()
    for i in range(4):
        ax.annotate("", xy=(W[0, i].item(), W[1, i].item()), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color[i], lw=2.4, shrinkA=0, shrinkB=0))
        ax.text(1.14 * W[0, i].item(), 1.14 * W[1, i].item(), f"$f_{i + 1}$", color=color[i], fontsize=15, ha="center", va="center")
    ax.axhline(0, color=MUTED, lw=0.6, alpha=0.5, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.6, alpha=0.5, zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    print(f"{arch}: seed {rec['seed']}, class {modal}, eval MSE {rec['eval_mse']:.4f}")
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=200)

print("\n| model | " + " | ".join(name for name, _ in TABLE) + " |")
for arch in ("relu_tied", "bilinear1", "bilinear2", "bilinear3", "bilinear4"):
    counts = Counter(classify(r["m"]) for r in runs[arch])
    print(f"| {arch} | " + " | ".join(str(sum(counts[k] for k in keys)) for _, keys in TABLE) + " |")
