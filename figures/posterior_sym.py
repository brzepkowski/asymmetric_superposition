# Figures for the symmetric antipodal pair (s = x1 - x3), all on the same frame:
#   posterior_x1_sym   -- best decoder for x1 alone, black line
#   posterior_pair_sym -- best decoders for x1 (blue) and x3 (orange)
#   samples_x1_sym     -- (s, x1) pairs drawn from the sampling process itself
#   binned_x1_sym      -- binned means of those samples over the closed-form curve
# The decoder plots mark the discontinuity at s = 0 (open circle at the one-sided
# limits p/2, closed at 0). Run from the repo root: python -m figures.posterior_sym
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import BLUE, MUTED, ORANGE, P, arms, x1_posterior, x3_posterior

OUT = Path("figures")
N_SAMPLES, SEED = 4096, 0


def frame(ax, marks=True, guide=True, lo=-1.0, hi=1.0, xticks=(-1, -0.5, 0, 0.5, 1)):
    """The shared dressing of every pair-decoder plot, so the figures of this file (and the
    panels of asym_compare, which imports it) all sit on the same frame: x-label s, fixed
    y-range [0, 1] with its ticks, no top/right spines. Options:
      - marks: the discontinuity markers at s = 0 — an open circle at the one-sided limit
        p/2, a filled one at the value 0. Correct for the symmetric pair only; a caller
        with other levers (asym_compare) passes False and draws its own.
      - guide: the dashed horizontal guide at p/2 with its label.
      - lo, hi: the reading's range (-a, b); sets the x-limits with a small margin.
      - xticks: the x tick positions, for callers whose range is not [-1, 1].
    """
    if marks:
        ax.plot([0], [P / 2], "o", ms=6.5, mfc="white", mec="black", mew=1.6, zorder=4)
        ax.plot([0], [0], "o", ms=6.5, color="black", zorder=5)
    if guide:
        ax.axhline(P / 2, color=MUTED, lw=1, ls="--", zorder=0)
        ax.text(lo + 0.02 * (hi - lo), P / 2 + 0.02, "$p/2$", color=MUTED, ha="left", va="bottom")
    ax.set_xlabel("$s$")
    ax.set_xticks(list(xticks))
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(lo - 0.05, hi + 0.05)
    ax.set_ylim(-0.04, 1.04)
    ax.spines[["top", "right"]].set_visible(False)


def new_fig():
    return plt.subplots(figsize=(5.0, 3.2), layout="constrained")


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13})

    fig, ax = new_fig()
    for s in arms(1.0, 1.0):
        ax.plot(s, x1_posterior(s, 1.0, 1.0), color="black", lw=2.5, zorder=3)
    frame(ax)
    ax.set_ylabel("$\\hat{x}_1(s)$")
    fig.savefig(OUT / "posterior_x1_sym.pdf")
    fig.savefig(OUT / "posterior_x1_sym.png", dpi=200)

    fig, ax = new_fig()
    for s in arms(1.0, 1.0):
        ax.plot(s, x1_posterior(s, 1.0, 1.0), color=BLUE, lw=2.5, zorder=3, label="$\\hat{x}_1(s)$")
        ax.plot(s, x3_posterior(s, 1.0, 1.0), color=ORANGE, lw=2.5, zorder=3, label="$\\hat{x}_3(s)$")
    frame(ax)
    ax.set_ylabel("predicted feature")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="upper center", frameon=False)
    fig.savefig(OUT / "posterior_pair_sym.pdf")
    fig.savefig(OUT / "posterior_pair_sym.png", dpi=200)

    rng = np.random.default_rng(SEED)
    x = (rng.random((N_SAMPLES, 2)) < P) * rng.random((N_SAMPLES, 2))  # (x1, x3) under the sparse prior
    case = (x[:, 0] > 0) + 2 * (x[:, 1] > 0)  # 0 neither, 1 only feature 1, 2 only feature 3, 3 both
    CASES = ((0, "black", "neither active"), (1, BLUE, "only feature 1 active"),
             (2, ORANGE, "only feature 3 active"), (3, "#2CA02C", "both active"))
    fig, ax = new_fig()
    #   - x is the (N_SAMPLES, 2) sample matrix drawn above: column 0 holds each sample's x1,
    #     column 1 its x3.
    #   - m = case == c is a boolean mask selecting the rows belonging to the case being drawn —
    #     neither / only feature 1 / only feature 3 / both.
    #   - x[m, 0] - x[m, 1] is those samples' reading s = x1 - x3 (unit levers here, so no a, b
    #     factors) — the horizontal coordinate.
    #   - x[m, 0] is those samples' true x1 — the vertical coordinate, the value the decoder is
    #     supposed to recover from s.
    for c, color, _ in CASES:
        m = case == c
        ax.scatter(x[m, 0] - x[m, 1], x[m, 0], s=9, color=color, alpha=0.35, lw=0, zorder=4 - c)
    frame(ax, marks=False, guide=False)
    ax.set_ylabel("$x_1$")
    ax.legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=7, color=color, label=label) for _, color, label in CASES],
              loc="upper left", frameon=False, handletextpad=0.3)
    fig.savefig(OUT / "samples_x1_sym.pdf")
    fig.savefig(OUT / "samples_x1_sym.png", dpi=200)

    # binned conditional means of the same samples, over the closed-form curve;
    # the exact-zero readings (the "neither active" atom) are a bin of their own
    N_BINS = 100
    # every sample's reading and true x1 — the same coordinates as the scatter, all cases together
    s_all, x1_all = x[:, 0] - x[:, 1], x[:, 0]
    # N_BINS equal bins over the reading's range [-1, 1], described by their N_BINS + 1 edges
    edges = np.linspace(-1, 1, N_BINS + 1)
    # the samples with reading exactly 0 (the "neither active" atom) are held out of the bins
    nz = s_all != 0
    #   - idx is aligned with s_all[nz], not s_all: the exact-zero readings are masked out before
    #     digitizing, so idx has one entry per nonzero reading — which is why the means line below
    #     selects x1_all[nz] first and only then applies idx == b.
    #   - digitize returns the index of the bin each reading belongs to, counting the bins from 1,
    #     so the -1 makes it a 0-based index.
    #   - the value b in a cell means "this reading lies in bin b", the interval from edges[b] to
    #     edges[b+1] (100 bins, 101 edges) — an index usable directly into means. E.g. if the 5th
    #     nonzero reading is s = -0.97, then idx[4] = 1: bin 1 spans [-0.98, -0.96), and that
    #     reading's x1 is averaged into means[1].
    #   - the clip guards one boundary case: a reading of exactly 1.0 sits on the last edge and
    #     would index one past the last bin.
    idx = np.clip(np.digitize(s_all[nz], edges) - 1, 0, N_BINS - 1)
    # the binned decoder: the mean of x1 over the samples in each bin
    means = np.array([x1_all[nz][idx == b].mean() for b in range(N_BINS)])
    fig, ax = new_fig()
    for s in arms(1.0, 1.0):
        ax.plot(s, x1_posterior(s, 1.0, 1.0), color="black", lw=2.5, zorder=3)
    # the binned means as a step function over the closed-form curve
    ax.stairs(means, edges, baseline=None, color=BLUE, lw=2, zorder=4)
    # the held-out atom: one dot at s = 0, the mean x1 of the exact-zero readings
    ax.plot([0], [x1_all[~nz].mean()], "o", ms=6.5, color=BLUE, zorder=6)
    frame(ax)
    ax.set_ylabel("$\\hat{x}_1(s)$")
    ax.legend(handles=[plt.Line2D([], [], color="black", lw=2.5, label="closed form"),
                       plt.Line2D([], [], color=BLUE, lw=2, label="mean of samples")],
              loc="upper left", frameon=False)
    fig.savefig(OUT / "binned_x1_sym.pdf")
    fig.savefig(OUT / "binned_x1_sym.png", dpi=200)
