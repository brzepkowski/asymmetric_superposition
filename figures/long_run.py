# The four-MLP model of class_check.py (seed 9, 2^20 samples) trained longer and measured the same way: MSE
# of the binned decoder of its encoder, of the best polynomial of degree 16, and of the model itself, and
# the MSE gap. Two kinds of runs: retrained from the same initialization over a longer cosine schedule
# (cached under checkpoints/big_long/ for 150k, checkpoints/big_<steps>/ otherwise), and one trajectory
# continued from the 20k checkpoint of class_check.py at a lower peak lr, which keeps the geometry that run
# discovered: each milestone continues the previous one's checkpoint and anneals its own cosine, so every
# measured point is an annealed endpoint (cached under checkpoints/big_cont_<total extra steps>/).
# Run from the repo root:
#   python -m figures.long_run                    all runs
#   python -m figures.long_run <steps ...>        retrained runs only
#   python -m figures.long_run cont <steps ...>   the continued trajectory, chained through the listed milestones
import sys
from collections.abc import Sequence

import torch

from common import CKPT, P, classify, load_model, measure, poly_predictor, sample_x, train

SEED, DEG, BINS = 9, 16, 40
INIT = CKPT / "big" / "bilinear4" / f"seed{SEED}" / "model.pt"
scratch = lambda s: (s, CKPT / ("big_long" if s == 150_000 else f"big_{s}"), None, 1e-3, s)


def chain(milestones: Sequence[int]):
    """The job list of the continued trajectory. `milestones` holds increasing step counts,
    e.g. (130_000, 280_000, 580_000) in the default run:
      - each milestone is a running total of extra steps beyond the 20k checkpoint (INIT);
      - its job trains for the difference from the previous milestone, at the lower peak
        lr 1e-4, starting from the previous milestone's cached model (INIT for the first);
      - the result is cached under checkpoints/big_cont_<milestone>/;
      - the returned jobs are the same 5-tuples as scratch()'s, which the main loop consumes:
        (steps to train, checkpoint root, init checkpoint, lr, the total for the printed label);
      - since train() skips cached runs, rerunning a chain only trains the milestones not
        yet on disk.
    """
    prev, path, jobs = 0, INIT, []
    for s in milestones:
        root = CKPT / f"big_cont_{s}"
        jobs.append((s - prev, root, path, 1e-4, s))
        prev, path = s, root / "bilinear4" / f"seed{SEED}" / "model.pt"
    return jobs


if sys.argv[1:]:
    is_cont = sys.argv[1] == "cont"
    steps_list = [int(s) for s in sys.argv[1 + is_cont:]]
    jobs = chain(steps_list) if is_cont else [scratch(s) for s in steps_list]
else:
    jobs = [scratch(s) for s in (150_000, 300_000, 600_000)] + chain((130_000, 280_000, 580_000))

x = sample_x(2 ** 20, P, torch.Generator().manual_seed(7))  # the sample of class_check.py
for steps, root, init, lr, total in jobs:
    model = load_model(train("bilinear4", SEED, steps, root, batch=2 ** 20, device="cuda", init=init, lr=lr) / "model.pt")
    W = model.w_enc.detach()
    u = x @ W.T
    lo, hi = u.min(0).values - 1e-3, u.max(0).values + 1e-3
    idx = ((u - lo) / (hi - lo) * BINS).long().clamp(0, BINS - 1)
    flat = idx[:, 0] * BINS + idx[:, 1]
    cnt = torch.zeros(BINS * BINS).index_add_(0, flat, torch.ones(len(u)))
    mean = torch.zeros(BINS * BINS, 4).index_add_(0, flat, x) / cnt.clamp(min=1)[:, None]
    with torch.no_grad():
        y_model = model(x)
    y_fit = poly_predictor(u, x, DEG)(u).float()
    mse = {k: (x - y).pow(2).mean().item() for k, y in (("binned", mean[flat]), ("fit", y_fit), ("model", y_model))}
    m = measure(W)
    label = f"20,000 + {total:,} steps continued at lr {lr:g}" if init else f"{total:,} steps"
    print(f"geometry: {classify(m)}; " + "; ".join(f"pair {p['pair']} ratio {p['ratio']:.2f} cos {p['cos']:.3f}" for p in m["pairs"]))
    print(f"four bilinear MLPs, seed {SEED}, {label}: MSE binned {mse['binned']:.4f}  best of class {mse['fit']:.4f}  "
          f"model {mse['model']:.4f}  gap {(mse['model'] - mse['fit']) / mse['model'] * 100:.1f}%", flush=True)
