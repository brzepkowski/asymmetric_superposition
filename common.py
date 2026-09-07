# Everything the scripts share: the models, the data, the catalogue of trained runs, the
# closed-form decoders and the best decoders of each class. Run everything from this directory:
#   python train.py             the 100 runs of the strategy table (skips runs already in checkpoints/)
#   python -m figures.<name>    one figure script (see reproduce.sh for the full list)
import json
import math
from collections import Counter
from itertools import combinations, combinations_with_replacement
from pathlib import Path

import numpy as np
import torch

INK, MUTED = "#0b0b0b", "#898781"
BLUE, ORANGE, AQUA, PBLUE = "#2a78d6", "#eb6834", "#1baf7a", "#3A82B5"
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

N, D, P, STEPS = 4, 2, 0.2, 20_000
MLP_LAYERS = {"bilinear1": [(N, N)], "bilinear": [(N, N)] * 2, "bilinear3": [(N, N)] * 3, "bilinear4": [(N, N)] * 4}
ARCHES = tuple(MLP_LAYERS) + ("relu_tied",)
SEEDS = range(20)
CKPT = Path("checkpoints")
DEAD_EPS, PAIR_COS, RHO_T, TILT_T = 0.05, 0.7, 1.2, 5.0  # geometry classification thresholds


class Bilinear(torch.nn.Linear):
    def __init__(self, d_in, d_out, bias):
        super().__init__(d_in, 2 * d_out, bias=bias)

    def forward(self, x):
        left, right = super().forward(x).chunk(2, dim=-1)
        return left * right


class MLP(torch.nn.Module):
    def __init__(self, d_in, d_hidden, d_out, bias=True):
        super().__init__()
        self.w = Bilinear(d_in, d_hidden, bias)
        self.p = torch.nn.Linear(d_hidden, d_out, bias=bias)

    def forward(self, x):
        return self.p(self.w(x))


class SuperModel(torch.nn.Module):  # linear encoder n -> d, then a stack of bilinear MLPs
    def __init__(self, n, d, mlp_layers, bias=True):
        super().__init__()
        self.encoder = torch.nn.Linear(n, d, bias=bias)
        d_in, mlps = d, []
        for d_hidden, d_out in mlp_layers:
            mlps.append(MLP(d_in, d_hidden, d_out, bias))
            d_in = d_out
        self.mlp_layers = torch.nn.ModuleList(mlps)

    @property
    def w_enc(self):
        return self.encoder.weight

    def forward(self, x):
        z = self.encoder(x)
        for layer in self.mlp_layers:
            z = layer(z)
        return z


class TiedReLU(torch.nn.Module):  # Toy Models of Superposition: relu(W^T W x + b)
    def __init__(self, n, d):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(d, n))
        torch.nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        self.b = torch.nn.Parameter(torch.zeros(n))

    @property
    def w_enc(self):
        return self.W

    def forward(self, x):
        return torch.relu(x @ self.W.T @ self.W + self.b)


def make_model(arch):
    return TiedReLU(N, D) if arch == "relu_tied" else SuperModel(N, D, MLP_LAYERS[arch])


def sparse_batch(m, n, p):  # x_i = Bernoulli(p) * U[0, 1], from the global RNG
    x = torch.rand(m, n)
    return x * (torch.rand(m, n) < p)


def sample_x(m, p, g):  # the same distribution from a dedicated generator
    return (torch.rand(m, N, generator=g) < p) * torch.rand(m, N, generator=g)


def load_model(path):
    blob = torch.load(path)
    model = SuperModel(blob["n"], blob["d"], blob["mlp_layers"]) if "mlp_layers" in blob else TiedReLU(blob["n"], blob["d"])
    model.load_state_dict(blob["state_dict"])
    return model


def train(arch, seed, steps=STEPS, root=CKPT, batch=4096, device="cpu"):  # one run: AdamW on a fixed batch, no penalty
    run_dir = root / arch / f"seed{seed}"
    if (run_dir / "model.pt").exists():
        return run_dir
    torch.manual_seed(seed)
    model = make_model(arch).to(device)
    X = sparse_batch(batch, N, P).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        mse = torch.nn.functional.mse_loss(model(X), X)
        opt.zero_grad()
        mse.backward()
        opt.step()
        sched.step()
    run_dir.mkdir(parents=True, exist_ok=True)
    blob = {"n": N, "d": D, "state_dict": model.cpu().state_dict()}
    blob.update({"mlp_layers": MLP_LAYERS[arch]} if arch in MLP_LAYERS else {"arch": arch})
    torch.save(blob, run_dir / "model.pt")
    json.dump({"arch": arch, "seed": seed, "p_active": P, "steps": steps, "batch": batch, "task_mse": mse.item()},
              open(run_dir / "metadata.json", "w"), indent=2)
    print(f"{arch:9s} seed {seed:2d}  train mse {mse.item():.2e}", flush=True)
    return run_dir


def measure(W):
    """Reads the encoder's geometry off its weight W (shape (d, n); column i is the direction
    feature i is written to). Repeatedly takes the two most collinear remaining columns and,
    when they point in nearly opposite directions (cosine <= -PAIR_COS), records them as an
    antipodal pair. Returns a dict with three entries:
      "dead"  — list of feature indices the model dropped: their column norm is below
                DEAD_EPS of the largest column's norm.
      "pairs" — list with one dict per antipodal pair found: "pair" is the two feature
                indices, "cos" the cosine between their columns (-1 would be exactly
                antipodal), "ratio" the longer column's norm over the shorter one's
                (1 would be perfectly symmetric).
      "class" — coarse label for the whole encoder: "sacrifice" if any feature is dead,
                "pairs" if instead all features sit in two antipodal pairs, else "other".
    """
    norms = W.norm(dim=0)
    dead = [i for i in range(N) if norms[i] < DEAD_EPS * norms.max()]
    alive = [i for i in range(N) if i not in dead]
    Wn = W[:, alive] / norms[alive].clamp(min=1e-8)  # live columns, each scaled to unit length
    C = Wn.T @ Wn  # Gram matrix of those unit columns: C[a, b] is the cosine between them
    left, pairs = list(range(len(alive))), []
    while len(left) >= 2:
        # find the two most collinear remaining columns, aligned or anti-aligned (largest |cos|)
        a, b = max(((a, b) for a in left for b in left if a < b), key=lambda ab: C[ab].abs().item())
        if C[a, b].item() <= -PAIR_COS:  # record them only if nearly opposite
            i, j = alive[a], alive[b]
            pairs.append({"pair": (i, j), "cos": C[a, b].item(),
                          "ratio": (norms[[i, j]].max() / norms[[i, j]].min()).item()})
        left = [c for c in left if c not in (a, b)]  # either way, both leave the pool
    cls = "pairs" if not dead and len(pairs) == 2 else ("sacrifice" if dead else "other")
    return {"dead": dead, "class": cls, "pairs": pairs}


def tilt(cos):
    """Turns a pair's cosine into how many degrees it falls short of exactly antipodal:
    cos = -1 gives 0° (perfect pair), cos = -0.94 gives ~20°. The min/max only clamp
    float rounding so acos never sees a value outside [-1, 1].
    """
    return 180.0 - math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def classify(m):
    """Turns measure()'s output m into the strategy label used in the write-up's table.
    "sacrifice" and "other" pass through as their labels; a two-pair encoder is examined
    further: a pair is asymmetric when its norm ratio is at least RHO_T, tilted when it is
    at least TILT_T degrees off exactly antipodal, and the label says which of the two
    deviations occur anywhere in the encoder ("neither" = two clean antipodal pairs).
    """
    if m["class"] == "sacrifice":
        return "sacrificed feature"
    if m["class"] == "other":
        return "other geometry"
    asym = any(p["ratio"] >= RHO_T for p in m["pairs"])
    til = any(tilt(p["cos"]) >= TILT_T for p in m["pairs"])
    return "asymmetry + tilt" if asym and til else "asymmetry only" if asym else "tilt only" if til else "neither"


def all_runs():
    """The catalogue of trained runs: loads every checkpoints/<arch>/seed<k>/model.pt and scores
    them all on one fixed batch (seeded 9999). Returns {arch name: list of runs sorted by
    seed}; each run is a dict with keys
      "arch", "seed", "path" — which run this is and where its model.pt lives,
      "W"        — the encoder weight (d, n),
      "m"        — measure(W), the encoder's geometry,
      "eval_mse" — the model's reconstruction MSE on the shared batch.
    E.g. all_runs()["bilinear4"][9]["eval_mse"].
    """
    torch.manual_seed(9999)
    X_eval = sparse_batch(65536, N, P)
    runs = {a: [] for a in ARCHES}
    for path in sorted(CKPT.glob("*/seed*/model.pt")):
        arch, seed = path.parent.parent.name, int(path.parent.name[4:])
        model = load_model(path)
        with torch.no_grad():
            mse = torch.nn.functional.mse_loss(model(X_eval), X_eval).item()
        W = model.w_enc.detach()
        runs[arch].append({"arch": arch, "seed": seed, "path": path, "W": W, "m": measure(W), "eval_mse": mse})
    return {a: sorted(rs, key=lambda r: r["seed"]) for a, rs in runs.items()}


# --- the closed-form decoder of an antipodal pair ---------------------------------------------------
# The exact Bayes decoder for one antipodal pair, derived in the write-up's appendix. The two
# features share one bottleneck axis with opposite signs, so all the decoder ever sees is the
# reading s = |f_self| x_self - |f_partner| x_partner, and the best possible guess for a feature is
# its posterior mean given that reading. posterior_mean(t, w, v, p) evaluates it: E[x_self | s = t],
# with w = the signed embedding length of the feature being decoded, v = the partner's (opposite
# sign), p = the activation probability. The appendix's symmetric case s = x1 - x3 is w = 1, v = -1;
# unequal |w|, |v| give the asymmetric pairs of the sweeps.

def posterior_mean(t, w, v, p):
    """E[x_self | s = t]: the best possible guess for the decoded feature given the reading.

    t — tensor of reading values; w, v — signed embedding lengths of the decoded feature and
    of its partner (opposite signs); p — activation probability. Returns a tensor like t,
    NaN where no combination of the two features can produce that reading.

    The body is the appendix's weighted average over cases: "only" flags readings the decoded
    feature can produce alone, "ghost" those the partner produces alone. L and R are the
    endpoints of the interval x_self is uniform on when both features are active (the
    appendix's [s, 1] for s > 0, generalized to arbitrary lever lengths), so (R + L)/2 is the
    co-active best guess and width = R - L takes the place of the triangle density; num/den
    are the case-weighted mean and total density.
    """
    if w < 0:
        t, w, v = -t, -w, -v
    # In the appendix_square figure: each diagonal of the unit square is the set of (x1, x3)
    # producing one reading. [L, R] is its projection onto the x1 axis, and width as a function
    # of t is the right panel's triangle density 1 - |s| (generalized to unequal lever lengths).
    # E.g. (with w = 1, v = -1):
    #   s = 0.75:  diagonal (0.75, 0) to (1, 0.25)  ->  L = 0.75, R = 1,    width = 0.25
    #   s = 0:     diagonal (0, 0)    to (1, 1)     ->  L = 0,    R = 1,    width = 1
    #   s = -0.25: diagonal (0, 0.25) to (0.75, 1)  ->  L = 0,    R = 0.75, width = 0.75
    L = (t / w).clamp(min=0)  # the reading in feature units with the partner at its minimum (0)
    R = ((t - v) / w).clamp(max=1)  # same with the partner at its maximum (1); both clamped to [0, 1]
    width = (R - L).clamp(min=0)
    # 0/1 indicators of the two single-feature cases:
    #   - "only": the decoded feature alone gives s = w * x_self, so it can produce exactly
    #     the readings in (0, w).
    #   - "ghost": the partner alone covers (v, 0); the decoded feature is then 0, so the case
    #     adds density (see den) but nothing to the mean (absent from num) — this is what drags
    #     the ghost region's guess toward 0.
    #   - the "neither" case needs no indicator: its point mass at s = 0 falls outside both
    #     intervals, where den = 0 returns NaN.
    only = ((t > 0) & (t < w)).float()
    ghost = ((t > v) & (t < 0)).float()
    # num, with the grouping made explicit:
    #     num = [p(1-p)]·[only/w]·(t/w)  +  [p²]·[width/(-v)]·((R+L)/2)
    #            prior  · density · mean     prior ·  density  ·  mean
    # The means:
    #   - self alone: s = w * x_self is deterministic, so the reading pins the feature
    #     exactly at t/w.
    #   - both active: the reading only confines x_self to [L, R], uniformly, so the best
    #     guess is the midpoint (R + L)/2.
    #   - ghost (partner alone): the decoded feature is off, mean 0, so its term isn't written.
    # The densities are explained below, at den.
    num = p * (1 - p) * only * t / w ** 2 + p ** 2 * width * (R + L) / (2 * -v)
    # den is the total density of the reading (the Bayes normalizer), prior x density per case:
    #   - self alone:    prior p(1-p), density 1/w on (0, w)
    #   - partner alone: prior p(1-p), density 1/(-v) on (v, 0)
    #   - both active:   prior p^2, density width/(-v)
    # Where the both-active density comes from: pin x_self at some value first. Then
    #     s = w * x_self + v * x_partner
    # is just "partner alone, shifted by the constant w * x_self", and a lone uniform feature on
    # lever v has density 1/(-v) (the partner-alone case above). So every allowed x_self value
    # contributes density 1/(-v), and the allowed values are exactly the interval [L, R]:
    #     density = integral over [L, R] of 1/(-v) dx_self = (R - L)/(-v) = width/(-v)
    # In the symmetric case w = 1, v = -1 this is width/1 = 1 - |s|, the appendix's triangle.
    den = p * (1 - p) * (only / w + ghost / -v) + p ** 2 * width / -v
    return torch.where(den > 0, num / den, torch.nan)


def x1_posterior(s, a, b, p=P):
    """E[x1 | s] on the pair with lever lengths |f1| = b, |f3| = a and reading s = b x1 - a x3.
    Same function as posterior_mean(s, b, -a, p), but in numpy, with the num/den algebra worked
    out into one explicit formula per region:
      - "own" (s > 0): f1 alone or both active,
      - "plateau" (mild negative s): the reading says nothing about x1 — exactly p/2,
      - "tail" (strongly negative s): even a maxed-out partner caps x1 below 1,
      - NaN outside (-a, b).
    Valid for a >= b only.
    """
    own, plateau, tail = (s > 0) & (s < b), (s > b - a) & (s < 0), (s > -a) & (s <= b - a)
    out = np.full_like(s, np.nan)
    out[own] = ((2 * (1 - p) * a * s[own] + p * (b ** 2 - s[own] ** 2))
                / (2 * b * ((1 - p) * a + p * (b - s[own]))))
    out[plateau] = p / 2
    out[tail] = p * (a + s[tail]) ** 2 / (2 * b * ((1 - p) * b + p * (a + s[tail])))
    return out


def x3_posterior(s, a, b, p=P):
    """E[x3 | s] on the same pair: the mirror of x1_posterior, i.e. posterior_mean(s, -a, b, p).
    x3 pulls the reading negative, so its regions run the other way:
      - "far" (strongly negative s): x3 alone or both active,
      - "own" (mild negative s): the linear stretch where the reading tracks x3 directly,
      - "other" (s > 0): f1 alone or both active,
      - NaN outside (-a, b).
    Valid for a >= b only.
    """
    far, own, other = (s > -a) & (s < b - a), (s >= b - a) & (s < 0), (s > 0) & (s < b)
    out = np.full_like(s, np.nan)
    out[far] = ((-2 * (1 - p) * b * s[far] + p * (a ** 2 - s[far] ** 2))
                / (2 * a * ((1 - p) * b + p * (a + s[far]))))
    out[own] = -s[own] / a + p * b / (2 * a)
    out[other] = p * (b - s[other]) ** 2 / (2 * a * ((1 - p) * a + p * (b - s[other])))
    return out


def arms(a, b):
    """Plotting grid for the posterior curves: the reading segment (-a, b) as two separate
    arrays, the s < 0 and s > 0 arms. Separate because the curves jump at s = 0 — one
    continuous array would draw a spurious vertical segment across the discontinuity.
    """
    return [np.linspace(lo + 1e-6, hi - 1e-6, 600) for lo, hi in ((-a, 0.0), (0.0, b))]


# --- the best decoder of a class ----------------------------------------------------------------------

CLASS_DEG = {"quadratic": 2, "quartic": 4, "deg8": 8, "deg16": 16}


def class_fit(kind, s, Y, a, b, grid):
    # on one pair (samples s, targets Y = (x1, x3)): least squares for the polynomial classes; for the
    # tied ReLU the slopes are fixed by the encoder (g*a for x1, g*b for x3) and the scale g and the two
    # thresholds are found by a grid search. Returns the fit on `grid` and the MSE per feature.
    if kind in CLASS_DEG:
        deg = CLASS_DEG[kind]
        smin, smax = s.min(), s.max()
        tr = lambda u: ((2 * u - smin - smax) / (smax - smin)).double()
        S = torch.from_numpy(np.polynomial.chebyshev.chebvander(tr(s).numpy(), deg))
        C = torch.linalg.lstsq(S, Y.double()).solution
        G = torch.from_numpy(np.polynomial.chebyshev.chebvander(tr(grid).numpy(), deg))
        return (G @ C).float(), ((S @ C - Y.double()) ** 2).mean(0).float()
    cs = torch.linspace(s.min().item() - 0.3, s.max().item() + 0.3, 221)
    ey2 = (Y ** 2).mean(0)
    stats = {}
    for sig in (1.0, -1.0):
        eyh, ehh = torch.empty(2, len(cs)), torch.empty(len(cs))
        for k, c in enumerate(cs.tolist()):
            h = torch.relu(sig * (s - c))
            ehh[k] = (h ** 2).mean()
            eyh[:, k] = (Y * h[:, None]).mean(0)
        stats[sig] = (eyh, ehh)
    best = None
    for g in torch.logspace(-1.5, 1.5, 61).tolist():
        tot, prs, mses = 0.0, [], []
        for i, (sig, slope) in enumerate(((1.0, g * a), (-1.0, g * b))):
            eyh, ehh = stats[sig]
            m = ey2[i] - 2 * slope * eyh[i] + slope ** 2 * ehh
            k = m.argmin().item()
            tot += m[k].item()
            prs.append((sig, slope, cs[k].item()))
            mses.append(m[k].item())
        if best is None or tot < best[0]:
            best = (tot, prs, mses)
    _, prs, mses = best
    F = torch.stack([slope * torch.relu(sig * (grid - c)) for sig, slope, c in prs], 1)
    return F, torch.tensor(mses)


def poly_predictor(z, x, degree):  # least-squares polynomial of total degree `degree` on the reading plane
    zmin, zmax = z.min(0).values, z.max(0).values

    def basis(zz):
        z01 = (2 * (zz - zmin) / (zmax - zmin) - 1).double()
        T = [torch.ones_like(z01), z01]
        for _ in range(degree - 1):
            T.append(2 * z01 * T[-1] - T[-2])
        cols = []
        for deg in range(degree + 1):
            for c in combinations_with_replacement(range(2), deg):
                col = torch.ones(len(zz), dtype=torch.float64)
                for v, e in Counter(c).items():
                    col = col * T[e][:, v]
                cols.append(col)
        return torch.stack(cols, 1)

    sol = torch.linalg.lstsq(basis(z), x.double()).solution
    return lambda zz: (basis(zz) @ sol).float()


def oracle_xhat(W, z):
    # the unconstrained decoder to leading order in p: a reading on an embedding line is read as that
    # feature alone; any other reading is the density-weighted average over the pairs of features whose
    # parallelogram contains it
    xhat = torch.zeros(len(z), N)
    resolved = z.norm(dim=1) < 1e-9
    for i in range(N):
        w = W[:, i]
        n2 = w @ w
        t = (z @ w) / n2
        on = (~resolved) & ((z[:, 0] * w[1] - z[:, 1] * w[0]).abs() < 1e-6 * n2.sqrt()) \
            & (t >= -1e-9) & (t <= 1 + 1e-9)
        xhat[on, i] = t[on]
        resolved = resolved | on
    wsum = torch.zeros(len(z))
    acc = torch.zeros(len(z), N)
    for i, j in combinations(range(N), 2):
        M = torch.stack([W[:, i], W[:, j]], 1)
        Dt = torch.det(M).abs()
        if Dt < 1e-9:
            continue
        u = z @ torch.linalg.inv(M).T
        inside = (~resolved) & (u >= -1e-9).all(1) & (u <= 1 + 1e-9).all(1)
        wgt = inside.float() / Dt
        wsum += wgt
        acc[:, i] += wgt * u[:, 0]
        acc[:, j] += wgt * u[:, 1]
    mix = (~resolved) & (wsum > 0)
    xhat[mix] = acc[mix] / wsum[mix, None]
    return xhat


def bayes2d(z, x, bins=64):  # MSE of the binned decoder on the reading plane (bins in the whitened frame)
    cov = torch.cov(z.T) + 1e-9 * torch.eye(2)
    zw = z @ torch.linalg.cholesky(torch.linalg.inv(cov))
    idx = torch.zeros(len(z), dtype=torch.long)
    for d in range(2):
        e = torch.linspace(zw[:, d].min(), zw[:, d].max() + 1e-5, bins + 1)
        idx = idx * bins + (torch.bucketize(zw[:, d], e) - 1).clamp(0, bins - 1)
    B = bins * bins
    cnt = torch.zeros(B).index_add_(0, idx, torch.ones(len(z)))
    mean = torch.zeros(B, x.shape[1]).index_add_(0, idx, x) / cnt.clamp(min=1)[:, None]
    return (x - mean[idx]).pow(2).mean().item()


def binned_fit(z, x, bins):
    # the binned decoder on the reading plane: bins x bins cells over the range of z, the origin at a cell
    # centre so that no cell edge runs along an embedding line; returns the cell-mean predictor (a cell
    # never seen predicts 0)
    h = (z.max(0).values - z.min(0).values) / bins
    cell = torch.floor(z / h + 0.5).long()
    lo, wide = cell.min(0).values, cell.max(0).values - cell.min(0).values + 1
    flat = lambda c: ((c - lo).clamp(min=0) * torch.tensor([wide[1], 1])).sum(1).clamp(max=wide.prod() - 1)
    idx = flat(cell)
    cnt = torch.zeros(wide.prod()).index_add_(0, idx, torch.ones(len(z)))
    mean = torch.zeros(wide.prod(), x.shape[1]).index_add_(0, idx, x) / cnt.clamp(min=1)[:, None]
    return lambda zz: mean[flat(torch.floor(zz / h + 0.5).long())]


def parallelogram(u, v):
    return [(0, 0), tuple(u), (u[0] + v[0], u[1] + v[1]), tuple(v)]
