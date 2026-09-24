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
MLP_LAYERS = {"bilinear1": [(N, N)], "bilinear2": [(N, N)] * 2, "bilinear3": [(N, N)] * 3, "bilinear4": [(N, N)] * 4}
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


def train(arch, seed, steps=STEPS, root=CKPT, batch=4096, device="cpu", init=None, lr=1e-3):  # one run: AdamW on a fixed batch, no penalty
    run_dir = root / arch / f"seed{seed}"
    if (run_dir / "model.pt").exists():
        return run_dir
    torch.manual_seed(seed)
    model = make_model(arch).to(device)
    if init:  # continue from a checkpoint instead of the seed's initialization (the RNG use stays the same, so the batch does too)
        model.load_state_dict(torch.load(init)["state_dict"])
    X = sparse_batch(batch, N, P).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
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
    json.dump({"arch": arch, "seed": seed, "p_active": P, "steps": steps, "batch": batch, "lr": lr,
               **({"init": str(init)} if init else {}), "task_mse": mse.item()},
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
    """Finds the best decoder of one antipodal pair within the function class named by `kind`
    (a polynomial class from CLASS_DEG, or the tied ReLU's pair of hinges), by fitting it to a
    sample of readings and their targets. Inputs:
      - `kind`: the decoder class.
      - `s`: the sampled readings on the pair (M scalars).
      - `Y`: the (M, 2) targets behind those readings, columns (x1, x3).
      - `a`, `b`: the pair's two embedding lengths (used only by the tied ReLU, which assumes
        the feature in column 0 of Y reads the positive arm with length `a`).
      - `grid`: where to evaluate the fitted decoder for plotting; plays no role in the fit.
    Returns the fitted curves on `grid` (len(grid), 2) and the per-feature MSE on the sample.
    """
    # ----- Polynomial case -----
    if kind in CLASS_DEG:
        deg = CLASS_DEG[kind]
        smin, smax = s.min(), s.max()
        tr = lambda u: ((2 * u - smin - smax) / (smax - smin)).double()
        # Build the design matrix for the fit: each row corresponds to one sample reading s_i,
        # and holds the values of all deg+1 Chebyshev basis polynomials T_0..T_deg at that
        # (rescaled) reading. Fitting in the Chebyshev basis instead of raw powers of s keeps
        # the least-squares problem well conditioned at degrees 8 and 16.
        S = torch.from_numpy(np.polynomial.chebyshev.chebvander(tr(s).numpy(), deg))  # (M, deg+1)
        # Solve the least-squares problem S @ C ~= Y. Each column of C holds the polynomial
        # coefficients for one feature (x1 or x3). Since the class is linear in these
        # coefficients, this solution is exactly the best decoder in the class — no search needed.
        C = torch.linalg.lstsq(S, Y.double()).solution  # (deg+1, 2)
        # Evaluate the same basis polynomials on the plotting grid, using the same rescaling
        # `tr` (fitted on the sample's range) so that the coefficients in C mean the same thing.
        G = torch.from_numpy(np.polynomial.chebyshev.chebvander(tr(grid).numpy(), deg))  # (len(grid), deg+1)
        # Return the fitted decoder curves evaluated on the grid (grid basis times coefficients),
        # and the mean squared residual of the fit on the sample, separately for each feature.
        return (G @ C).float(), ((S @ C - Y.double()) ** 2).mean(0).float()  # (len(grid), 2), (2,)
    # ----- Tied ReLU case -----
    # Each feature's decoder is a hinge, slope * relu(±(s - c)): flat at zero on one side of a
    # threshold c, linear on the other. Weight tying fixes the slopes to g*a (x1) and g*b (x3)
    # with one shared gain g, so only three numbers are free — g and the two thresholds — but
    # they sit inside the relu, so they are found by search rather than least squares.
    # Candidate thresholds: 221 values covering the readings plus 0.3 of margin on each side,
    # so "the hinge never fires" (threshold past the data) is a reachable candidate.
    cs = torch.linspace(s.min().item() - 0.3, s.max().item() + 0.3, 221)  # (221,)
    # Precompute sufficient statistics, the one pass over the M samples. For a hinge
    # h(s) = relu(sig * (s - c)) the sample MSE of the prediction slope * h expands as
    #   E[(y - slope*h)^2] = E[y^2] - 2*slope*E[y*h] + slope^2 * E[h^2],
    # where the slope appears only outside the expectations. So storing E[y^2] once and, per
    # sign and candidate threshold, E[h^2] and E[y*h], makes the MSE of any slope at any
    # threshold three multiplications — the data is never touched again.
    ey2 = (Y ** 2).mean(0)  # (2,): E[y^2] per feature
    stats = {}
    for sig in (1.0, -1.0):
        eyh, ehh = torch.empty(2, len(cs)), torch.empty(len(cs))  # (2, 221) and (221,): E[y*h] and E[h^2] per threshold
        for k, c in enumerate(cs.tolist()):
            h = torch.relu(sig * (s - c))  # (M,): the hinge at threshold c, evaluated on every sample
            ehh[k] = (h ** 2).mean()
            eyh[:, k] = (Y * h[:, None]).mean(0)
        stats[sig] = (eyh, ehh)
    # The search. The gain g couples the two features (one g, both slopes), so it is searched
    # jointly; given g the features share nothing, so each picks its own best threshold
    # independently from the precomputed stats, and the g with the lowest summed MSE wins.
    best = None
    for g in torch.logspace(-1.5, 1.5, 61).tolist():
        total_mse, params, mses = 0.0, [], []
        for i, (sig, slope) in enumerate(((1.0, g * a), (-1.0, g * b))):
            eyh, ehh = stats[sig]
            # the MSE expansion above, evaluated for all 221 thresholds at once:
            # m is (221,), entry k = this feature's MSE at threshold cs[k] under the current slope
            m = ey2[i] - 2 * slope * eyh[i] + slope ** 2 * ehh
            k = m.argmin().item()
            total_mse += m[k].item()
            params.append((sig, slope, cs[k].item()))
            mses.append(m[k].item())
        if best is None or total_mse < best[0]:
            best = (total_mse, params, mses)
    _, params, mses = best
    # Evaluate the two winning hinges on the plotting grid; return the curves and the
    # per-feature MSEs, the same shape the polynomial branch returns.
    F = torch.stack([slope * torch.relu(sig * (grid - c)) for sig, slope, c in params], 1)  # (len(grid), 2)
    return F, torch.tensor(mses)  # (len(grid), 2), (2,)


def poly_predictor(z, x, degree):
    """The best decoder of a polynomial class on the 2D reading plane: fits, by least squares,
    one polynomial of total degree at most `degree` in the two plane coordinates per feature.
    This is the ceiling for a bilinear stack on a fixed geometry, since k bilinear MLPs can only
    implement a polynomial of degree 2^k in the reading. Inputs:
      - `z`: the sampled readings, (M, 2), with M the size of the training sample.
      - `x`: the features behind those readings, (M, k) — one fitted polynomial per column.
      - `degree`: the class's total degree (2 for one bilinear MLP, 16 for four).
    Returns a predictor zz (m, 2) -> xhat (m, k), where m is the number of query points —
    arbitrary, and distinct from M: the predictor is evaluated on the sample (m = M) as well
    as on plotting grids. Because the class is linear in its
    coefficients, the lstsq solution is exactly the best decoder of the class on the sample —
    and, the fit being over samples, best in the density-weighted sense: it is the closest
    polynomial to E[x | z] where the readings actually land.
    """
    zmin, zmax = z.min(0).values, z.max(0).values  # (2,) each: the training range, reused at evaluation

    def basis(zz):
        """The design matrix at m query points zz (m, 2): one row per point, one column per
        basis polynomial — the products of Chebyshev polynomials with total degree at most
        `degree`, evaluated after mapping zz onto [-1, 1]^2. Returns (m, (degree+1)(degree+2)/2).
        Called twice: at fit time on the training sample z (so m = M there), and inside the
        returned predictor on arbitrary points — with the same zmin/zmax mapping in both,
        so the fitted coefficients keep their meaning.
        """
        # each coordinate mapped affinely onto [-1, 1] over the training range (outside it the
        # Chebyshev argument leaves [-1, 1] and a high-degree polynomial explodes, so evaluation
        # is only meaningful on the data's support)
        z01 = (2 * (zz - zmin) / (zmax - zmin) - 1).double()  # (m, 2)
        # the Chebyshev recurrence T_0 = 1, T_1 = u, T_{n+1} = 2 u T_n - T_{n-1}. Since z01
        # holds both coordinates and the recurrence is applied to it elementwise, each entry
        # T[e] is an (m, 2) tensor: its column v holds the degree-e polynomial evaluated at
        # the rescaled coordinate v of every point. So T[e][:, v] below reads "T_e(u_v) at
        # each of the m points" — the factors the basis columns are assembled from.
        T = [torch.ones_like(z01), z01]  # degree+1 entries of (m, 2)
        for _ in range(degree - 1):
            T.append(2 * z01 * T[-1] - T[-2])
        # the 2D basis. A polynomial in two variables of total degree <= `degree` is a linear
        # combination of the monomials u0^i u1^j with i + j <= degree, so one basis function is
        # needed per pair (i, j). Here the monomial of each pair is replaced by T_i(u0) T_j(u1),
        # which has the same degrees and spans the same space, but is better conditioned. The
        # loops enumerate the pairs: for every total degree deg, combinations_with_replacement
        # lists the ways deg units of degree can be split between the two coordinates — the
        # tuple (0, 0, 1), say, gives two units to coordinate 0 and one to coordinate 1, so
        # Counter turns it into {0: 2, 1: 1}, i.e. (i, j) = (2, 1), and the inner loop
        # multiplies the factors into the column T_2(u0) * T_1(u1).
        cols = []
        for deg in range(degree + 1):
            for c in combinations_with_replacement(range(2), deg):
                col = torch.ones(len(zz), dtype=torch.float64)  # (m,)
                for v, e in Counter(c).items():
                    col = col * T[e][:, v]
                cols.append(col)
        return torch.stack(cols, 1)  # (m, (degree+1)(degree+2)/2): 6 columns for degree 2, 153 for 16

    sol = torch.linalg.lstsq(basis(z), x.double()).solution  # ((degree+1)(degree+2)/2, k)
    return lambda zz: (basis(zz) @ sol).float()  # (m, k)


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
