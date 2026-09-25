# common.py contains the same piece of mathematics twice, written two different ways:
#   - posterior_mean(t, w, v, p) — the general formula, built exactly as the appendix derives
#     it: a weighted average over the three cases (feature alone, partner alone, both active),
#     with the case priors and densities written out as code.
#   - x1_posterior(s, a, b) and x3_posterior(s, a, b) — the same posteriors, but with all that
#     num/den algebra worked out by hand into one explicit closed-form expression per region
#     ("own", "plateau", "tail", ...), valid for a >= b.
# The hand-worked pair is what four figure scripts actually plot (posterior_sym, asym_sweep,
# asym_compare, class_compare), so a typo in that algebra would silently corrupt published
# figures. This script exists to rule that out: it numerically verifies that the two
# implementations agree everywhere. It's run standalone (python check_posteriors.py) and is
# wired into reproduce.sh, so every full reproduction of the repo re-validates the algebra.
# Nothing imports it — it's a test, not a library.
import numpy as np
import torch

from common import P, posterior_mean, x1_posterior, x3_posterior

for a, b in [(1.0, 1.0), (3.35, 1.0), (2.0, 0.5), (1.3, 1.2)]:
    s = np.linspace(-a + 1e-9, b - 1e-9, 200_001)
    st = torch.tensor(s, dtype=torch.float64)
    for name, f, w, v in (("x1", x1_posterior, b, -a), ("x3", x3_posterior, -a, b)):
        ref, got = posterior_mean(st, w, v, P).numpy(), f(s, a, b)
        assert np.array_equal(np.isnan(ref), np.isnan(got)), f"NaN regions differ for {name} at a={a}, b={b}"
        diff = np.nanmax(np.abs(ref - got))
        print(f"a={a:4.2f} b={b:4.2f} {name}: max |diff| = {diff:.2e}")
        assert diff < 1e-6, f"{name} disagrees with posterior_mean at a={a}, b={b}"
print("ok")
