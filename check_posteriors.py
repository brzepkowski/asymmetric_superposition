# Numerical check: the piecewise formulas x1_posterior / x3_posterior match posterior_mean
# on their valid domain a >= b. Run as: python check_posteriors.py
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
