# The four-MLP model of class_check.py (seed 9, 2^20 samples) retrained from the same initialization and
# on the same sample for 150 000 steps instead of 20 000, and measured the same way: MSE of the binned decoder of
# its encoder, of the best polynomial of degree 16, and of the model itself, the MSE gap and the score.
# The run is cached under checkpoints/big_long/. Run from the repo root: python -m figures.long_run
import torch

from common import CKPT, P, classify, load_model, measure, poly_predictor, sample_x, train

STEPS, SEED, DEG, BINS = 150_000, 9, 16, 40

path = train("bilinear4", SEED, STEPS, CKPT / "big_long", batch=2 ** 20, device="cuda") / "model.pt"
model = load_model(path)
W = model.w_enc.detach()
x = sample_x(2 ** 20, P, torch.Generator().manual_seed(7))  # the sample of class_check.py
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
score = 1 - (y_model - y_fit).pow(2).sum().item() / (y_model - y_model.mean(0)).pow(2).sum().item()
m = measure(W)
print(f"geometry: {classify(m)}; " + "; ".join(f"pair {p['pair']} ratio {p['ratio']:.2f} cos {p['cos']:.3f}" for p in m["pairs"]))
print(f"four bilinear MLPs, seed {SEED}, {STEPS} steps: MSE binned {mse['binned']:.4f}  best of class {mse['fit']:.4f}  "
      f"model {mse['model']:.4f}  gap {(mse['model'] - mse['fit']) / mse['model'] * 100:.1f}%  score {score:.4f}")
