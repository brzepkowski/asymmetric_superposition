# Five features in a 2D bottleneck

The pipeline of the write-up rerun with 5 instead of 4 input features (same distribution, $p = 0.2$, same 2D bottleneck and architectures). Code: `five_features.py` (modes `search` / `train` / `table`), checkpoints under `checkpoints/n5/`.

## Free geometry search

The free search of "Search for the best strategy", generalized: $f_1$ pinned to the $+x$ axis, the remaining 4 angles and all 5 lengths free (9 parameters), every geometry scored by the MSE of a binned decoder ($96 \times 96$ cells, $2^{19}$ samples). Nelder-Mead from the best 3 of 150 random geometries and from the uniform pentagon, winners re-scored on a fresh sample.

| start | MSE (search sample) | MSE (fresh) | resulting geometry |
|---|---|---|---|
| random start 1 | 0.00562 | 0.00564 | two asymmetric, opened pairs (ratios 4.0 / 8.3, tilts 3° / 17°), fifth feature short |
| random start 2 | 0.00539 | 0.00547 | two loose asymmetric pairs (ratios 2.3 / 1.8), no dominant feature |
| random start 3 | 0.00540 | **0.00543** | one nearly closed asymmetric pair (ratio 2.0), the other three features short |
| pentagon | 0.00916 | 0.00930 | stays uniform: all lengths ≈ 1, $f_3$, $f_5$ rotate to 4° off antipodal |

- MSE floor per feature: **0.00543** (the 4-feature task's floor at the same resolution was 0.00221 — the fifth feature roughly doubles the per-feature error).
- The uniform pentagon (0.01009 unoptimized) is a shallow local optimum: the search barely improves it and never leaves it, while every random start finds a non-uniform geometry at ~0.0054, about 40% less error. The near-optimal geometries are degenerate — several different arrangements of short features and asymmetric near-antipodal pairs score within a few percent of each other.

## Trained models

20 runs per architecture from random initializations (seeds 0–19), the training identical to the strategy table of the write-up (AdamW, lr $10^{-3}$ cosine-annealed over 20,000 steps, a fixed batch of 4096 samples per seed). MSE evaluated on a shared batch of 65,536 samples.

| model | median MSE | best MSE | geometries (20 runs) |
|---|---|---|---|
| tied ReLU | 0.0167 | 0.0160 | 15× symmetric cross of four + $f_5$ dropped or tiny; 4× regular pentagon; 1 other |
| 1 bilinear MLP | 0.0188 | 0.0185 | 16× symmetric cross of four + $f_5$ dropped or tiny; 3× pentagon-like; 1 other |
| 2 bilinear MLPs | 0.0139 | 0.0137 | 16× two asymmetric pairs (ratios 1.8–4.3), all five features alive; 3× one feature dead; 1 other |
| 3 bilinear MLPs | 0.0114 | 0.0109 | 19× two asymmetric pairs (ratios up to ~12, some tilted), all alive; 1 other |
| 4 bilinear MLPs | 0.0116 | 0.0104 | 15× two asymmetric, tilted pairs (ratios up to ~15), all alive; 5× degraded runs (dead features or a single pair) |

## Takeaways

1. **Shallow models give up on the fifth feature**: the tied ReLU and the single bilinear MLP almost always build the familiar symmetric antipodal cross out of four features and shrink the fifth to (near) zero.
2. **The uniform pentagon appears only in shallow models, and rarely** (4/20 tied ReLU, ~3/20 single bilinear). For those weak decoders it slightly beats the sacrifice solution (0.0160 vs 0.0167) — the uniform polytope is a good solution precisely where the decoder is weak.
3. **From two bilinear MLPs on, asymmetry rescues the fifth feature**: virtually every run keeps all five features alive as two asymmetric (and, deeper, increasingly tilted) antipodal pairs plus a fifth intermediate-length feature, cutting the error by ~40% relative to the shallow solutions.
4. **The gap to the floor remains**: even the best four-layer run (0.0104) sits about 2× above the binned-search floor (0.0054), the same kind of shortfall the write-up documents for 4 features.
