# The 100 runs of the strategy table: 20 seeds for each of the tied ReLU and the bilinear stacks of
# depth 1-4. A run that already has a model.pt in checkpoints/ is skipped, so this is a no-op on the
# shipped checkpoints; delete them to retrain (each run is deterministic given its seed).
import sys

from common import ARCHES, SEEDS, train

for seed in [int(s) for s in sys.argv[1:]] or SEEDS:
    for arch in ARCHES:
        train(arch, seed)
