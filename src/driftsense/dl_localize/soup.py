#!/usr/bin/env python3
"""Average the weights of two or more dl_localize checkpoints ("model soup").

    python -m driftsense.dl_localize.soup ^
        --checkpoints <run-dir>\\phase3_all.pt <run-dir>\\phase3_all.pt ^
        --out <run-dir>\\phase3_all.pt

WHY THIS IS WORTH ONE EVAL RUN
==============================

Both scaling axes are exhausted -- more steps plateaued at step 5000 of
10000, and 3.6x more hard data bought +1.4 points, inside the noise band.
Seven approaches to the easy/medium residual all returned nothing. What has
NOT been tried is combining the checkpoints that already exist.

Run C and run D were trained on different data volumes (11,600 vs 41,600 hard
samples) with different schedules (5,000 vs 10,000 steps), so their errors are
not identical. Averaging weights (Wortsman et al., 2022, "Model soups")
typically recovers part of that difference for free.

The caveat, stated honestly: weight averaging only works when the models lie
in the same loss basin. Run C and run D both resume from run B's phase-2
checkpoint, which makes a shared basin likely -- but it also makes them
CORRELATED, so the gain should be smaller than for independently trained
models. If the soup scores clearly below both parents, they are not in the
same basin and a logit-level ensemble is the alternative.

This script therefore prints a warning if the averaged weights differ wildly
from the parents, which is the cheap signal for that failure.

EVALUATION DISCIPLINE
=====================

Test the soup ONCE on golden. Testing many variants and reporting the best is
how a held-out set stops being held out. A soup is a single a-priori method
choice, not a search over hyperparameters, so one measurement is legitimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch


# Checkpoints record the architecture that produced them. The shipped weights
# were trained before this package was renamed, so their stored string is the
# historical one. Both are accepted: the network definition did not change, only
# the module and class names did.
#
# The historical constant is built from parts on purpose. Written literally it
# would be rewritten by any future rename sweep across this file, and the
# failure is silent -- inference falls back to the classical path and reports a
# plausible but much less accurate coordinate.
_LEGACY_ARCH = "localize" + "88." + "SiameseCorrelationNet" + "88"
_ACCEPTED_ARCH = ("dl_localize.SiameseCorrelationNet", _LEGACY_ARCH)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="two or more dl_localize .pt files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="optional mixing weights, normalised. Default: equal")
    ap.add_argument("--key", default="ema", choices=["ema", "model"],
                    help="which weight set to average. EMA is what inference "
                         "loads by default")
    args = ap.parse_args()

    paths: List[str] = args.checkpoints
    if len(paths) < 2:
        raise SystemExit("give at least two checkpoints")

    w = args.weights or [1.0] * len(paths)
    if len(w) != len(paths):
        raise SystemExit("--weights must match --checkpoints in length")
    total = float(sum(w))
    w = [x / total for x in w]

    cks = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if ck.get("arch") not in _ACCEPTED_ARCH:
            raise SystemExit(f"{p} is not a dl_localize checkpoint")
        cks.append(ck)
        print(f"  loaded {p}  (phase {ck.get('phase')}, step {ck.get('step')})")

    ref = cks[0][args.key]
    for i, ck in enumerate(cks[1:], 1):
        if set(ck[args.key].keys()) != set(ref.keys()):
            raise SystemExit(
                f"checkpoint {paths[i]} has a different parameter set -- the "
                f"architectures differ and cannot be averaged")

    souped = {}
    cosines = []
    for k in ref:
        v0 = ref[k]
        if not torch.is_floating_point(v0):
            souped[k] = v0.clone()          # integer buffers: take the first
            continue
        acc = torch.zeros_like(v0, dtype=torch.float32)
        for ck, wi in zip(cks, w):
            acc += wi * ck[args.key][k].float()
        souped[k] = acc
        # Basin agreement is measured by COSINE SIMILARITY between parents,
        # not by relative drift. Drift divides by a tensor's own magnitude and
        # so explodes on near-zero bias vectors -- it reported 10.4 on a pair
        # of models whose median cosine similarity was 0.989, i.e. it was
        # measuring float noise on a scalar, not disagreement.
        if v0.numel() >= 2 and len(cks) == 2:
            x = cks[0][args.key][k].float().flatten()
            y = cks[1][args.key][k].float().flatten()
            cosines.append(float(torch.nn.functional.cosine_similarity(
                x, y, dim=0)))

    print(f"\n  averaged {len(souped)} tensors with weights "
          f"{[round(x, 3) for x in w]}")
    if cosines:
        cosines.sort()
        med = cosines[len(cosines) // 2]
        print(f"  parent cosine similarity over {len(cosines)} tensors: "
              f"median {med:.4f}, min {cosines[0]:.4f}")
        if med < 0.90:
            print("  WARNING: the parents are far apart in weight space and "
                  "may not share a loss basin, in which case the soup can "
                  "score below BOTH of them. If the eval confirms that, use "
                  "a logit-level ensemble instead of weight averaging.")
        else:
            print("  (near 1.0 -- a shared loss basin, which is what makes "
                  "weight averaging valid)")

    out = dict(cks[-1])
    out["ema"] = souped
    out["model"] = souped
    out["phase"] = "soup"
    out["soup_sources"] = list(paths)
    out["soup_weights"] = w
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"\n  wrote {args.out}")
    print("  Evaluate it ONCE on golden. If it does not beat both parents, "
          "keep the better parent and stop.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
