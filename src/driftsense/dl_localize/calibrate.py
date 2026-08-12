#!/usr/bin/env python3
"""Pre-flight geometric calibration.  RUN THIS BEFORE ANY TRAINING.

    python -m driftsense.dl_localize.calibrate
    python -m driftsense.dl_localize.calibrate --dataset <dataset-root>

WHY THIS SCRIPT EXISTS
======================

The most expensive bug on this project was silent.  The mapping from
correlation-output index to search pixel disagreed between the target builder
and the decoder by an amount that depended on the sample's template size.
Nothing crashed.  The loss curve looked healthy.  The overfit test passed on
two machines.  The only symptom was an accuracy bucket that would not move,
and roughly 40 GPU-hours went into "more data" and "more steps" before anyone
asked whether the coordinates were right.

A coordinate-convention bug cannot be caught by watching a loss curve,
because the network simply learns whatever convention the loss used.  It can
only be caught by an independent measurement: plant a pattern at a KNOWN
location and check that the decoded answer is that location.  That is all
this script does, and it needs no training, no GPU and about a minute.

Two independent checks:

SYNTHETIC -- crop a patch out of a noise image at a stride-aligned offset and
correlate it back against the same image.  The decoded pixel must equal the
crop's true centre EXACTLY (to float precision), because with an untrained
encoder the patch still matches itself perfectly.  Any non-zero error here
is a pure convention bug in coords or the encoder's spatial alignment.

REAL DATA -- take a real search frame, cut the 100x100 window centred on
ground truth with sub-pixel accuracy, and use that as the template.  This
exercises data's template builder, the encoder, and coords together on
real SEM imagery.  The residual must be under half a cell in each axis
(<= 2.83 px Euclidean), which is the argmax quantisation floor before the
sub-pixel offset head is trained.

Exit code is non-zero if either check fails.  Wire it into whatever you use
as a pre-run gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

from driftsense.dl_localize.coords import (REF_EMB_PX, STRIDE, TEMPLATE_PX,
                                            out_to_pixel, pixel_to_out)
from driftsense.dl_localize.data import normalize_image
from driftsense.dl_localize.model import SiameseCorrelationNet, peak_xy


def synthetic_check(net: SiameseCorrelationNet, device: torch.device,
                    size: int = 400) -> float:
    """Plant exact crops at stride-aligned offsets.  Returns the worst error.

    Offsets must be multiples of STRIDE.  At a half-cell offset the true
    response splits between two cells, which tests the encoder's feature
    quality rather than the coordinate convention.
    """
    rng = np.random.default_rng(0)
    s = torch.from_numpy(
        rng.normal(0, 1, (1, 1, size, size)).astype(np.float32)).to(device)
    worst = 0.0
    print("  SYNTHETIC (exact crops, stride-aligned)")
    for (x0, y0) in ((0, 0), (120, 80), (200, 152),
                     (size - TEMPLATE_PX, size - TEMPLATE_PX)):
        t = s[..., y0:y0 + TEMPLATE_PX, x0:x0 + TEMPLATE_PX].contiguous()
        with torch.no_grad():
            logits, _ = net(t, s)
        ox, oy = peak_xy(logits)
        gx, gy = out_to_pixel(ox), out_to_pixel(oy)
        tx = x0 + (TEMPLATE_PX - 1) / 2.0
        ty = y0 + (TEMPLATE_PX - 1) / 2.0
        e = math.hypot(gx - tx, gy - ty)
        worst = max(worst, e)
        print(f"    crop ({x0:>4},{y0:>4})  decoded ({gx:8.2f},{gy:8.2f})  "
              f"true ({tx:8.2f},{ty:8.2f})  err {e:6.2f} px")
    return worst


def real_check(net: SiameseCorrelationNet, device: torch.device,
               dataset: str, n_pairs: int = 4) -> float:
    """Cut the template out of the search frame at ground truth.

    This bypasses the reference image entirely on purpose: the question is
    whether the geometry is right, not whether the encoder can match across
    a 10x magnification change.  Returns the worst error.
    """
    import cv2
    root = Path(dataset)
    rows = []
    with open(root / "labels.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["difficulty"] == "easy":
                rows.append(r)
            if len(rows) >= n_pairs:
                break

    worst = 0.0
    print(f"\n  REAL DATA ({len(rows)} easy pairs, template cut from the "
          f"search frame at ground truth)")
    for r in rows:
        p = r["search_path"]
        srch = cv2.imread(p if os.path.isabs(p) else str(root / p), 0)
        if srch is None:
            print(f"    id={r['id']}: could not read {p}")
            continue
        gx, gy = float(r["gt_x"]), float(r["gt_y"])
        # Sub-pixel exact extraction: getRectSubPix's output centre maps to
        # `center` exactly, so no half-pixel crop bias is introduced here.
        tmpl = cv2.getRectSubPix(srch.astype(np.float32),
                                 (TEMPLATE_PX, TEMPLATE_PX), (gx, gy))
        t = torch.from_numpy(
            ((tmpl - tmpl.mean()) / (tmpl.std() + 1e-6)).astype(np.float32)
        )[None, None].to(device)
        s = normalize_image(srch).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _ = net(t, s)
        ox, oy = peak_xy(logits)
        px, py = out_to_pixel(ox), out_to_pixel(oy)
        e = math.hypot(px - gx, py - gy)
        worst = max(worst, e)
        print(f"    id={r['id']:>7}  decoded ({px:7.2f},{py:7.2f})  "
              f"gt ({gx:7.2f},{gy:7.2f})  err {e:5.2f} px")
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=None,
                    help="dataset root; enables the real-data check")
    ap.add_argument("--device", default="cpu",
                    help="cpu is fine and is what you want here -- this is a "
                         "geometry check, not a throughput test")
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--checkpoint", default=None,
                    help="optional; the check is valid with random weights "
                         "and that is the point -- geometry does not depend "
                         "on training")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    net = SiameseCorrelationNet().to(device).eval()
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location=device,
                        weights_only=False)
        net.load_state_dict(ck.get("ema") or ck["model"])
        print(f"  loaded {args.checkpoint}")

    print(f"  TEMPLATE_PX={TEMPLATE_PX}  STRIDE={STRIDE}  "
          f"REF_EMB_PX={REF_EMB_PX}")
    print(f"  out_to_pixel(o) = {STRIDE} * o + {(TEMPLATE_PX - 1) / 2}\n")

    ok = True
    w_syn = synthetic_check(net, device)
    if w_syn > 1e-6:
        ok = False
        print(f"\n  FAIL: synthetic worst error {w_syn:.3f} px. The encoder "
              f"and coords disagree about the coordinate convention. This "
              f"is exactly the class of bug that cost this project 40 GPU-"
              f"hours. DO NOT TRAIN until it is zero.")
    else:
        print(f"  PASS: synthetic worst error {w_syn:.2e} px (exact)")

    if args.dataset:
        w_real = real_check(net, device, args.dataset, args.pairs)
        limit = STRIDE / 2.0 * math.sqrt(2.0) + 1e-6
        if w_real > limit:
            ok = False
            print(f"\n  FAIL: real-data worst error {w_real:.2f} px exceeds "
                  f"the {limit:.2f} px argmax-quantisation floor. Something "
                  f"in data's template pipeline is off-centre.")
        else:
            print(f"  PASS: real-data worst error {w_real:.2f} px "
                  f"(<= {limit:.2f} px quantisation floor)")

    print("\n  " + ("ALL CHECKS PASSED -- geometry is sound, safe to train."
                    if ok else
                    "CALIBRATION FAILED -- do not start a training run."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
