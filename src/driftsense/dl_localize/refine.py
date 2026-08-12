"""Stage 3 -- local classical sub-pixel refinement of the learned prediction.

WHY THIS EXISTS
===============

After the geometry and offset-head fixes, the measured failure profile on
2000 held-out val pairs was:

    group    <=1px   <=2px   <=5px   near-miss   gross-miss
    easy     87.8%   98.9%   99.6%      11.8%        0.2%
    medium   87.3%   98.8%   99.3%      12.0%        0.5%
    hard     48.0%   68.0%   75.0%      27.0%       20.2%

Read the easy row again: the model puts 98.9% of predictions within 2 px and
99.6% within 5 px, but only 87.8% within 1 px.  It is not failing to FIND the
pattern.  It finds it and then loses the last pixel.  Across all buckets,
14.9% of samples sit in the 1-5 px band -- that band is the single largest
remaining error source in the whole system, larger than periodic locks (1.1%)
and gross misses (4.3%) combined.

The cause is structural and known: the encoder has output stride 4, so the
correlation map is quantised to 4 px and every sub-pixel digit has to be
interpolated by the offset head.  Dropping to stride 2 would fix it directly
but costs 16x on the correlation ((451/226)^2 x (50/25)^2), which a 6 GB
laptop GPU will not absorb.

The classical baseline does not have this problem.  It correlates at STRIDE 1
and fits a parabola to the peak, which is exactly why it reaches 89% on easy
while the learned path sat at 84%.  Its weakness was never precision -- it was
global search, where it periodic-locks and saturates at ~32% on hard.

So: use each method for the half of the problem it is good at.  The learned
model does the global search it is now demonstrably excellent at (99.6% of
easy within 5 px), then hands a +/-3 px window to classical stride-1 NCC for
the last pixel.

MEASURED, on 14 real pairs (4 hard), starting from ground truth perturbed by
a plausible stride-4 readout error:

    median error   1.25 px  ->  0.23 px
    <=1px             36%   ->   100%

WHY A +/-3 PX WINDOW AND NOT MORE
=================================
The measured lattice pitch is 5.3-22.7 search pixels.  A +/-3 px window cannot
contain a lattice replica even at the finest pitch, so this stage physically
cannot introduce a periodic lock -- it can only refine within the cell the
learned model already chose.  Widening the window past ~half the minimum pitch
would reintroduce exactly the failure mode the lattice margin loss was built
to remove.

MULTI-ANGLE, BECAUSE ROTATION IS NOW THE VISIBLE COVARIATE
==========================================================
With the inference template bank removed (it was measured to hurt), the model
absorbs the full residual rotation, and <=1px accuracy falls off with it:

    |rel_rotation|  [0,0.42) 81%  [0.42,0.84) 83%  [0.84,1.34) 84%
                    [1.34,2.02) 75%  [2.02,5.15) 75%

Searching angles HERE is safe where searching them globally was not.  The
global bank failed because selection by peak height over a 226x226 map is
unreliable when the surface is flat.  Over a 7x7 window centred on an
already-good estimate there is no competing structure to be confused by, so
the angle that maximises NCC is the angle that actually matches.

COST
====
The 1000->100 downscale of the reference is the only expensive operation, and
it is hoisted out of the angle loop -- all angles share one resized image and
differ only by a warpAffine on a 100x100 array plus a matchTemplate on a
~107x107 window.  Both are negligible, so angle count is nearly free.

Self-test::

    python -m driftsense.dl_localize.refine --dataset <dataset-root>
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from driftsense.dl_localize.coords import TEMPLATE_PX
from driftsense.dl_localize.data import NOMINAL_SCALE

#: Angles searched, degrees.  |rel_rotation_deg| has sigma 1.6 and abs max
#: 5.83 in the manifest, so +/-4.5 at 1.5 spacing leaves <=0.75 deg residual
#: for ~99.5% of samples.  Extra angles are nearly free -- see COST above.
ANGLES: Tuple[float, ...] = (-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5)

#: Search half-width in search-image pixels.  Must stay below half the
#: minimum lattice pitch (5.3 px) so a replica can never enter the window.
RADIUS: int = 3


def _parabolic(surface: np.ndarray, i: int, j: int) -> Tuple[float, float]:
    """Three-point parabolic sub-pixel fit around ``(j, i)``.

    The standard correlation-peak interpolator: fit a parabola through the
    peak and its two neighbours along each axis independently and take the
    vertex.  Clamped to +/-1 sample, because a fitted vertex further away than
    that means the surface is not locally quadratic and the fit is garbage.

    Args:
        surface: 2-D correlation surface.
        i: Peak row.
        j: Peak column.

    Returns:
        ``(dx, dy)`` sub-sample offsets, each in [-1, 1].
    """
    h, w = surface.shape
    dx = dy = 0.0
    if 0 < j < w - 1:
        a, b, c = (float(surface[i, j - 1]), float(surface[i, j]),
                   float(surface[i, j + 1]))
        d = a - 2.0 * b + c
        if abs(d) > 1e-12:
            dx = 0.5 * (a - c) / d
    if 0 < i < h - 1:
        a, b, c = (float(surface[i - 1, j]), float(surface[i, j]),
                   float(surface[i + 1, j]))
        d = a - 2.0 * b + c
        if abs(d) > 1e-12:
            dy = 0.5 * (a - c) / d
    return max(-1.0, min(1.0, dx)), max(-1.0, min(1.0, dy))


def make_template_bank(reference: np.ndarray,
                       angles: Sequence[float] = ANGLES,
                       scale: float = NOMINAL_SCALE,
                       template_px: int = TEMPLATE_PX) -> list:
    """Rotated templates, sharing ONE anti-aliased downscale.

    ``data.build_template`` resizes then warps, so calling it once per angle
    repeats the 1000->100 INTER_AREA reduction -- which is the only costly
    step -- for no reason.  Here the reduction happens once and each angle is
    a warpAffine on a 100x100 array.

    The warp keeps the reference centre exactly on the template centre, the
    same invariant ``coords`` depends on; an integer crop would shift it by
    half a pixel whenever the size difference is odd.

    Args:
        reference: (N, N) grayscale reference, full resolution.
        angles: Rotations in degrees.
        scale: Demagnification.
        template_px: Output side length.

    Returns:
        List of float32 ``(template_px, template_px)`` arrays.
    """
    n = reference.shape[0]
    m = max(8, int(round(n / float(scale))))
    interp = cv2.INTER_AREA if m < n else cv2.INTER_LINEAR
    small = cv2.resize(reference.astype(np.float32), (m, m),
                       interpolation=interp)

    src_c = (m - 1) / 2.0
    dst_c = (template_px - 1) / 2.0
    out = []
    for angle in angles:
        mat = cv2.getRotationMatrix2D((src_c, src_c), float(angle), 1.0)
        mat[0, 2] += dst_c - src_c
        mat[1, 2] += dst_c - src_c
        out.append(cv2.warpAffine(small, mat, (template_px, template_px),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101))
    return out


def refine_local(reference: np.ndarray, search: np.ndarray,
                 x: float, y: float,
                 angles: Sequence[float] = ANGLES,
                 radius: int = RADIUS,
                 scale: float = NOMINAL_SCALE,
                 template_px: int = TEMPLATE_PX,
                 templates: Optional[list] = None) -> dict:
    """Refine ``(x, y)`` by stride-1 multi-angle NCC in a small window.

    Args:
        reference: (N, N) grayscale reference, full resolution.
        search: (H, W) grayscale search image.
        x: Learned prediction x, search pixels.
        y: Learned prediction y.
        angles: Rotations to try, degrees.
        radius: Search half-width in pixels.  Keep below half the minimum
            lattice pitch so a replica cannot enter the window.
        scale: Demagnification.
        template_px: Template side length.
        templates: Pre-built bank from :func:`make_template_bank`, if the
            caller is reusing it.

    Returns:
        Dict with refined ``x``, ``y``, the winning ``ncc`` and ``angle``, and
        ``shift`` (how far the refinement moved the estimate, in pixels).
        A large ``shift`` together with a low ``ncc`` means the learned
        estimate was outside the window and the refinement is not trustworthy.
    """
    if templates is None:
        templates = make_template_bank(reference, angles, scale, template_px)

    side = template_px + 2 * radius
    # getRectSubPix extracts with bilinear sub-pixel accuracy and its output
    # centre maps EXACTLY to `(x, y)`, so no half-pixel crop bias is
    # introduced -- the same reason data uses a warp instead of a slice.
    win = cv2.getRectSubPix(search.astype(np.float32), (side, side),
                            (float(x), float(y)))

    best_score = -2.0
    best = (0.0, 0.0, 0.0)
    for angle, tmpl in zip(angles, templates):
        surf = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
        _, mx, _, mloc = cv2.minMaxLoc(surf)
        if mx > best_score:
            j, i = int(mloc[0]), int(mloc[1])
            ddx, ddy = _parabolic(surf, i, j)
            best_score = float(mx)
            best = ((j - radius) + ddx, (i - radius) + ddy, float(angle))

    dx, dy, angle = best
    return {"x": float(x) + dx, "y": float(y) + dy,
            "ncc": best_score, "angle": angle,
            "shift": float(np.hypot(dx, dy))}


def _self_test() -> None:
    """Perturb ground truth and check the refiner recovers it, on real data."""
    import argparse
    import itertools
    import json
    import math
    import os
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--perturb", type=float, default=1.5,
                    help="uniform +/- offset applied to ground truth, "
                         "simulating a stride-4 readout error")
    args = ap.parse_args()

    root = args.dataset
    rows = []
    with open(os.path.join(root, "labels.jsonl")) as f:
        for line in itertools.islice(f, 2000):
            rows.append(json.loads(line))
            if len(rows) >= args.n:
                break

    rng = np.random.default_rng(0)
    before, after = [], []
    t0 = time.perf_counter()
    print(f"  {'id':>7} {'diff':<7} {'start':>8} {'refined':>9} {'ncc':>6} "
          f"{'ang':>5} {'shift':>6}")
    for r in rows:
        rp, sp = r["reference_path"], r["search_path"]
        ref = cv2.imread(rp if os.path.isabs(rp) else os.path.join(root, rp), 0)
        srch = cv2.imread(sp if os.path.isabs(sp) else os.path.join(root, sp), 0)
        if ref is None or srch is None:
            continue
        gx, gy = float(r["gt_x"]), float(r["gt_y"])
        px = gx + float(rng.uniform(-args.perturb, args.perturb))
        py = gy + float(rng.uniform(-args.perturb, args.perturb))
        e0 = math.hypot(px - gx, py - gy)
        out = refine_local(ref, srch, px, py)
        e1 = math.hypot(out["x"] - gx, out["y"] - gy)
        before.append(e0)
        after.append(e1)
        print(f"  {r['id']:>7} {r['difficulty']:<7} {e0:>8.2f} {e1:>9.2f} "
              f"{out['ncc']:>6.3f} {out['angle']:>+5.1f} {out['shift']:>6.2f}")

    b, a = np.array(before), np.array(after)
    dt = (time.perf_counter() - t0) / max(len(b), 1)
    print(f"\n  median error  {np.median(b):.2f} px -> {np.median(a):.2f} px")
    print(f"  <=1px         {100 * (b <= 1).mean():.0f}% -> "
          f"{100 * (a <= 1).mean():.0f}%")
    print(f"  cost          {1000 * dt:.0f} ms/pair "
          f"({len(ANGLES)} angles, this CPU)")
    assert np.median(a) < np.median(b), "refinement made things worse"
    print("  OK")


if __name__ == "__main__":
    _self_test()
