"""``visualize.py`` -- look at the data before you train on it.

    python -m driftsense.cli.visualize --pairs 6 --out data/preview

Three views, because they answer three different questions:

``montage`` (default)
    Reference beside search, with the ground-truth quad and centre drawn on the
    search image.  Answers "does the label point at the right place, and does
    this look like an SEM image at all".

``zoom``
    The reference box-downsampled by the recorded scale ratio, beside the search
    image cropped at the recorded location.  These two panels should show the
    *same structure* through different noise.  This is the view that catches a
    labelling bug instantly -- if the panels disagree, the label is wrong, and no
    amount of staring at a montage would have told you.

``grid``
    Many samples at thumbnail size, to check that variety is actually present:
    both architectures, all difficulty tiers, a spread of noise and rotation.  A
    generator that has quietly collapsed to one look is a common failure and is
    invisible one sample at a time.

Everything is drawn with OpenCV so there is no matplotlib dependency and the
output is deterministic.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from driftsense import geometry as G
from driftsense.config import PRESETS, GeneratorConfig, preset
from driftsense.metadata import iter_rows
from driftsense.pipeline import Plan, plan_sample, render, verify_pair
from driftsense.rng import sample_seeds

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

#: BGR colours (OpenCV order).
GREEN = (90, 255, 90)
RED = (60, 60, 255)
WHITE = (235, 235, 235)
GREY = (150, 150, 150)
BG = (24, 24, 28)


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    """Grayscale to 3-channel BGR."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _text(img: np.ndarray, s: str, org: Tuple[int, int],
          colour: Tuple[int, int, int] = WHITE, scale: float = 0.5) -> None:
    """Draw a caption."""
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1,
                cv2.LINE_AA)


def draw_ground_truth(search_bgr: np.ndarray, gt: Dict[str, Any]) -> None:
    """Overlay the footprint quad and centre cross on a search image.

    The rotated quad is drawn rather than the axis-aligned box because the box
    overstates the footprint by up to 41 % under rotation, and a viewer who
    checks the box will conclude the label is loose when it is exact.
    """
    quad = np.asarray(gt["quad"], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(search_bgr, [quad], True, GREEN, 1, cv2.LINE_AA)
    x, y = int(round(gt["gt_x"])), int(round(gt["gt_y"]))
    cv2.line(search_bgr, (x - 14, y), (x + 14, y), RED, 1, cv2.LINE_AA)
    cv2.line(search_bgr, (x, y - 14), (x, y + 14), RED, 1, cv2.LINE_AA)


def montage(plan: Plan, ref: np.ndarray, search: np.ndarray,
            zncc: Optional[float] = None) -> np.ndarray:
    """Reference beside search with the label drawn.

    Args:
        plan: The sample plan.
        ref: Reference frame.
        search: Search frame.
        zncc: Optional label-check score to print in the caption.

    Returns:
        A BGR image.
    """
    gt = plan.ground_truth()
    n = ref.shape[0]
    canvas = np.full((n + 64, 2 * n + 30, 3), BG, np.uint8)
    canvas[46:46 + n, 10:10 + n] = _to_bgr(ref)
    right = _to_bgr(search)
    draw_ground_truth(right, gt)
    canvas[46:46 + n, n + 20:n + 20 + n] = right

    marker = (f"landmark={plan.target.kind}" if plan.target
              else f"defect={plan.defect.kind}")
    _text(canvas, f"REFERENCE  {n}x{n} @ {plan.reference.px_nm:.3f} nm/px   "
                  f"[{plan.style} / {plan.difficulty}]  {marker}", (12, 20))
    _text(canvas, f"seed={plan.seed}  dose={plan.ref_capture.noise.dose:.0f} e/px"
                  f"  sigma={plan.ref_capture.probe.sigma_px:.2f} px",
          (12, 38), GREY, 0.42)
    _text(canvas, f"SEARCH  @ {plan.search.px_nm:.3f} nm/px   "
                  f"GT=({gt['gt_x']:.1f}, {gt['gt_y']:.1f})  "
                  f"rot={gt['rel_rotation_deg']:+.2f} deg  "
                  f"scale={gt['scale_ratio']:.3f}", (n + 22, 20))
    tail = f"  ZNCC={zncc:.3f}" if zncc is not None else ""
    _text(canvas, f"dose={plan.search_capture.noise.dose:.0f} e/px  "
                  f"footprint={gt['footprint_px']:.1f} px  "
                  f"label_corr={gt['label_correction_px']:.2f} px{tail}",
          (n + 22, 38), GREY, 0.42)
    return canvas


def zoom(plan: Plan, ref: np.ndarray, search: np.ndarray,
         panel: int = 320) -> np.ndarray:
    """Demagnified reference beside the search crop at the label.

    Args:
        plan: The sample plan.
        ref: Reference frame.
        search: Search frame.
        panel: Size of each panel in pixels.

    Returns:
        A BGR image with two panels that should show the same structure.
    """
    gt = plan.ground_truth()
    factor = max(1, int(round(gt["scale_ratio"])))
    small = G.box_downsample(ref.astype(np.float32), factor).astype(np.uint8)

    k = small.shape[0]
    x0 = int(round(gt["gt_x"] - k / 2.0))
    y0 = int(round(gt["gt_y"] - k / 2.0))
    x0 = max(0, min(search.shape[1] - k, x0))
    y0 = max(0, min(search.shape[0] - k, y0))
    crop = search[y0:y0 + k, x0:x0 + k]

    up = cv2.resize(small, (panel, panel), interpolation=cv2.INTER_NEAREST)
    vp = cv2.resize(crop, (panel, panel), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((panel + 46, 2 * panel + 30, 3), BG, np.uint8)
    canvas[36:36 + panel, 10:10 + panel] = _to_bgr(up)
    canvas[36:36 + panel, panel + 20:panel + 20 + panel] = _to_bgr(vp)
    _text(canvas, f"reference / {factor}", (12, 16))
    _text(canvas, "search @ ground truth", (panel + 22, 16))
    _text(canvas, f"{plan.style} / {plan.difficulty}  seed={plan.seed}",
          (12, 32), GREY, 0.42)
    return canvas


def grid(plans: Sequence[Plan], renders: Sequence[Tuple[np.ndarray, np.ndarray]],
         cell: int = 200, cols: int = 4) -> np.ndarray:
    """Thumbnail grid of search images with labels drawn.

    Args:
        plans: Sample plans.
        renders: Matching ``(reference, search)`` pairs.
        cell: Thumbnail size.
        cols: Grid width.

    Returns:
        A BGR image.
    """
    rows = int(math.ceil(len(plans) / cols))
    canvas = np.full((rows * (cell + 22), cols * (cell + 8), 3), BG, np.uint8)
    for i, (plan, (_, search)) in enumerate(zip(plans, renders)):
        gt = plan.ground_truth()
        img = _to_bgr(search)
        draw_ground_truth(img, gt)
        thumb = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y, x = r * (cell + 22) + 18, c * (cell + 8) + 4
        canvas[y:y + cell, x:x + cell] = thumb
        _text(canvas, f"{plan.style[:6]}/{plan.difficulty[:4]} "
                      f"d={plan.search_capture.noise.dose:.0f}",
              (x, y - 5), GREY, 0.38)
    return canvas


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _plans_from(args: argparse.Namespace, cfg: GeneratorConfig) -> List[Plan]:
    """Plans either from an existing dataset or freshly sampled."""
    if args.dataset:
        root = Path(args.dataset)
        rows = sorted(iter_rows(root / "labels.jsonl"), key=lambda r: r["id"])
        if args.difficulty:
            rows = [r for r in rows if r["difficulty"] == args.difficulty]
        if args.style:
            rows = [r for r in rows if r["style"] == args.style]
        rows = rows[:args.pairs]
        cfg_path = root / "config.yaml"
        if cfg_path.exists():
            cfg = GeneratorConfig.from_yaml(cfg_path)
        return [plan_sample(int(r["seed"]), int(r["id"]), cfg) for r in rows]

    seeds = sample_seeds(args.seed, args.pairs)
    return [plan_sample(int(s), i, cfg) for i, s in enumerate(seeds)]


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point."""
    if not _HAS_CV2:
        sys.exit("visualize needs opencv: pip install opencv-python-headless")

    ap = argparse.ArgumentParser(
        prog="python -m driftsense.cli.visualize",
        description="Render annotated previews of Drift-Sense samples.")
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--out", default="data/preview")
    ap.add_argument("--dataset", default=None,
                    help="visualise samples from an existing dataset root")
    ap.add_argument("--preset", choices=PRESETS, default="default")
    ap.add_argument("--config", default=None)
    ap.add_argument("--style", choices=("dram", "finfet"), default=None)
    ap.add_argument("--difficulty", choices=("easy", "medium", "hard"),
                    default=None)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--mode", choices=("montage", "zoom", "grid", "all"),
                    default="all")
    ap.add_argument("--quality", type=int, default=88, help="JPEG quality")
    args = ap.parse_args(argv)

    cfg = GeneratorConfig.from_yaml(args.config) if args.config \
        else preset(args.preset)
    if args.style:
        cfg = cfg.override({"styles": [args.style]})
    if args.difficulty == "hard":
        cfg = cfg.override({"hard_fraction": 1.0})
    elif args.difficulty in ("easy", "medium"):
        cfg = cfg.override({"hard_fraction": 0.0})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plans = _plans_from(args, cfg)
    renders = []
    scores = []
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]

    for i, plan in enumerate(plans):
        ref, search = render(plan)
        renders.append((ref, search))
        z = verify_pair(plan, ref, search)
        scores.append(z)
        if args.mode in ("montage", "all"):
            cv2.imwrite(str(out / f"{i:03d}_montage.jpg"),
                        montage(plan, ref, search, z), enc)
        if args.mode in ("zoom", "all"):
            cv2.imwrite(str(out / f"{i:03d}_zoom.jpg"),
                        zoom(plan, ref, search), enc)
        sys.stdout.write(f"\r  {i+1}/{len(plans)}  ZNCC {z:.3f}   ")
        sys.stdout.flush()

    if args.mode in ("grid", "all") and plans:
        cv2.imwrite(str(out / "grid.jpg"), grid(plans, renders), enc)

    arr = np.array(scores)
    print(f"\n\n  wrote {len(plans)} samples to {out}")
    print(f"  label ZNCC : mean {arr.mean():.3f}  min {arr.min():.3f}  "
          f"max {arr.max():.3f}")
    print(f"  mix        : "
          + ", ".join(f"{d}={sum(p.difficulty == d for p in plans)}"
                      for d in ("easy", "medium", "hard")))
    if arr.min() < 0.15:
        print("  WARNING: at least one sample has a suspicious label; "
              "check its *_zoom.jpg -- the two panels should match.")


if __name__ == "__main__":
    main()
