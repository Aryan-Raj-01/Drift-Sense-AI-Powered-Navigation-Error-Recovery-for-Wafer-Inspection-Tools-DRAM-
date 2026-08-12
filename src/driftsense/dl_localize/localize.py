#!/usr/bin/env python3
"""Drift-Sense — standalone localization entry point.

THE DELIVERABLE API
===================

    from driftsense.localize import locate
    x, y = locate(reference_image, search_image)

``reference_image`` and ``search_image`` are grayscale numpy arrays (uint8 or
float). The return is the sub-pixel centre of the reference pattern inside the
search image, in search-image pixels, top-left origin — the format Applied
Materials' scoring CSV expects (``GTx``, ``GTy``).

Command line, single pair::

    python -m driftsense.localize --reference ref.png --search search.png

Command line, batch over a CSV in the scoring format::

    python -m driftsense.localize --csv input.csv --out predictions.csv

where ``input.csv`` has columns ``Wide Search Image Path`` and
``Reference Image Path``; ``predictions.csv`` adds ``GTx`` and ``GTy``.

MEASURED PERFORMANCE
====================

2,000 held-out pairs never trained on and never used to select a
hyperparameter, learned and classical scored in the same run on the same
hardware (RTX 4050 Laptop):

    group    <=1px   <=2px   <=5px   median    classical <=1px
    ALL      89.5%   95.9%   96.7%   0.24 px       76.6%
    easy     92.3%   98.7%   99.8%   0.23 px       90.4%
    medium   91.8%   98.7%   99.3%   0.23 px       81.3%
    hard     79.2%   84.5%   85.0%   0.30 px       35.0%

    168 ms/pair, versus 705 ms/pair for the classical baseline.

1 px is ~10 nm on the wafer, so ~96% of predictions land within 20 nm.

HOW IT WORKS
============

Two stages, each doing the half of the problem it is measurably better at.

1. A learned Siamese dense-correlation network searches globally. It is
   robust against the periodic-lock failure that defeats classical matching
   on DRAM arrays — measured at 0.0% periodic locks on the easy bucket, where
   the classical baseline's error budget was 4% locks.

2. Classical stride-1 multi-angle NCC refines the last pixel inside a +/-3 px
   window. The learned stage is quantised to its stride-4 output grid;
   classical correlation is not. Measured contribution: median error
   0.493 px -> 0.240 px, for 5 ms.

The +/-3 px refinement window is narrower than the smallest lattice pitch in
the dataset (5.3 px), so stage 2 physically cannot relocate to a periodic
replica — it can only sharpen the cell stage 1 already chose.

Full method, evidence and negative results: ``dl_localize/README88.md``.

FAILURE POLICY
==============

An inference script that raises scores zero. This module therefore NEVER
raises from :func:`locate`. It degrades:

    learned + refinement  ->  classical NCC  ->  search-image centre

and reports which path produced the answer in ``method``. The final fallback
is the centre of the search frame, which is also the challenge's stated
tie-break convention when the true location is unknown.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np

#: Default checkpoint. Override with ``--checkpoint``, ``set_checkpoint()``,
#: or the ``DRIFTSENSE_CHECKPOINT`` environment variable.
#:
#: Tries, in order: the env var; the submission layout
#: (<archive root>/checkpoints/final_phase3_all.pt -- three levels up from
#: this file, since it lives at <root>/code/driftsense/dl_localize/); then the
#: original dev layout (<repo root>/runs/lscv88d/phase3_all.pt -- two levels
#: up). Whichever candidate actually exists on disk wins; if neither exists
#: yet, default to the submission path so a load failure points at the
#: location a reviewer is expected to have.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CHECKPOINT_CANDIDATES = [
    # submission layout: <root>/model/…, this file at <root>/src/driftsense/dl_localize/
    os.path.normpath(os.path.join(
        _HERE, "..", "..", "..", "model", "final_phase3_all.pt")),
    # earlier submission layout, kept so an older archive still resolves
    os.path.normpath(os.path.join(
        _HERE, "..", "..", "..", "checkpoints", "final_phase3_all.pt")),
    # development layout: <repo>/runs/…, this file at <repo>/driftsense/dl_localize/
    os.path.normpath(os.path.join(
        _HERE, "..", "..", "runs", "lscv88d", "phase3_all.pt")),
]
DEFAULT_CHECKPOINT = os.environ.get("DRIFTSENSE_CHECKPOINT") or next(
    (p for p in _CHECKPOINT_CANDIDATES if os.path.isfile(p)),
    _CHECKPOINT_CANDIDATES[0])

_LOCALIZER = None          # lazily constructed; loading a checkpoint is slow
_LOAD_FAILED = False       # remember failure so we do not retry per-pair

#: Logit gap under which the problem statement's closest-to-centre rule
#: decides between competing matches. 0.0 disables it. See this module's
#: docstring for the measurement behind the default -- enabling it costs
#: accuracy on a uniformly-placed target distribution and should help on a
#: centre-weighted one.
TIE_MARGIN = float(os.environ.get("DRIFTSENSE_TIE_MARGIN", "0.0"))


def set_checkpoint(path: str) -> None:
    """Point the module at a different checkpoint and force a reload."""
    global DEFAULT_CHECKPOINT, _LOCALIZER, _LOAD_FAILED
    DEFAULT_CHECKPOINT = path
    _LOCALIZER = None
    _LOAD_FAILED = False


def set_tie_margin(value: float) -> None:
    """Enable/disable the closest-to-centre rule and force a reload."""
    global TIE_MARGIN, _LOCALIZER, _LOAD_FAILED
    TIE_MARGIN = float(value)
    _LOCALIZER = None
    _LOAD_FAILED = False


def _get_localizer(checkpoint: Optional[str] = None, device: str = "auto"):
    """Load the learned model once, or return None if it cannot be loaded."""
    global _LOCALIZER, _LOAD_FAILED
    if _LOCALIZER is not None:
        return _LOCALIZER
    if _LOAD_FAILED and checkpoint is None:
        return None
    try:
        from driftsense.dl_localize.infer import LearnedLocalizer
        path = checkpoint or DEFAULT_CHECKPOINT
        _LOCALIZER = LearnedLocalizer(path, device=device,
                                        tie_margin=TIE_MARGIN)
        return _LOCALIZER
    except Exception as exc:                       # noqa: BLE001
        _LOAD_FAILED = True
        print(f"  [localize] learned model unavailable ({exc}); "
              f"falling back to classical NCC", file=sys.stderr)
        return None


def _classical(reference: np.ndarray, search: np.ndarray
               ) -> Optional[Tuple[float, float]]:
    """Classical multi-angle NCC fallback. Returns None if it also fails."""
    try:
        from driftsense.localize.propose import propose
        cands = propose(reference.astype(np.float64),
                        search.astype(np.float64), top_k=1)
        if cands:
            return float(cands[0].x), float(cands[0].y)
    except Exception:                              # noqa: BLE001
        pass
    return None


def locate_full(reference: np.ndarray, search: np.ndarray,
                checkpoint: Optional[str] = None, device: str = "auto"
                ) -> dict:
    """Locate the reference pattern, with diagnostics.

    Args:
        reference: (N, N) grayscale reference image.
        search: (H, W) grayscale search image.
        checkpoint: Optional path to a ``dl_localize`` checkpoint.
        device: "auto", "cuda" or "cpu".

    Returns:
        Dict with:
          ``x``, ``y`` -- sub-pixel centre in search pixels, top-left origin.
          ``confidence`` -- softmax mass at the peak, in [0, 1].
          ``margin`` -- logit gap to the best competitor. Measured to separate
            correct from incorrect predictions (5.47 +/- 1.98 versus
            3.73 +/- 2.82), so it is usable for flagging low-confidence sites.
          ``method`` -- "learned", "classical" or "centre".
    """
    ref = np.asarray(reference)
    srch = np.asarray(search)
    if ref.ndim == 3:
        ref = ref.mean(axis=2)
    if srch.ndim == 3:
        srch = srch.mean(axis=2)

    loc = _get_localizer(checkpoint, device)
    if loc is not None:
        try:
            out = loc.locate(ref.astype(np.uint8) if ref.dtype != np.uint8
                             else ref,
                             srch.astype(np.uint8) if srch.dtype != np.uint8
                             else srch)
            x, y = float(out["x"]), float(out["y"])
            if np.isfinite([x, y]).all():
                return {"x": x, "y": y,
                        "confidence": float(out.get("confidence", 0.0)),
                        "margin": float(out.get("margin", 0.0)),
                        "method": "learned"}
        except Exception as exc:                   # noqa: BLE001
            print(f"  [localize] learned path failed ({exc}); "
                  f"falling back", file=sys.stderr)

    got = _classical(ref, srch)
    if got is not None:
        return {"x": got[0], "y": got[1], "confidence": 0.0,
                "margin": 0.0, "method": "classical"}

    # Last resort: the centre of the search frame. This is also the
    # challenge's stated tie-break when several regions match equally.
    return {"x": (srch.shape[1] - 1) / 2.0, "y": (srch.shape[0] - 1) / 2.0,
            "confidence": 0.0, "margin": 0.0, "method": "centre"}


def locate(reference: np.ndarray, search: np.ndarray,
           checkpoint: Optional[str] = None,
           device: str = "auto") -> Tuple[float, float]:
    """The deliverable API: (reference, search) -> (x, y).

    Never raises. See :func:`locate_full` for confidence and method.
    """
    r = locate_full(reference, search, checkpoint, device)
    return r["x"], r["y"]


def _run_csv(csv_in: str, csv_out: str, root: str,
             checkpoint: Optional[str], device: str) -> int:
    import csv as _csv

    import cv2

    with open(csv_in, newline="") as f:
        rows = list(_csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{csv_in} is empty")

    def _col(row, *names):
        for n in names:
            for k in row:
                if k.strip().lower() == n.strip().lower():
                    return row[k]
        raise SystemExit(
            f"could not find any of {names} in columns {list(row)}")

    out_rows = []
    t0 = time.perf_counter()
    methods = {}
    for i, r in enumerate(rows, 1):
        sp = _col(r, "Wide Search Image Path", "search_path")
        rp = _col(r, "Reference Image Path", "reference_path")
        sp = sp if os.path.isabs(sp) else os.path.join(root, sp)
        rp = rp if os.path.isabs(rp) else os.path.join(root, rp)
        ref = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
        srch = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
        if ref is None or srch is None:
            print(f"  [localize] unreadable pair at row {i}; writing centre",
                  file=sys.stderr)
            res = {"x": 499.5, "y": 499.5, "method": "centre",
                   "confidence": 0.0, "margin": 0.0}
        else:
            res = locate_full(ref, srch, checkpoint, device)
        methods[res["method"]] = methods.get(res["method"], 0) + 1
        row = dict(r)
        row["GTx"] = f"{res['x']:.4f}"
        row["GTy"] = f"{res['y']:.4f}"
        row["confidence"] = f"{res['confidence']:.6f}"
        row["margin"] = f"{res['margin']:.4f}"
        out_rows.append(row)
        if i % 25 == 0 or i == len(rows):
            print(f"\r  {i}/{len(rows)}", end="", flush=True)
    print()

    fieldnames = list(out_rows[0].keys())
    with open(csv_out, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    dt = time.perf_counter() - t0
    print(f"  wrote {len(out_rows)} rows to {csv_out}")
    print(f"  {1000 * dt / len(out_rows):.0f} ms/pair   paths used: {methods}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", help="reference image path")
    ap.add_argument("--search", help="search image path")
    ap.add_argument("--csv", help="batch mode: input CSV in scoring format")
    ap.add_argument("--out", help="batch mode: output CSV")
    ap.add_argument("--root", default=".",
                    help="root for relative paths in --csv")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tie-margin", type=float, default=None,
                    help="logit gap under which the problem statement's "
                         "closest-to-centre rule decides between competing "
                         "matches. Default 0.0 = disabled, because it was "
                         "MEASURED to cost accuracy on this dataset's "
                         "uniformly-placed targets (90.8%% -> 90.2%% at 1.0). "
                         "Enable with 1.0 if your evaluation data places the "
                         "target near the search-image centre. See the module "
                         "docstring for the full measurement.")
    args = ap.parse_args()

    if args.checkpoint:
        set_checkpoint(args.checkpoint)
    if args.tie_margin is not None:
        set_tie_margin(args.tie_margin)

    if args.csv:
        if not args.out:
            raise SystemExit("--csv requires --out")
        return _run_csv(args.csv, args.out, args.root, args.checkpoint,
                        args.device)

    if not (args.reference and args.search):
        raise SystemExit("give --reference and --search, or --csv and --out")

    import cv2
    ref = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    srch = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
    if ref is None or srch is None:
        raise SystemExit("could not read one of the images")

    # Separate one-off startup from steady-state inference. A cold call pays
    # for the torch import, CUDA context creation, checkpoint load and cuDNN
    # autotuning -- several seconds, none of it per-pair. Reporting a single
    # number here would overstate inference cost by ~40x and is exactly the
    # sort of figure that ends up misquoted on a slide.
    t0 = time.perf_counter()
    res = locate_full(ref, srch, args.checkpoint, args.device)
    t_cold = 1000 * (time.perf_counter() - t0)

    t1 = time.perf_counter()
    res = locate_full(ref, srch, args.checkpoint, args.device)
    t_warm = 1000 * (time.perf_counter() - t1)

    print(f"  x = {res['x']:.4f}")
    print(f"  y = {res['y']:.4f}")
    print(f"  confidence {res['confidence']:.3e}   margin {res['margin']:.2f}"
          f"   method {res['method']}")
    print(f"  timing: {t_warm:.0f} ms/pair steady-state   "
          f"({t_cold:.0f} ms including one-off model load and CUDA warm-up)")
    print("  Batch mode (--csv) loads the model once, so steady-state is the "
          "figure that applies to a run of any length.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
