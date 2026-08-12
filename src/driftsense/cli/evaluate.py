"""``evaluate.py`` -- score a localisation algorithm.

    # the classical baseline, on a generated dataset
    python -m driftsense.cli.evaluate --dataset data/eval --pairs 60

    # your own predictions
    python -m driftsense.cli.evaluate --dataset data/eval \\
        --predictions preds.csv

    # robustness sweep against the noisier hidden-test simulation
    python -m driftsense.cli.evaluate --preset hidden_test_sim --pairs 60

A prediction file is a CSV with columns ``id,pred_x,pred_y`` and optionally
``score``.  Ground truth is joined from the dataset manifest, so predictions can
come from anywhere -- another language, another machine, a teammate.

Read the ``hard`` row, not the aggregate.  The aggregate moves when the
difficulty mix moves, so it measures your dataset as much as your algorithm.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from driftsense.config import PRESETS, GeneratorConfig, preset
from driftsense.eval.baseline_ncc import dominant_pitch_px, locate
from driftsense.eval.metrics import Prediction, report, save
from driftsense.metadata import iter_rows
from driftsense.pipeline import plan_sample, render
from driftsense.rng import sample_seeds


def _load_predictions(path: Path) -> Dict[int, Dict[str, float]]:
    """Read ``id,pred_x,pred_y[,score]`` from CSV."""
    import csv

    out: Dict[int, Dict[str, float]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            out[int(row["id"])] = {
                "pred_x": float(row["pred_x"]),
                "pred_y": float(row["pred_y"]),
                "score": float(row.get("score", "nan") or "nan"),
            }
    if not out:
        raise ValueError(f"{path}: no predictions found")
    return out


def _rows_for(args: argparse.Namespace, cfg: GeneratorConfig) -> List[dict]:
    """Manifest rows, either from a dataset or planned on the spot."""
    if args.dataset:
        root = Path(args.dataset)
        rows = sorted(iter_rows(root / "labels.jsonl"), key=lambda r: r["id"])
        return rows[:args.pairs]
    seeds = sample_seeds(args.seed, args.pairs)
    return [{"id": i, "seed": int(s)} for i, s in enumerate(seeds)]


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(
        prog="python -m driftsense.cli.evaluate",
        description="Score a localisation algorithm on Drift-Sense data.")
    ap.add_argument("--dataset", default=None, help="dataset root")
    ap.add_argument("--predictions", default=None,
                    help="CSV with id,pred_x,pred_y[,score]; runs the NCC "
                         "baseline when omitted")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--preset", choices=PRESETS, default="default")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--angles", type=int, default=5,
                    help="rotations the baseline searches over")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args(argv)

    cfg = GeneratorConfig.from_yaml(args.config) if args.config \
        else preset(args.preset)
    if args.dataset:
        p = Path(args.dataset) / "config.yaml"
        if p.exists():
            cfg = GeneratorConfig.from_yaml(p)

    rows = _rows_for(args, cfg)
    external = _load_predictions(Path(args.predictions)) if args.predictions else None
    angles = tuple(np.linspace(-2.5, 2.5, max(1, args.angles))) \
        if args.angles > 1 else (0.0,)

    preds: List[Prediction] = []
    t_start = time.time()
    for k, row in enumerate(rows):
        plan = plan_sample(int(row["seed"]), int(row["id"]), cfg)
        gt = plan.ground_truth()

        if external is not None:
            if int(row["id"]) not in external:
                continue
            e = external[int(row["id"])]
            px, py, score, seconds = e["pred_x"], e["pred_y"], e["score"], float("nan")
            pitch = float("nan")
        else:
            ref, search = render(plan)
            t0 = time.time()
            px, py, score, _tied = locate(ref, search, scale=gt["scale_ratio"],
                                          angles=angles)
            seconds = time.time() - t0
            pitch = dominant_pitch_px(search)

        preds.append(Prediction(
            id=int(row["id"]), pred_x=px, pred_y=py,
            gt_x=gt["gt_x"], gt_y=gt["gt_y"],
            difficulty=plan.difficulty, style=plan.style,
            dose=plan.search_capture.noise.dose,
            pitch_px=pitch, score=score, seconds=seconds))

        err = preds[-1].error_px
        sys.stdout.write(f"\r  {k+1}/{len(rows)}  last error {err:8.2f} px   ")
        sys.stdout.flush()

    if not preds:
        sys.exit("no predictions matched the manifest ids")

    name = "external predictions" if external else "baseline NCC"
    print(f"\n  {len(preds)} samples in {time.time() - t_start:.1f} s")
    print(report(preds, name))

    if args.out:
        path = save(preds, args.out,
                    extra={"algorithm": name, "config_hash": cfg.version_hash(),
                           "dataset": args.dataset})
        print(f"\n  results written to {path}")


if __name__ == "__main__":
    main()
