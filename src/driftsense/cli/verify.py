"""``verify.py`` -- check a dataset before you train on it.

    python -m driftsense.cli.verify --dataset data/train --pairs 60

Runs the checks that catch the failures which are invisible by eye:

**Label correctness (ZNCC).**  Demagnify the reference by the recorded scale,
undo the recorded rotation, correlate against the search image at the recorded
centre.  Healthy is 0.4-0.9: high enough that the label points at the right
structure, low enough that the two frames are independent captures rather than
copies.  A mean above ~0.95 is as much of a red flag as one below 0.2 -- it
means the frames share noise, which is a shortcut a network will find and the
hidden test set will not provide.

**Determinism.**  Re-render two samples and compare bytes.  If a dataset is not
reproducible from its seed then the manifest is not a dataset, it is a
suggestion.

**Duplicate seeds.**  The classic ``fork`` bug produces N identical copies of
the data and the images look perfectly fine.

**Balance and coverage.**  Difficulty mix, style mix, dose range, rotation
spread, footprint spread.  A generator that has quietly collapsed to one
operating point is a real failure and an aggregate accuracy number will not
reveal it.

**Saturation.**  Clipped highlights and crushed blacks carry no information.
More than a few percent means the video stage is mis-tuned.

Exit code is non-zero when a check fails, so this can gate a pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from driftsense.config import PRESETS, GeneratorConfig, preset
from driftsense.metadata import iter_rows
from driftsense.pipeline import DIFFICULTIES, plan_sample, render, verify_pair
from driftsense.rng import sample_seeds

#: Thresholds that decide pass/fail.
MIN_MEAN_ZNCC = 0.35
MIN_SAMPLE_ZNCC = 0.10
MAX_MEAN_ZNCC = 0.97
MAX_SATURATION = 0.06


class Check:
    """One named check with a pass/fail verdict and a detail line."""

    def __init__(self, name: str, ok: bool, detail: str) -> None:
        self.name = name
        self.ok = ok
        self.detail = detail

    def __str__(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        return f"  [{mark}] {self.name:<26} {self.detail}"


def verify_dataset(root: Optional[Path], cfg: GeneratorConfig, pairs: int,
                   seed: int) -> List[Check]:
    """Run every check and return the verdicts.

    Args:
        root: Existing dataset root, or ``None`` to check freshly drawn samples.
        cfg: Configuration to render with.
        pairs: How many samples to render.
        seed: Root seed when sampling fresh.

    Returns:
        The list of :class:`Check` results.
    """
    checks: List[Check] = []

    if root is not None:
        rows = sorted(iter_rows(root / "labels.jsonl"), key=lambda r: r["id"])
        cfg_path = root / "config.yaml"
        if cfg_path.exists():
            cfg = GeneratorConfig.from_yaml(cfg_path)
        sample_rows = rows[:pairs]
        plans = [plan_sample(int(r["seed"]), int(r["id"]), cfg)
                 for r in sample_rows]

        seeds_all = [r["seed"] for r in rows]
        dupes = len(seeds_all) - len(set(seeds_all))
        checks.append(Check("unique seeds", dupes == 0,
                            f"{dupes} duplicates in {len(rows)} rows"
                            + ("  <- the fork bug" if dupes else "")))

        hashes = {r.get("config_hash") for r in rows}
        checks.append(Check("single config hash", len(hashes) == 1,
                            f"{', '.join(sorted(str(h) for h in hashes))}"))

        # The stored label must equal the freshly planned label, or the manifest
        # and the renderer disagree about what the data is.
        worst = 0.0
        for r, p in zip(sample_rows, plans):
            gt = p.ground_truth()
            worst = max(worst, abs(gt["gt_x"] - r["gt_x"]),
                        abs(gt["gt_y"] - r["gt_y"]))
        checks.append(Check("manifest matches planner", worst < 1e-3,
                            f"max label delta {worst:.2e} px"))
    else:
        seeds = sample_seeds(seed, pairs)
        plans = [plan_sample(int(s), i, cfg) for i, s in enumerate(seeds)]
        checks.append(Check("unique seeds", len(set(int(s) for s in seeds)) == pairs,
                            f"{pairs} freshly drawn seeds"))

    # -- render and measure -------------------------------------------------- #
    zncc: List[float] = []
    sat_hi: List[float] = []
    sat_lo: List[float] = []
    contrast: List[float] = []
    first_ref = None
    for i, plan in enumerate(plans):
        ref, search = render(plan)
        if i == 0:
            first_ref = ref
        zncc.append(verify_pair(plan, ref, search))
        for img in (ref, search):
            sat_hi.append(float((img >= 254).mean()))
            sat_lo.append(float((img <= 1).mean()))
            contrast.append(float(img.std()))
        sys.stdout.write(f"\r  rendering {i+1}/{len(plans)}   ")
        sys.stdout.flush()
    print()

    z = np.array(zncc)
    checks.append(Check("label correctness", z.mean() >= MIN_MEAN_ZNCC
                        and z.min() >= MIN_SAMPLE_ZNCC,
                        f"ZNCC mean {z.mean():.3f}, min {z.min():.3f} "
                        f"(want >= {MIN_MEAN_ZNCC})"))
    checks.append(Check("frames independent", z.mean() <= MAX_MEAN_ZNCC,
                        f"ZNCC mean {z.mean():.3f} (want <= {MAX_MEAN_ZNCC}; "
                        f"higher means shared noise)"))

    ref2, _ = render(plans[0])
    checks.append(Check("deterministic render",
                        first_ref is not None and np.array_equal(first_ref, ref2),
                        "re-render is byte-identical"))

    checks.append(Check("no saturation",
                        max(sat_hi) < MAX_SATURATION and max(sat_lo) < MAX_SATURATION,
                        f"max clipped {100*max(sat_hi):.1f}% high, "
                        f"{100*max(sat_lo):.1f}% low"))
    checks.append(Check("contrast present", min(contrast) > 8.0,
                        f"min std {min(contrast):.1f} of 255"))

    # -- coverage ------------------------------------------------------------ #
    mix = {d: sum(p.difficulty == d for p in plans) for d in DIFFICULTIES}
    styles = {s: sum(p.style == s for p in plans) for s in cfg.styles}
    checks.append(Check("difficulty coverage", all(v > 0 for v in mix.values()),
                        ", ".join(f"{k}={v}" for k, v in mix.items())))
    checks.append(Check("style coverage", all(v > 0 for v in styles.values()),
                        ", ".join(f"{k}={v}" for k, v in styles.items())))

    dose = np.array([p.search_capture.noise.dose for p in plans])
    rot = np.array([p.ground_truth()["rel_rotation_deg"] for p in plans])
    foot = np.array([p.ground_truth()["footprint_px"] for p in plans])
    corr = np.array([p.ground_truth()["label_correction_px"] for p in plans])

    checks.append(Check("dose spread", dose.max() / max(dose.min(), 1e-9) > 2.0,
                        f"{dose.min():.0f} .. {dose.max():.0f} e/px"))
    checks.append(Check("rotation spread", rot.std() > 0.3,
                        f"sigma {rot.std():.2f} deg, |max| {np.abs(rot).max():.2f}"))
    checks.append(Check("footprint varies", foot.std() > 0.05,
                        f"{foot.min():.2f} .. {foot.max():.2f} px "
                        f"(never exactly 100)"))
    checks.append(Check("scan error corrected", corr.max() > 0.02 and corr.max() < 8.0,
                        f"label correction mean {corr.mean():.2f} px, "
                        f"max {corr.max():.2f} px"))
    checks.append(Check("targets on-screen",
                        all(g["footprint_px"] / 2 < g["gt_x"] < 1000 - g["footprint_px"] / 2
                            and g["footprint_px"] / 2 < g["gt_y"] < 1000 - g["footprint_px"] / 2
                            for g in (p.ground_truth() for p in plans)),
                        "every footprint fits inside the search frame"))
    checks.append(Check("well posed",
                        all((p.target is None) != (p.defect is None) for p in plans),
                        "every sample has exactly one unique marker"))
    return checks


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(
        prog="python -m driftsense.cli.verify",
        description="Health-check a Drift-Sense dataset or configuration.")
    ap.add_argument("--dataset", default=None, help="dataset root to check")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--preset", choices=PRESETS, default="default")
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--json", default=None, help="write results to this file")
    args = ap.parse_args(argv)

    cfg = GeneratorConfig.from_yaml(args.config) if args.config \
        else preset(args.preset)
    root = Path(args.dataset) if args.dataset else None

    print(f"\nverifying {'dataset ' + str(root) if root else 'preset ' + args.preset}"
          f"  ({args.pairs} samples)\n")
    checks = verify_dataset(root, cfg, args.pairs, args.seed)
    print()
    for c in checks:
        print(c)

    failed = [c for c in checks if not c.ok]
    print(f"\n  {len(checks) - len(failed)}/{len(checks)} checks passed")
    if args.json:
        Path(args.json).write_text(json.dumps(
            [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
            indent=2))
    if failed:
        print("\n  FAILED: " + ", ".join(c.name for c in failed))
        sys.exit(1)
    print("  dataset looks healthy\n")


if __name__ == "__main__":
    main()
