#!/usr/bin/env python3
"""Add more TRAIN samples to an existing subset split, without moving val.

    python -m driftsense.dl_localize.extend_subset ^
        --dataset "<dataset-root>" ^
        --split-csv "<dataset-root>\\subset_split.csv" ^
        --add hard=10000 ^
        --out "<dataset-root>\\subset_split_hard.csv"

WHY
===
Measured on 2000 held-out val pairs, with the learned model plus local
refinement:

    group   <=1px    hit   near-miss   periodic-lock   gross-miss
    easy    92.6%   92.6%      7.0%            0.2%         0.2%
    medium  92.8%   92.8%      6.5%            0.2%         0.5%
    hard    69.8%   69.8%      5.0%            9.8%        15.5%

easy and medium are essentially solved by this architecture -- their residual
7% is near-misses driven by non-rigid scan distortion (failing samples have
label_correction_px 1.42 vs 0.65 for the population, while their rotation is
identical at 1.28 vs 1.25).  A global affine ECC warp was tried against that
and measured NOT to help (median 0.617 -> 0.609 px, <=1px unchanged at 62.5%).

Hard is different, and it is data-starved rather than model-starved:

  * The subset has 1,600 hard TRAIN samples.  The manifest has 87,720.
  * Phase 3 draws uniformly from an 8,000-sample pool of which hard is 20%,
    so 2250 steps x 8 samples x 0.2 = 3,600 hard draws -- each of the 1,600
    unique hard samples is seen about 2.25 times in the entire run.
  * 9.8% of hard predictions are periodic locks and 15.5% are gross misses,
    and gross misses carry a mean logit margin of 0.63 against 3.49 for hits.
    The model is not confidently wrong; it is unconfidently wrong, which is
    what too little data on a hard distribution looks like.

WHAT THIS SCRIPT GUARANTEES
===========================
New rows are drawn ONLY from seeds absent from the input CSV, and are all
marked "train".  The existing train/val assignment is copied through
unchanged.  So the 2,000-pair val set stays byte-identical and every number
measured before and after this change is directly comparable -- which is the
whole point of doing it this way rather than regenerating a split.

The script refuses to write if any seed would end up in more than one split.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="directory containing labels.jsonl")
    ap.add_argument("--split-csv", required=True,
                    help="existing split to extend; never modified")
    ap.add_argument("--add", nargs="+", required=True,
                    help="difficulty=count, e.g. hard=10000 medium=2000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rng-seed", type=int, default=8888)
    ap.add_argument("--split", default="train",
                    choices=["train", "golden"],
                    help="which split the new rows join. Use 'golden' to "
                         "carve a clean held-out set from seeds no run has "
                         "ever touched -- neither trained on NOR used to "
                         "choose a hyperparameter. The val set has been used "
                         "to pick the template bank, the refinement radius "
                         "and angle count, the readout mode, the sampling "
                         "mix and the phase LR schedule, so numbers measured "
                         "on it are mildly optimistic by construction. "
                         "Golden is never trained on and is read ONCE.")
    args = ap.parse_args()

    want = {}
    for spec in args.add:
        k, v = spec.split("=")
        want[k.strip()] = int(v)

    old = pd.read_csv(args.split_csv,
                      dtype={"id": "int64", "seed": "int64",
                             "difficulty": "string", "split": "string"})
    used = set(int(s) for s in old["seed"])
    print(f"  existing split: {len(old)} rows, {len(used)} seeds")
    print(pd.crosstab(old["difficulty"], old["split"], margins=True).to_string())

    # Stream the manifest -- it is ~570 MB, so do not materialise it.
    pool = {k: [] for k in want}
    root = Path(args.dataset)
    with open(root / "labels.jsonl") as f:
        for line in f:
            r = json.loads(line)
            d = r["difficulty"]
            if d in pool and int(r["seed"]) not in used:
                pool[d].append((int(r["id"]), int(r["seed"]), d))
    for d, rows in pool.items():
        print(f"  candidate {d}: {len(rows)} unused seeds available")
        if len(rows) < want[d]:
            raise SystemExit(
                f"asked for {want[d]} {d} rows but only {len(rows)} seeds are "
                f"unused. Lower the count.")

    rng = np.random.default_rng(args.rng_seed)
    new_rows = []
    for d, n in want.items():
        idx = rng.permutation(len(pool[d]))[:n]
        new_rows.extend((pool[d][i][0], pool[d][i][1], d, args.split)
                        for i in idx)
    print(f"  adding {len(new_rows)} rows, all marked '{args.split}': "
          f"{dict(Counter(r[2] for r in new_rows))}")

    add_df = pd.DataFrame(new_rows,
                          columns=["id", "seed", "difficulty", "split"])
    out_df = pd.concat([old[["id", "seed", "difficulty", "split"]], add_df],
                       ignore_index=True)

    spanning = int((out_df.groupby("seed")["split"].nunique() > 1).sum())
    dup_ids = int(out_df["id"].duplicated().sum())
    if spanning or dup_ids:
        raise SystemExit(f"REFUSING TO WRITE: {spanning} seeds span more than "
                         f"one split, {dup_ids} duplicate ids.")

    val_before = set(old.loc[old["split"] == "val", "seed"])
    val_after = set(out_df.loc[out_df["split"] == "val", "seed"])
    if val_before != val_after:
        raise SystemExit("REFUSING TO WRITE: the val set changed. Every "
                         "number measured against the old split would become "
                         "incomparable.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\n  wrote {len(out_df)} rows to {args.out}")
    print(pd.crosstab(out_df["difficulty"], out_df["split"],
                      margins=True).to_string())
    print(f"\n  seeds spanning >1 split: {spanning}   duplicate ids: {dup_ids}")
    print(f"  val set unchanged: {len(val_after)} seeds, identical to input")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
