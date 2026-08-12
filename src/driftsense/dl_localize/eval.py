#!/usr/bin/env python3
"""Evaluate the fixed model -- and diagnose its failures.

    python -m driftsense.dl_localize.eval ^
        --dataset <dataset-root> ^
        --checkpoint <run-dir>/phase3_all.pt ^
        --split-csv <dataset-root>/subset_split.csv --split val ^
        --pairs 2000 --out <run-dir>/eval_val.json

WHAT THIS ADDS OVER ``tools/eval_dl.py``
========================================

1. THE ACCURACY BANDS THE CHALLENGE ACTUALLY ASKS FOR.  Their slide says
   "publish confusion matrix for 1px-5px accuracy".  The old script reported
   only <=1 px, which understates a model whose failures are near-misses and
   overstates nothing.  1/2/3/5 px are all reported.

2. THE FAILURE DIAGNOSTIC THAT WAS NEVER BUILT.  Section 12 of the project
   context lists "diagnose why easy is stuck at 84%" as the top unfinished
   item.  ``--diagnose`` classifies every failure as:

     * periodic-lock  -- the error is close to an integer number of lattice
       pitches.  This is the confusion the whole design is aimed at, and it
       is the one the lattice margin term should reduce.
     * near-miss      -- 1 px < error <= 5 px.  Right region, bad sub-pixel.
       Fixed by the readout, not by more data.
     * gross-miss     -- everything else.  Wrong region entirely.

   and then reports the failure rate against |rel_rotation_deg|,
   scale_ratio, lattice pitch, dose and SNR.  A monotone trend in rotation
   or scale means the bank is too narrow.  A flat trend across all of them
   means the signal genuinely is not there, and more data will not help.

3. CLASSICAL COMPARISON IS OPT-IN.  ``propose()`` costs ~1000 ms/pair; on
   2000 pairs that is over half an hour of wall clock spent re-measuring a
   number that has not changed.  ``--classical`` when you want it.

4. READOUT A/B.  All three sub-pixel readouts are scored on the same
   predictions, so the choice between the offset head and the soft-argmax is
   made on measured numbers rather than on an argument.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from driftsense.dl_localize.coords import STRIDE
from driftsense.dl_localize.data import load_split_seeds
from driftsense.dl_localize.infer import LearnedLocalizer

BANDS = (1.0, 2.0, 3.0, 4.0, 5.0)


def _resolve(root: Path, path: str) -> str:
    return path if os.path.isabs(path) else str(root / path)


def classify_failure(err: float, dx: float, dy: float,
                     pitch_x_px: float, pitch_y_px: float,
                     tol: float = 1.5) -> str:
    """Label one prediction.

    Args:
        err: Euclidean error, pixels.
        dx: Signed x error, pixels.
        dy: Signed y error.
        pitch_x_px: Lattice pitch along x, search pixels (0 = unknown).
        pitch_y_px: Lattice pitch along y.
        tol: How close to an exact pitch multiple counts as a lock.

    Returns:
        ``"hit"``, ``"near-miss"``, ``"periodic-lock"`` or ``"gross-miss"``.
    """
    if err <= 1.0:
        return "hit"
    if pitch_x_px > 0 and pitch_y_px > 0:
        kx = dx / pitch_x_px
        ky = dy / pitch_y_px
        on_x = abs(kx - round(kx)) * pitch_x_px <= tol
        on_y = abs(ky - round(ky)) * pitch_y_px <= tol
        if on_x and on_y and (abs(round(kx)) >= 1 or abs(round(ky)) >= 1):
            return "periodic-lock"
    if err <= 5.0:
        return "near-miss"
    return "gross-miss"


def _bucket_report(errs: np.ndarray, name: str) -> Dict[str, float]:
    # `mean` and `max` are reported because the challenge's validation
    # requirements ask for "mean, median and worst-case error" explicitly.
    # Note that on a bucket with gross misses the mean is dominated by the
    # tail and is NOT a useful central estimate -- the median is. Both are
    # printed so a reader can see that divergence rather than infer it.
    return {"n": int(errs.size),
            **{f"acc{int(b)}px": 100.0 * float((errs <= b).mean())
               for b in BANDS},
            "mean": float(errs.mean()),
            "median": float(np.median(errs)),
            "p90": float(np.percentile(errs, 90)),
            "max": float(errs.max())}


def _trend(rows: List[Dict], key: str, ok: np.ndarray, n_bins: int = 5
           ) -> List[Dict]:
    """Failure rate as a function of one metadata field, in quantile bins."""
    vals = np.array([abs(float(r.get(key, 0.0) or 0.0)) for r in rows])
    if float(vals.std()) < 1e-9:
        return []
    edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    out = []
    for i in range(n_bins):
        m = (vals >= edges[i]) & (vals < edges[i + 1])
        if m.sum() < 5:
            continue
        out.append({"lo": float(edges[i]), "hi": float(edges[i + 1]),
                    "n": int(m.sum()),
                    "acc1px": 100.0 * float(ok[m].mean())})
    return out


def run(cfg: argparse.Namespace) -> Dict:
    root = Path(cfg.dataset)
    rows = [json.loads(l) for l in open(root / "labels.jsonl")]

    if cfg.split_csv:
        wanted = load_split_seeds(cfg.split_csv, cfg.split)
        rows = [r for r in rows if int(r["seed"]) in wanted]
        print(f"  split '{cfg.split}': {len(rows)} candidates")
    else:
        print("  WARNING: no --split-csv. These rows may include seeds the "
              "model was trained on; the resulting number is not a "
              "generalisation estimate.")

    idx = np.random.default_rng(1).permutation(len(rows))[:cfg.pairs]
    rows = [rows[i] for i in idx]

    print(f"  loading {cfg.checkpoint}")
    parse = lambda s: tuple(float(v) for v in s.split(",")) if s else None
    loc = LearnedLocalizer(cfg.checkpoint, device=cfg.device,
                             use_ema=not cfg.no_ema, readout=cfg.readout,
                             angle_bank=parse(cfg.angle_bank),
                             scale_bank=parse(cfg.scale_bank),
                             tie_margin=cfg.tie_margin,
                             refine=cfg.refine,
                             refine_radius=cfg.refine_radius)
    print(f"  bank: {len(loc.bank)} members "
          f"(angles {loc.angle_bank}, scales {loc.scale_bank})")
    print(f"  local refinement: "
          + (f"ON (+/-{loc.refine_radius}px, {len(loc.refine_angles)} angles)"
             if loc.refine else "OFF"))

    classical = None
    if cfg.classical:
        from driftsense.localize.propose import propose
        classical = propose

    recs: List[Dict] = []
    t_learned = t_classical = 0.0
    for n, r in enumerate(rows, 1):
        a = cv2.imread(_resolve(root, r["reference_path"]), 0)
        b = cv2.imread(_resolve(root, r["search_path"]), 0)
        if a is None or b is None:
            continue
        gx, gy = float(r["gt_x"]), float(r["gt_y"])
        px_nm = float(r.get("search_px_nm", 0.0)) or 1.0
        pitch_x = float(r.get("layout_bl_pitch", 0.0)) / px_nm
        pitch_y = float(r.get("layout_wl_pitch", 0.0)) / px_nm

        t0 = time.perf_counter()
        out = loc.locate(a, b, pitch_x_px=pitch_x if cfg.use_pitch else 0.0,
                         pitch_y_px=pitch_y if cfg.use_pitch else 0.0)
        t_learned += time.perf_counter() - t0

        dx, dy = out["x"] - gx, out["y"] - gy
        err = math.hypot(dx, dy)
        rec = dict(r)
        # pred_x/pred_y are written out explicitly rather than left implicit in
        # dx/dy: the challenge's manifest requirement asks for ground-truth AND
        # predicted coordinates side by side, and a reader should not have to
        # reconstruct one from the other.
        rec.update({"pred_x": out["x"], "pred_y": out["y"],
                    "err": err, "dx": dx, "dy": dy,
                    "pitch_x_px": pitch_x, "pitch_y_px": pitch_y,
                    "confidence": out["confidence"], "margin": out["margin"],
                    "sel_angle": out["angle"], "sel_scale": out["scale"],
                    "tie_break": out["tie_break"],
                    "klass": classify_failure(err, dx, dy, pitch_x, pitch_y)})
        # Refinement diagnostics. `err_learned` is the error BEFORE the
        # classical stage, so err_learned vs err says whether refinement
        # helped or hurt that sample, and `shift` saturating at the radius
        # says the window was too small to reach the truth.
        if "x_learned" in out:
            el = math.hypot(out["x_learned"] - gx, out["y_learned"] - gy)
            rec.update({"x_learned": out["x_learned"],
                        "y_learned": out["y_learned"],
                        "err_learned": el, "ncc": out["ncc"],
                        "shift": out["shift"],
                        "refine_gain": el - err})

        if classical is not None:
            t0 = time.perf_counter()
            cands = classical(a.astype(np.float64), b.astype(np.float64),
                              top_k=1)
            t_classical += time.perf_counter() - t0
            rec["err_classical"] = (
                math.hypot(cands[0].x - gx, cands[0].y - gy)
                if cands else 1e6)
        recs.append(rec)
        if n % 25 == 0 or n == len(rows):
            print(f"\r  {n}/{len(rows)}", end="", flush=True)
    print()

    if not recs:
        raise SystemExit("no rows evaluated -- check the dataset path")

    err = np.array([r["err"] for r in recs])
    diff = np.array([r["difficulty"] for r in recs])
    results: Dict[str, Dict] = {}

    hdr = (f"\n  {'group':<8}{'n':>6}" +
           "".join(f"{'<=' + str(int(b)) + 'px':>9}" for b in BANDS) +
           f"{'mean':>9}{'median':>9}{'p90':>9}{'worst':>10}")
    if classical is not None:
        hdr += f"{'clas<=1':>9}"
    print(hdr)
    for g in ("ALL", "easy", "medium", "hard"):
        m = np.ones(len(diff), bool) if g == "ALL" else diff == g
        if not m.any():
            continue
        rep = _bucket_report(err[m], g)
        line = (f"  {g:<8}{rep['n']:>6}" +
                "".join(f"{rep['acc' + str(int(b)) + 'px']:>8.1f}%"
                        for b in BANDS) +
                f"{rep['mean']:>9.2f}{rep['median']:>9.2f}"
                f"{rep['p90']:>9.2f}{rep['max']:>10.2f}")
        if classical is not None:
            ec = np.array([r["err_classical"] for r in recs])
            rep["classical_acc1px"] = 100.0 * float((ec[m] <= 1.0).mean())
            line += f"{rep['classical_acc1px']:>8.1f}%"
        print(line)
        results[g] = rep

    # ---- failure taxonomy -------------------------------------------------
    print(f"\n  {'group':<8}{'hit':>8}{'near-miss':>11}"
          f"{'periodic-lock':>15}{'gross-miss':>12}")
    for g in ("ALL", "easy", "medium", "hard"):
        m = np.ones(len(diff), bool) if g == "ALL" else diff == g
        if not m.any():
            continue
        sub = [r["klass"] for r, keep in zip(recs, m) if keep]
        c = {k: 100.0 * sub.count(k) / len(sub)
             for k in ("hit", "near-miss", "periodic-lock", "gross-miss")}
        print(f"  {g:<8}{c['hit']:>7.1f}%{c['near-miss']:>10.1f}%"
              f"{c['periodic-lock']:>14.1f}%{c['gross-miss']:>11.1f}%")
        results.setdefault(g, {})["failure_mix"] = c

    print("\n  How to read the taxonomy:")
    print("    near-miss dominant     -> readout/sub-pixel problem. More "
          "data will not help; fix the offset head or drop the stride.")
    print("    periodic-lock dominant -> the lattice margin term is the "
          "lever. Raise --lattice-weight and re-train.")
    print("    gross-miss dominant    -> the signal is not being found at "
          "all. This is the only case where more data is the right answer.")

    # ---- covariate trends -------------------------------------------------
    if cfg.diagnose:
        ok = (err <= 1.0).astype(float)
        print("\n  <=1px accuracy vs metadata (a monotone trend names the "
              "cause; a flat one rules it out)")
        for key, label in (("rel_rotation_deg", "|rel_rotation| deg"),
                           ("scale_ratio", "scale_ratio"),
                           ("layout_bl_pitch", "bitline pitch nm"),
                           ("search_dose_e_per_px", "search dose e/px"),
                           ("search_shot_snr", "search shot SNR"),
                           ("landmark_size_nm", "landmark size nm"),
                           ("label_correction_px", "label correction px")):
            tr = _trend(recs, key, ok)
            if not tr:
                continue
            cells = "  ".join(f"[{t['lo']:.2f},{t['hi']:.2f}) "
                              f"{t['acc1px']:.0f}%(n={t['n']})" for t in tr)
            print(f"    {label:<22} {cells}")
            results.setdefault("trends", {})[key] = tr

        sel = defaultdict(int)
        for r in recs:
            sel[(r["sel_angle"], r["sel_scale"])] += 1
        print("\n  bank members selected (edge-heavy selection means the "
              "bank is too narrow -- widen it and re-run):")
        for (a_, s_), c in sorted(sel.items(), key=lambda kv: -kv[1]):
            print(f"    angle {a_:+5.1f}  scale {s_:5.2f}  "
                  f"{100.0 * c / len(recs):5.1f}%")
        results["bank_selection"] = {f"{a_}|{s_}": c
                                     for (a_, s_), c in sel.items()}

        conf_hit = np.array([r["margin"] for r in recs if r["err"] <= 1.0])
        conf_bad = np.array([r["margin"] for r in recs if r["err"] > 1.0])
        if conf_hit.size and conf_bad.size:
            print(f"\n  logit margin: hits {conf_hit.mean():.2f} +/- "
                  f"{conf_hit.std():.2f}   failures {conf_bad.mean():.2f} "
                  f"+/- {conf_bad.std():.2f}")
            print("    (a real separation here means `margin` is a usable "
                  "confidence signal and can gate a classical fallback)")
            results["margin_hits"] = float(conf_hit.mean())
            results["margin_failures"] = float(conf_bad.mean())

    # ---- did the classical refinement help or hurt, and where? -----------
    if cfg.refine and cfg.diagnose and "err_learned" in recs[0]:
        el = np.array([r["err_learned"] for r in recs])
        sh = np.array([r["shift"] for r in recs])
        near = (err > 1.0) & (err <= 5.0)
        print(f"\n  refinement audit ({int(cfg.refine_radius)}px window)")
        print(f"    median err  learned {np.median(el):.3f} -> refined "
              f"{np.median(err):.3f} px")
        print(f"    helped {100 * (err < el).mean():.0f}% of samples, "
              f"hurt {100 * (err > el).mean():.0f}%")
        if near.any():
            print(f"    among the {int(near.sum())} near-misses (1-5px): "
                  f"refinement hurt {100 * (err[near] > el[near]).mean():.0f}%"
                  f", mean shift {sh[near].mean():.2f}px, "
                  f"{100 * (sh[near] >= cfg.refine_radius - 0.05).mean():.0f}%"
                  f" pinned at the window edge")
            print("      >50% hurt  -> refinement is chasing the rigid match; "
                  "gate it on label distortion or margin")
            print("      >20% pinned -> window too small; raise "
                  "--refine-radius (stay under half the min pitch, 5.3px)")
        results["refine_median_before"] = float(np.median(el))
        results["refine_median_after"] = float(np.median(err))

    ms = 1000.0 * t_learned / len(recs)
    print(f"\n  timing: learned {ms:.0f} ms/pair"
          + (f"   classical {1000 * t_classical / len(recs):.0f} ms/pair"
             if classical is not None else ""))
    results["ms_per_pair"] = ms

    if cfg.dump_csv:
        import csv
        keys = ["id", "seed", "difficulty",
                "reference_path", "search_path",
                "gt_x", "gt_y", "pred_x", "pred_y",
                "err", "err_learned",
                "refine_gain", "shift", "ncc", "dx", "dy", "klass",
                "margin", "confidence", "sel_angle", "sel_scale", "tie_break",
                "rel_rotation_deg", "scale_ratio", "layout_bl_pitch",
                "layout_wl_pitch", "search_dose_e_per_px", "search_shot_snr",
                "landmark_size_nm", "defect_size_nm", "label_correction_px"]
        Path(cfg.dump_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.dump_csv, "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            wcsv.writeheader()
            for r in recs:
                wcsv.writerow(r)
        print(f"  per-sample results -> {cfg.dump_csv}")
        print("    Sort by err descending and open the image pairs at the "
              "top. Twenty minutes of looking at the worst twenty is worth "
              "more than another training run.")
    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pairs", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--readout", default="offset",
                    choices=["offset", "softargmax", "both"])
    ap.add_argument("--angle-bank", default=None,
                    help="override the checkpoint's inference rotation bank, "
                         "e.g. '-2,0,2' or '0'. MEASURED on the 10k run: the "
                         "bank helps easy (83->85%%) but hurts medium "
                         "(86->82%%) and hard (47->34%%), because selection "
                         "is by max peak logit and a flat correlation "
                         "surface lets a wrong member win with a spurious "
                         "peak. Narrow it for the weak-peak buckets.")
    ap.add_argument("--scale-bank", default=None,
                    help="override the scale bank, e.g. '10.0'. The measured "
                         "<=1px-vs-scale_ratio trend is FLAT, so scale "
                         "variation is already absorbed by augmentation and "
                         "extra scale members contribute selection noise "
                         "without contributing signal.")
    ap.add_argument("--tie-margin", type=float, default=0.0,
                    help="logit gap under which the problem statement's "
                         "closest-to-centre rule decides between competing "
                         "matches. 0 disables it. Calibrate against the "
                         "hits-vs-failures margin distribution this script "
                         "reports: on the final model, gross misses sit at "
                         "median margin 0.35 (p90 0.94) while hits sit at "
                         "5.70 (p10 2.69), so a value near 1.0-1.5 fires "
                         "almost exclusively on genuine failures. Tune on "
                         "val, confirm once on golden.")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--classical", action="store_true",
                    help="also run the classical baseline (~1 s/pair)")
    ap.add_argument("--diagnose", action="store_true", default=True)
    ap.add_argument("--no-diagnose", dest="diagnose", action="store_false")
    ap.add_argument("--refine", action="store_true", default=True,
                    help="local stride-1 multi-angle NCC refinement of the "
                         "learned prediction. On by default: the measured "
                         "failure profile is dominated by the 1-5px band "
                         "(14.9%% of all samples), which is a stride-4 "
                         "quantisation artefact, not a localisation failure")
    ap.add_argument("--no-refine", dest="refine", action="store_false")
    ap.add_argument("--refine-radius", type=int, default=3,
                    help="must stay below half the minimum lattice pitch "
                         "(5.3px measured) so a periodic replica can never "
                         "enter the refinement window")
    ap.add_argument("--use-pitch", action="store_true",
                    help="feed the manifest's lattice pitch to the readout. "
                         "Applied Materials' test set will NOT have it, so "
                         "any gain measured this way is not bankable")
    ap.add_argument("--split-csv", default=None)
    ap.add_argument("--split", default=None,
                    choices=["train", "val", "golden"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump-csv", default=None)
    cfg = ap.parse_args()

    if cfg.split_csv and not cfg.split:
        raise SystemExit("--split-csv given without --split")

    res = run(cfg)
    if cfg.out:
        Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
        Path(cfg.out).write_text(json.dumps(res, indent=2, default=float))
        print(f"  written to {cfg.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
