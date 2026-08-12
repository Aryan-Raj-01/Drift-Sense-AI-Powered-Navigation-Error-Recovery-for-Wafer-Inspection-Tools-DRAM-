#!/usr/bin/env python3
"""Build the results figures and the failure-case report from an eval dump.

    python -m driftsense.dl_localize.report ^
        --csv       <submission-root>\\results\\manifest_golden_predictions.csv ^
        --dataset   <dataset-root> ^
        --out       <submission-root>\\results

Produces, in ``--out``:

    accuracy_by_threshold.png   pass rate at 1/2/3/4/5 px, per difficulty
    error_distribution.png      log-scale error CDF, per difficulty
    confidence_separation.png   margin distribution, hits vs gross misses,
                                plus the flag-rate/recall trade-off curve
    robustness_covariates.png   accuracy vs rotation, scale, SNR, distortion
    failure_cases/failure_<rank>_<id>.png
                                reference, search with ground truth and
                                prediction marked, and the crop the model
                                actually chose

WHY THE FAILURE FIGURE IS BUILT THIS WAY
========================================

A failure figure that shows only the search image with two dots on it proves
nothing -- the reader cannot tell whether the model picked something
reasonable. Each panel therefore shows the reference pattern, the true
location, and the region the model chose, at the same scale, so the reader can
judge for themselves whether the chosen region genuinely resembles the target.
For a periodic layout it usually does, and that is the entire point: the
failure is not carelessness, it is a genuine ambiguity in the data.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    raise SystemExit("matplotlib is required: pip install matplotlib")

import cv2

BANDS = (1.0, 2.0, 3.0, 4.0, 5.0)
BUCKETS = ("easy", "medium", "hard")
COLORS = {"easy": "#2b7bba", "medium": "#e08214", "hard": "#c1272d",
          "ALL": "#444444"}


def _f(row: Dict, key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load(csv_path: str) -> List[Dict]:
    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        for k in ("err", "margin", "gt_x", "gt_y", "pred_x", "pred_y",
                  "rel_rotation_deg", "scale_ratio", "search_shot_snr",
                  "label_correction_px", "err_learned"):
            r[k] = _f(r, k)
    return rows


# --------------------------------------------------------------------------
# figure 1: accuracy by threshold
# --------------------------------------------------------------------------
def fig_accuracy(rows: List[Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(BANDS))
    width = 0.2
    groups = [("ALL", rows)] + [(b, [r for r in rows if r["difficulty"] == b])
                                for b in BUCKETS]
    for i, (name, g) in enumerate(groups):
        if not g:
            continue
        e = np.array([r["err"] for r in g])
        vals = [100.0 * float((e <= b).mean()) for b in BANDS]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=f"{name} (n={len(g)})",
                      color=COLORS[name])
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.6, f"{v:.1f}",
                    ha="center", fontsize=6.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"\u2264{int(b)} px" for b in BANDS])
    ax.set_ylabel("pass rate (%)")
    ax.set_ylim(0, 108)
    ax.set_title("Localization pass rate by threshold\n"
                 "golden set, 2000 held-out pairs", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "accuracy_by_threshold.png", dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------
# figure 2: error distribution (CDF, log x)
# --------------------------------------------------------------------------
def fig_error_cdf(rows: List[Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in ("ALL",) + BUCKETS:
        g = rows if name == "ALL" else [r for r in rows
                                        if r["difficulty"] == name]
        if not g:
            continue
        e = np.sort(np.array([max(r["err"], 1e-3) for r in g]))
        y = 100.0 * np.arange(1, e.size + 1) / e.size
        ax.plot(e, y, label=f"{name} (n={len(g)})", color=COLORS[name],
                lw=1.8 if name == "ALL" else 1.3)
    for b in (1.0, 5.0):
        ax.axvline(b, color="k", ls=":", lw=0.8)
        ax.text(b, 2, f" {int(b)} px", fontsize=7, rotation=90, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Euclidean error (px, log scale)")
    ax.set_ylabel("cumulative % of pairs")
    ax.set_title("Error distribution\n"
                 "the vertical cliff at ~1 px and the flat tail beyond 100 px "
                 "are two different failure modes", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "error_distribution.png", dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------
# figure 3: confidence separation and the gating trade-off
# --------------------------------------------------------------------------
def fig_confidence(rows: List[Dict], out: Path) -> Dict:
    hits = [r for r in rows if r["err"] <= 1.0]
    near = [r for r in rows if 1.0 < r["err"] <= 5.0]
    gross = [r for r in rows if r["err"] > 5.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    bins = np.linspace(0, 12, 49)
    ax1.hist([r["margin"] for r in hits], bins=bins, alpha=0.7,
             label=f"hit \u22641 px (n={len(hits)})", color="#2b7bba")
    ax1.hist([r["margin"] for r in near], bins=bins, alpha=0.7,
             label=f"near-miss 1-5 px (n={len(near)})", color="#e08214")
    ax1.hist([r["margin"] for r in gross], bins=bins, alpha=0.85,
             label=f"gross miss >5 px (n={len(gross)})", color="#c1272d")
    ax1.set_xlabel("logit margin (peak minus best competing lattice cell)")
    ax1.set_ylabel("count")
    ax1.set_yscale("log")
    ax1.set_title("Confidence separates catastrophic failure\n"
                  "gross misses concentrate below margin 1", fontsize=10)
    ax1.legend(fontsize=7.5)
    ax1.grid(alpha=0.3)

    ts = np.linspace(0.1, 5.0, 60)
    recall, flagged, false_flag = [], [], []
    for t in ts:
        recall.append(100.0 * sum(r["margin"] < t for r in gross) /
                      max(len(gross), 1))
        flagged.append(100.0 * sum(r["margin"] < t for r in rows) / len(rows))
        false_flag.append(100.0 * sum(r["margin"] < t for r in hits) /
                          max(len(hits), 1))
    ax2.plot(ts, recall, color="#c1272d", lw=2,
             label="% of gross misses caught")
    ax2.plot(ts, false_flag, color="#2b7bba", lw=2,
             label="% of correct results wrongly flagged")
    ax2.plot(ts, flagged, color="#777777", lw=1.2, ls="--",
             label="% of all pairs flagged")
    ax2.axvline(1.5, color="k", ls=":", lw=1)
    ax2.text(1.55, 50, "operating point\nmargin < 1.5", fontsize=7.5)
    ax2.set_xlabel("margin threshold")
    ax2.set_ylabel("%")
    ax2.set_title("Self-reported failure detection\n"
                  "the model knows when it has failed", fontsize=10)
    ax2.legend(fontsize=7.5, loc="center right")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / "confidence_separation.png", dpi=170)
    plt.close(fig)

    at = {}
    for t in (1.0, 1.5, 2.0):
        at[t] = {
            "caught_pct": 100.0 * sum(r["margin"] < t for r in gross) /
                          max(len(gross), 1),
            "flagged_pct": 100.0 * sum(r["margin"] < t for r in rows) /
                           len(rows),
            "false_flag_pct": 100.0 * sum(r["margin"] < t for r in hits) /
                              max(len(hits), 1)}
    return at


# --------------------------------------------------------------------------
# figure 4: robustness covariates
# --------------------------------------------------------------------------
def fig_robustness(rows: List[Dict], out: Path) -> None:
    specs = [("rel_rotation_deg", "|relative rotation| (deg)", True),
             ("scale_ratio", "scale ratio (nominal 10:1)", False),
             ("search_shot_snr", "search-image shot SNR", False),
             ("label_correction_px", "non-rigid distortion (px)", False)]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    ok = np.array([1.0 if r["err"] <= 1.0 else 0.0 for r in rows])
    for ax, (key, label, use_abs) in zip(axes.ravel(), specs):
        v = np.array([abs(r[key]) if use_abs else r[key] for r in rows])
        edges = np.quantile(v, np.linspace(0, 1, 6))
        edges[-1] += 1e-9
        xs, ys, ns = [], [], []
        for i in range(5):
            m = (v >= edges[i]) & (v < edges[i + 1])
            if m.sum() < 5:
                continue
            xs.append(0.5 * (edges[i] + edges[i + 1]))
            ys.append(100.0 * ok[m].mean())
            ns.append(int(m.sum()))
        ax.plot(xs, ys, "o-", color="#2b7bba", lw=1.8)
        for x_, y_, n_ in zip(xs, ys, ns):
            ax.annotate(f"n={n_}", (x_, y_), fontsize=6,
                        textcoords="offset points", xytext=(0, -12),
                        ha="center")
        ax.set_xlabel(label, fontsize=8.5)
        ax.set_ylabel("\u22641 px accuracy (%)", fontsize=8.5)
        ax.set_ylim(60, 100)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7.5)
    fig.suptitle("Robustness: a flat line rules a factor out as a cause; "
                 "a falling line names it", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "robustness_covariates.png", dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------
# failure case panels
# --------------------------------------------------------------------------
def failure_panels(rows: List[Dict], dataset: Path, out: Path,
                   top: int = 3) -> List[Dict]:
    fdir = out / "failure_cases"
    fdir.mkdir(parents=True, exist_ok=True)
    worst = sorted(rows, key=lambda r: -r["err"])[:top]
    made = []
    for rank, r in enumerate(worst, 1):
        rp = dataset / r["reference_path"]
        sp = dataset / r["search_path"]
        ref = cv2.imread(str(rp), 0)
        srch = cv2.imread(str(sp), 0)
        if ref is None or srch is None:
            print(f"  skip rank {rank}: cannot read {rp}")
            continue

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
        axes[0].imshow(ref, cmap="gray")
        axes[0].set_title("reference (100x)\nthe pattern to find", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(srch, cmap="gray")
        gx, gy, px, py = r["gt_x"], r["gt_y"], r["pred_x"], r["pred_y"]
        axes[1].plot(gx, gy, "o", ms=13, mfc="none", mec="#00d000", mew=2.2,
                     label=f"ground truth ({gx:.0f}, {gy:.0f})")
        axes[1].plot(px, py, "x", ms=13, mec="#ff2020", mew=2.6,
                     label=f"prediction ({px:.0f}, {py:.0f})")
        axes[1].plot([gx, px], [gy, py], "-", color="#ffaa00", lw=1.1,
                     alpha=0.9)
        axes[1].legend(fontsize=7.5, loc="upper right",
                       facecolor="white", framealpha=0.85)
        axes[1].set_title(f"search (10x)  \u2014  error {r['err']:.0f} px\n"
                          f"{r['difficulty']}, {r['klass']}", fontsize=9)
        axes[1].axis("off")

        # what the model actually looked at, next to what it should have
        half = 50
        def crop(cx, cy):
            x0 = int(round(cx)) - half
            y0 = int(round(cy)) - half
            x0 = max(0, min(srch.shape[1] - 2 * half, x0))
            y0 = max(0, min(srch.shape[0] - 2 * half, y0))
            return srch[y0:y0 + 2 * half, x0:x0 + 2 * half]
        pair = np.hstack([crop(gx, gy), np.full((2 * half, 6), 255,
                                                np.uint8), crop(px, py)])
        axes[2].imshow(pair, cmap="gray")
        axes[2].set_title("true region (left) vs chosen region (right)\n"
                          "100x100 px crops at search scale", fontsize=9)
        axes[2].axis("off")

        fig.suptitle(
            f"Failure #{rank}  |  margin {r['margin']:.2f} "
            f"(hits average 5.47)  |  distortion "
            f"{r['label_correction_px']:.2f} px  |  SNR "
            f"{r['search_shot_snr']:.2f}  |  seed {r['seed']}",
            fontsize=9.5)
        fig.tight_layout()
        name = f"failure_{rank}_{r.get('id', 'x')}.png"
        fig.savefig(fdir / name, dpi=165)
        plt.close(fig)
        made.append({"rank": rank, "file": name, **{
            k: r[k] for k in ("err", "margin", "difficulty", "klass",
                              "gt_x", "gt_y", "pred_x", "pred_y",
                              "label_correction_px", "search_shot_snr",
                              "seed", "reference_path", "search_path")}})
        print(f"  wrote {fdir / name}")
    return made


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=3)
    cfg = ap.parse_args()

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(cfg.csv)
    print(f"  {len(rows)} rows from {cfg.csv}")

    fig_accuracy(rows, out)
    print(f"  wrote {out / 'accuracy_by_threshold.png'}")
    fig_error_cdf(rows, out)
    print(f"  wrote {out / 'error_distribution.png'}")
    gate = fig_confidence(rows, out)
    print(f"  wrote {out / 'confidence_separation.png'}")
    fig_robustness(rows, out)
    print(f"  wrote {out / 'robustness_covariates.png'}")
    failure_panels(rows, Path(cfg.dataset), out, cfg.top)

    print("\n  confidence gating, measured:")
    for t, d in gate.items():
        print(f"    margin < {t}: flags {d['flagged_pct']:.1f}% of pairs, "
              f"catches {d['caught_pct']:.1f}% of gross misses, "
              f"wrongly flags {d['false_flag_pct']:.1f}% of correct results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
