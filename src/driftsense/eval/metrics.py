"""Localisation metrics for Drift-Sense.

The score the hackathon cares about is the distance in search pixels between the
predicted centre and the true one.  Everything here derives from that, but the
*aggregation* is where the useful information is:

**Report the median and a hit rate, not the mean.**  A matcher that locks onto
the wrong periodic repeat is off by one lattice pitch (10-20 px) or by hundreds
of pixels.  Those failures dominate a mean and hide the fact that the successful
cases are sub-pixel.  ``median`` plus ``hit@5px`` says "usually exact, sometimes
catastrophically wrong", which is the truth and is actionable.

**Stratify by difficulty.**  An aggregate number over a mixed dataset is
uninterpretable because you can move it by changing the difficulty mix rather
than by improving the algorithm.  The ``hard`` row is the one the hackathon is
actually testing: highly periodic regions where classical template matching
breaks down.

**Stratify by dose.**  The hidden test set is stated to be noisier than
anything you generate.  An accuracy-versus-SNR curve extrapolates; a single
number does not.  :func:`by_dose` produces that curve.

**Count the one-pitch failures separately.**  A prediction that is off by
exactly one line pitch is a different bug from one that is off by 400 px: the
first means the matcher found the right structure and the wrong instance, the
second means it found nothing.  :func:`failure_modes` separates them, which is
the "failure mode awareness" the brief asks for.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

#: Distance thresholds reported as hit rates, in search pixels.
THRESHOLDS_PX: Tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 25.0)

#: A prediction within this many pixels of an integer number of lattice pitches
#: counts as a "periodic lock" rather than a random miss.
PITCH_TOLERANCE_PX = 3.0


@dataclass
class Prediction:
    """One algorithm output, paired with its ground truth.

    Attributes:
        id: Sample id.
        pred_x: Predicted centre column, search pixels.
        pred_y: Predicted centre row.
        gt_x: True centre column.
        gt_y: True centre row.
        difficulty: ``"easy"``, ``"medium"`` or ``"hard"``.
        style: ``"dram"`` or ``"finfet"``.
        dose: Search-frame dose, e/px, for SNR stratification.
        pitch_px: Dominant layout pitch in search pixels, for periodic-lock
            detection.  Optional.
        score: Optional confidence, for risk-coverage analysis.
        seconds: Optional wall time.
    """

    id: int
    pred_x: float
    pred_y: float
    gt_x: float
    gt_y: float
    difficulty: str = "unknown"
    style: str = "unknown"
    dose: float = float("nan")
    pitch_px: float = float("nan")
    score: float = float("nan")
    seconds: float = float("nan")

    @property
    def error_px(self) -> float:
        """Euclidean distance between prediction and truth, in search pixels."""
        return math.hypot(self.pred_x - self.gt_x, self.pred_y - self.gt_y)


def summarise(preds: Sequence[Prediction]) -> Dict[str, Any]:
    """Aggregate metrics over a set of predictions.

    Args:
        preds: Predictions to score.

    Returns:
        A dictionary with ``n``, ``median_px``, ``mean_px``, ``p90_px``,
        ``p99_px`` and a ``hit`` sub-dictionary keyed by threshold.
    """
    if not preds:
        return {"n": 0}
    err = np.array([p.error_px for p in preds], dtype=np.float64)
    out: Dict[str, Any] = {
        "n": int(err.size),
        "median_px": float(np.median(err)),
        "mean_px": float(err.mean()),
        "p90_px": float(np.percentile(err, 90)),
        "p99_px": float(np.percentile(err, 99)),
        "max_px": float(err.max()),
    }
    out["hit"] = {f"{t:g}px": float((err <= t).mean()) for t in THRESHOLDS_PX}
    times = np.array([p.seconds for p in preds], dtype=np.float64)
    if np.isfinite(times).any():
        out["seconds_per_sample"] = float(np.nanmedian(times))
    return out


def by_group(preds: Sequence[Prediction], key: str) -> Dict[str, Dict[str, Any]]:
    """Metrics split by an attribute of :class:`Prediction`.

    Args:
        preds: Predictions to score.
        key: Attribute name, e.g. ``"difficulty"`` or ``"style"``.

    Returns:
        Mapping from group value to its metric dictionary.
    """
    groups: Dict[str, List[Prediction]] = {}
    for p in preds:
        groups.setdefault(str(getattr(p, key)), []).append(p)
    return {k: summarise(v) for k, v in sorted(groups.items())}


def by_dose(preds: Sequence[Prediction], bins: int = 4) -> List[Dict[str, Any]]:
    """Accuracy as a function of search-frame dose.

    Bins are log-spaced because dose is log-distributed by design; linear bins
    would put almost every sample in the lowest bin and report one meaningful
    number instead of four.

    Args:
        preds: Predictions carrying a finite ``dose``.
        bins: Number of dose bins.

    Returns:
        One dictionary per bin, each with ``dose_lo``, ``dose_hi`` and the
        metrics for that bin.
    """
    usable = [p for p in preds if np.isfinite(p.dose) and p.dose > 0]
    if not usable:
        return []
    dose = np.array([p.dose for p in usable])
    edges = np.exp(np.linspace(np.log(dose.min()), np.log(dose.max() + 1e-9),
                               bins + 1))
    out: List[Dict[str, Any]] = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = [p for p in usable if lo <= p.dose <= hi]
        if not sel:
            continue
        row = {"dose_lo": float(lo), "dose_hi": float(hi)}
        row.update(summarise(sel))
        out.append(row)
    return out


def failure_modes(preds: Sequence[Prediction],
                  hit_px: float = 5.0) -> Dict[str, Any]:
    """Classify failures into periodic locks and outright misses.

    A "periodic lock" is a prediction whose error is close to an integer
    multiple of the layout pitch: the algorithm found the right *structure* and
    the wrong *instance*.  That is the characteristic failure of correlation on
    a repeating array, and it calls for a different fix (wider context, unique
    feature detection) than a random miss (better features).

    Args:
        preds: Predictions; those with a finite ``pitch_px`` participate in the
            periodic classification.
        hit_px: Success threshold.

    Returns:
        Counts and fractions for ``hit``, ``periodic_lock`` and ``miss``.
    """
    hit = lock = miss = 0
    lock_multiples: List[float] = []
    for p in preds:
        e = p.error_px
        if e <= hit_px:
            hit += 1
            continue
        if np.isfinite(p.pitch_px) and p.pitch_px > 1e-6:
            k = e / p.pitch_px
            if abs(k - round(k)) * p.pitch_px <= PITCH_TOLERANCE_PX and k < 40:
                lock += 1
                lock_multiples.append(k)
                continue
        miss += 1
    n = max(1, len(preds))
    return {
        "n": len(preds),
        "hit": hit, "hit_frac": hit / n,
        "periodic_lock": lock, "periodic_lock_frac": lock / n,
        "miss": miss, "miss_frac": miss / n,
        "median_lock_multiple": (float(np.median(lock_multiples))
                                 if lock_multiples else None),
    }


def report(preds: Sequence[Prediction], title: str = "results") -> str:
    """Format a human-readable report.

    Args:
        preds: Predictions to score.
        title: Heading.

    Returns:
        A multi-line string ready to print or paste into a slide.
    """
    lines: List[str] = [f"\n{title}  (n={len(preds)})", "=" * 66]
    header = (f"{'group':>12} {'n':>5} {'median':>9} {'p90':>9} "
              f"{'<=1px':>7} {'<=5px':>7} {'<=25px':>7}")

    def row(name: str, m: Dict[str, Any]) -> str:
        if not m.get("n"):
            return f"{name:>12}     0"
        return (f"{name:>12} {m['n']:5d} {m['median_px']:9.2f} {m['p90_px']:9.2f} "
                f"{100*m['hit']['1px']:6.0f}% {100*m['hit']['5px']:6.0f}% "
                f"{100*m['hit']['25px']:6.0f}%")

    lines.append(header)
    lines.append("-" * len(header))
    lines.append(row("ALL", summarise(preds)))
    lines.append("")
    for group_key in ("difficulty", "style"):
        for name, m in by_group(preds, group_key).items():
            if name != "unknown":
                lines.append(row(name, m))
        lines.append("")

    doses = by_dose(preds)
    if doses:
        lines.append(f"{'dose e/px':>12} {'n':>5} {'median':>9} {'p90':>9} "
                     f"{'<=5px':>7}")
        lines.append("-" * 46)
        for d in doses:
            lines.append(f"{d['dose_lo']:5.0f}-{d['dose_hi']:<6.0f} {d['n']:5d} "
                         f"{d['median_px']:9.2f} {d['p90_px']:9.2f} "
                         f"{100*d['hit']['5px']:6.0f}%")
        lines.append("")

    fm = failure_modes(preds)
    lines.append(f"failure modes: {100*fm['hit_frac']:.0f}% hit, "
                 f"{100*fm['periodic_lock_frac']:.0f}% periodic lock, "
                 f"{100*fm['miss_frac']:.0f}% miss")
    if fm["median_lock_multiple"]:
        lines.append(f"  typical lock is {fm['median_lock_multiple']:.1f} "
                     f"lattice pitches away -- right structure, wrong instance")
    return "\n".join(lines)


def save(preds: Sequence[Prediction], path: Union[str, Path],
         extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write predictions and metrics to JSON.

    Args:
        preds: Predictions to save.
        path: Output file.
        extra: Additional metadata to embed (algorithm name, config hash).

    Returns:
        The path written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summarise(preds),
        "by_difficulty": by_group(preds, "difficulty"),
        "by_style": by_group(preds, "style"),
        "by_dose": by_dose(preds),
        "failure_modes": failure_modes(preds),
        "predictions": [
            {"id": q.id, "pred_x": q.pred_x, "pred_y": q.pred_y,
             "gt_x": q.gt_x, "gt_y": q.gt_y, "error_px": q.error_px,
             "difficulty": q.difficulty, "style": q.style, "dose": q.dose,
             "score": q.score}
            for q in preds],
    }
    if extra:
        payload["meta"] = extra
    p.write_text(json.dumps(payload, indent=2, default=float))
    return p


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.eval.metrics``."""
    import tempfile

    rng = np.random.default_rng(0)

    # 1. a perfect algorithm
    perfect = [Prediction(i, 100.0, 100.0, 100.0, 100.0, "easy", "dram", 50.0)
               for i in range(20)]
    m = summarise(perfect)
    assert m["median_px"] == 0.0 and m["hit"]["1px"] == 1.0

    # 2. why the mean is the wrong statistic: 90 % sub-pixel, 10 % catastrophic
    mixed: List[Prediction] = []
    for i in range(90):
        mixed.append(Prediction(i, 100.4, 100.0, 100.0, 100.0, "easy", "dram", 60.0,
                                pitch_px=14.0))
    for i in range(10):
        mixed.append(Prediction(90 + i, 500.0, 100.0, 100.0, 100.0, "hard",
                                "dram", 20.0, pitch_px=14.0))
    m = summarise(mixed)
    assert m["median_px"] < 1.0 and m["mean_px"] > 30.0
    assert abs(m["hit"]["5px"] - 0.9) < 1e-9

    # 3. stratification separates what the aggregate hides
    g = by_group(mixed, "difficulty")
    assert g["easy"]["median_px"] < 1.0
    assert g["hard"]["median_px"] > 100.0

    # 4. periodic locks are told apart from random misses
    locks = [Prediction(i, 100.0 + 14.0 * (1 + i % 3), 100.0, 100.0, 100.0,
                        "hard", "dram", 30.0, pitch_px=14.0) for i in range(10)]
    misses = [Prediction(100 + i, 700.0, 400.0, 100.0, 100.0, "hard", "dram",
                         30.0, pitch_px=14.0) for i in range(5)]
    fm = failure_modes(locks + misses)
    assert fm["periodic_lock"] == 10, fm
    assert fm["miss"] == 5, fm
    assert fm["median_lock_multiple"] is not None

    # 5. dose bins are log-spaced and ordered
    spread = [Prediction(i, 100.0 + rng.normal(0, 1), 100.0, 100.0, 100.0,
                         "easy", "dram", float(d))
              for i, d in enumerate(np.exp(rng.uniform(np.log(8), np.log(700), 200)))]
    bins = by_dose(spread, bins=4)
    assert len(bins) == 4
    assert all(bins[i]["dose_hi"] <= bins[i + 1]["dose_hi"] + 1e-9
               for i in range(3))
    assert sum(b["n"] for b in bins) == len(spread)

    # 6. the report renders and mentions the hard row
    text = report(mixed, "self-test")
    assert "hard" in text and "failure modes" in text

    # 7. saving round-trips
    with tempfile.TemporaryDirectory() as td:
        p = save(mixed, Path(td) / "r.json", extra={"algo": "test"})
        data = json.loads(p.read_text())
        assert data["summary"]["n"] == 100
        assert data["meta"]["algo"] == "test"
        assert len(data["predictions"]) == 100

    print("eval/metrics.py self-test OK")
    print(f"  mixed set: median {m['median_px']:.2f} px vs mean "
          f"{m['mean_px']:.2f} px  <- why the mean is useless here")
    print(f"  stratified: easy median {g['easy']['median_px']:.2f} px, "
          f"hard median {g['hard']['median_px']:.1f} px")
    print(f"  failure modes: {fm['periodic_lock']} periodic locks vs "
          f"{fm['miss']} misses")
    print(f"  dose curve: {len(bins)} log-spaced bins")


if __name__ == "__main__":
    _self_test()
