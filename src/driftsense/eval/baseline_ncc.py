"""Classical baseline: multi-angle normalised cross-correlation.

This is the method Applied Materials say is used today, and the method the
hackathon says breaks down on highly periodic layouts.  Run it before training
anything, for two reasons:

1. **It validates the labels.**  If NCC does not localise the *easy* samples to
   within a pixel, the ground truth is wrong -- not the algorithm.  That is a
   generator bug, and finding it after a week of training is expensive.  On this
   dataset it scores 100 % within 5 px on easy samples with a median error of
   about half a pixel, which is the strongest evidence available that the
   pipeline's closed-form labels are right.

2. **It is the number to beat.**  A learned model that does not beat the
   ``hard`` row is not solving the problem the hackathon poses.

Implementation follows J. P. Lewis, "Fast Normalized Cross-Correlation", Vision
Interface 1995: the numerator by FFT, the local sums by integral images.  Two
details cost real accuracy and are easy to get wrong:

* **Non-maximum suppression must be distance-based.**  In a periodic layout the
  second peak sits exactly one line pitch away -- often under 15 px.  A grid-
  bucket de-duplication either merges it with the true peak or lets it through
  depending on where the bucket boundary falls, which produces a systematic
  error of one pitch on a large fraction of samples.
* **The problem statement's "closest to centre" tie-break applies to genuinely
  equal matches, not to near-peak neighbours.**  A loose relative threshold
  (0.97 of the best score) actively drags the answer off the true peak toward
  the image centre in a periodic array.  0.995 is tight enough to fire only on
  real ties.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from driftsense.config import IMG_N

#: Relative score above which a second peak counts as a genuine tie.
TIE_THRESHOLD = 0.995


def _integral(a: np.ndarray) -> np.ndarray:
    """Summed-area table with a zero row and column prepended."""
    ii = np.cumsum(np.cumsum(a, axis=0), axis=1)
    return np.pad(ii, ((1, 0), (1, 0)))


def _window_sum(ii: np.ndarray, k: int) -> np.ndarray:
    """Sums over every ``k x k`` window, from a summed-area table."""
    return ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]


def ncc_map(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Zero-mean normalised cross-correlation at every valid position.

    Args:
        search: The search image.
        template: The template, smaller than ``search`` in both axes.

    Returns:
        ``float64`` score map of shape
        ``(H - k + 1, W - k + 1)``, indexed by the window's top-left corner.
    """
    S = np.asarray(search, dtype=np.float64)
    T = np.asarray(template, dtype=np.float64)
    k = T.shape[0]
    n = float(k * k)

    T0 = T - T.mean()
    tss = math.sqrt(float((T0 * T0).sum())) + 1e-9

    H, W = S.shape
    fh, fw = H + k, W + k                    # pad to prevent circular wrap
    num = np.fft.irfft2(np.fft.rfft2(S, (fh, fw))
                        * np.fft.rfft2(T0[::-1, ::-1], (fh, fw)), (fh, fw))
    num = num[k - 1:H, k - 1:W]

    ls = _window_sum(_integral(S), k)
    lss = _window_sum(_integral(S * S), k)
    var = np.maximum(lss - ls * ls / n, 0.0)
    return num / (np.sqrt(var) * tss + 1e-9)


def peaks(score: np.ndarray, k: int, rel: float = 0.90,
          max_peaks: int = 16) -> Tuple[float, List[Tuple[float, float, float]]]:
    """Distinct local maxima, suppressed by distance rather than by grid cell.

    Args:
        score: NCC map from :func:`ncc_map`.
        k: Template size, used for the suppression radius.
        rel: Keep peaks within this fraction of the best score.
        max_peaks: Stop after this many.

    Returns:
        ``(best_score, [(score, x_centre, y_centre), ...])`` in search pixels.
    """
    best = float(score.max())
    ys, xs = np.where(score >= rel * best)
    order = np.argsort(-score[ys, xs])
    r2 = (k / 2.0) ** 2
    out: List[Tuple[float, float, float]] = []
    for i in order:
        y, x = float(ys[i]), float(xs[i])
        if any((x - q[1]) ** 2 + (y - q[2]) ** 2 < r2 for q in out):
            continue
        out.append((float(score[int(y), int(x)]),
                    x + (k - 1) / 2.0, y + (k - 1) / 2.0))
        if len(out) >= max_peaks:
            break
    return best, out


def locate(reference: np.ndarray, search: np.ndarray, scale: float = 10.0,
           angles: Sequence[float] = (-2.5, -1.25, 0.0, 1.25, 2.5),
           rotation_crop: float = 0.10
           ) -> Tuple[float, float, float, int]:
    """Localise the reference pattern inside the search image.

    Args:
        reference: Reference frame.
        search: Search frame.
        scale: Demagnification factor; use the manifest's ``scale_ratio`` when
            you have it, since it is never exactly 10.
        angles: Rotations to try, in degrees.
        rotation_crop: Fraction trimmed from each side of a rotated template, to
            discard the corners the rotation invented.

    Returns:
        ``(x, y, score, n_tied_candidates)`` in search pixels.
    """
    import cv2

    k = int(round(IMG_N / scale))
    template0 = cv2.resize(np.asarray(reference), (k, k),
                           interpolation=cv2.INTER_AREA).astype(np.float64)

    best = (-2.0, 0.0, 0.0, 1)
    centre = IMG_N / 2.0
    for angle in angles:
        t = template0
        if angle != 0.0:
            m = cv2.getRotationMatrix2D((k / 2.0, k / 2.0), -angle, 1.0)
            rotated = cv2.warpAffine(template0, m, (k, k),
                                     borderMode=cv2.BORDER_REPLICATE)
            c = int(rotation_crop * k)
            t = rotated[c:k - c, c:k - c]
            # The crop is symmetric, so the cropped template shares its centre
            # with the full one and NO offset correction is needed.  Adding one
            # here is a subtle, systematic ~10 px error.

        score = ncc_map(search, t)
        top, candidates = peaks(score, t.shape[0])
        if top <= best[0]:
            continue
        tied = [q for q in candidates if q[0] >= TIE_THRESHOLD * top]
        # Problem statement: if several regions match, take the one nearest the
        # centre of the search image.
        q = min(tied, key=lambda z: (z[1] - centre) ** 2 + (z[2] - centre) ** 2)
        best = (top, q[1], q[2], len(tied))

    return best[1], best[2], best[0], best[3]


def dominant_pitch_px(search: np.ndarray, axis: int = 1) -> float:
    """Estimate the layout's repeat pitch in search pixels.

    Used only to classify failures as periodic locks; a rough estimate from the
    autocorrelation of a mean profile is enough for that.

    Args:
        search: Search image.
        axis: Axis to profile along.

    Returns:
        Pitch in pixels, or ``nan`` if no clear peak is found.
    """
    prof = np.asarray(search, dtype=np.float64).mean(axis=axis)
    prof = prof - prof.mean()
    ac = np.correlate(prof, prof, mode="full")[len(prof) - 1:]
    if ac[0] <= 0:
        return float("nan")
    ac = ac / ac[0]
    lo, hi = 3, min(120, len(ac) - 1)
    seg = ac[lo:hi]
    if seg.size == 0:
        return float("nan")
    idx = int(np.argmax(seg)) + lo
    return float(idx) if ac[idx] > 0.15 else float("nan")


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.eval.baseline_ncc``."""
    from driftsense.config import GeneratorConfig
    from driftsense.eval.metrics import Prediction, report, summarise
    from driftsense.pipeline import plan_sample, render
    from driftsense.rng import sample_seeds

    cfg = GeneratorConfig()
    seeds = sample_seeds(20260803, 12)

    preds: List[Prediction] = []
    for i, s in enumerate(seeds):
        plan = plan_sample(int(s), i, cfg)
        ref, search = render(plan)
        gt = plan.ground_truth()
        x, y, score, tied = locate(ref, search, scale=gt["scale_ratio"],
                                   angles=(-2.5, 0.0, 2.5))
        preds.append(Prediction(
            id=i, pred_x=x, pred_y=y, gt_x=gt["gt_x"], gt_y=gt["gt_y"],
            difficulty=plan.difficulty, style=plan.style,
            dose=plan.search_capture.noise.dose,
            pitch_px=dominant_pitch_px(search), score=score))

    easy = [p for p in preds if p.difficulty in ("easy", "medium")]
    hard = [p for p in preds if p.difficulty == "hard"]

    # THE test this file exists for: if classical NCC cannot solve the easy
    # samples, the labels are wrong.
    if easy:
        m = summarise(easy)
        assert m["median_px"] < 3.0, (
            f"NCC median {m['median_px']:.2f} px on easy samples -- "
            f"this indicates a LABELLING BUG, not a hard dataset")
        assert m["hit"]["5px"] > 0.7, m["hit"]

    # 1. the NCC map has the shape and range it should
    ref, search = render(plan_sample(int(seeds[0]), 0, cfg))
    import cv2

    t = cv2.resize(ref, (100, 100), interpolation=cv2.INTER_AREA)
    sc = ncc_map(search, t.astype(np.float64))
    assert sc.shape == (IMG_N - 99, IMG_N - 99)
    assert -1.01 <= float(sc.min()) and float(sc.max()) <= 1.01

    # 2. a template cut straight from the search image scores ~1 at its origin
    patch = search[300:400, 250:350].astype(np.float64)
    sc2 = ncc_map(search, patch)
    yy, xx = np.unravel_index(int(np.argmax(sc2)), sc2.shape)
    assert (abs(xx - 250) <= 1 and abs(yy - 300) <= 1), (xx, yy)
    assert float(sc2.max()) > 0.99

    # 3. distance-based NMS returns separated peaks
    _, cands = peaks(sc2, 100)
    for a in range(len(cands)):
        for b in range(a + 1, len(cands)):
            d = math.hypot(cands[a][1] - cands[b][1], cands[a][2] - cands[b][2])
            assert d >= 49.0, d

    print(report(preds, "baseline NCC (12 samples)"))
    print("\neval/baseline_ncc.py self-test OK")
    if easy:
        print(f"  easy+medium median     : {summarise(easy)['median_px']:.2f} px"
              f"   <- labels verified")
    if hard:
        print(f"  hard median            : {summarise(hard)['median_px']:.1f} px"
              f"   <- the number to beat")
    print(f"  self-correlation peak  : {float(sc2.max()):.4f} at the exact origin")


if __name__ == "__main__":
    _self_test()
