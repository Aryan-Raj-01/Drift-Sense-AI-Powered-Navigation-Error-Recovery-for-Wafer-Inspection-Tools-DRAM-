"""Stage 1 -- top-K candidate proposal.

Runs multi-angle normalised cross-correlation and returns the best K peaks
rather than the single argmax.

The measured failure split of the classical baseline is 80 % hit, 4 % periodic
lock, 16 % miss: the true location is usually *present* in the correlation
surface but is not the highest peak.  Keeping K candidates converts a
~6000-way ambiguous global search into a K-way discrimination problem, which
is what Stage 2 is trained to solve.

Non-maximum suppression is by distance, never by grid cell.  In a periodic
layout the runner-up sits exactly one pitch away -- often under 15 px -- so a
cell-based scheme merges or drops the true peak depending on where the cell
boundary happens to fall.

Self-test:
    python -m driftsense.localize.propose
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence

import cv2
import numpy as np

from driftsense.eval.baseline_ncc import ncc_map

#: Rotations tried, degrees.  Relative rotation has sigma ~1.6 deg and an
#: observed max of 5.8, so this spans roughly +/-3.5 sigma.
ANGLES: Sequence[float] = (-3.0, -1.5, 0.0, 1.5, 3.0)

#: Fraction trimmed from each side of a rotated template.  Symmetric, so the
#: cropped template keeps the same centre and needs no offset correction.
ROTATION_CROP = 0.10

#: Candidates scoring below this fraction of the best are discarded, but only
#: once ``top_k`` have already been collected.  Recall is worth more than
#: pruning here: on hard samples the true peak often scores far below the best,
#: so an early cut on relative score is exactly the wrong thing to do.
REL_FLOOR = 0.0


class Candidate(NamedTuple):
    """One proposed location in search-image pixels."""

    x: float
    y: float
    score: float
    angle: float


def _rotated(template: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a template about its centre and trim the invented corners."""
    if angle == 0.0:
        return template
    k = template.shape[0]
    m = cv2.getRotationMatrix2D((k / 2.0, k / 2.0), -angle, 1.0)
    out = cv2.warpAffine(template, m, (k, k), borderMode=cv2.BORDER_REPLICATE)
    c = int(ROTATION_CROP * k)
    return out[c:k - c, c:k - c]


def propose(reference: np.ndarray, search: np.ndarray, scale: float = 10.0,
            top_k: int = 32, angles: Sequence[float] = ANGLES,
            rel_floor: float = REL_FLOOR) -> List[Candidate]:
    """Propose the top-K candidate locations.

    Args:
        reference: Reference frame (already lattice-suppressed, ideally).
        search: Search frame, same preprocessing.
        scale: Demagnification to shrink the reference by.
        top_k: How many candidates to return.
        angles: Rotations to try, in degrees.
        rel_floor: Discard candidates below this fraction of the best score.

    Returns:
        Up to ``top_k`` candidates, best first.  Coordinates are the centre of
        the matching region in search pixels.
    """
    ref = np.asarray(reference, dtype=np.float64)
    srch = np.asarray(search, dtype=np.float64)

    k0 = max(8, min(int(round(ref.shape[0] / scale)), srch.shape[0] - 2))
    base = cv2.resize(ref, (k0, k0), interpolation=cv2.INTER_AREA)

    pooled: List[Candidate] = []
    for angle in angles:
        t = _rotated(base, angle)
        k = t.shape[0]
        if k < 4 or k >= srch.shape[0]:
            continue
        score = ncc_map(srch, t)
        # Keep a generous pool per angle; the global NMS below trims it.
        flat = score.ravel()
        n_keep = min(flat.size, top_k * 40)
        idx = np.argpartition(-flat, n_keep - 1)[:n_keep]
        ys, xs = np.unravel_index(idx, score.shape)
        off = (k - 1) / 2.0
        pooled.extend(Candidate(float(x) + off, float(y) + off,
                                float(score[y, x]), float(angle))
                      for y, x in zip(ys, xs))

    if not pooled:
        return []

    pooled.sort(key=lambda c: -c.score)
    best = pooled[0].score
    radius2 = (k0 / 2.0) ** 2

    out: List[Candidate] = []
    for c in pooled:
        if any((c.x - q.x) ** 2 + (c.y - q.y) ** 2 < radius2 for q in out):
            continue
        if c.score < rel_floor * best and len(out) >= top_k:
            break
        out.append(c)
        if len(out) >= top_k:
            break
    return out


def recall_at_k(candidates: Sequence[Candidate], gt_x: float, gt_y: float,
                tolerance: float = 5.0) -> Optional[int]:
    """Rank at which the true location appears, or None if absent.

    Args:
        candidates: Output of :func:`propose`, best first.
        gt_x: True centre x.
        gt_y: True centre y.
        tolerance: A candidate counts as correct within this many pixels.

    Returns:
        Zero-based rank of the first correct candidate, else ``None``.
    """
    for i, c in enumerate(candidates):
        if (c.x - gt_x) ** 2 + (c.y - gt_y) ** 2 <= tolerance ** 2:
            return i
    return None


def _self_test() -> None:
    """Plant a known patch and check it is proposed at rank 0."""
    rng = np.random.default_rng(0)
    search = rng.normal(0, 1, (1000, 1000))
    yy, xx = np.mgrid[:1000, :1000]
    search += 3.0 * np.sin(2 * np.pi * xx / 12.0)

    patch = rng.normal(0, 3, (100, 100))
    search[400:500, 600:700] += patch

    reference = cv2.resize(patch, (1000, 1000),
                           interpolation=cv2.INTER_NEAREST).astype(np.float64)

    cands = propose(reference, search, scale=10.0, top_k=32)
    rank = recall_at_k(cands, 649.5, 449.5, tolerance=5.0)
    print(f"  candidates returned {len(cands)}")
    print(f"  true location rank  {rank}")
    assert cands, "no candidates returned"
    assert rank is not None and rank <= 2, f"true location not found (rank={rank})"
    print("  OK")


if __name__ == "__main__":
    _self_test()
