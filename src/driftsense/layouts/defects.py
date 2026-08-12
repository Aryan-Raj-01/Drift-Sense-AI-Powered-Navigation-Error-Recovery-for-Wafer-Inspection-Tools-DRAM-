"""Process defects: the unique marker for hard samples.

A "hard" sample is one where the target sits on plain periodic array with no
landmark anywhere near it.  The temptation is to generate exactly that and call
it hard -- but a target with no unique feature has no correct answer, and
training on unanswerable samples teaches a model to hedge toward the image
centre (see :mod:`driftsense.layouts.landmarks`).

So hard samples are made hard by *shrinking* the unique feature rather than
removing it.  A single process defect -- one missing contact, one bridge between
adjacent lines, one broken line -- is 14-34 nm across, which is 1.4-3.4 pixels
in the wide-search frame.  It is genuinely near the limit of what is findable,
it is exactly the kind of feature a real inspection tool navigates to, and it
keeps the sample well posed: there is one right answer and it is recoverable in
principle.

Defects here are placed deliberately at the target, so they are unique by
construction.  That is a different mechanism from the *population* defects in
:class:`~driftsense.layouts.dram.DramLayout` (``missing_contact_rate``,
``broken_line_rate``), which are scattered position-hashed across the whole die
and act as distractors.  Both exist, and the distinction matters: without the
scattered population, "the one defect in the frame" would itself be a giveaway.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from driftsense import geometry as G
from driftsense.config import DefectConfig

#: Defect vocabulary.  Each maps to a real failure mode of the process.
DEFECT_TYPES: Tuple[str, ...] = ("missing_via", "extra_via", "bridge",
                                 "line_break")


@dataclass(frozen=True)
class Defect:
    """A single localised process defect.

    Attributes:
        kind: One of :data:`DEFECT_TYPES`.
        x: Die x coordinate (nm), snapped to the layout's contact lattice.
        y: Die y coordinate (nm).
        size_nm: Characteristic size (nm).
        rot: Orientation in radians (0 or pi/2) for the anisotropic kinds.
    """

    kind: str
    x: float
    y: float
    size_nm: float
    rot: float = 0.0

    @property
    def extent_nm(self) -> float:
        """Radius of a circle that certainly contains the defect (nm)."""
        return 3.0 * self.size_nm

    def size_px(self, px_nm: float) -> float:
        """Defect size in pixels at a given sampling pitch."""
        return self.size_nm / px_nm

    def to_dict(self) -> dict:
        """JSON-serialisable summary for the manifest."""
        return {"defect_type": self.kind,
                "defect_size_nm": round(self.size_nm, 3)}


def snap_to_lattice(point_nm: Tuple[float, float],
                    lattice: Tuple[float, float, float, float]
                    ) -> Tuple[float, float]:
    """Move a point to the nearest contact-lattice site.

    A "missing contact" that lands between contacts is not a missing contact,
    it is a smudge.  Snapping makes the defect structurally meaningful, which
    also makes it a fair test: the algorithm has to find a *modified lattice
    site*, not an arbitrary blob.

    Args:
        point_nm: Requested ``(x, y)`` in die coordinates.
        lattice: ``(pitch_x, pitch_y, phase_x, phase_y)`` from
            :meth:`driftsense.layouts.base.Layout.lattice`.

    Returns:
        The snapped ``(x, y)``.
    """
    px, py, phx, phy = lattice
    sx = phx + round((point_nm[0] - phx) / px) * px
    sy = phy + round((point_nm[1] - phy) / py) * py
    return float(sx), float(sy)


def sample_defect(rng: np.random.Generator, cfg: DefectConfig,
                  point_nm: Tuple[float, float],
                  lattice: Tuple[float, float, float, float]) -> Defect:
    """Draw a defect to mark a target position.

    Args:
        rng: The sample's ``"plan"`` stream.
        cfg: Defect configuration.
        point_nm: Desired location in die coordinates (nm).
        lattice: Contact lattice of the layout, for snapping.

    Returns:
        A :class:`Defect` at the snapped position.
    """
    kinds = tuple(cfg.types) or DEFECT_TYPES
    kind = kinds[int(rng.integers(len(kinds)))]
    x, y = snap_to_lattice(point_nm, lattice)
    return Defect(
        kind=kind,
        x=x,
        y=y,
        size_nm=cfg.size_nm.draw(rng),
        rot=float(int(rng.integers(2)) * (math.pi / 2.0)),
    )


def render(img: np.ndarray, x: np.ndarray, y: np.ndarray, frame,
           defect: Defect, layout, aa: float) -> np.ndarray:
    """Composite a defect into the frame, in place.

    Args:
        img: Working material map.
        x: Die x coordinates sampled by each pixel (nm).
        y: Die y coordinates (nm).
        frame: The :class:`~driftsense.optics.acquisition.FrameGeometry`, used
            to restrict the composite to a local window.
        defect: The defect to draw.
        layout: The layout, queried for the grey level to paint with, so a
            missing contact is filled with *line* material rather than an
            arbitrary grey.
        aa: Sampling pitch (nm).

    Returns:
        ``img``, modified in place.
    """
    n = img.shape[0]
    u, v = frame.to_pixel((defect.x, defect.y))
    radius = int(defect.extent_nm / aa) + 4
    sl = _window_slice(u, v, radius, n)
    if sl is None:
        return img

    xs, ys = x[sl], y[sl]
    s, cx, cy = defect.size_nm, defect.x, defect.y

    if defect.kind == "missing_via":
        mask = G.soft_step(0.5 * s - np.hypot(xs - cx, ys - cy), aa)
        level = layout.feature_level("line")
    elif defect.kind == "extra_via":
        mask = G.soft_step(0.5 * s - np.hypot(xs - cx, ys - cy), aa)
        level = layout.feature_level("contact")
    elif defect.kind == "bridge":
        mask = G.soft_rect(xs, ys, cx, cy, s * 3.2, s * 0.8, defect.rot, aa, 0.0)
        level = layout.feature_level("line")
    else:  # line_break
        mask = G.soft_rect(xs, ys, cx, cy, s * 1.0, s * 2.6, defect.rot, aa, 0.0)
        level = layout.feature_level("background")

    if layout.polarity < 0:
        level = 1.0 - level
    img[sl] = img[sl] * (1.0 - mask) + mask * np.float32(level)
    return img


def _window_slice(u: float, v: float, radius: int, n: int
                  ) -> Optional[Tuple[slice, slice]]:
    """Clipped ``(rows, cols)`` slice around a pixel, or ``None`` if off-frame."""
    c0 = int(max(0, math.floor(u - radius)))
    c1 = int(min(n, math.ceil(u + radius)))
    r0 = int(max(0, math.floor(v - radius)))
    r1 = int(min(n, math.ceil(v + radius)))
    if c1 <= c0 or r1 <= r0:
        return None
    return slice(r0, r1), slice(c0, c1)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.layouts.defects``."""
    from driftsense.config import GeneratorConfig
    from driftsense.layouts.base import sample_layout
    from driftsense.optics.acquisition import FrameGeometry

    cfg = GeneratorConfig()
    layout = sample_layout("dram", np.random.default_rng(3), cfg)
    lat = layout.lattice()

    # 1. snapping really lands on a lattice site
    for _ in range(50):
        p = (float(np.random.uniform(-5000, 5000)),
             float(np.random.uniform(-5000, 5000)))
        sx, sy = snap_to_lattice(p, lat)
        assert abs((sx - lat[2]) % lat[0]) < 1e-6 or \
               abs((sx - lat[2]) % lat[0] - lat[0]) < 1e-6
        assert math.hypot(sx - p[0], sy - p[1]) <= math.hypot(lat[0], lat[1])

    # 2. every kind draws, is local, and is visible at reference resolution
    fine = FrameGeometry(center_nm=(0.0, 0.0), px_nm=1.0, theta=0.0, n=400)
    xf, yf = fine.grid()
    base_fine = layout.evaluate(xf, yf, 1.0)
    stats = {}
    for kind in DEFECT_TYPES:
        d = Defect(kind=kind, **dict(zip(("x", "y"), snap_to_lattice((0.0, 0.0), lat))),
                   size_nm=28.0, rot=0.0)
        img = base_fine.copy()
        render(img, xf, yf, fine, d, layout, 1.0)
        diff = np.abs(img - base_fine)
        changed = int((diff > 0.02).sum())
        assert changed > 0, f"{kind} drew nothing"
        # strictly local: nothing changes more than ~100 nm away
        assert float(diff[:120, :120].max()) < 1e-6, f"{kind} leaked"
        stats[kind] = changed

    # 3. at 10 nm/px the same defect is a handful of pixels -- hard, not absent
    coarse = FrameGeometry(center_nm=(0.0, 0.0), px_nm=10.0, theta=0.0, n=200)
    xc, yc = coarse.grid()
    base_coarse = layout.evaluate(xc, yc, 10.0)
    coarse_changed = {}
    for kind in DEFECT_TYPES:
        d = Defect(kind, *snap_to_lattice((0.0, 0.0), lat), 28.0, 0.0)
        img = base_coarse.copy()
        render(img, xc, yc, coarse, d, layout, 10.0)
        chg = int((np.abs(img - base_coarse) > 0.02).sum())
        coarse_changed[kind] = chg
        assert 0 < chg < 400, f"{kind}: {chg} px changed at 10 nm/px"

    # 4. sampling is reproducible and lands where asked
    d1 = sample_defect(np.random.default_rng(9), cfg.defects, (1234.0, -567.0), lat)
    d2 = sample_defect(np.random.default_rng(9), cfg.defects, (1234.0, -567.0), lat)
    assert d1 == d2
    assert math.hypot(d1.x - 1234.0, d1.y + 567.0) <= math.hypot(lat[0], lat[1])
    assert cfg.defects.size_nm.lo <= d1.size_nm <= cfg.defects.size_nm.hi

    # 5. polarity inversion flips the paint level too
    inv = type(layout)(**{**layout.__dict__, "polarity": -1})
    img = np.full((50, 50), 0.5, np.float32)
    xs, ys = fine.grid()
    render(img, xs[:50, :50], ys[:50, :50], fine,
           Defect("line_break", float(xs[25, 25]), float(ys[25, 25]), 28.0, 0.0),
           inv, 1.0)
    assert float(img.max()) > 0.5

    print("layouts/defects.py self-test OK")
    print(f"  snapped to lattice     : pitch {lat[0]:.1f} x {lat[1]:.1f} nm")
    print(f"  28 nm defect @1 nm/px  : "
          + ", ".join(f"{k}={v}px" for k, v in stats.items()))
    print(f"  28 nm defect @10 nm/px : "
          + ", ".join(f"{k}={v}px" for k, v in coarse_changed.items()))
    print(f"  locality               : zero change beyond ~100 nm")


if __name__ == "__main__":
    _self_test()
