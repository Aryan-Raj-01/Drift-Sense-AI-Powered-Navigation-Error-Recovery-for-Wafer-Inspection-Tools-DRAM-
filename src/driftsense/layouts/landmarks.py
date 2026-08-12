"""Navigation landmarks: the features a target is actually placed on.

This is the module that decides whether a training sample has a correct answer
at all, so it is worth stating the reasoning explicitly.

**Why landmarks exist.**  A DRAM array is periodic to within a nanometre over
tens of micrometres.  If the reference frame is centred on plain array, then
hundreds of positions in the search image are pixel-identical to the true one,
and no algorithm -- classical or learned -- can prefer the right one.  Such a
sample does not have a hard answer, it has no answer.  Training on a diet of
them puts an irreducible floor under the loss, and the loss-minimising response
is to predict the image centre every time.  Real navigation targets are not
placed on plain array either; they are placed on something locally unique.

**Why the field is position-hashed.**  The reference frame and the search frame
are rendered independently, at different magnifications, possibly in different
processes.  They must nevertheless agree on exactly which landmarks exist and
where.  Hashing the lattice cell index means any window of the die can be
queried in isolation and always returns the same answer -- no shared canvas, no
shared RNG, no ordering dependency.  See :func:`driftsense.rng.position_rng`.

**Why signatures.**  Uniqueness is enforced on a *signature*
``(type, orientation, size bucket, polarity)`` rather than on position.  With
eight shapes x four orientations x three size buckets x two polarities there
are ~192 signatures, so a search window holding ~30 landmarks usually contains
several unique ones -- but the model cannot shortcut by memorising "the target
is the bright pad", because two thirds of the time it is not.

**Why they are drawn into a local window.**  A 400 nm landmark covers 40 px of a
search image.  Compositing it across the full 1000x1000 frame does 625x more
work than necessary; with ~30 landmarks per frame that is the difference
between ~1.2 s and ~0.05 s per frame, which at 100k pairs is days.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from driftsense import geometry as G
from driftsense.config import LandmarkConfig
from driftsense.rng import position_rng

#: Shapes available to the landmark field.  Adding one widens the signature
#: space; it does not require changes anywhere else.
LANDMARK_TYPES: Tuple[str, ...] = ("pad", "cross", "tee", "ell", "ring",
                                   "block", "hbar", "vbar")


@dataclass(frozen=True)
class Landmark:
    """One landmark on the die.

    Attributes:
        x: Die x coordinate of the centre (nm).
        y: Die y coordinate of the centre (nm).
        kind: One of :data:`LANDMARK_TYPES`.
        size_nm: Nominal extent (nm); the actual bounding box depends on shape.
        rot: Orientation in radians, always a multiple of 90 degrees.
        bright: ``True`` for a bright feature, ``False`` for a dark one.
    """

    x: float
    y: float
    kind: str
    size_nm: float
    rot: float
    bright: bool

    def signature(self, bucket_nm: float) -> str:
        """Identity used for the uniqueness test.

        Two landmarks with the same signature are interchangeable to a matcher,
        so a target is only usable if its signature is unique in its window.

        Args:
            bucket_nm: Size quantisation; sizes within one bucket count as the
                same. Coarser buckets are stricter (fewer usable targets) and
                safer.
        """
        quarter = int(round(self.rot / (math.pi / 2.0))) % 4
        return (f"{self.kind}|{quarter}|{int(self.bright)}"
                f"|{int(self.size_nm // bucket_nm)}")

    @property
    def extent_nm(self) -> float:
        """Radius of a circle that certainly contains the shape (nm)."""
        return 1.0 * self.size_nm      # hbar/vbar are 1.8x size on one axis


@dataclass(frozen=True)
class LandmarkField:
    """A die's landmark population, defined implicitly by a hash.

    The field is infinite: no landmark is ever stored, only regenerated on
    demand for whatever window is asked for.

    Attributes:
        die_seed: Identifies this die.
        spacing_nm: Lattice cell size (nm).
        density: Probability a cell is occupied.
        size_lo: Minimum landmark size (nm).
        size_hi: Maximum landmark size (nm).
        size_bucket_nm: Size quantisation for signatures.
        bright_probability: Fraction of landmarks drawn bright.
        types: Shape vocabulary.
    """

    die_seed: int
    spacing_nm: float
    density: float
    size_lo: float
    size_hi: float
    size_bucket_nm: float
    bright_probability: float
    types: Tuple[str, ...] = LANDMARK_TYPES

    @classmethod
    def sample(cls, rng: np.random.Generator, cfg: LandmarkConfig,
               die_seed: int) -> "LandmarkField":
        """Draw a landmark population for one die.

        Args:
            rng: The sample's ``"layout"`` stream.
            cfg: Landmark configuration.
            die_seed: Seed identifying the die.
        """
        return cls(
            die_seed=int(die_seed),
            spacing_nm=cfg.spacing_nm.draw(rng),
            density=cfg.density.draw(rng),
            size_lo=cfg.size_nm.lo,
            size_hi=cfg.size_nm.hi,
            size_bucket_nm=cfg.size_bucket_nm,
            bright_probability=cfg.bright_probability,
            types=tuple(cfg.types),
        )

    # -- population ---------------------------------------------------------- #
    def in_window(self, x0: float, y0: float, x1: float, y1: float,
                  pad_nm: float = 600.0) -> List[Landmark]:
        """Every landmark whose centre falls in a padded die-coordinate window.

        Args:
            x0: Window left edge (nm).
            y0: Window top edge (nm).
            x1: Window right edge (nm).
            y1: Window bottom edge (nm).
            pad_nm: Extra margin so landmarks straddling the edge are still
                drawn.  A landmark clipped by the frame edge is a legitimate
                (and informative) sight; a landmark *missing* because it was
                culled would differ between the two frames.

        Returns:
            Landmarks in deterministic order (row-major over lattice cells),
            so two callers always see the same list.
        """
        sp = self.spacing_nm
        i0 = int(math.floor((x0 - pad_nm) / sp))
        i1 = int(math.ceil((x1 + pad_nm) / sp))
        j0 = int(math.floor((y0 - pad_nm) / sp))
        j1 = int(math.ceil((y1 + pad_nm) / sp))

        out: List[Landmark] = []
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                r = position_rng(self.die_seed, i, j, "landmark")
                if r.random() > self.density:
                    continue
                cx = (i + 0.5 + (r.random() - 0.5) * 0.55) * sp
                cy = (j + 0.5 + (r.random() - 0.5) * 0.55) * sp
                if not (x0 - pad_nm <= cx <= x1 + pad_nm
                        and y0 - pad_nm <= cy <= y1 + pad_nm):
                    continue
                out.append(Landmark(
                    x=cx, y=cy,
                    kind=self.types[int(r.integers(len(self.types)))],
                    size_nm=float(r.uniform(self.size_lo, self.size_hi)),
                    rot=float(int(r.integers(4)) * (math.pi / 2.0)),
                    bright=bool(r.random() < self.bright_probability),
                ))
        return out

    def in_frame(self, frame, pad_nm: float = 600.0) -> List[Landmark]:
        """Landmarks visible in a :class:`~driftsense.optics.acquisition.FrameGeometry`."""
        cx, cy = frame.center_nm
        h = frame.half_fov_nm
        return self.in_window(cx - h, cy - h, cx + h, cy + h, pad_nm)

    def unique_candidates(self, landmarks: Sequence[Landmark]
                          ) -> List[Landmark]:
        """Landmarks whose signature occurs exactly once in the given list.

        Args:
            landmarks: Population of the search window.

        Returns:
            The subset usable as navigation targets.  May be empty, in which
            case the planner falls back to a defect-marked hard sample.
        """
        counts: Dict[str, int] = {}
        for lm in landmarks:
            s = lm.signature(self.size_bucket_nm)
            counts[s] = counts.get(s, 0) + 1
        return [lm for lm in landmarks
                if counts[lm.signature(self.size_bucket_nm)] == 1]

    # -- rendering ----------------------------------------------------------- #
    def render(self, img: np.ndarray, x: np.ndarray, y: np.ndarray,
               frame, aa: float, landmarks: Optional[Sequence[Landmark]] = None
               ) -> np.ndarray:
        """Composite every visible landmark into the frame, in place.

        Args:
            img: Working material map, shape ``(n, n)``.
            x: Die x coordinates sampled by each pixel (nm).
            y: Die y coordinates (nm).
            frame: The :class:`FrameGeometry` being rendered, used only to find
                each landmark's approximate pixel location so the composite can
                be restricted to a local window.
            aa: Sampling pitch (nm).
            landmarks: Precomputed population; queried from the frame if absent.

        Returns:
            ``img``, modified in place.
        """
        if landmarks is None:
            landmarks = self.in_frame(frame)
        n = img.shape[0]
        for lm in landmarks:
            u, v = frame.to_pixel((lm.x, lm.y))
            radius = int(1.6 * lm.extent_nm / aa) + 3
            sl = G.window_slice(u, v, radius, n) if hasattr(G, "window_slice") \
                else _window_slice(u, v, radius, n)
            if sl is None:
                continue
            _draw(img, x, y, lm, aa, sl)
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


def _draw(img: np.ndarray, x: np.ndarray, y: np.ndarray, lm: Landmark,
          aa: float, sl: Tuple[slice, slice]) -> None:
    """Composite one landmark into a local slice.

    Shapes are built from rounded-rectangle SDFs so that every one of them is
    anti-aliased by the same one-pixel rule as the array itself: a landmark that
    aliased differently from its surroundings would be trivially separable and
    would teach the network nothing transferable.
    """
    xs, ys = x[sl], y[sl]
    level = 0.93 if lm.bright else 0.10
    s, cx, cy, th = lm.size_nm, lm.x, lm.y, lm.rot
    r = 0.12 * s

    if lm.kind == "pad":
        mask = G.soft_rect(xs, ys, cx, cy, s, s * 0.8, th, aa, r)
    elif lm.kind == "block":
        mask = G.soft_rect(xs, ys, cx, cy, s * 1.3, s * 0.55, th, aa, r * 0.4)
    elif lm.kind == "cross":
        mask = np.maximum(G.soft_rect(xs, ys, cx, cy, s, s * 0.30, th, aa, 0.0),
                          G.soft_rect(xs, ys, cx, cy, s * 0.30, s, th, aa, 0.0))
    elif lm.kind == "tee":
        mask = np.maximum(
            G.soft_rect(xs, ys, cx, cy - s * 0.30, s, s * 0.28, th, aa, 0.0),
            G.soft_rect(xs, ys, cx, cy, s * 0.28, s, th, aa, 0.0))
    elif lm.kind == "ell":
        mask = np.maximum(
            G.soft_rect(xs, ys, cx - s * 0.28, cy, s * 0.30, s, th, aa, 0.0),
            G.soft_rect(xs, ys, cx, cy + s * 0.30, s, s * 0.30, th, aa, 0.0))
    elif lm.kind == "ring":
        outer = G.soft_rect(xs, ys, cx, cy, s, s, th, aa, 0.45 * s)
        inner = G.soft_rect(xs, ys, cx, cy, s * 0.52, s * 0.52, th, aa, 0.24 * s)
        mask = np.clip(outer - inner, 0.0, 1.0)
    elif lm.kind == "hbar":
        mask = G.soft_rect(xs, ys, cx, cy, s * 1.8, s * 0.22, 0.0, aa, 0.0)
    else:  # vbar
        mask = G.soft_rect(xs, ys, cx, cy, s * 0.22, s * 1.8, 0.0, aa, 0.0)

    img[sl] = img[sl] * (1.0 - mask) + mask * np.float32(level)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.layouts.landmarks``."""
    from driftsense.config import GeneratorConfig
    from driftsense.optics.acquisition import FrameGeometry

    cfg = GeneratorConfig()
    rng = np.random.default_rng(5)
    fieldset = LandmarkField.sample(rng, cfg.landmarks, die_seed=987654321)

    # 1. determinism and window independence: querying a big window and four
    #    overlapping sub-windows must agree exactly.  This is the property that
    #    lets the two frames be rendered in different processes.
    big = fieldset.in_window(-5000, -5000, 5000, 5000, pad_nm=0.0)
    assert big == fieldset.in_window(-5000, -5000, 5000, 5000, pad_nm=0.0)
    seen = set()
    for (a, b) in ((-5000, 0), (0, 5000)):
        for (c, d) in ((-5000, 0), (0, 5000)):
            for lm in fieldset.in_window(a, c, b, d, pad_nm=0.0):
                seen.add((round(lm.x, 6), round(lm.y, 6), lm.kind))
    assert seen == {(round(l.x, 6), round(l.y, 6), l.kind) for l in big}

    # 2. population density in a 10 um search window is workable
    counts = []
    for k in range(30):
        f = LandmarkField.sample(np.random.default_rng(k), cfg.landmarks, k)
        counts.append(len(f.in_window(-5000, -5000, 5000, 5000, pad_nm=0.0)))
    mean_count = float(np.mean(counts))
    assert 10 < mean_count < 120, mean_count

    # 3. unique candidates exist most of the time -- this is the whole point
    uniq_frac = []
    for k in range(60):
        f = LandmarkField.sample(np.random.default_rng(1000 + k), cfg.landmarks,
                                 5000 + k)
        pop = f.in_window(-5000, -5000, 5000, 5000, pad_nm=0.0)
        uniq_frac.append(len(f.unique_candidates(pop)) / max(1, len(pop)))
    mean_uniq = float(np.mean(uniq_frac))
    assert mean_uniq > 0.4, mean_uniq

    # 4. a unique candidate really is unique
    f = LandmarkField.sample(np.random.default_rng(7), cfg.landmarks, 42)
    pop = f.in_window(-5000, -5000, 5000, 5000, pad_nm=0.0)
    for lm in f.unique_candidates(pop):
        same = [o for o in pop
                if o.signature(f.size_bucket_nm) == lm.signature(f.size_bucket_nm)]
        assert len(same) == 1

    # 5. rendering: every shape draws, changes the image, and stays local
    frame = FrameGeometry(center_nm=(0.0, 0.0), px_nm=10.0, theta=0.0, n=200)
    x, y = frame.grid()
    for kind in LANDMARK_TYPES:
        img = np.full((200, 200), 0.5, dtype=np.float32)
        lm = Landmark(x=0.0, y=0.0, kind=kind, size_nm=400.0, rot=0.0, bright=True)
        f.render(img, x, y, frame, 10.0, landmarks=[lm])
        assert float(img.max()) > 0.9, kind
        # a 400 nm shape is 40 px: nothing may change 150 px away
        assert abs(float(img[0, 0]) - 0.5) < 1e-6, kind

    # 6. the same landmark renders consistently at both magnifications
    fine = FrameGeometry(center_nm=(0.0, 0.0), px_nm=1.0, theta=0.0, n=1000)
    coarse = FrameGeometry(center_nm=(0.0, 0.0), px_nm=10.0, theta=0.0, n=100)
    lm = Landmark(0.0, 0.0, "cross", 400.0, 0.0, True)
    a_img = np.full((1000, 1000), 0.5, np.float32)
    b_img = np.full((100, 100), 0.5, np.float32)
    xf, yf = fine.grid()
    xc, yc = coarse.grid()
    f.render(a_img, xf, yf, fine, 1.0, [lm])
    f.render(b_img, xc, yc, coarse, 10.0, [lm])
    corr = float(np.corrcoef(G.box_downsample(a_img, 10).ravel(),
                             b_img.ravel())[0, 1])
    assert corr > 0.97, corr

    # 7. signature space is wide enough to matter
    sigs = {lm.signature(cfg.landmarks.size_bucket_nm)
            for k in range(200)
            for lm in LandmarkField.sample(np.random.default_rng(k),
                                           cfg.landmarks, k)
            .in_window(-3000, -3000, 3000, 3000, 0.0)}
    assert len(sigs) > 60, len(sigs)

    print("layouts/landmarks.py self-test OK")
    print(f"  landmarks per 10um FOV : {mean_count:.1f} (mean of 30 dies)")
    print(f"  unique-signature share : {100*mean_uniq:.0f} %")
    print(f"  distinct signatures    : {len(sigs)}")
    print(f"  1nm vs 10nm landmark   : r = {corr:.4f}")
    print(f"  window independence    : exact over 4 overlapping sub-windows")


if __name__ == "__main__":
    _self_test()
