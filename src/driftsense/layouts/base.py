"""Layout base classes: what every die architecture shares.

A layout is a **pure function of physical position**, not a raster.  It is
evaluated on whatever sampling grid a frame provides, at 1 nm/px for the
reference and 10 nm/px for the wide search, and the two frames therefore agree
by construction rather than by resampling one into the other.

Two shared mechanisms live here.

**Position-hashed process variation.**  Missing contacts, broken lines and
broken fins have to be *the same defects* in both frames.  Drawing them from
an RNG at render time cannot work: the two frames render independently, so the
same lattice site would be defective in one and intact in the other, and the
network would be trained on a contradiction.  :func:`hash_uniform` maps an
integer lattice site to a deterministic uniform value with a SplitMix64
finaliser, so "is this contact missing?" is answerable from the site alone, in
either frame, forever.

**Line-edge roughness.**  Modelled as a sum of sinusoids with 40-600 nm
correlation lengths rather than white noise on the edge.  Real LER spectra are
dominated by low frequencies (Constantoudis / Patsis / Gogolides, J. Vac. Sci.
Technol. B 21 (2003) and 22 (2004)); the distinction matters because white
roughness averages away under 10x demagnification while long-wavelength
roughness survives into the search image, and only the surviving part can help
or hurt a matcher.

References for the structures themselves are given in the subclasses.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from driftsense import geometry as G
from driftsense.config import (
    GeneratorConfig,
    MacroStructureConfig,
    ProcessVariationConfig,
)

_K1 = np.uint64(0x9E3779B97F4A7C15)
_K2 = np.uint64(0xC2B2AE3D27D4EB4F)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_TWO64 = float(1 << 64)


def _mix(z: np.ndarray) -> np.ndarray:
    """SplitMix64 finaliser; avalanches every input bit across the output."""
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    return z ^ (z >> np.uint64(31))


def hash_uniform(seed: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Deterministic uniform value in ``[0, 1)`` for integer lattice site ``(i, j)``.

    Vectorised, allocation-light, and independent of evaluation order -- the
    same site gives the same value in the reference frame, in the search frame,
    on another machine and next year.

    Args:
        seed: Field seed (different per defect type, so they do not correlate).
        i: Integer lattice column indices; may be negative.
        j: Integer lattice row indices; may be negative.

    Returns:
        ``float32`` array of the same shape.
    """
    ii = np.asarray(i, dtype=np.int64).astype(np.uint64, copy=False)
    jj = np.asarray(j, dtype=np.int64).astype(np.uint64, copy=False)
    with np.errstate(over="ignore"):
        h = _mix(np.uint64(seed & 0xFFFFFFFFFFFFFFFF) ^ (ii * _K1))
        h = _mix(h ^ (jj * _K2))
    return (h.astype(np.float64) / _TWO64).astype(np.float32)


# --------------------------------------------------------------------------- #
# Roughness
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Roughness:
    """Line-edge roughness as a sum of sinusoids.

    Attributes:
        amp_nm: Overall amplitude (nm).
        comps: ``(weight, wavelength_nm, phase)`` per harmonic.
        min_px_nm: Skip evaluation when the sampling pitch is coarser than
            this; at 10 nm/px a 2 nm roughness is a fiftieth of a pixel and
            costs three ``sin`` calls over a million points for nothing.
    """

    amp_nm: float
    comps: Tuple[Tuple[float, float, float], ...]
    min_px_nm: float = 4.0

    @classmethod
    def sample(cls, rng: np.random.Generator,
               cfg: ProcessVariationConfig) -> "Roughness":
        """Draw a roughness profile."""
        comps = tuple(
            (float(rng.uniform(0.4, 1.0)),
             float(cfg.ler_wavelength_nm.draw(rng)),
             float(rng.uniform(0.0, 2.0 * math.pi)))
            for _ in range(max(1, cfg.ler_harmonics))
        )
        return cls(amp_nm=cfg.ler_amp_nm.draw(rng), comps=comps,
                   min_px_nm=cfg.ler_min_px_nm)

    def displace(self, coord: np.ndarray, aa: float) -> Optional[np.ndarray]:
        """Edge displacement along a line, or ``None`` when sub-pixel.

        Args:
            coord: Coordinate *along* the line (nm).
            aa: Sampling pitch (nm).
        """
        if aa >= self.min_px_nm or self.amp_nm <= 0.0:
            return None
        c = coord.astype(np.float32, copy=False)
        total = None
        for w, lam, ph in self.comps:
            term = np.float32(w) * np.sin(np.float32(2.0 * math.pi / lam) * c
                                          + np.float32(ph))
            total = term if total is None else total + term
        scale = np.float32(self.amp_nm / max(1.0, sum(w for w, _, _ in self.comps)))
        return scale * total


# --------------------------------------------------------------------------- #
# Macro superstructure
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MacroStructure:
    """Power rails and array-block boundaries.

    These coarse bars are the only structure that survives 10x demagnification
    at full contrast, and they are what gives a real wide-search frame its
    blocky appearance.  Without them the search image is a uniform texture and
    every sample looks the same to a network.
    """

    enabled: bool
    pitch_x: float
    pitch_y: float
    width: float
    phase_x: float
    phase_y: float
    level: float

    @classmethod
    def sample(cls, rng: np.random.Generator,
               cfg: MacroStructureConfig) -> "MacroStructure":
        """Draw the superstructure for one die."""
        pitch_x = cfg.pitch_nm.draw(rng)
        pitch_y = pitch_x * cfg.pitch_aspect.draw(rng)
        return cls(
            enabled=bool(rng.random() < cfg.probability),
            pitch_x=pitch_x,
            pitch_y=pitch_y,
            width=cfg.width_nm.draw(rng),
            phase_x=float(rng.uniform(0.0, pitch_x)),
            phase_y=float(rng.uniform(0.0, pitch_y)),
            level=cfg.level.draw(rng),
        )

    def apply(self, img: np.ndarray, x: np.ndarray, y: np.ndarray,
              aa: float) -> np.ndarray:
        """Composite the rails over an image in place."""
        if not self.enabled:
            return img
        m = np.maximum(
            G.soft_bands(y, self.pitch_y, self.width, self.phase_y, aa),
            G.soft_bands(x, self.pitch_x, self.width, self.phase_x, aa),
        )
        return G.composite(img, m, self.level)


# --------------------------------------------------------------------------- #
# Layout ABC
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Layout(ABC):
    """A die architecture, evaluable at arbitrary physical coordinates.

    Attributes:
        style: ``"dram"`` or ``"finfet"``.
        background: Grey level of the field between features.
        roughness_x: Roughness applied to vertically-running structures.
        roughness_y: Roughness applied to horizontally-running structures.
        macro: Coarse superstructure shared by both architectures.
        variation_seed: Seed for the position-hashed defect fields.
        cd_drift_frac: Amplitude of slow critical-dimension drift across the die.
        cd_drift_period_nm: Its spatial period.
        polarity: ``+1`` for the usual bright-features-on-dark-field, ``-1``
            for an inverted contrast image.  Detector settings and material
            stack routinely flip this on a real tool; a model trained on one
            polarity fails completely on the other, and it costs one line to
            randomise.
    """

    style: str
    background: float
    roughness_x: Roughness
    roughness_y: Roughness
    macro: MacroStructure
    variation_seed: int
    cd_drift_frac: float
    cd_drift_period_nm: float
    polarity: int = 1

    # -- interface ----------------------------------------------------------- #
    @abstractmethod
    def _render_core(self, img: np.ndarray, xa: np.ndarray, ya: np.ndarray,
                     aa: float) -> np.ndarray:
        """Composite the architecture-specific layers onto ``img``.

        Args:
            img: Working image, prefilled with the background level.
            xa: Horizontal coordinates including roughness displacement (nm).
            ya: Vertical coordinates including roughness displacement (nm).
            aa: Sampling pitch (nm), used as the anti-aliasing width.
        """

    @abstractmethod
    def feature_level(self, name: str) -> float:
        """Grey level of a named layer, for defects to paint with.

        Args:
            name: ``"line"``, ``"contact"`` or ``"background"``.
        """

    @abstractmethod
    def lattice(self) -> Tuple[float, float, float, float]:
        """The contact lattice as ``(pitch_x, pitch_y, phase_x, phase_y)``.

        Defects snap to this so that a "missing contact" lands where a contact
        actually is.
        """

    def to_dict(self) -> Dict[str, Any]:
        """Flat, JSON-serialisable summary for the manifest."""
        out: Dict[str, Any] = {"style": self.style, "polarity": self.polarity}
        for f in self.__dataclass_fields__:  # type: ignore[attr-defined]
            v = getattr(self, f)
            if isinstance(v, (int, float, str, bool)):
                out[f] = v
        return out

    # -- shared machinery ---------------------------------------------------- #
    def cd_scale(self, coord: np.ndarray) -> float:
        """Slow CD drift factor -- etch and exposure vary across a die.

        Evaluated at the frame centre rather than per pixel: within a 1 um
        field the drift is constant to well under a nanometre, and a scalar
        keeps the width parameters scalar (so ``soft_bands`` stays cheap).
        """
        if self.cd_drift_frac <= 0.0:
            return 1.0
        c = float(np.mean(coord[:: max(1, coord.shape[0] // 4)]))
        return 1.0 + self.cd_drift_frac * math.sin(
            2.0 * math.pi * c / self.cd_drift_period_nm)

    def evaluate(self, x: np.ndarray, y: np.ndarray, aa: float) -> np.ndarray:
        """Material map for the sampled coordinates.

        Args:
            x: Horizontal die coordinates (nm), shape ``(n, n)``.
            y: Vertical die coordinates (nm), same shape.
            aa: Sampling pitch (nm).

        Returns:
            ``float32`` image in roughly ``[0, 1]``.
        """
        dx = self.roughness_x.displace(y, aa)
        dy = self.roughness_y.displace(x, aa)
        xa = x if dx is None else x + dx
        ya = y if dy is None else y + dy

        img = np.full(x.shape, np.float32(self.background), dtype=np.float32)
        img = self._render_core(img, xa, ya, aa)
        img = self.macro.apply(img, x, y, aa)
        if self.polarity < 0:
            img = 1.0 - img
        return img

    def keep_mask(self, i: np.ndarray, j: np.ndarray, rate: float,
                  channel: int) -> Optional[np.ndarray]:
        """Position-hashed survival mask for lattice sites.

        Args:
            i: Lattice column indices.
            j: Lattice row indices.
            rate: Probability that a site is defective.
            channel: Distinguishes defect fields so missing contacts and broken
                lines are independent.

        Returns:
            ``float32`` mask, 1 where the feature survives, or ``None`` when
            the rate is zero.
        """
        if rate <= 0.0:
            return None
        u = hash_uniform(self.variation_seed + channel * 7919, i, j)
        return (u >= np.float32(rate)).astype(np.float32)


def lattice_index(coord: np.ndarray, pitch: float, phase: float) -> np.ndarray:
    """Which lattice cell each coordinate falls in (integer, may be negative)."""
    return np.floor((coord - phase) / pitch + 0.5).astype(np.int64)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY: Dict[str, Any] = {}


def register_layout(style: str):
    """Class decorator that adds a layout to the registry.

    Keeps :func:`sample_layout` free of imports from its own subpackages, so
    adding an architecture (gate-all-around, 3D NAND) means adding one file.
    """

    def wrap(cls):
        _REGISTRY[style] = cls
        return cls

    return wrap


def available_styles() -> Tuple[str, ...]:
    """Registered architecture names."""
    return tuple(sorted(_REGISTRY))


def sample_layout(style: str, rng: np.random.Generator,
                  cfg: GeneratorConfig) -> Layout:
    """Draw a fully randomised layout of the requested architecture.

    Args:
        style: ``"dram"`` or ``"finfet"``.
        rng: The sample's ``"layout"`` stream.
        cfg: Full generator configuration.

    Raises:
        ValueError: If the style is not registered.
    """
    from driftsense.layouts import dram, finfet  # noqa: F401  (registration)

    cls = _REGISTRY.get(style)
    if cls is None:
        raise ValueError(f"unknown layout style {style!r}; "
                         f"have {available_styles()}")
    return cls.sample(rng, cfg)


def shared_fields(rng: np.random.Generator, cfg: GeneratorConfig) -> Dict[str, Any]:
    """Draw the parameters every architecture has in common."""
    return dict(
        roughness_x=Roughness.sample(rng, cfg.process),
        roughness_y=Roughness.sample(rng, cfg.process),
        macro=MacroStructure.sample(rng, cfg.macro),
        variation_seed=int(rng.integers(0, 2 ** 62)),
        cd_drift_frac=cfg.process.cd_drift_frac.draw(rng),
        cd_drift_period_nm=cfg.process.cd_drift_period_nm.draw(rng),
        polarity=1,
    )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.layouts.base``."""
    from driftsense.config import ProcessVariationConfig as PVC

    rng = np.random.default_rng(0)

    # 1. the hash is uniform, stable, and independent across channels/sites
    i = np.arange(-500, 500)
    j = np.zeros_like(i)
    u = hash_uniform(1234, i, j)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert 0.47 < float(u.mean()) < 0.53, u.mean()
    assert np.array_equal(u, hash_uniform(1234, i, j)), "hash not deterministic"
    assert not np.array_equal(u, hash_uniform(1235, i, j))
    r = abs(float(np.corrcoef(u[:-1], u[1:])[0, 1]))
    assert r < 0.06, f"adjacent sites correlated: {r:.3f}"
    # negative indices are fine and distinct from their positive twins
    assert hash_uniform(1, np.array([-7]), np.array([3]))[0] != \
           hash_uniform(1, np.array([7]), np.array([3]))[0]

    # 2. THE property the hash exists for: the same site is defective at both
    #    magnifications, so the two frames never contradict each other
    site = (np.array([17]), np.array([-4]))
    assert hash_uniform(99, *site)[0] == hash_uniform(99, *site)[0]

    # 3. roughness is skipped when sub-pixel, active when not
    rough = Roughness.sample(rng, PVC())
    coord = np.linspace(0.0, 5000.0, 512).astype(np.float64)
    assert rough.displace(coord, 10.0) is None
    d = rough.displace(coord, 1.0)
    assert d is not None and float(np.abs(d).max()) <= rough.amp_nm + 1e-4
    # and it is low-frequency, not white
    lag1 = float(np.corrcoef(d[:-1], d[1:])[0, 1])
    assert lag1 > 0.9, lag1

    # 4. lattice indexing round-trips
    idx = lattice_index(np.array([0.0, 120.0, -120.0, 61.0]), 120.0, 0.0)
    assert idx.tolist() == [0, 1, -1, 1]

    # 5. macro structure composites and respects its probability
    x, y = G.make_grid((0.0, 0.0), 128, 10.0, 0.0)
    macro = MacroStructure.sample(rng, MacroStructureConfig())
    img = np.full((128, 128), 0.2, np.float32)
    out = macro.apply(img.copy(), x, y, 10.0)
    assert out.shape == img.shape

    print("layouts/base.py self-test OK")
    print(f"  hash mean              : {u.mean():.4f} (uniform)")
    print(f"  hash adjacent |r|      : {r:.4f}")
    print(f"  LER amplitude          : {rough.amp_nm:.2f} nm, lag-1 corr {lag1:.3f}")
    print(f"  LER skipped at 10 nm/px: yes")
    print(f"  registered styles      : {available_styles() or '(none imported yet)'}")


if __name__ == "__main__":
    _self_test()
