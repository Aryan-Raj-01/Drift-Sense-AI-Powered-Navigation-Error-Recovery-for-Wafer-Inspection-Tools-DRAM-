"""Detector-side SEM effects: edge brightening, charging and shading.

These are what make a synthetic image *look* like an SEM rather than like a
rendered CAD layout, and each one is a specific physical mechanism:

* **Edge brightening.**  Secondary-electron yield rises as the local surface
  tilts away from normal, because the escape depth is shorter along an inclined
  path.  Feature sidewalls therefore appear as bright fringes.  This is the
  single most recognisable signature of an SE image and the brief calls it out
  explicitly.  Modelled as a gain on the gradient magnitude, blurred by the
  escape length.  (Reimer 1998, ch. 4; Goldstein et al. 2018, topographic
  contrast.)

* **Charging.**  Dielectric layers accumulate charge under the beam, which
  deflects incoming electrons and modulates the collected signal on a scale of
  micrometres.  It appears as a smooth multiplicative field, sometimes with
  bright blooming.  (J. Cazaux, *Scanning* 26 (2004).)

* **Illumination gradient and detector non-uniformity.**  The Everhart-Thornley
  detector sits to one side of the chamber, so collection efficiency varies
  across the field; the detector and video amplifier add their own fixed-pattern
  gain.  Both are low-frequency multiplicative terms.

Why they matter for robustness rather than realism alone: all three are
*multiplicative, low-frequency* nuisances.  A matcher that normalises locally is
immune to them; a matcher that compares raw intensities is not.  Including them
in training is what forces the network to learn the invariant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from driftsense import geometry as G
from driftsense.config import OpticsConfig


@dataclass(frozen=True)
class DetectorParams:
    """Sampled detector-side effects for one frame.

    Attributes:
        edge_gain: Strength of the secondary-electron edge fringe.
        edge_sigma_px: Width of the fringe, i.e. the escape length in pixels.
        charging: Amplitude of the multiplicative charging field, ``0``-``1``.
        charging_cells: Spatial frequency of that field (cells across the frame).
        nonuniformity: Amplitude of fixed-pattern detector gain variation.
        illumination_gradient: Peak-to-peak linear shading across the frame.
        illumination_angle_deg: Direction of that shading.
    """

    edge_gain: float
    edge_sigma_px: float
    charging: float
    charging_cells: int
    nonuniformity: float
    illumination_gradient: float
    illumination_angle_deg: float

    @classmethod
    def sample(cls, rng: np.random.Generator,
               cfg: OpticsConfig) -> "DetectorParams":
        """Draw detector effects for one frame."""
        return cls(
            edge_gain=cfg.edge_gain.draw(rng),
            edge_sigma_px=cfg.edge_sigma_px.draw(rng),
            charging=cfg.charging.draw(rng),
            charging_cells=cfg.charging_cells.draw(rng),
            nonuniformity=cfg.detector_nonuniformity.draw(rng),
            illumination_gradient=cfg.illumination_gradient.draw(rng),
            illumination_angle_deg=float(rng.uniform(0.0, 360.0)),
        )

    # -- individual effects -------------------------------------------------- #
    def edge_effect(self, img: np.ndarray) -> np.ndarray:
        """Add the SE edge fringe.

        The gradient is normalised by its own maximum before being scaled, so
        ``edge_gain`` means the same thing regardless of the layout's contrast.
        Without that normalisation a high-contrast DRAM array would get a
        fringe ten times stronger than a low-contrast one from the same
        parameter, and the parameter would stop being interpretable.
        """
        if self.edge_gain <= 0.0:
            return img
        g = G.gradient_magnitude(img)
        peak = float(g.max())
        if peak <= 1e-6:
            return img
        g /= peak
        return img + np.float32(self.edge_gain) * G.gaussian_blur(g, self.edge_sigma_px)

    def charging_field(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Smooth multiplicative charging field centred on 1.0."""
        if self.charging <= 0.0:
            return np.float32(1.0)
        f = G.smooth_random_field(rng, n, self.charging_cells,
                                  float(rng.uniform(25.0, 70.0)))
        return (1.0 - np.float32(self.charging) * (f - 0.5)).astype(np.float32)

    def shading_field(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Combined illumination gradient and fixed-pattern gain, centred on 1.0."""
        out = None
        if self.illumination_gradient > 0.0:
            th = math.radians(self.illumination_angle_deg)
            ax = np.linspace(-0.5, 0.5, n, dtype=np.float32)
            ramp = (np.cos(th) * ax[None, :] + np.sin(th) * ax[:, None])
            out = 1.0 + np.float32(self.illumination_gradient) * ramp
        if self.nonuniformity > 0.0:
            f = G.smooth_random_field(rng, n, int(rng.integers(10, 24)), 6.0)
            gain = 1.0 + np.float32(self.nonuniformity) * (f - 0.5) * 2.0
            out = gain if out is None else out * gain
        return np.float32(1.0) if out is None else out.astype(np.float32)

    def apply(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply every detector effect in physical order.

        Args:
            img: Blurred material map (``float32``).
            rng: This frame's optics stream.

        Returns:
            ``float32`` image, still in signal units (not yet quantised).
        """
        out = self.edge_effect(np.ascontiguousarray(img, dtype=np.float32))
        out = out * self.charging_field(rng, out.shape[0])
        out = out * self.shading_field(rng, out.shape[0])
        return out.astype(np.float32, copy=False)

    def to_dict(self) -> Dict[str, Any]:
        """Manifest view."""
        return {
            "edge_gain": round(self.edge_gain, 4),
            "edge_sigma_px": round(self.edge_sigma_px, 4),
            "charging": round(self.charging, 4),
            "charging_cells": int(self.charging_cells),
            "detector_nonuniformity": round(self.nonuniformity, 4),
            "illumination_gradient": round(self.illumination_gradient, 4),
            "illumination_angle_deg": round(self.illumination_angle_deg, 1),
        }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.optics.sem``."""
    import json

    from driftsense.config import GeneratorConfig

    cfg = GeneratorConfig()
    p = DetectorParams.sample(np.random.default_rng(8), cfg.search_optics)
    assert p == DetectorParams.sample(np.random.default_rng(8), cfg.search_optics)

    # 1. the edge fringe really is at the edge, and really is brighter
    step = np.zeros((128, 128), np.float32)
    step[:, 64:] = 0.8
    e = DetectorParams(0.6, 1.5, 0.0, 5, 0.0, 0.0, 0.0).edge_effect(step)
    band = e[:, 60:68]
    flat_dark = e[:, :40]
    flat_bright = e[:, 90:]
    assert float(band.max()) > float(flat_bright.max()) + 0.05, "no edge fringe"
    assert float(np.abs(flat_dark - step[:, :40]).max()) < 1e-5, "fringe leaked"

    # 2. edge gain is calibrated against contrast, not absolute intensity: the
    #    same gain on a 10x weaker pattern gives the same relative fringe
    weak = step * 0.1
    e_weak = DetectorParams(0.6, 1.5, 0.0, 5, 0.0, 0.0, 0.0).edge_effect(weak)
    lift_strong = float(e[:, 60:68].max() - step.max())
    lift_weak = float(e_weak[:, 60:68].max() - weak.max())
    assert abs(lift_strong - lift_weak) < 0.1, (lift_strong, lift_weak)

    # 3. charging is multiplicative, smooth and mean-preserving to first order
    rng = np.random.default_rng(2)
    field = DetectorParams(0.0, 1.0, 0.25, 6, 0.0, 0.0, 0.0).charging_field(rng, 256)
    assert 0.9 < float(field.mean()) < 1.1
    assert float(field.std()) > 0.01
    # smooth: neighbouring pixels are nearly identical
    assert float(np.abs(np.diff(field, axis=1)).mean()) < 0.01

    # 4. shading really shades, in the requested direction
    sh = DetectorParams(0.0, 1.0, 0.0, 5, 0.0, 0.4, 0.0).shading_field(rng, 128)
    assert float(sh[:, -1].mean()) > float(sh[:, 0].mean())
    sh90 = DetectorParams(0.0, 1.0, 0.0, 5, 0.0, 0.4, 90.0).shading_field(rng, 128)
    assert float(sh90[-1, :].mean()) > float(sh90[0, :].mean())

    # 5. the full chain keeps dtype/shape and stays finite
    img = np.random.default_rng(1).random((256, 256)).astype(np.float32) * 0.6 + 0.2
    out = p.apply(img, np.random.default_rng(3))
    assert out.dtype == np.float32 and out.shape == img.shape
    assert np.isfinite(out).all()

    # 6. every effect is low-frequency EXCEPT the edge fringe: a locally
    #    normalised view of the image must be nearly unchanged by charging and
    #    shading, which is precisely the invariance we want the model to learn
    plain = DetectorParams(0.0, 1.0, 0.3, 5, 0.05, 0.15, 30.0)
    nuisance = plain.apply(img, np.random.default_rng(3))
    a = (img - G.gaussian_blur(img, 12.0))
    b = (nuisance - G.gaussian_blur(nuisance, 12.0))
    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    assert r > 0.97, f"nuisance terms are not low-frequency: r={r:.3f}"

    json.dumps(p.to_dict())

    print("optics/sem.py self-test OK")
    print(f"  edge gain / sigma      : {p.edge_gain:.2f} / {p.edge_sigma_px:.2f} px")
    print(f"  fringe lift            : {lift_strong:.3f} (strong) "
          f"{lift_weak:.3f} (10x weaker pattern)")
    print(f"  charging field         : mean {float(field.mean()):.3f}, "
          f"std {float(field.std()):.3f}")
    print(f"  high-pass invariance   : r = {r:.4f}")


if __name__ == "__main__":
    _self_test()
