"""Probe point-spread function: focus, astigmatism and scan smear.

Three distinct blurs, because they are three distinct physical effects and a
network trained on only the first learns an isotropic, symmetric notion of
"same structure" that breaks the moment the test tool is slightly mis-tuned.

* **Focus blur** -- the finite probe diameter.  Isotropic Gaussian.
* **Astigmatism** -- the probe focuses at different working distances along two
  perpendicular axes, so the spot is an ellipse at an arbitrary angle.  This is
  the single most common real SEM mis-tune, and it is directional: it changes
  which features survive demagnification, not just how many.
* **Motion smear** -- stage or beam movement during the dwell, a line kernel
  along the direction of travel.

The reference and search frames draw these independently, so the model cannot
assume the two captures share a PSF.  On a real tool they never do: they are
taken at different magnifications, minutes or hours apart.

References: L. Reimer, *Scanning Electron Microscopy*, 2nd ed., Springer 1998
(probe formation); J. Goldstein et al., *Scanning Electron Microscopy and X-Ray
Microanalysis*, 4th ed., Springer 2018 (astigmatism correction and its imaging
signature); D. C. Joy, *J. Microscopy* 208 (2002) on measuring SEM resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from driftsense import geometry as G
from driftsense.config import OpticsConfig


@dataclass(frozen=True)
class ProbeParams:
    """Sampled point-spread function for one frame.

    Attributes:
        sigma_px: Geometric-mean Gaussian width of the probe, in pixels.
        astig_ratio: Ratio of major to minor axis; ``1.0`` is isotropic.
        astig_angle_deg: Orientation of the major axis.
        motion_px: Length of the motion smear in pixels; ``0`` disables it.
        motion_angle_deg: Direction of the smear.
    """

    sigma_px: float
    astig_ratio: float = 1.0
    astig_angle_deg: float = 0.0
    motion_px: float = 0.0
    motion_angle_deg: float = 0.0

    @classmethod
    def sample(cls, rng: np.random.Generator, cfg: OpticsConfig) -> "ProbeParams":
        """Draw a PSF for one frame.

        Args:
            rng: This frame's optics stream.
            cfg: The frame's :class:`~driftsense.config.OpticsConfig`.
        """
        motion = 0.0
        if rng.random() < cfg.motion_blur_probability:
            motion = cfg.motion_blur_px.draw(rng)
        return cls(
            sigma_px=cfg.psf_sigma_px.draw(rng),
            astig_ratio=cfg.astigmatism_ratio.draw(rng),
            astig_angle_deg=cfg.astigmatism_angle_deg.draw(rng),
            motion_px=motion,
            motion_angle_deg=float(rng.uniform(0.0, 180.0)),
        )

    @property
    def is_astigmatic(self) -> bool:
        """Whether the elliptical path is needed at all."""
        return self.astig_ratio > 1.05

    def apply(self, img: np.ndarray) -> np.ndarray:
        """Convolve a material map with this PSF.

        Args:
            img: ``float32`` image.

        Returns:
            A blurred ``float32`` image.  Separable Gaussian is used for the
            isotropic case because it is O(N) via OpenCV; the elliptical case
            needs a genuine 2-D kernel.
        """
        out = np.ascontiguousarray(img, dtype=np.float32)
        if self.is_astigmatic:
            # Keep the geometric mean equal to sigma_px so astigmatism changes
            # the shape of the probe without changing its area -- otherwise
            # "more astigmatism" would silently mean "more blur" and the two
            # effects could not be separated.
            root = math.sqrt(self.astig_ratio)
            kernel = G.elliptical_gaussian_kernel(self.sigma_px * root,
                                                  self.sigma_px / root,
                                                  self.astig_angle_deg)
            out = G.convolve(out, kernel)
        else:
            out = G.gaussian_blur(out, self.sigma_px)

        if self.motion_px > 0.15:
            out = G.convolve(out, G.line_kernel(self.motion_px,
                                                self.motion_angle_deg))
        return out

    def to_dict(self) -> Dict[str, Any]:
        """Manifest view."""
        return {
            "psf_sigma_px": round(self.sigma_px, 4),
            "astig_ratio": round(self.astig_ratio, 4),
            "astig_angle_deg": round(self.astig_angle_deg, 2),
            "motion_px": round(self.motion_px, 4),
            "motion_angle_deg": round(self.motion_angle_deg, 2),
        }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.optics.psf``."""
    from driftsense.config import GeneratorConfig

    cfg = GeneratorConfig()

    # 1. sampling respects the configured ranges and is reproducible
    p1 = ProbeParams.sample(np.random.default_rng(4), cfg.search_optics)
    p2 = ProbeParams.sample(np.random.default_rng(4), cfg.search_optics)
    assert p1 == p2
    assert cfg.search_optics.psf_sigma_px.lo <= p1.sigma_px <= cfg.search_optics.psf_sigma_px.hi

    # 2. blur reduces high-frequency energy monotonically
    rng = np.random.default_rng(0)
    img = rng.random((256, 256)).astype(np.float32)
    sharp = float(G.gradient_magnitude(img).mean())
    prev = sharp
    for s in (0.5, 1.0, 2.0, 4.0):
        blurred = ProbeParams(sigma_px=s).apply(img)
        cur = float(G.gradient_magnitude(blurred).mean())
        assert cur < prev, (s, cur, prev)
        prev = cur
    assert float(np.abs(ProbeParams(sigma_px=2.0).apply(img).mean()
                        - img.mean())) < 0.01, "blur must preserve mean"

    # 3. astigmatism is genuinely directional: a horizontal edge and a vertical
    #    edge must blur by different amounts
    step = np.zeros((128, 128), np.float32)
    step[:, 64:] = 1.0                                   # vertical edge
    step_t = step.T.copy()                               # horizontal edge
    astig = ProbeParams(sigma_px=2.0, astig_ratio=3.0, astig_angle_deg=0.0)
    gx = float(G.gradient_magnitude(astig.apply(step)).max())
    gy = float(G.gradient_magnitude(astig.apply(step_t)).max())
    assert abs(gx - gy) / max(gx, gy) > 0.15, (gx, gy)
    iso = ProbeParams(sigma_px=2.0, astig_ratio=1.0)
    ix = float(G.gradient_magnitude(iso.apply(step)).max())
    iy = float(G.gradient_magnitude(iso.apply(step_t)).max())
    assert abs(ix - iy) / max(ix, iy) < 0.05, (ix, iy)

    # 4. astigmatism preserves total blur "area": the isotropic and elliptical
    #    probes of the same sigma must smooth a random field comparably
    e_iso = float(G.gradient_magnitude(iso.apply(img)).mean())
    e_ast = float(G.gradient_magnitude(astig.apply(img)).mean())
    assert 0.5 < e_ast / e_iso < 1.6, e_ast / e_iso

    # 5. motion smear is directional and preserves the mean
    motion = ProbeParams(sigma_px=0.3, motion_px=9.0, motion_angle_deg=0.0)
    m = motion.apply(step)
    assert abs(float(m.mean() - step.mean())) < 0.02
    m_along = motion.apply(step_t)                        # smear along the edge
    assert float(G.gradient_magnitude(m_along).max()) > \
           float(G.gradient_magnitude(m).max())

    # 6. dtype and shape discipline
    out = p1.apply(img)
    assert out.dtype == np.float32 and out.shape == img.shape

    import json
    json.dumps(p1.to_dict())

    print("optics/psf.py self-test OK")
    print(f"  sampled sigma          : {p1.sigma_px:.3f} px "
          f"(astig x{p1.astig_ratio:.2f} @ {p1.astig_angle_deg:.0f} deg)")
    print(f"  edge sharpness ratio   : astigmatic {gx/gy:.2f}, isotropic {ix/iy:.2f}")
    print(f"  blur monotonic in sigma: yes (0.5 -> 4.0 px)")


if __name__ == "__main__":
    _self_test()
