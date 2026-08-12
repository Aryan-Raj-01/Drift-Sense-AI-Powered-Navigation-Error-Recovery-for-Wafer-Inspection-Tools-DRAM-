"""Sensor noise: shot, read, stripe and quantisation.

The brief's first mandatory requirement is *independent* sensor noise on the two
frames, and the reason is worth spelling out because it is the difference
between a dataset that teaches matching and one that teaches cheating.  If the
reference and the search image share a noise realisation, then the noise itself
is a correspondence signal: cross-correlating the noise alone localises the
patch perfectly.  A network will find that shortcut, score beautifully on your
validation set, and collapse on the hidden test set where the two captures are
genuinely separate physical events.  Independence here is enforced structurally
-- the two frames are handed different RNG streams by
:class:`driftsense.rng.SeedBook` and never touch each other's.

The noise model itself is not one Gaussian:

* **Shot noise** dominates and is Poisson.  Secondary-electron emission is a
  counting process, so the variance equals the mean and the SNR goes as the
  square root of the dose.  This is why a low-dose wide scan is grainy in a way
  that no additive Gaussian reproduces: the noise is signal-dependent, so dark
  regions are *quieter* than bright ones.  (Reimer 1998; Bunday et al.,
  Proc. SPIE 5375 (2004) on the CD-SEM dose/noise trade-off.)
* **Read noise** is the detector and amplifier floor: additive, Gaussian,
  signal-independent.
* **Stripe noise** is per-scan-line offset from 1/f noise in the scan amplifier
  -- horizontal banding, strongly correlated along a row and independent between
  rows.  Whitening it away is easy for a network only if it has seen it.
* **Quantisation** to 8 bits, because that is what the tool stores.

The search frame gets far less dose than the reference: it covers 100x the area
in a comparable frame time.  That asymmetry is physical, and it is also exactly
what the hackathon says the hidden test set will exaggerate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from driftsense.config import OpticsConfig

#: Above this mean count the Gaussian limit of the Poisson distribution is
#: indistinguishable from the real thing (relative error < 1 %), and roughly 3x
#: cheaper to sample over a million pixels.
GAUSSIAN_LIMIT_DOSE = 150.0


@dataclass(frozen=True)
class NoiseParams:
    """Sampled sensor noise for one frame.

    Attributes:
        dose: Mean electrons per pixel at unit signal.  The single most
            important robustness knob in the whole generator.
        read: Standard deviation of additive read noise, in signal units.
        stripe: Standard deviation of per-row offsets, in signal units.
    """

    dose: float
    read: float
    stripe: float

    @classmethod
    def sample(cls, rng: np.random.Generator, cfg: OpticsConfig,
               noise_boost: float = 1.0) -> "NoiseParams":
        """Draw noise parameters for one frame.

        Args:
            rng: This frame's optics stream.
            cfg: The frame's optics configuration.
            noise_boost: Multiplier for how much worse this frame should be.
                Dose is divided by it and the additive terms multiplied, so a
                single number moves the whole SNR.  Used to build the
                "hidden test is noisier" evaluation set.
        """
        boost = max(noise_boost, 1e-6)
        return cls(
            dose=cfg.dose_e_per_px.draw(rng) / boost,
            read=cfg.read_noise.draw(rng) * boost,
            stripe=cfg.stripe_noise.draw(rng) * boost,
        )

    @property
    def shot_snr(self) -> float:
        """Approximate SNR from shot noise alone at half-scale signal."""
        return float(np.sqrt(max(self.dose, 1e-9) * 0.5))

    def apply(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Add shot, read and stripe noise.

        Args:
            img: Video-stage image (``float32``), nominally ``[0, 1]``.
            rng: This frame's optics stream.  Must not be shared with the other
                frame -- see the module docstring.

        Returns:
            ``float32`` noisy image, still unquantised.
        """
        out = np.ascontiguousarray(img, dtype=np.float32)

        # --- shot noise ----------------------------------------------------- #
        lam = np.clip(out, 0.0, 4.0) * np.float32(self.dose)
        if self.dose >= GAUSSIAN_LIMIT_DOSE:
            out = (lam + np.sqrt(lam) * rng.standard_normal(lam.shape,
                                                            dtype=np.float32))
            out /= np.float32(self.dose)
        else:
            out = rng.poisson(lam).astype(np.float32) / np.float32(self.dose)

        # --- read noise ----------------------------------------------------- #
        if self.read > 0.0:
            out += rng.normal(0.0, self.read, out.shape).astype(np.float32)

        # --- scan-line offsets ---------------------------------------------- #
        if self.stripe > 0.0:
            out += rng.normal(0.0, self.stripe,
                              (out.shape[0], 1)).astype(np.float32)

        return out

    def to_dict(self) -> Dict[str, Any]:
        """Manifest view."""
        return {
            "dose_e_per_px": round(self.dose, 3),
            "read_noise": round(self.read, 5),
            "stripe_noise": round(self.stripe, 5),
            "shot_snr": round(self.shot_snr, 3),
        }


def quantise(img: np.ndarray) -> np.ndarray:
    """Clip to ``[0, 1]`` and convert to 8-bit, as the tool stores it.

    Args:
        img: ``float32`` image in signal units.

    Returns:
        ``uint8`` image.
    """
    return np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.optics.noise``."""
    import json

    from driftsense.config import GeneratorConfig
    from driftsense.rng import SeedBook

    cfg = GeneratorConfig()
    p = NoiseParams.sample(np.random.default_rng(12), cfg.search_optics)
    assert p == NoiseParams.sample(np.random.default_rng(12), cfg.search_optics)

    flat = np.full((512, 512), 0.5, np.float32)

    # 1. shot noise variance scales as 1/dose -- the defining property
    for dose in (30.0, 300.0, 3000.0):
        n = NoiseParams(dose=dose, read=0.0, stripe=0.0)
        v = float(n.apply(flat, np.random.default_rng(1)).var())
        expected = 0.5 / dose                       # Var(P(l)/l) = l/l^2 = 1/l
        assert 0.7 < v / expected < 1.4, (dose, v, expected)

    # 2. shot noise is signal-dependent: bright regions are noisier than dark
    ramp = np.tile(np.linspace(0.05, 0.95, 512, dtype=np.float32), (512, 1))
    noisy = NoiseParams(60.0, 0.0, 0.0).apply(ramp, np.random.default_rng(2))
    resid = noisy - ramp
    dark_var = float(resid[:, :100].var())
    bright_var = float(resid[:, -100:].var())
    assert bright_var > 2.0 * dark_var, (dark_var, bright_var)

    # 3. the Gaussian limit agrees with true Poisson where it is used
    hi = 400.0
    a = NoiseParams(hi, 0.0, 0.0).apply(flat, np.random.default_rng(4))
    true_poisson = (np.random.default_rng(4).poisson(0.5 * hi, size=flat.shape)
                    .astype(np.float32) / hi)
    assert abs(float(a.var()) - float(np.var(true_poisson))) / float(a.var()) < 0.15

    # 4. read noise is additive and signal-independent
    r = NoiseParams(1e9, 0.05, 0.0).apply(flat, np.random.default_rng(5))
    assert abs(float(r.std()) - 0.05) < 0.005

    # 5. stripes are constant along a row and independent between rows
    s = NoiseParams(1e9, 0.0, 0.04).apply(flat, np.random.default_rng(6))
    row_means = s.mean(axis=1)
    assert float(row_means.std()) > 0.02, "stripes missing"
    assert float(s.std(axis=1).mean()) < 0.005, "stripes not constant along rows"

    # 6. THE key property: two frames must get independent noise
    book = SeedBook(4242)
    n1 = NoiseParams(50.0, 0.03, 0.0)
    f1 = n1.apply(flat, book.stream("optics.reference"))
    f2 = n1.apply(flat, book.stream("optics.search"))
    corr = abs(float(np.corrcoef((f1 - 0.5).ravel(), (f2 - 0.5).ravel())[0, 1]))
    assert corr < 0.01, f"noise correlated between frames: r={corr:.4f}"

    # ... and the failure mode it guards against, for contrast
    shared = np.random.default_rng(77)
    g1 = n1.apply(flat, np.random.default_rng(77))
    g2 = n1.apply(flat, np.random.default_rng(77))
    assert float(np.corrcoef((g1 - 0.5).ravel(), (g2 - 0.5).ravel())[0, 1]) > 0.99

    # 7. noise_boost moves SNR in the right direction
    quiet = NoiseParams.sample(np.random.default_rng(3), cfg.search_optics, 1.0)
    loud = NoiseParams.sample(np.random.default_rng(3), cfg.search_optics, 2.0)
    assert loud.dose < quiet.dose and loud.read > quiet.read
    assert loud.shot_snr < quiet.shot_snr

    # 8. quantisation
    q = quantise(np.array([[-1.0, 0.0, 0.5, 1.0, 2.0]], np.float32))
    assert q.dtype == np.uint8
    assert q.tolist() == [[0, 0, 127, 255, 255]]

    json.dumps(p.to_dict())

    print("optics/noise.py self-test OK")
    print(f"  sampled dose           : {p.dose:.1f} e/px  (shot SNR {p.shot_snr:.1f})")
    print(f"  var(noise) ~ 1/dose    : verified at 30 / 300 / 3000 e/px")
    print(f"  signal-dependence      : bright/dark variance "
          f"{bright_var/dark_var:.1f}x")
    print(f"  cross-frame noise |r|  : {corr:.5f}  (shared-seed control: >0.99)")


if __name__ == "__main__":
    _self_test()
