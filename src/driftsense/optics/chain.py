"""The SEM capture chain: material map in, 8-bit image out.

This module exists so that the *order* of the imaging chain is written down
exactly once:

    PSF  ->  edge effect + charging + shading  ->  video gain/gamma
         ->  shot noise  ->  read noise  ->  stripes  ->  quantisation

Order is not a detail here, it is the difference between a dataset that
generalises and one that does not.

**Blur must precede noise.**  Convolving after the noise is added leaves the
noise spatially correlated with a kernel that is a direct function of the blur
parameter.  A network can then read the magnification, the focus, or worse the
*correspondence* between two frames straight out of the noise autocorrelation --
a cue that is an artefact of generation order and does not exist in a real
capture, where the detector noise enters after the optics.  The self-test below
measures this explicitly.

**Noise must be the last thing before quantisation.**  Any smooth multiplicative
term applied afterwards (charging, shading) would scale the noise with it and
make the effective SNR position-dependent in a way the physics does not produce.

Each frame of a pair gets its own :class:`CaptureParams` *and* its own RNG
stream.  Nothing in this module ever sees both frames, which is the structural
guarantee behind the brief's independent-noise requirement.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as _replace
from typing import Any, Dict

import numpy as np

from driftsense.config import OpticsConfig
from driftsense.optics.noise import NoiseParams, quantise
from driftsense.optics.psf import ProbeParams
from driftsense.optics.sem import DetectorParams
from driftsense.optics.video import VideoParams


@dataclass(frozen=True)
class CaptureParams:
    """Everything that turns one material map into one 8-bit frame.

    Attributes:
        probe: Point-spread function.
        detector: Edge effect, charging and shading.
        video: Contrast, brightness, gamma, polarity.
        noise: Shot, read and stripe noise.
    """

    probe: ProbeParams
    detector: DetectorParams
    video: VideoParams
    noise: NoiseParams

    @classmethod
    def sample(cls, rng: np.random.Generator, cfg: OpticsConfig,
               noise_boost: float = 1.0) -> "CaptureParams":
        """Draw a complete imaging configuration for one frame.

        Args:
            rng: This frame's optics stream (``"optics.reference"`` or
                ``"optics.search"``).  Never share it between frames.
            cfg: The frame's :class:`~driftsense.config.OpticsConfig`.
            noise_boost: SNR degradation factor; see
                :meth:`~driftsense.optics.noise.NoiseParams.sample`.
        """
        return cls(
            probe=ProbeParams.sample(rng, cfg),
            detector=DetectorParams.sample(rng, cfg),
            video=VideoParams.sample(rng, cfg),
            noise=NoiseParams.sample(rng, cfg, noise_boost),
        )

    def apply(self, material: np.ndarray, rng: np.random.Generator,
              quantised: bool = True) -> np.ndarray:
        """Run the full chain on a material map.

        Args:
            material: ``float32`` material map from
                :meth:`driftsense.layouts.base.Layout.evaluate`, with landmarks
                and defects already composited in.
            rng: This frame's optics stream.
            quantised: Return ``uint8`` (the default, matching what a tool
                stores) or ``float32`` signal units.

        Returns:
            ``uint8`` image, or ``float32`` if ``quantised`` is ``False``.
        """
        img = self.probe.apply(material)
        img = self.detector.apply(img, rng)
        img = self.video.apply(img)
        img = self.noise.apply(img, rng)
        return quantise(img) if quantised else img

    def to_dict(self, prefix: str = "") -> Dict[str, Any]:
        """Flat manifest view, optionally prefixed (``"ref_"`` / ``"search_"``)."""
        out: Dict[str, Any] = {}
        for part in (self.probe, self.detector, self.video, self.noise):
            for k, v in part.to_dict().items():
                out[f"{prefix}{k}"] = v
        return out


def sample_pair(book, cfg, noise_boost: float = 1.0):
    """Draw the reference and search capture parameters for one sample.

    Args:
        book: The sample's :class:`~driftsense.rng.SeedBook`.
        cfg: The full :class:`~driftsense.config.GeneratorConfig`.
        noise_boost: Applied to the search frame only -- the reference is a
            deliberate, dwelled capture on a known target and does not degrade
            the same way.

    Returns:
        ``(reference_params, search_params)``.

    Note:
        Contrast polarity is drawn ONCE and applied to both frames.  Whether
        features come out bright on dark depends on detector selection, landing
        energy and the material stack -- properties of the recipe and the wafer,
        not of the individual capture.  Two revisits of the same site on the same
        tool share it.  Drawing it per frame (as an earlier version did) made
        ~12 % of samples arrive with mismatched polarity, which is not a
        realistic acquisition and which inverts the sign of every correlation
        based verification.  Randomising it per SAMPLE still forces polarity
        invariance across the dataset, which is the robustness that matters.
    """
    ref = CaptureParams.sample(book.stream("optics.reference"), cfg.ref_optics,
                               noise_boost=1.0)
    search = CaptureParams.sample(book.stream("optics.search"), cfg.search_optics,
                                  noise_boost=noise_boost)

    invert = bool(book.fresh("polarity").random() < cfg.search_optics.invert_probability)
    ref = _replace(ref, video=_replace(ref.video, invert=invert))
    search = _replace(search, video=_replace(search.video, invert=invert))
    return ref, search


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.optics.chain``."""
    import json

    from driftsense import geometry as G
    from driftsense.config import GeneratorConfig
    from driftsense.layouts.base import sample_layout
    from driftsense.optics.acquisition import FrameGeometry
    from driftsense.rng import SeedBook

    cfg = GeneratorConfig()
    book = SeedBook(31337)
    layout = sample_layout("dram", book.stream("layout"), cfg)

    fine = FrameGeometry(center_nm=(0.0, 0.0), px_nm=1.0, theta=0.0, n=1000)
    coarse = FrameGeometry(center_nm=(0.0, 0.0), px_nm=10.0, theta=0.0, n=1000)
    mat_fine = layout.evaluate(*fine.grid(), 1.0)
    mat_coarse = layout.evaluate(*coarse.grid(), 10.0)

    ref_p, search_p = sample_pair(book, cfg)
    ref = ref_p.apply(mat_fine, book.stream("optics.reference"))
    search = search_p.apply(mat_coarse, book.stream("optics.search"))

    # 1. output discipline
    for im in (ref, search):
        assert im.dtype == np.uint8 and im.shape == (1000, 1000)
        assert float((im >= 254).mean()) < 0.06, "saturating highlights"
        assert float((im <= 1).mean()) < 0.06, "crushed blacks"
        assert 40 < float(im.mean()) < 215
        assert float(im.std()) > 12, "no contrast left"

    # 2. the search frame is the noisier of the two, as the physics demands
    assert search_p.noise.dose < ref_p.noise.dose

    # 3. reproducibility: same seed, same bytes
    b2 = SeedBook(31337)
    _ = sample_layout("dram", b2.stream("layout"), cfg)
    r2, s2 = sample_pair(b2, cfg)
    assert np.array_equal(ref, r2.apply(mat_fine, b2.stream("optics.reference")))
    assert np.array_equal(search, s2.apply(mat_coarse, b2.stream("optics.search")))

    # 4. independent noise end to end.  Isolating the noise matters here: two
    #    renders of the same die correlate strongly through their SHARED
    #    STRUCTURE, which is not a defect.  So subtract a noise-free render of
    #    the identical chain and correlate what is left, which is noise alone.
    from dataclasses import replace as _replace

    quiet = _replace(search_p, noise=NoiseParams(dose=1e12, read=0.0, stripe=0.0))
    clean = quiet.apply(mat_coarse, SeedBook(5).stream("optics.search"),
                        quantised=False)
    fa = search_p.apply(mat_coarse, SeedBook(5).stream("optics.reference"),
                        quantised=False) - clean
    fb = search_p.apply(mat_coarse, SeedBook(5).stream("optics.search"),
                        quantised=False) - clean
    cross = abs(float(np.corrcoef(fa.ravel(), fb.ravel())[0, 1]))
    assert cross < 0.15, f"frames share noise structure: r={cross:.3f}"

    # 5. ORDER TEST: with blur before noise the residual noise is white; with
    #    noise before blur it is strongly autocorrelated and leaks the kernel.
    flat = np.full((512, 512), 0.5, np.float32)
    correct = search_p.noise.apply(search_p.probe.apply(flat),
                                   np.random.default_rng(1))
    wrong = search_p.probe.apply(search_p.noise.apply(flat,
                                                      np.random.default_rng(1)))

    def lag1(a: np.ndarray) -> float:
        a = a - a.mean()
        return abs(float(np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1]))

    assert lag1(correct) < 0.25, lag1(correct)
    assert lag1(wrong) > lag1(correct) + 0.2, (lag1(wrong), lag1(correct))

    # 6. the pair still corresponds after the whole chain -- this is the end-to-
    #    end statement that the two magnifications image the same die
    m = 60
    small = G.box_downsample(ref.astype(np.float32), 10)
    crop = search[500 - 50:500 + 50, 500 - 50:500 + 50].astype(np.float32)
    a = small[10:-10, 10:-10]
    b = crop[10:-10, 10:-10]
    zncc = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    assert zncc > 0.3, f"pair does not correspond: {zncc:.3f}"

    d = ref_p.to_dict("ref_")
    d.update(search_p.to_dict("search_"))
    json.dumps(d)
    assert len(d) == 2 * len(ref_p.to_dict())

    print("optics/chain.py self-test OK")
    print(f"  ref   : dose {ref_p.noise.dose:7.1f} e/px  sigma "
          f"{ref_p.probe.sigma_px:.2f} px  mean {ref.mean():5.1f}  "
          f"std {ref.std():5.1f}")
    print(f"  search: dose {search_p.noise.dose:7.1f} e/px  sigma "
          f"{search_p.probe.sigma_px:.2f} px  mean {search.mean():5.1f}  "
          f"std {search.std():5.1f}")
    print(f"  cross-frame noise |r|  : {cross:.4f}")
    print(f"  noise lag-1 corr       : {lag1(correct):.3f} (blur->noise, correct) "
          f"vs {lag1(wrong):.3f} (noise->blur, wrong)")
    print(f"  ref/10 vs search ZNCC  : {zncc:.3f}")
    print(f"  manifest fields        : {len(d)}")


if __name__ == "__main__":
    _self_test()
