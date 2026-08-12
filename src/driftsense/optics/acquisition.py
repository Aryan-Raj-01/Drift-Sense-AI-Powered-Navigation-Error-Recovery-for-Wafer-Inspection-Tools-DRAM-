"""Acquisition geometry: how the tool actually lands on the die.

This module owns every error between "where the tool meant to look" and "where
the beam went", and it expresses all of them as a warp of the SAMPLING GRID.

That choice is the reason the ground truth in this dataset is exact.  The usual
pipeline renders a frame and then augments it -- ``cv2.warpAffine`` for
rotation, a resize for magnification error, ``remap`` for distortion.  Each of
those resamples the image, which:

* leaves interpolation blur, ringing and black corners that correlate with the
  augmentation parameters, giving the network shortcut cues that do not exist
  on the hidden test set;
* moves the target by a sub-pixel amount the label does not know about, so the
  recorded centre drifts from the true one -- while the hackathon scores in
  pixels.

Warping the grid has neither problem.

Label fidelity under a warped grid
----------------------------------
A warped grid raises its own question: if jitter, drift and distortion move the
beam, the target is no longer at the *nominal* pixel, and a label computed from
the rigid mapping is wrong by exactly that displacement.  Measured on the first
draft of this module: up to 3 px.  That would put a 3 px floor under every
score in a task judged in pixels.

The fix is why the distortion model here is a **sum of sinusoids rather than a
smoothed random field**: it can be evaluated at a single point in O(1), so the
warp is invertible.  :meth:`FrameGeometry.locate` runs a few fixed-point
iterations of ``u <- u_target - delta(u)`` and returns the pixel that actually
samples the target.  The label then describes the pixels rather than the
intention, and distortion stays a genuine augmentation instead of label noise.

Modelled errors
---------------
============================  ====================================================
stage rotation                 revisit orientation error between the two captures
magnification error            calibration drift; footprint is never exactly 100 px
per-row scan jitter            AR(1) mechanical vibration, line-to-line
intra-frame drift              thermal expansion during the scan (top row -> bottom)
scan-field distortion          smooth deflection non-linearity, bounded ~1 px
============================  ====================================================

The last one is what the brief calls "local elastic deformation".  It is
deliberately bounded, low-frequency and invertible: a die is rigid silicon and
does not deform locally.

References: stage drift, vibration and frame-averaging effects in CD-SEM
metrology are surveyed in B. Bunday et al., Proc. SPIE 5375 (2004); scan-field
distortion and its calibration are standard SEM instrument topics
(Goldstein et al., 2018, ch. on image formation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np

from driftsense import geometry as G
from driftsense.config import AcquisitionConfig

Number = Union[float, np.ndarray]


# --------------------------------------------------------------------------- #
# Scan-field distortion
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DistortionField:
    """Smooth, bounded, analytically evaluable scan-field distortion.

    Each axis is a sum of plane waves whose periods are a sizeable fraction of
    the field of view, so the warp is low-frequency by construction --
    deflection non-linearity, not rubber-sheeting.

    Attributes:
        amp_nm: Peak displacement in nanometres.
        comps_u: ``(fx, fy, phase)`` per component for the horizontal axis,
            frequencies in cycles per nanometre.
        comps_v: Same for the vertical axis.
    """

    amp_nm: float = 0.0
    comps_u: Tuple[Tuple[float, float, float], ...] = ()
    comps_v: Tuple[Tuple[float, float, float], ...] = ()

    @classmethod
    def sample(cls, rng: np.random.Generator, amp_nm: float, fov_nm: float,
               n_components: int = 3) -> "DistortionField":
        """Draw a field with periods between 0.4 and 2.5 field widths."""
        if amp_nm <= 0.0:
            return cls()

        def comps() -> Tuple[Tuple[float, float, float], ...]:
            out = []
            for _ in range(max(1, int(n_components))):
                period = fov_nm * float(rng.uniform(0.4, 2.5))
                angle = float(rng.uniform(0.0, 2.0 * math.pi))
                f = 1.0 / period
                out.append((f * math.cos(angle), f * math.sin(angle),
                            float(rng.uniform(0.0, 2.0 * math.pi))))
            return tuple(out)

        return cls(amp_nm=float(amp_nm), comps_u=comps(), comps_v=comps())

    def _axis(self, comps: Tuple[Tuple[float, float, float], ...],
              u: Number, v: Number) -> Number:
        """Sum the plane waves for one axis.

        Full-frame evaluation is done in float32.  The frame-local coordinates
        reach a few thousand nanometres, where float32 resolves ~0.0005 nm --
        four orders of magnitude below the nanometre-scale displacement this
        function returns -- while float64 sines over a million points cost
        ~285 ms per frame, which at 100k pairs is eight hours of pure trig.
        Scalar calls (used to invert the label) stay in float64.
        """
        is_array = isinstance(u, np.ndarray)
        if is_array:
            uu = np.ascontiguousarray(u, dtype=np.float32)
            vv = np.ascontiguousarray(v, dtype=np.float32)
        else:
            uu, vv = u, v

        total = None
        for fx, fy, ph in comps:
            if is_array:
                arg = uu * np.float32(2.0 * math.pi * fx)
                arg += vv * np.float32(2.0 * math.pi * fy)
                arg += np.float32(ph)
                term = np.sin(arg, out=arg)
            else:
                term = np.sin(2.0 * math.pi * (fx * uu + fy * vv) + ph)
            total = term if total is None else total + term
        return (self.amp_nm / len(comps)) * total

    def evaluate(self, u: Number, v: Number) -> Tuple[Number, Number]:
        """Displacement ``(du, dv)`` in nm at frame coordinates ``(u, v)``.

        Works on scalars (label inversion) and on full arrays (rendering).
        """
        if not self.active:
            zero = np.zeros_like(u) if isinstance(u, np.ndarray) else 0.0
            return zero, zero
        return self._axis(self.comps_u, u, v), self._axis(self.comps_v, u, v)

    @property
    def active(self) -> bool:
        """Whether this field does anything at all."""
        return self.amp_nm > 0.0 and bool(self.comps_u)


# --------------------------------------------------------------------------- #
# Frame geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FrameGeometry:
    """Everything needed to sample one frame, and to invert it exactly.

    Attributes:
        center_nm: Frame centre in die coordinates (nm).
        px_nm: Actual pixel pitch including magnification error (nm).
        theta: Stage rotation (radians).
        n: Frame size in pixels.
        jitter_amp_nm: AR(1) per-row jitter amplitude (nm, 1 sigma).
        ar1: AR(1) correlation coefficient of that jitter.
        drift_nm: Total ``(dx, dy)`` drift from first to last scan line (nm).
        distortion: Analytic scan-field distortion.
        jitter_seed: Sub-seed for the jitter sequence, so the grid is a pure
            function of this dataclass and a frame can be re-rendered without
            replaying the plan.
    """

    center_nm: Tuple[float, float]
    px_nm: float
    theta: float
    n: int
    jitter_amp_nm: float = 0.0
    ar1: float = 0.85
    drift_nm: Tuple[float, float] = (0.0, 0.0)
    distortion: DistortionField = field(default_factory=DistortionField)
    jitter_seed: int = 0

    # -- derived ------------------------------------------------------------- #
    @property
    def fov_nm(self) -> float:
        """Field of view (nm) along one side."""
        return self.n * self.px_nm

    @property
    def half_fov_nm(self) -> float:
        """Half the field of view (nm)."""
        return 0.5 * self.fov_nm

    @property
    def rotation_deg(self) -> float:
        """Stage rotation in degrees."""
        return math.degrees(self.theta)

    # -- the scan-error field ------------------------------------------------ #
    def jitter_sequence(self) -> np.ndarray:
        """Per-row horizontal jitter in nm, shape ``(n,)``.

        O(n) and about 50 us, so the planner can afford to call it -- which is
        what lets the label account for jitter instead of inheriting it as
        error.
        """
        if self.jitter_amp_nm <= 0.0:
            return np.zeros(self.n, dtype=np.float64)
        rng = np.random.default_rng([int(self.jitter_seed), 1])
        return G.ar1_sequence(rng, self.n, self.jitter_amp_nm, self.ar1)

    def offset_nm(self, u: Number, v: Number,
                  jitter: Optional[np.ndarray] = None) -> Tuple[Number, Number]:
        """Total scan-error displacement at frame coordinates ``(u, v)``.

        Args:
            u: Horizontal frame coordinate in nm (0 at frame centre).
            v: Vertical frame coordinate in nm.
            jitter: Precomputed :meth:`jitter_sequence`, to avoid regenerating
                it inside a loop.

        Returns:
            ``(du, dv)`` in nanometres, same shape as the inputs.
        """
        if jitter is None:
            jitter = self.jitter_sequence()
        half = (self.n - 1) / 2.0
        row = np.clip(np.round(np.asarray(v, dtype=np.float64) / self.px_nm + half),
                      0, self.n - 1).astype(np.int64)
        ramp = row / max(self.n - 1, 1)
        du = jitter[row] + self.drift_nm[0] * ramp
        dv = self.drift_nm[1] * ramp
        if self.distortion.active:
            ddu, ddv = self.distortion.evaluate(u, v)
            du = du + ddu
            dv = dv + ddv
        if np.ndim(u) == 0:
            return float(du), float(dv)
        return du, dv

    # -- sampling ------------------------------------------------------------ #
    def grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Die coordinates (nm) sampled by every pixel of this frame.

        Returns:
            ``(x, y)`` float64 arrays of shape ``(n, n)``.
        """
        idx = (np.arange(self.n, dtype=np.float64) - (self.n - 1) / 2.0) * self.px_nm
        u = np.broadcast_to(idx[None, :], (self.n, self.n)).copy()
        v = np.broadcast_to(idx[:, None], (self.n, self.n)).copy()

        if self.jitter_amp_nm > 0.0:
            u += self.jitter_sequence()[:, None]
        if self.drift_nm != (0.0, 0.0):
            ramp = np.linspace(0.0, 1.0, self.n, dtype=np.float64)[:, None]
            u += self.drift_nm[0] * ramp
            v += self.drift_nm[1] * ramp
        if self.distortion.active:
            du, dv = self.distortion.evaluate(u, v)
            u += du
            v += dv

        ct, st = math.cos(self.theta), math.sin(self.theta)
        return (self.center_nm[0] + ct * u - st * v,
                self.center_nm[1] + st * u + ct * v)

    # -- exact mappings ------------------------------------------------------ #
    def to_pixel(self, point_nm: Tuple[float, float]) -> Tuple[float, float]:
        """Nominal (rigid) die -> pixel mapping, ignoring scan errors."""
        return G.die_to_pixel(point_nm, self.center_nm, self.theta,
                              self.px_nm, self.n)

    def to_die(self, point_px: Tuple[float, float]) -> Tuple[float, float]:
        """Nominal (rigid) pixel -> die mapping, ignoring scan errors."""
        return G.pixel_to_die(point_px, self.center_nm, self.theta,
                              self.px_nm, self.n)

    def locate(self, point_nm: Tuple[float, float],
               iterations: int = 3) -> Tuple[float, float]:
        """The pixel that actually *samples* ``point_nm``, scan errors included.

        Solves ``u + delta(u, v) = u_target`` by fixed-point iteration.  With
        ``|delta|`` around a pixel and a tiny gradient, two iterations converge
        to well under 0.01 px; three is free insurance.

        This -- not :meth:`to_pixel` -- is the ground truth.

        Args:
            point_nm: Die coordinate to locate.
            iterations: Fixed-point iterations.

        Returns:
            Fractional ``(x_px, y_px)``.
        """
        half = (self.n - 1) / 2.0
        ct, st = math.cos(-self.theta), math.sin(-self.theta)
        dx = point_nm[0] - self.center_nm[0]
        dy = point_nm[1] - self.center_nm[1]
        ut = ct * dx - st * dy
        vt = st * dx + ct * dy

        jitter = self.jitter_sequence()
        u, v = ut, vt
        for _ in range(max(1, iterations)):
            du, dv = self.offset_nm(u, v, jitter=jitter)
            u, v = ut - du, vt - dv
        return u / self.px_nm + half, v / self.px_nm + half

    def contains_nm(self, point_nm: Tuple[float, float],
                    margin_px: float = 0.0) -> bool:
        """Is a die point inside this frame, with a pixel margin to spare?"""
        u, v = self.to_pixel(point_nm)
        return (margin_px <= u <= self.n - 1 - margin_px
                and margin_px <= v <= self.n - 1 - margin_px)


# --------------------------------------------------------------------------- #
# The pair
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AcquisitionPair:
    """The reference and search frame geometries for one sample."""

    reference: FrameGeometry
    search: FrameGeometry

    @property
    def relative_rotation(self) -> float:
        """Reference rotation minus search rotation (radians).

        This is what the matcher must cope with; absolute stage angle is
        unobservable from a single frame.
        """
        return self.reference.theta - self.search.theta

    @property
    def scale_ratio(self) -> float:
        """How much the reference pattern is shrunk inside the search image.

        Nominally 10, never exactly 10: both frames carry independent
        magnification error.
        """
        return self.search.px_nm / self.reference.px_nm

    @property
    def footprint_px(self) -> float:
        """Side length of the reference footprint, in search pixels."""
        return self.reference.fov_nm / self.search.px_nm

    def target_pixel(self) -> Tuple[float, float]:
        """Ground-truth centre in search pixels, scan errors accounted for."""
        return self.search.locate(self.reference.center_nm)

    def nominal_target_pixel(self) -> Tuple[float, float]:
        """Ground truth ignoring scan errors -- diagnostics only."""
        return self.search.to_pixel(self.reference.center_nm)

    def label_correction_px(self) -> float:
        """How far scan errors moved the target from its nominal pixel.

        Recorded per sample: a dataset that does not measure this is unaware of
        its own label noise.
        """
        a = np.asarray(self.target_pixel())
        b = np.asarray(self.nominal_target_pixel())
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def sample_acquisition(rng: np.random.Generator, cfg: AcquisitionConfig,
                       n: int, ref_px_nm: float, search_px_nm: float,
                       search_center_nm: Optional[Tuple[float, float]] = None,
                       ) -> Tuple[FrameGeometry, float, float]:
    """Draw the wide-search frame plus the reference's relative errors.

    The reference centre is not known yet -- it depends on where the planner
    finds a usable landmark -- so this returns the search frame and the two
    scalars needed to finish the reference frame later.

    Args:
        rng: The sample's ``"acquisition"`` stream.
        cfg: Acquisition configuration.
        n: Frame size in pixels.
        ref_px_nm: Nominal reference pixel pitch (nm).
        search_px_nm: Nominal search pixel pitch (nm).
        search_center_nm: Optional fixed centre; drawn from the die extent when
            omitted.

    Returns:
        ``(search_frame, ref_px_nm_actual, ref_theta)``.
    """
    search_theta = math.radians(rng.normal(0.0, cfg.search_rotation_deg_sigma))
    relative_theta = math.radians(rng.normal(0.0, cfg.relative_rotation_deg_sigma))
    ref_theta = search_theta + relative_theta

    search_px = search_px_nm * float(rng.normal(1.0, cfg.search_mag_error_sigma))
    ref_px = ref_px_nm * float(rng.normal(1.0, cfg.ref_mag_error_sigma))

    if search_center_nm is None:
        e = cfg.die_extent_nm
        search_center_nm = (float(rng.uniform(-e, e)), float(rng.uniform(-e, e)))

    fov = n * search_px
    search = FrameGeometry(
        center_nm=search_center_nm,
        px_nm=search_px,
        theta=search_theta,
        n=n,
        jitter_amp_nm=abs(float(rng.normal(0.0, cfg.search_jitter_px_sigma))) * search_px,
        ar1=cfg.ar1_coefficient,
        drift_nm=(float(rng.normal(0.0, cfg.search_drift_nm_sigma)),
                  float(rng.normal(0.0, cfg.search_drift_nm_sigma))),
        distortion=DistortionField.sample(
            rng, cfg.field_distortion_px.draw(rng) * search_px, fov,
            cfg.field_distortion_cells.draw(rng)),
        jitter_seed=int(rng.integers(0, 2 ** 62)),
    )
    return search, ref_px, ref_theta


def build_reference_frame(rng: np.random.Generator, cfg: AcquisitionConfig,
                          center_nm: Tuple[float, float], n: int,
                          px_nm: float, theta: float) -> FrameGeometry:
    """Complete the reference frame once the planner has chosen its centre.

    Args:
        rng: The sample's ``"acquisition"`` stream (continued).
        cfg: Acquisition configuration.
        center_nm: Reference centre in die coordinates -- the quantity the
            algorithm has to recover.
        n: Frame size in pixels.
        px_nm: Actual reference pixel pitch from :func:`sample_acquisition`.
        theta: Actual reference rotation from :func:`sample_acquisition`.
    """
    fov = n * px_nm
    return FrameGeometry(
        center_nm=center_nm,
        px_nm=px_nm,
        theta=theta,
        n=n,
        jitter_amp_nm=abs(float(rng.normal(0.0, cfg.ref_jitter_px_sigma))) * px_nm,
        ar1=cfg.ar1_coefficient,
        drift_nm=(float(rng.normal(0.0, cfg.ref_drift_nm_sigma)),
                  float(rng.normal(0.0, cfg.ref_drift_nm_sigma))),
        distortion=DistortionField.sample(
            rng, cfg.field_distortion_px.draw(rng) * px_nm, fov,
            cfg.field_distortion_cells.draw(rng)),
        jitter_seed=int(rng.integers(0, 2 ** 62)),
    )


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.optics.acquisition``."""
    from driftsense.config import IMG_N, REF_PX_NM, SEARCH_PX_NM
    from driftsense.rng import SeedBook

    cfg = AcquisitionConfig()
    rng = SeedBook(4242).stream("acquisition")

    search, ref_px, ref_theta = sample_acquisition(
        rng, cfg, IMG_N, REF_PX_NM, SEARCH_PX_NM)
    ref_center = (search.center_nm[0] + 300.0, search.center_nm[1] - 120.0)
    ref = build_reference_frame(rng, cfg, ref_center, IMG_N, ref_px, ref_theta)
    pair = AcquisitionPair(ref, search)

    # 1. magnification error present but small; footprint near but not at 100
    assert 9.0 < pair.scale_ratio < 11.0
    assert 95.0 < pair.footprint_px < 105.0
    assert pair.footprint_px != 100.0

    # 2. nominal mapping round-trips exactly
    gx, gy = pair.nominal_target_pixel()
    back = search.to_die((gx, gy))
    assert abs(back[0] - ref_center[0]) < 1e-6
    assert abs(back[1] - ref_center[1]) < 1e-6

    # 3. THE test: the label must point at the pixel that samples the target.
    #    Found by brute force over the rendered grid, which knows nothing about
    #    how the label was computed.
    gxs, gys = search.grid()
    lx, ly = pair.target_pixel()
    r0, c0 = int(round(ly)), int(round(lx))
    win = 8
    sub_x = gxs[r0 - win:r0 + win + 1, c0 - win:c0 + win + 1]
    sub_y = gys[r0 - win:r0 + win + 1, c0 - win:c0 + win + 1]
    d = np.hypot(sub_x - ref_center[0], sub_y - ref_center[1])
    br, bc = np.unravel_index(int(np.argmin(d)), d.shape)
    true_px = (c0 - win + bc, r0 - win + br)
    err = math.hypot(true_px[0] - lx, true_px[1] - ly)
    assert err < 0.75, f"label off by {err:.3f} px"

    nx, ny = pair.nominal_target_pixel()
    nominal_err = math.hypot(true_px[0] - nx, true_px[1] - ny)
    correction = pair.label_correction_px()

    # 4. distortion is analytic: scalars and arrays agree, bound is respected.
    #    The array path evaluates in float32 for speed (see _axis), so they
    #    agree to ~1e-4 nm rather than exactly.  That is four orders of
    #    magnitude below the nanometre displacement being computed and six
    #    below one search pixel, so the label and the render still describe the
    #    same physical warp.
    dfield = DistortionField.sample(np.random.default_rng(0), 12.0, 10000.0, 3)
    du_s, dv_s = dfield.evaluate(123.0, -456.0)
    du_a, dv_a = dfield.evaluate(np.array([123.0]), np.array([-456.0]))
    scalar_vs_array_nm = max(abs(du_s - float(du_a[0])),
                             abs(dv_s - float(dv_a[0])))
    assert scalar_vs_array_nm < 1e-3, scalar_vs_array_nm
    assert abs(du_s) <= 12.0 + 1e-9

    # 5. grid reproducibility
    again = search.grid()
    assert np.array_equal(gxs, again[0]) and np.array_equal(gys, again[1])

    # 6. containment margin logic used by the planner
    assert search.contains_nm(ref_center, margin_px=60.0)
    far = (search.center_nm[0] + search.half_fov_nm * 0.99, search.center_nm[1])
    assert not search.contains_nm(far, margin_px=60.0)

    # 7. population statistics, plus convergence of the inverse map
    rels, foots, corr, resid = [], [], [], []
    for s in range(120):
        b = SeedBook(s).stream("acquisition")
        sf, rpx, rth = sample_acquisition(b, cfg, IMG_N, REF_PX_NM, SEARCH_PX_NM)
        c = (sf.center_nm[0] + float(b.uniform(-2000, 2000)),
             sf.center_nm[1] + float(b.uniform(-2000, 2000)))
        rf = build_reference_frame(b, cfg, c, IMG_N, rpx, rth)
        p = AcquisitionPair(rf, sf)
        rels.append(math.degrees(p.relative_rotation))
        foots.append(p.footprint_px)
        corr.append(p.label_correction_px())
        # forward-map the label back and measure the residual
        ux, uy = p.target_pixel()
        half = (IMG_N - 1) / 2.0
        u = (ux - half) * sf.px_nm
        v = (uy - half) * sf.px_nm
        du, dv = sf.offset_nm(u, v)
        ct, st = math.cos(sf.theta), math.sin(sf.theta)
        px_ = sf.center_nm[0] + ct * (u + du) - st * (v + dv)
        py_ = sf.center_nm[1] + st * (u + du) + ct * (v + dv)
        resid.append(math.hypot(px_ - c[0], py_ - c[1]) / sf.px_nm)

    rels = np.array(rels)
    foots = np.array(foots)
    corr = np.array(corr)
    resid = np.array(resid)
    assert 1.2 < rels.std() < 2.0, rels.std()
    assert abs(rels.mean()) < 0.4, rels.mean()
    assert 0.5 < foots.std() < 2.0, foots.std()
    assert resid.max() < 0.05, f"fixed point did not converge: {resid.max():.4f} px"

    print("acquisition.py self-test OK")
    print(f"  scale ratio             : {pair.scale_ratio:.4f} (nominal 10)")
    print(f"  footprint               : {pair.footprint_px:.2f} px (nominal 100)")
    print(f"  relative rotation       : {math.degrees(pair.relative_rotation):+.3f} deg")
    print(f"  label vs sampled grid   : {err:.3f} px "
          f"(uncorrected label: {nominal_err:.3f} px off)")
    print(f"  scan-error correction   : {correction:.3f} px on this sample")
    print(f"  over 120 samples        : correction mean {corr.mean():.2f} px, "
          f"max {corr.max():.2f} px")
    print(f"  scalar vs array warp    : {scalar_vs_array_nm:.2e} nm")
    print(f"  inverse-map residual    : max {resid.max():.4f} px")
    print(f"                          : rot sigma {rels.std():.2f} deg, "
          f"footprint {foots.mean():.2f} +- {foots.std():.2f} px")


if __name__ == "__main__":
    _self_test()
