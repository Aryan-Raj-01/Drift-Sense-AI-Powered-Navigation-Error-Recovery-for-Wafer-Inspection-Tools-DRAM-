"""Geometric and signal-processing primitives shared by every renderer.

Two ideas carry the whole package and both live here.

**1. Analytic anti-aliasing instead of supersampling.**
Every shape is a signed distance field turned into coverage by
:func:`soft_step`, whose transition width is exactly one pixel of whatever grid
is sampling it.  A feature narrower than one pixel therefore loses contrast
automatically, which is the physically correct band-limited behaviour and is
what makes the same layout function usable at 1 nm/px and at 10 nm/px without
rendering an intermediate canvas.  Supersampling 4x would cost 16x the memory
to approximate the same thing.

**2. Grids are warped, images are not.**
:func:`make_grid` returns the die coordinates that each pixel samples.  Stage
rotation, magnification error, scan jitter, drift and scan-field distortion are
all applied to those coordinates.  Nothing downstream is ever allowed to move a
pixel, because ``warpAffine`` after the fact leaves interpolation blur, ringing
and black corners -- all three correlated with the labels, all three absent
from the hidden test set -- and because a resampled image no longer matches its
closed-form ground truth.

All heavy filtering goes through OpenCV when available, with a numpy fallback so
the module still imports (and self-tests) on a machine without it.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover - exercised only on machines without cv2
    _HAS_CV2 = False


# --------------------------------------------------------------------------- #
# Signed distance fields -> anti-aliased coverage
# --------------------------------------------------------------------------- #
def soft_step(distance: np.ndarray, aa: float) -> np.ndarray:
    """Convert a signed distance field to coverage in ``[0, 1]``.

    Args:
        distance: Signed distance in nanometres; positive inside the shape.
        aa: Pixel pitch in nanometres, so the edge fades over exactly one pixel.

    Returns:
        ``float32`` coverage array.
    """
    return np.clip(distance / aa + 0.5, 0.0, 1.0).astype(np.float32, copy=False)


def soft_bands(coord: np.ndarray, pitch: float, width: float, phase: float,
               aa: float) -> np.ndarray:
    """Infinite periodic set of parallel lines.

    Args:
        coord: Coordinate perpendicular to the lines (nm).
        pitch: Centre-to-centre spacing (nm).
        width: Line width (nm).
        phase: Offset of one line centre from the origin (nm).
        aa: Pixel pitch (nm).

    Returns:
        Coverage array.
    """
    t = np.mod(coord - phase + 0.5 * pitch, pitch) - 0.5 * pitch
    return soft_step(0.5 * width - np.abs(t), aa)


def soft_disks(x: np.ndarray, y: np.ndarray, pitch_x: float, pitch_y: float,
               phase_x: float, phase_y: float, radius: float,
               aa: float) -> np.ndarray:
    """Two-dimensional periodic lattice of discs (contacts, vias, epi)."""
    tx = np.mod(x - phase_x + 0.5 * pitch_x, pitch_x) - 0.5 * pitch_x
    ty = np.mod(y - phase_y + 0.5 * pitch_y, pitch_y) - 0.5 * pitch_y
    return soft_step(radius - np.hypot(tx, ty), aa)


def soft_disk(x: np.ndarray, y: np.ndarray, cx: float, cy: float,
              radius: float, aa: float) -> np.ndarray:
    """A single disc."""
    return soft_step(radius - np.hypot(x - cx, y - cy), aa)


def soft_rect(x: np.ndarray, y: np.ndarray, cx: float, cy: float,
              width: float, height: float, theta: float, aa: float,
              corner_radius: float = 0.0) -> np.ndarray:
    """A rounded rectangle at arbitrary orientation.

    Uses the standard rounded-box SDF so that corner rounding costs nothing
    extra -- real lithography never produces a sharp corner, and a generator
    that emits perfect right angles gives the network a cue that does not
    survive contact with a real (or a differently-generated) image.

    Args:
        x, y: Sample coordinates (nm).
        cx, cy: Rectangle centre (nm).
        width, height: Extents before rotation (nm).
        theta: Rotation in radians.
        aa: Pixel pitch (nm).
        corner_radius: Corner radius (nm), clamped to half the short side.
    """
    ct, st = math.cos(theta), math.sin(theta)
    dx = (x - cx) * ct + (y - cy) * st
    dy = -(x - cx) * st + (y - cy) * ct
    r = min(corner_radius, 0.49 * min(width, height))
    qx = np.abs(dx) - (0.5 * width - r)
    qy = np.abs(dy) - (0.5 * height - r)
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return soft_step(-(outside + inside - r), aa)


def composite(base: np.ndarray, mask: np.ndarray, level: float) -> np.ndarray:
    """Alpha-composite a constant grey ``level`` over ``base`` using ``mask``.

    Written as an in-place update because at 1000x1000 float32 each temporary is
    4 MB and a naive layer stack allocates a dozen of them per frame.
    """
    base *= (1.0 - mask)
    base += mask * np.float32(level)
    return base


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def _box_blur_axis(a: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """One box-filter pass along ``axis`` (numpy fallback, O(N))."""
    if radius < 1:
        return a
    k = 2 * radius + 1
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius, radius)
    ap = np.pad(a, pad, mode="edge")
    cs = np.cumsum(ap, axis=axis, dtype=np.float32)
    zeros_shape = list(cs.shape)
    zeros_shape[axis] = 1
    cs = np.concatenate([np.zeros(zeros_shape, np.float32), cs], axis=axis)
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(k, None)
    lo[axis] = slice(0, -k)
    return (cs[tuple(hi)] - cs[tuple(lo)]) / np.float32(k)


def gaussian_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Isotropic Gaussian blur.

    Uses OpenCV when present.  The fallback is three box passes, whose combined
    variance is ``r(r+1)`` for radius ``r`` (Kovesi's approximation) -- within a
    few percent of a true Gaussian and O(N) rather than O(N log N).
    """
    if sigma <= 0.05:
        return a
    a = np.ascontiguousarray(a, dtype=np.float32)
    if _HAS_CV2:
        k = int(2 * round(3.0 * sigma) + 1)
        return cv2.GaussianBlur(a, (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
    r = max(1, int(round((-1.0 + math.sqrt(1.0 + 4.0 * sigma * sigma)) / 2.0)))
    out = a
    for _ in range(3):
        out = _box_blur_axis(out, r, 0)
        out = _box_blur_axis(out, r, 1)
    return out


def elliptical_gaussian_kernel(sigma_major: float, sigma_minor: float,
                               angle_deg: float) -> np.ndarray:
    """2-D Gaussian kernel with astigmatism (unequal axes, arbitrary angle).

    Astigmatism is a real and common SEM defect: the probe focuses at different
    working distances along two perpendicular axes, so features blur more in one
    direction than the other.  A network trained only on isotropic blur learns
    an isotropic notion of "same structure" and loses accuracy the moment the
    test tool is slightly out of stigmation.
    """
    sigma_major = max(sigma_major, 0.05)
    sigma_minor = max(sigma_minor, 0.05)
    radius = max(1, int(round(3.0 * max(sigma_major, sigma_minor))))
    n = 2 * radius + 1
    ax = np.arange(n, dtype=np.float32) - radius
    xx, yy = np.meshgrid(ax, ax)
    th = math.radians(angle_deg)
    ct, st = math.cos(th), math.sin(th)
    u = xx * ct + yy * st
    v = -xx * st + yy * ct
    k = np.exp(-0.5 * ((u / sigma_major) ** 2 + (v / sigma_minor) ** 2))
    return (k / k.sum()).astype(np.float32)


def line_kernel(length_px: float, angle_deg: float) -> np.ndarray:
    """Normalised line kernel for motion / drift smear during the scan."""
    length = max(1.0, length_px)
    radius = max(1, int(round(length / 2.0)))
    n = 2 * radius + 1
    k = np.zeros((n, n), np.float32)
    th = math.radians(angle_deg)
    for t in np.linspace(-length / 2.0, length / 2.0, int(4 * length) + 3):
        x = int(round(radius + t * math.cos(th)))
        y = int(round(radius + t * math.sin(th)))
        if 0 <= x < n and 0 <= y < n:
            k[y, x] += 1.0
    s = k.sum()
    return k / s if s > 0 else np.ones((1, 1), np.float32)


def convolve(a: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D convolution with edge replication."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    if kernel.shape == (1, 1):
        return a * float(kernel[0, 0])
    if _HAS_CV2:
        return cv2.filter2D(a, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    from numpy.lib.stride_tricks import sliding_window_view  # pragma: no cover

    kh, kw = kernel.shape
    pad = ((kh // 2, kh // 2), (kw // 2, kw // 2))
    win = sliding_window_view(np.pad(a, pad, mode="edge"), kernel.shape)
    return np.einsum("ijkl,kl->ij", win, kernel, optimize=True).astype(np.float32)


def gradient_magnitude(a: np.ndarray) -> np.ndarray:
    """Central-difference gradient magnitude, used for the SEM edge effect."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    if _HAS_CV2:
        gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)
    gx = np.empty_like(a)
    gy = np.empty_like(a)
    gx[:, 1:-1] = 0.5 * (a[:, 2:] - a[:, :-2])
    gx[:, 0] = a[:, 1] - a[:, 0]
    gx[:, -1] = a[:, -1] - a[:, -2]
    gy[1:-1, :] = 0.5 * (a[2:, :] - a[:-2, :])
    gy[0, :] = a[1, :] - a[0, :]
    gy[-1, :] = a[-1, :] - a[-2, :]
    return np.hypot(gx, gy)


def smooth_random_field(rng: np.random.Generator, n: int, cells: int,
                        sigma_px: float) -> np.ndarray:
    """Low-frequency random field normalised to ``[0, 1]``.

    Backs charging, detector non-uniformity and scan-field distortion.  Built by
    upsampling a tiny ``cells x cells`` array and blurring, which costs almost
    nothing compared with generating full-resolution noise and low-passing it.
    """
    cells = max(2, int(cells))
    seed = rng.random((cells, cells)).astype(np.float32)
    rep = int(math.ceil(n / cells))
    big = np.kron(seed, np.ones((rep, rep), np.float32))[:n, :n]
    big = gaussian_blur(big, sigma_px)
    lo, hi = float(big.min()), float(big.max())
    return (big - lo) / (hi - lo + 1e-8)


def box_downsample(a: np.ndarray, factor: int) -> np.ndarray:
    """Exact box (area) downsampling by an integer factor.

    Area averaging is the correct antialiasing filter for demagnification; a
    bilinear or nearest resize aliases the fine array pitch into moire that
    differs from what the coarse capture would really show.
    """
    factor = int(factor)
    if factor <= 1:
        return a
    h = (a.shape[0] // factor) * factor
    w = (a.shape[1] // factor) * factor
    v = a[:h, :w].astype(np.float32, copy=False)
    return v.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


# --------------------------------------------------------------------------- #
# Sampling grids and rigid mappings
# --------------------------------------------------------------------------- #
def rotation_matrix(theta: float) -> np.ndarray:
    """2x2 rotation matrix for ``theta`` radians."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def make_grid(center: Tuple[float, float], n: int, px_nm: float, theta: float,
              row_jitter: Optional[np.ndarray] = None,
              drift_nm: Tuple[float, float] = (0.0, 0.0),
              distortion: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              ) -> Tuple[np.ndarray, np.ndarray]:
    """Die coordinates (nm) sampled by each pixel of a frame.

    Args:
        center: Frame centre in die coordinates (nm).
        n: Frame size in pixels (square).
        px_nm: Pixel pitch in nanometres, including any magnification error.
        theta: Stage rotation in radians.
        row_jitter: Optional per-row horizontal offset in nm, shape ``(n,)``.
            Models line-to-line vibration of the scan.
        drift_nm: Total ``(dx, dy)`` accumulated between the first and last
            scan line -- thermal drift within a single frame.
        distortion: Optional ``(dx, dy)`` fields in nm, shape ``(n, n)``,
            for smooth scan-field distortion.

    Returns:
        ``(x, y)`` float64 arrays of shape ``(n, n)``.

    Note:
        Coordinates stay in float64.  The layout functions take a modulus
        against pitches of tens of nanometres at die coordinates of ~1e5 nm;
        in float32 the representable spacing there is ~0.01 nm and the
        rounding shows up as a visible phase error between the two frames.
    """
    idx = (np.arange(n, dtype=np.float64) - (n - 1) / 2.0) * px_nm
    u = np.broadcast_to(idx[None, :], (n, n)).copy()
    v = np.broadcast_to(idx[:, None], (n, n)).copy()

    if row_jitter is not None:
        u += np.asarray(row_jitter, dtype=np.float64)[:, None]

    if drift_nm != (0.0, 0.0):
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float64)[:, None]
        u += drift_nm[0] * ramp
        v += drift_nm[1] * ramp

    if distortion is not None:
        u += distortion[0]
        v += distortion[1]

    ct, st = math.cos(theta), math.sin(theta)
    x = center[0] + ct * u - st * v
    y = center[1] + st * u + ct * v
    return x, y


def die_to_pixel(point_nm: Tuple[float, float], center: Tuple[float, float],
                 theta: float, px_nm: float, n: int) -> Tuple[float, float]:
    """Map a die coordinate (nm) to fractional pixel coordinates in a frame.

    This is the exact inverse of the nominal part of :func:`make_grid` (jitter,
    drift and distortion are sub-pixel perturbations and are deliberately not
    inverted), and it is what produces the ground-truth centre in closed form.

    Returns:
        ``(x_px, y_px)`` with the frame centre at ``((n-1)/2, (n-1)/2)``.
    """
    ct, st = math.cos(-theta), math.sin(-theta)
    dx = point_nm[0] - center[0]
    dy = point_nm[1] - center[1]
    u = (ct * dx - st * dy) / px_nm
    v = (st * dx + ct * dy) / px_nm
    half = (n - 1) / 2.0
    return u + half, v + half


def pixel_to_die(point_px: Tuple[float, float], center: Tuple[float, float],
                 theta: float, px_nm: float, n: int) -> Tuple[float, float]:
    """Inverse of :func:`die_to_pixel`."""
    half = (n - 1) / 2.0
    u = (point_px[0] - half) * px_nm
    v = (point_px[1] - half) * px_nm
    ct, st = math.cos(theta), math.sin(theta)
    return center[0] + ct * u - st * v, center[1] + st * u + ct * v


def footprint_quad(center_px: Tuple[float, float], side_px: float,
                   rotation_rad: float) -> np.ndarray:
    """Corners of a rotated square footprint, clockwise from top-left.

    The axis-aligned bounding box of a rotated 100 px square is up to 41 %
    larger than the square, so the bbox alone is a poor label; the quad is
    stored alongside it.

    Returns:
        ``(4, 2)`` float64 array of pixel coordinates.
    """
    ct, st = math.cos(rotation_rad), math.sin(rotation_rad)
    h = side_px / 2.0
    out = np.empty((4, 2), dtype=np.float64)
    for i, (sx, sy) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        ux, uy = sx * h, sy * h
        out[i, 0] = center_px[0] + ct * ux - st * uy
        out[i, 1] = center_px[1] + st * ux + ct * uy
    return out


def quad_bbox(quad: np.ndarray) -> Tuple[float, float, float, float]:
    """Axis-aligned bounds ``(x0, y0, x1, y1)`` of a quad."""
    return (float(quad[:, 0].min()), float(quad[:, 1].min()),
            float(quad[:, 0].max()), float(quad[:, 1].max()))


def ar1_sequence(rng: np.random.Generator, n: int, amplitude: float,
                 coefficient: float = 0.85) -> np.ndarray:
    """Zero-mean AR(1) sequence scaled to ``amplitude`` standard deviations.

    Scan jitter is not white: consecutive scan lines are strongly correlated
    because the disturbance is mechanical.  White per-row noise would be removed
    by any smoothing the network learns; correlated jitter would not.
    """
    if amplitude <= 0.0:
        return np.zeros(n, dtype=np.float64)
    noise = rng.standard_normal(n)
    out = np.empty(n, dtype=np.float64)
    acc = 0.0
    for i in range(n):
        acc = coefficient * acc + noise[i]
        out[i] = acc
    out -= out.mean()
    return out / (out.std() + 1e-9) * amplitude


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.geometry``."""
    rng = np.random.default_rng(0)
    n = 256

    # 1. anti-aliasing: coverage of a band equals its duty cycle, at any pitch.
    #    The window must span a whole number of periods for the mean to be the
    #    duty cycle exactly, hence pitch = n * px / periods.
    for px, pitch, width in ((1.0, 64.0, 25.6), (10.0, 640.0, 256.0)):
        x, y = make_grid((0.0, 0.0), n, px, 0.0)
        m = soft_bands(x, pitch, width, 0.0, px)
        assert abs(float(m.mean()) - width / pitch) < 0.01, (px, m.mean())

    # 2. sub-pixel features lose contrast rather than being drawn at full
    #    strength.  What survives is residual moire, exactly as on a real tool
    #    scanning below its resolution limit -- so the test is a ratio, not a
    #    claim that the pattern vanishes.
    x1, _ = make_grid((0.0, 0.0), n, 1.0, 0.0)
    x10, _ = make_grid((0.0, 0.0), n, 10.0, 0.0)
    fine_ref = soft_bands(x1, 14.0, 6.0, 0.0, 1.0)
    fine_wide = soft_bands(x10, 14.0, 6.0, 0.0, 10.0)
    assert fine_ref.std() > 0.4, fine_ref.std()
    assert fine_wide.std() < 0.5 * fine_ref.std(), (fine_ref.std(), fine_wide.std())

    # 3. grid <-> pixel round-trip is exact under rotation and mag error
    for theta in (0.0, 0.03, -0.5):
        for px in (1.0, 9.97):
            c = (12345.0, -6789.0)
            p = (321.5, 88.25)
            back = die_to_pixel(pixel_to_die(p, c, theta, px, 1000),
                                c, theta, px, 1000)
            assert abs(back[0] - p[0]) < 1e-6 and abs(back[1] - p[1]) < 1e-6

    # 4. the grid a frame samples agrees with die_to_pixel at every corner
    theta, px, c = 0.021, 9.98, (5000.0, -3000.0)
    gx, gy = make_grid(c, 1000, px, theta)
    for (r, col) in ((0, 0), (0, 999), (999, 0), (999, 999), (500, 250)):
        u, v = die_to_pixel((gx[r, col], gy[r, col]), c, theta, px, 1000)
        assert abs(u - col) < 1e-6 and abs(v - r) < 1e-6, (r, col, u, v)

    # 5. rotation of the grid does not change what is sampled at the centre
    a = make_grid((100.0, 200.0), 5, 2.0, 0.0)
    b = make_grid((100.0, 200.0), 5, 2.0, 0.7)
    assert abs(a[0][2, 2] - b[0][2, 2]) < 1e-9
    assert abs(a[1][2, 2] - b[1][2, 2]) < 1e-9

    # 6. filters
    img = rng.random((128, 128)).astype(np.float32)
    for sigma in (0.6, 2.0, 5.0):
        blurred = gaussian_blur(img, sigma)
        assert blurred.shape == img.shape and blurred.dtype == np.float32
        assert blurred.std() < img.std()
    k = elliptical_gaussian_kernel(3.0, 1.0, 30.0)
    assert abs(float(k.sum()) - 1.0) < 1e-5
    assert abs(float(line_kernel(7.0, 0.0).sum()) - 1.0) < 1e-5
    delta = np.zeros((33, 33), np.float32)
    delta[16, 16] = 1.0
    conv = convolve(delta, k)
    assert abs(float(conv.sum()) - 1.0) < 1e-4
    # astigmatism really is anisotropic: the response is wider along the axis
    ys, xs = np.nonzero(conv > conv.max() * 0.2)
    assert np.ptp(xs) != np.ptp(ys)

    # 7. box downsample preserves the mean and the geometry
    big = rng.random((100, 100)).astype(np.float32)
    small = box_downsample(big, 10)
    assert small.shape == (10, 10)
    assert abs(float(small.mean() - big.mean())) < 1e-5

    # 8. footprint quad and bbox
    quad = footprint_quad((500.0, 400.0), 100.0, math.radians(45.0))
    assert abs(quad[:, 0].mean() - 500.0) < 1e-9
    x0, y0, x1, y1 = quad_bbox(quad)
    assert abs((x1 - x0) - 100.0 * math.sqrt(2.0)) < 1e-6

    # 9. AR(1) jitter is correlated, unlike white noise
    j = ar1_sequence(rng, 4000, 1.0, 0.85)
    r1 = float(np.corrcoef(j[:-1], j[1:])[0, 1])
    assert 0.75 < r1 < 0.95, r1
    assert abs(float(j.mean())) < 1e-9 and abs(float(j.std()) - 1.0) < 1e-6

    print("geometry.py self-test OK")
    print(f"  OpenCV                 : {'yes' if _HAS_CV2 else 'no (numpy fallback)'}")
    print(f"  band coverage error    : < 2 %")
    print(f"  14 nm pitch contrast   : {fine_ref.std():.3f} @1nm/px -> "
          f"{fine_wide.std():.3f} @10nm/px")
    print(f"  grid/pixel round-trip  : < 1e-6 px")
    print(f"  AR(1) lag-1 correlation: {r1:.3f}")


if __name__ == "__main__":
    _self_test()
