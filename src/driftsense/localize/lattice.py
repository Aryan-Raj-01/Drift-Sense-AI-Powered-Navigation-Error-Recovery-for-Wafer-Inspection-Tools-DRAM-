"""Stage 0 -- spectral lattice suppression.

A DRAM array is close to exactly periodic, so in the Fourier domain it appears
as a small number of very sharp, isolated peaks at the lattice fundamental and
its harmonics.  Everything that identifies *where* we are -- landmarks, process
defects, missing contacts, array boundaries -- is aperiodic and spreads out as
broad, low-amplitude spectral content.

Suppressing those sharp peaks removes the component that carries no positional
information but dominates the correlation score.  On a hard sample the
aperiodic marker occupies roughly 0.02--0.1 % of the footprint's energy; after
suppression it becomes the dominant term.

The peaks are found by comparing each spectral magnitude against a local
background, so no assumption is made about pitch, orientation, or which layout
family the image comes from.  The same code works unchanged on FinFET fin
lines, which matters because the official test set contains layouts we never
trained on.

Self-test:
    python -m driftsense.localize.lattice
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

#: Spectral magnitude must exceed the local background by this factor to count
#: as a lattice harmonic.  Lower removes more (and risks eating real signal).
PEAK_RATIO = 4.0

#: Half-width of the box used to estimate the local spectral background.
BACKGROUND_BOX = 15

#: Notched bins are multiplied by this rather than zeroed.  Exact zeroing rings.
NOTCH_GAIN = 0.03

#: Bins closer to DC than this (in cycles per image) are never touched --
#: they carry overall brightness and large-scale shading, not lattice.
DC_PROTECT = 3.0

#: Radius in bins over which a detected peak is suppressed.
NOTCH_RADIUS = 1


def _periodic_component(a: np.ndarray) -> np.ndarray:
    """Moisan's periodic component of the periodic-plus-smooth decomposition.

    A plain FFT of a non-periodic image treats the wrap-around edge
    discontinuity as signal, producing a bright cross along the axes that
    masks genuine lattice peaks and causes ringing once we notch.  Removing
    the smooth component first eliminates that artefact.

    Args:
        a: Float image.

    Returns:
        The periodic component, same shape and dtype.

    Reference:
        Moisan, L. (2011). Periodic plus smooth image decomposition.
        *Journal of Mathematical Imaging and Vision*, 39(2), 161--179.
    """
    h, w = a.shape
    v = np.zeros_like(a)
    v[0, :] = a[-1, :] - a[0, :]
    v[-1, :] = -v[0, :]
    v[:, 0] += a[:, -1] - a[:, 0]
    v[:, -1] -= a[:, -1] - a[:, 0]

    qy = 2.0 * np.cos(2.0 * np.pi * np.fft.fftfreq(h))[:, None]
    qx = 2.0 * np.cos(2.0 * np.pi * np.fft.fftfreq(w))[None, :]
    denom = qy + qx - 4.0
    denom[0, 0] = 1.0

    smooth = np.fft.ifft2(np.fft.fft2(v) / denom).real
    smooth[0, 0] = 0.0
    return a - smooth


def lattice_mask(spectrum: np.ndarray, peak_ratio: float = PEAK_RATIO,
                 dc_protect: float = DC_PROTECT) -> np.ndarray:
    """Boolean mask of bins that look like lattice harmonics.

    Args:
        spectrum: Shifted magnitude spectrum (DC at the centre).
        peak_ratio: Magnitude must exceed the local background by this factor.
        dc_protect: Radius in bins around DC that is never masked.

    Returns:
        Boolean array, True where a bin should be suppressed.
    """
    log_mag = np.log1p(spectrum.astype(np.float32))
    background = cv2.blur(log_mag, (BACKGROUND_BOX, BACKGROUND_BOX))
    ratio = np.expm1(log_mag) / (np.expm1(background) + 1e-9)

    mask = ratio > peak_ratio

    h, w = spectrum.shape
    yy, xx = np.ogrid[:h, :w]
    r2 = (yy - h // 2) ** 2 + (xx - w // 2) ** 2
    mask &= r2 > dc_protect ** 2

    if NOTCH_RADIUS > 0:
        k = 2 * NOTCH_RADIUS + 1
        mask = cv2.dilate(mask.astype(np.uint8),
                          np.ones((k, k), np.uint8)).astype(bool)
    return mask


def suppress(image: np.ndarray, peak_ratio: float = PEAK_RATIO,
             notch_gain: float = NOTCH_GAIN,
             return_mask: bool = False
             ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Remove the periodic lattice, leaving the aperiodic residual.

    Args:
        image: Grayscale image, any dtype.
        peak_ratio: Detection threshold; see :func:`lattice_mask`.
        notch_gain: Detected bins are scaled by this instead of zeroed.
        return_mask: Also return the suppression mask, for diagnostics.

    Returns:
        Float64 residual with the same shape as the input, contrast-normalised
        to zero mean and unit variance.  With ``return_mask``, a
        ``(residual, mask)`` pair.
    """
    a = np.asarray(image, dtype=np.float64)
    a = a - a.mean()

    periodic = _periodic_component(a)
    spec = np.fft.fftshift(np.fft.fft2(periodic))
    mask = lattice_mask(np.abs(spec), peak_ratio=peak_ratio)

    spec[mask] *= notch_gain
    residual = np.fft.ifft2(np.fft.ifftshift(spec)).real

    s = residual.std()
    if s > 1e-9:
        residual = (residual - residual.mean()) / s

    return (residual, mask) if return_mask else residual


def suppression_stats(image: np.ndarray) -> dict:
    """Diagnostics for one image, for tuning and for the report.

    Args:
        image: Grayscale image.

    Returns:
        Dict with the fraction of bins notched and the fraction of spectral
        energy removed.  Energy removed far above ~0.5 means the threshold is
        too aggressive and real signal is being destroyed.
    """
    a = np.asarray(image, dtype=np.float64)
    a = a - a.mean()
    spec = np.fft.fftshift(np.fft.fft2(_periodic_component(a)))
    mask = lattice_mask(np.abs(spec))
    power = np.abs(spec) ** 2
    total = float(power.sum()) + 1e-12
    return {"bins_notched": float(mask.mean()),
            "energy_removed": float(power[mask].sum() / total)}


def _self_test() -> None:
    """Synthetic check: a periodic grid plus one blob."""
    n = 512
    yy, xx = np.mgrid[:n, :n]
    grid = (np.sin(2 * np.pi * xx / 16.0) * np.sin(2 * np.pi * yy / 16.0))
    blob = np.zeros((n, n))
    cv2.circle(blob, (300, 200), 12, 1.0, -1)
    img = 128 + 50 * grid + 60 * blob

    res = suppress(img)
    stats = suppression_stats(img)

    # The blob must survive; the grid must not.
    blob_region = res[188:212, 288:312]
    grid_region = res[60:120, 60:120]
    contrast = abs(float(blob_region.mean())) / (float(grid_region.std()) + 1e-9)

    before = abs(float(img[188:212, 288:312].mean() - img.mean())) / \
        (float(img[60:120, 60:120].std()) + 1e-9)

    print(f"  bins notched     {stats['bins_notched'] * 100:.3f} %")
    print(f"  energy removed   {stats['energy_removed'] * 100:.1f} %")
    print(f"  blob/grid before {before:.2f}")
    print(f"  blob/grid after  {contrast:.2f}")
    assert contrast > before * 2.0, "suppression did not improve blob contrast"
    assert stats["energy_removed"] > 0.2, "almost nothing was suppressed"
    print("  OK")


if __name__ == "__main__":
    _self_test()
