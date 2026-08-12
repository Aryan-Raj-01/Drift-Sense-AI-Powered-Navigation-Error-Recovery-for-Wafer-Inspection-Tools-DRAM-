"""Coordinate mapping between the correlation-output grid and search pixels.

Pure arithmetic -- no torch dependency -- so it is fully unit-tested in this
environment even though the model itself cannot be run here.

The encoder reduces spatial resolution by a factor of ``stride`` (4).  The
correlation between a reference embedding of size ``ref_emb_size`` and a
search embedding is computed in "valid" mode, so correlation-output index 0
does not correspond to search-pixel 0 -- it corresponds to the centre of the
first window the reference kernel covers.

``out_to_pixel`` and ``pixel_to_out`` must be exact inverses, and the SAME
pair of functions must be used to build training targets and to decode
inference output.  If the two ever disagree, the network calibrates to
whatever convention the loss used, and inference silently reports the wrong
answer.  That is the one bug class this module exists to prevent.
"""

from __future__ import annotations


def out_to_pixel(out_xy: float, ref_emb_size: int, stride: float = 4.0) -> float:
    """Map a correlation-output-grid coordinate to an original search pixel.

    Args:
        out_xy: Coordinate in the correlation output grid (post soft-argmax).
        ref_emb_size: Spatial size of the reference embedding (``re.shape[-1]``
            at runtime -- read from the tensor, never hard-coded, since it
            varies slightly with the sampled scale ratio).
        stride: Encoder output stride.

    Returns:
        Coordinate in original search-image pixels.
    """
    return (out_xy + (ref_emb_size - 1) / 2.0) * stride + stride / 2.0


def pixel_to_out(px_xy: float, ref_emb_size: int, stride: float = 4.0) -> float:
    """Inverse of :func:`out_to_pixel`."""
    return (px_xy - stride / 2.0) / stride - (ref_emb_size - 1) / 2.0


def shrink_size(full_size: int, scale: float) -> int:
    """Reference crop size after shrinking by the demagnification.

    Same clamping as the classical proposal stage (``propose.py``), factored
    out so both the classical and learned paths agree on template size.

    Args:
        full_size: Reference image side length (1000).
        scale: Demagnification (~10).

    Returns:
        Shrunk side length, clamped to a sane range.
    """
    return max(8, min(round(full_size / scale), full_size - 2))


def _self_test() -> None:
    """Round-trip and monotonicity checks -- pure math, no torch required."""
    import random
    random.seed(0)

    # Round trip must be exact (up to float error) for every plausible
    # ref_emb_size (24-26, per the encoder's stride-4 output for k in 97-103)
    # and every pixel in the search frame.
    for ref_emb_size in (24, 25, 26):
        for _ in range(2000):
            px = random.uniform(-50, 1050)
            out = pixel_to_out(px, ref_emb_size)
            back = out_to_pixel(out, ref_emb_size)
            assert abs(back - px) < 1e-9, (ref_emb_size, px, out, back)

    # Monotonic: larger output index must map to a larger pixel coordinate.
    for ref_emb_size in (24, 25, 26):
        a = out_to_pixel(10.0, ref_emb_size)
        b = out_to_pixel(11.0, ref_emb_size)
        assert b > a

    # shrink_size sanity: known DRAM values map close to 100.
    for scale in (9.73, 10.0, 10.27):
        s = shrink_size(1000, scale)
        assert 95 <= s <= 105, (scale, s)

    print("  round-trip OK on 3 ref_emb_size x 2000 random pixels")
    print("  monotonicity OK")
    print("  shrink_size OK for scale range 9.73-10.27")
    print("  OK")


if __name__ == "__main__":
    _self_test()
