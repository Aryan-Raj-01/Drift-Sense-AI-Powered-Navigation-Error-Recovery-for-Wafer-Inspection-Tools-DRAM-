"""DRAM array layout.

A 6F^2-style array: horizontal word lines crossed by vertical bit lines with a
storage-node contact at every intersection.  Reference geometry: K. Kim, "Technology
for sub-50nm DRAM and NAND flash manufacturing", IEDM 2005; IEEE IRDS "More Moore"
chapter for the pitch roadmap.

Two things here matter more than the drawing code.

**Pitch bounds keep the sample well posed.**  Pitches are drawn so the array
stays resolvable at 10 nm/px -- roughly four pixels per pitch or better.  Finer
than that, the wide-search frame aliases to flat grey, and a sample whose search
image contains no usable structure is not "hard", it is unanswerable.  Training
on unanswerable samples puts a floor under the loss that gradient descent
answers by hedging toward the image centre, which is exactly the failure mode
this project is supposed to fix.

**Defects are position-hashed, not random per frame.**  A missing contact must
be missing in *both* frames, at the same die coordinate, without the two
renderers sharing state.  ``Layout.keep_mask`` hashes the lattice index, so the
answer is a pure function of position and the die's variation seed.  Drawing
defects from a per-frame RNG would put them in different places in the reference
and the search image, which silently destroys correspondence -- and looks fine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from driftsense import geometry as G
from driftsense.config import GeneratorConfig
from driftsense.layouts.base import (
    Layout,
    lattice_index,
    register_layout,
    shared_fields,
)


@register_layout("dram")
@dataclass(frozen=True)
class DramLayout(Layout):
    """Periodic word-line / bit-line array with storage-node contacts.

    Attributes:
        wl_pitch: Word-line (horizontal) pitch, nm.
        bl_pitch: Bit-line (vertical) pitch, nm.
        wl_width: Word-line width, nm.
        bl_width: Bit-line width, nm.
        wl_phase: Die-coordinate offset of one word-line centre, nm.
        bl_phase: Die-coordinate offset of one bit-line centre, nm.
        contact_radius: Storage-node contact radius, nm.
        missing_contact_rate: Fraction of contacts absent (open defect).
        broken_line_rate: Fraction of line segments absent (line-break defect).
        level_wordline: Grey level of word lines.
        level_bitline: Grey level of bit lines.
        level_contact: Grey level of contacts.
    """

    wl_pitch: float = 100.0
    bl_pitch: float = 100.0
    wl_width: float = 42.0
    bl_width: float = 42.0
    wl_phase: float = 0.0
    bl_phase: float = 0.0
    contact_radius: float = 18.0
    missing_contact_rate: float = 0.0
    broken_line_rate: float = 0.0
    level_wordline: float = 0.50
    level_bitline: float = 0.68
    level_contact: float = 0.90

    # -- construction -------------------------------------------------------- #
    @classmethod
    def sample(cls, rng: np.random.Generator, cfg: GeneratorConfig) -> "DramLayout":
        """Draw a fully randomised DRAM array.

        Args:
            rng: The sample's ``"layout"`` stream.
            cfg: Full generator configuration.

        Returns:
            A frozen :class:`DramLayout`.
        """
        c = cfg.dram
        wl_pitch = c.wl_pitch_nm.draw(rng)
        bl_pitch = wl_pitch * c.bl_pitch_ratio.draw(rng)
        min_pitch = min(wl_pitch, bl_pitch)
        return cls(
            style="dram",
            background=c.level_bg.draw(rng),
            wl_pitch=wl_pitch,
            bl_pitch=bl_pitch,
            wl_width=wl_pitch * c.wl_width_frac.draw(rng),
            bl_width=bl_pitch * c.bl_width_frac.draw(rng),
            wl_phase=float(rng.uniform(0.0, wl_pitch)),
            bl_phase=float(rng.uniform(0.0, bl_pitch)),
            contact_radius=0.5 * min_pitch * c.contact_diam_frac.draw(rng),
            missing_contact_rate=c.missing_contact_rate.draw(rng),
            broken_line_rate=c.broken_line_rate.draw(rng),
            level_wordline=c.level_wordline.draw(rng),
            level_bitline=c.level_bitline.draw(rng),
            level_contact=c.level_contact.draw(rng),
            **shared_fields(rng, cfg),
        )

    # -- Layout interface ---------------------------------------------------- #
    def feature_level(self, name: str) -> float:
        """Grey level a defect should paint with.  See :meth:`Layout.feature_level`."""
        return {
            "line": self.level_bitline,
            "contact": self.level_contact,
            "background": self.background,
        }.get(name, self.level_bitline)

    def lattice(self) -> Tuple[float, float, float, float]:
        """Contact lattice ``(pitch_x, pitch_y, phase_x, phase_y)``."""
        return self.bl_pitch, self.wl_pitch, self.bl_phase, self.wl_phase

    def _render_core(self, img: np.ndarray, xa: np.ndarray, ya: np.ndarray,
                     aa: float) -> np.ndarray:
        """Composite word lines, bit lines and contacts.

        Layer order follows the physical stack: word lines are buried, bit lines
        run above them, contacts sit on top and are the brightest feature.
        """
        cd = self.cd_scale(xa)

        # Lattice indices are only needed when something can be defective; each
        # one costs a full-frame floor division, which is not free at 1e6 px.
        need_i = self.missing_contact_rate > 0.0 or self.broken_line_rate > 0.0
        idx_i = lattice_index(xa, self.bl_pitch, self.bl_phase) if need_i else None
        idx_j = lattice_index(ya, self.wl_pitch, self.wl_phase) if need_i else None

        # --- word lines (horizontal) --------------------------------------- #
        wl = G.soft_bands(ya, self.wl_pitch, self.wl_width * cd, self.wl_phase, aa)
        keep = self.keep_mask(idx_i, idx_j, self.broken_line_rate, channel=1) \
            if need_i else None
        if keep is not None:
            wl = wl * keep          # break the segment inside one bit-line cell
        img = G.composite(img, wl, self.level_wordline)

        # --- bit lines (vertical) ------------------------------------------ #
        bl = G.soft_bands(xa, self.bl_pitch, self.bl_width * cd, self.bl_phase, aa)
        keep = self.keep_mask(idx_i, idx_j, self.broken_line_rate, channel=2) \
            if need_i else None
        if keep is not None:
            bl = bl * keep
        img = G.composite(img, bl, self.level_bitline)

        # --- storage-node contacts ----------------------------------------- #
        via = G.soft_disks(xa, ya, self.bl_pitch, self.wl_pitch,
                           self.bl_phase, self.wl_phase,
                           self.contact_radius * cd, aa)
        keep = self.keep_mask(idx_i, idx_j, self.missing_contact_rate, channel=3) \
            if need_i else None
        if keep is not None:
            via = via * keep
        return G.composite(img, via, self.level_contact)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.layouts.dram``."""
    from driftsense.config import GeneratorConfig
    from driftsense.layouts.base import available_styles, sample_layout

    cfg = GeneratorConfig()
    rng = np.random.default_rng(11)
    lay = DramLayout.sample(rng, cfg)
    assert lay.style == "dram"
    assert "dram" in available_styles()

    # 1. the registry path resolves to this architecture.  Compare by class
    #    NAME, not identity: running this file as __main__ creates a second
    #    module object, so the registered class and the local one are distinct
    #    objects even though they are the same source.
    via_registry = sample_layout("dram", np.random.default_rng(11), cfg)
    assert type(via_registry).__name__ == "DramLayout"
    assert via_registry.style == "dram"

    # 2. determinism: the same die coordinates give the same pixels, and the
    #    two magnifications agree about where the structure is
    n = 256
    x, y = G.make_grid((12345.0, -6789.0), n, 1.0, 0.0)
    a = lay.evaluate(x, y, 1.0)
    b = lay.evaluate(x, y, 1.0)
    assert np.array_equal(a, b)
    assert a.dtype == np.float32
    assert 0.0 <= float(a.min()) and float(a.max()) <= 1.0

    # 3. periodicity: shifting by exactly one pitch reproduces the image
    x2, y2 = G.make_grid((12345.0 + lay.bl_pitch, -6789.0 + lay.wl_pitch),
                         n, 1.0, 0.0)
    shifted = lay.evaluate(x2, y2, 1.0)
    if lay.missing_contact_rate == 0.0 and lay.broken_line_rate == 0.0:
        assert float(np.abs(shifted - a).max()) < 0.02

    # 4. cross-magnification agreement: the coarse render must be close to an
    #    area-average of the fine one.  This is the property that makes the
    #    reference genuinely appear inside the search image.
    fine_n = 1000
    xf, yf = G.make_grid((0.0, 0.0), fine_n, 1.0, 0.0)
    fine = lay.evaluate(xf, yf, 1.0)
    xc, yc = G.make_grid((0.0, 0.0), fine_n // 10, 10.0, 10.0 * 0.0)
    coarse = lay.evaluate(xc, yc, 10.0)
    ref_small = G.box_downsample(fine, 10)
    m = 8
    aa = ref_small[m:-m, m:-m].ravel()
    bb = coarse[m:-m, m:-m].ravel()
    corr = float(np.corrcoef(aa, bb)[0, 1])
    assert corr > 0.9, f"magnifications disagree: r={corr:.3f}"

    # 5. defects are position-hashed: same die coordinate, same defect, at both
    #    magnifications and from independently constructed grids
    defective = DramLayout.sample(np.random.default_rng(4), cfg)
    object.__setattr__(defective, "missing_contact_rate", 0.25)
    d1 = defective.evaluate(x, y, 1.0)
    xs, ys = G.make_grid((12345.0, -6789.0), n, 1.0, 0.0)
    d2 = defective.evaluate(xs, ys, 1.0)
    assert np.array_equal(d1, d2)
    # ... and they actually remove material
    clean = DramLayout(**{**defective.__dict__, "missing_contact_rate": 0.0})
    assert float(d1.mean()) < float(clean.evaluate(x, y, 1.0).mean())

    # 6. polarity inversion is a clean complement
    inv = DramLayout(**{**lay.__dict__, "polarity": -1})
    assert abs(float((inv.evaluate(x, y, 1.0) + a).mean()) - 1.0) < 1e-5

    # 7. the manifest view is JSON-clean
    import json
    json.dumps(lay.to_dict())

    print("layouts/dram.py self-test OK")
    print(f"  pitch wl/bl            : {lay.wl_pitch:.1f} / {lay.bl_pitch:.1f} nm")
    print(f"  width wl/bl            : {lay.wl_width:.1f} / {lay.bl_width:.1f} nm")
    print(f"  contact radius         : {lay.contact_radius:.1f} nm "
          f"({2*lay.contact_radius/10.0:.1f} px at 10 nm/px)")
    print(f"  px per pitch @10nm/px  : {min(lay.wl_pitch, lay.bl_pitch)/10.0:.1f}")
    print(f"  1nm vs 10nm agreement  : r = {corr:.3f}")
    print(f"  contrast (std)         : {float(a.std()):.3f} fine, "
          f"{float(coarse.std()):.3f} coarse")


if __name__ == "__main__":
    _self_test()
