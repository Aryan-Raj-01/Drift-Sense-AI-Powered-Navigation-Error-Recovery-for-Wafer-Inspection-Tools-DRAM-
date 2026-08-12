"""FinFET logic layout.

Dense parallel vertical fins crossed by one or two horizontal gate bars, with
raised source/drain epi contacts between gates and shallow-trench-isolation
bands breaking the array into cells.  Reference geometry: D. Hisamoto et al.,
"FinFET - a self-aligned double-gate MOSFET scalable to 20 nm", IEEE TED 47(12),
2000; C. Auth et al., VLSI Symp. 2012, for fin pitch and contacted gate pitch.

The STI bands matter more than they look.  Without them a FinFET die is an
infinite fin array, and at 10 um field of view every part of it looks like every
other part -- the search image becomes a uniform texture with no long-range
structure for a matcher to lock onto, and every sample degenerates into the
hardest possible case.  Real logic is not like that: it is cells separated by
isolation, and that cellular structure is most of what survives the 10x
demagnification.

Fin pitch is bounded below at ~42 nm so the array remains resolvable at
10 nm/px.  See the note in :mod:`driftsense.layouts.dram` on why unanswerable
samples are worse than hard ones.
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


@register_layout("finfet")
@dataclass(frozen=True)
class FinfetLayout(Layout):
    """Fin array with gate bars, epi contacts and STI isolation.

    Attributes:
        fin_pitch: Fin-to-fin pitch, nm.
        gate_pitch: Contacted gate pitch (the gate period), nm.
        fin_width: Fin width, nm.
        gate_width: Gate-bar width, nm.
        fin_phase: Die-coordinate offset of one fin centre, nm.
        gate_phase: Die-coordinate offset of one gate period, nm.
        gate_count: Gate bars drawn per gate period (1 or 2).
        gate_split: Separation of the second bar as a fraction of gate pitch.
        epi_radius: Source/drain epi contact radius, nm.
        sti_period: Spacing of shallow-trench-isolation bands, nm.
        sti_width: Width of an STI band, nm.
        broken_fin_rate: Fraction of fin segments absent.
        level_fin: Grey level of fins.
        level_gate: Grey level of gate bars.
        level_epi: Grey level of epi contacts.
        level_sti: Grey level of isolation trenches (darker than the field).
    """

    fin_pitch: float = 60.0
    gate_pitch: float = 210.0
    fin_width: float = 20.0
    gate_width: float = 56.0
    fin_phase: float = 0.0
    gate_phase: float = 0.0
    gate_count: int = 1
    gate_split: float = 0.42
    epi_radius: float = 13.0
    sti_period: float = 1200.0
    sti_width: float = 150.0
    broken_fin_rate: float = 0.0
    level_fin: float = 0.58
    level_gate: float = 0.80
    level_epi: float = 0.90
    level_sti: float = 0.28

    # -- construction -------------------------------------------------------- #
    @classmethod
    def sample(cls, rng: np.random.Generator,
               cfg: GeneratorConfig) -> "FinfetLayout":
        """Draw a fully randomised FinFET layout.

        Args:
            rng: The sample's ``"layout"`` stream.
            cfg: Full generator configuration.

        Returns:
            A frozen :class:`FinfetLayout`.
        """
        c = cfg.finfet
        fin_pitch = c.fin_pitch_nm.draw(rng)
        gate_pitch = fin_pitch * c.gate_pitch_ratio.draw(rng)
        return cls(
            style="finfet",
            background=c.level_bg.draw(rng),
            fin_pitch=fin_pitch,
            gate_pitch=gate_pitch,
            fin_width=fin_pitch * c.fin_width_frac.draw(rng),
            gate_width=gate_pitch * c.gate_width_frac.draw(rng),
            fin_phase=float(rng.uniform(0.0, fin_pitch)),
            gate_phase=float(rng.uniform(0.0, gate_pitch)),
            gate_count=c.gate_count.draw(rng),
            gate_split=float(rng.uniform(0.34, 0.48)),
            epi_radius=0.5 * fin_pitch * c.epi_diam_frac.draw(rng),
            sti_period=c.sti_period_nm.draw(rng),
            sti_width=c.sti_width_nm.draw(rng),
            broken_fin_rate=c.broken_fin_rate.draw(rng),
            level_fin=c.level_fin.draw(rng),
            level_gate=c.level_gate.draw(rng),
            level_epi=c.level_epi.draw(rng),
            level_sti=c.level_sti.draw(rng),
            **shared_fields(rng, cfg),
        )

    # -- Layout interface ---------------------------------------------------- #
    def feature_level(self, name: str) -> float:
        """Grey level a defect should paint with."""
        return {
            "line": self.level_fin,
            "contact": self.level_epi,
            "background": self.background,
        }.get(name, self.level_fin)

    def lattice(self) -> Tuple[float, float, float, float]:
        """Epi-contact lattice ``(pitch_x, pitch_y, phase_x, phase_y)``.

        The contact row sits mid-way between gate bars, which is where a real
        source/drain contact lands.
        """
        return (self.fin_pitch, self.gate_pitch,
                self.fin_phase, self.gate_phase + 0.5 * self.gate_pitch)

    def _render_core(self, img: np.ndarray, xa: np.ndarray, ya: np.ndarray,
                     aa: float) -> np.ndarray:
        """Composite STI, fins, epi contacts and gate bars, in stack order."""
        cd = self.cd_scale(ya)

        # --- shallow trench isolation --------------------------------------- #
        # Drawn first and reused as a suppression mask: fins do not cross an
        # isolation trench, which is what turns an infinite array into cells.
        sti = G.soft_bands(ya, self.sti_period, self.sti_width,
                           self.gate_phase, aa)
        img = G.composite(img, sti, self.level_sti)
        active = 1.0 - sti

        # --- fins (vertical) ------------------------------------------------ #
        fin = G.soft_bands(xa, self.fin_pitch, self.fin_width * cd,
                           self.fin_phase, aa)
        if self.broken_fin_rate > 0.0:
            idx_i = lattice_index(xa, self.fin_pitch, self.fin_phase)
            idx_j = lattice_index(ya, self.gate_pitch, self.gate_phase)
            keep = self.keep_mask(idx_i, idx_j, self.broken_fin_rate, channel=4)
            if keep is not None:
                fin = fin * keep
        img = G.composite(img, fin * active, self.level_fin)

        # --- raised source/drain epi contacts ------------------------------- #
        px, py, phx, phy = self.lattice()
        epi = G.soft_disks(xa, ya, px, py, phx, phy, self.epi_radius * cd, aa)
        img = G.composite(img, epi * active, self.level_epi)

        # --- gate bars (horizontal, on top) --------------------------------- #
        gate = G.soft_bands(ya, self.gate_pitch, self.gate_width * cd,
                            self.gate_phase, aa)
        if self.gate_count > 1:
            second = G.soft_bands(ya, self.gate_pitch, self.gate_width * 0.72 * cd,
                                  self.gate_phase + self.gate_split * self.gate_pitch,
                                  aa)
            gate = np.maximum(gate, second)
        return G.composite(img, gate, self.level_gate)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.layouts.finfet``."""
    import json

    from driftsense.config import GeneratorConfig
    from driftsense.layouts.base import available_styles, sample_layout

    cfg = GeneratorConfig()
    lay = FinfetLayout.sample(np.random.default_rng(21), cfg)
    assert lay.style == "finfet"
    assert "finfet" in available_styles()
    # compare by class NAME: running this file as __main__ creates a second
    # module object, so registry identity would spuriously fail
    via_registry = sample_layout("finfet", np.random.default_rng(21), cfg)
    assert type(via_registry).__name__ == "FinfetLayout"
    assert via_registry.style == "finfet"

    n = 256
    x, y = G.make_grid((-4321.0, 8765.0), n, 1.0, 0.0)
    a = lay.evaluate(x, y, 1.0)
    assert a.dtype == np.float32
    assert np.array_equal(a, lay.evaluate(x, y, 1.0))
    assert 0.0 <= float(a.min()) and float(a.max()) <= 1.0

    # 1. fin periodicity along x (STI only modulates y)
    x2, y2 = G.make_grid((-4321.0 + lay.fin_pitch, 8765.0), n, 1.0, 0.0)
    if lay.broken_fin_rate == 0.0:
        assert float(np.abs(lay.evaluate(x2, y2, 1.0) - a).max()) < 0.02

    # 2. cross-magnification agreement
    xf, yf = G.make_grid((0.0, 0.0), 1000, 1.0, 0.0)
    fine = lay.evaluate(xf, yf, 1.0)
    xc, yc = G.make_grid((0.0, 0.0), 100, 10.0, 0.0)
    coarse = lay.evaluate(xc, yc, 10.0)
    m = 8
    corr = float(np.corrcoef(G.box_downsample(fine, 10)[m:-m, m:-m].ravel(),
                             coarse[m:-m, m:-m].ravel())[0, 1])
    assert corr > 0.9, f"magnifications disagree: r={corr:.3f}"

    # 3. STI genuinely breaks the array: row variance across a 4 um span must
    #    show the isolation bands, or the search image is a uniform texture
    xl, yl = G.make_grid((0.0, 0.0), 400, 10.0, 0.0)
    wide = lay.evaluate(xl, yl, 10.0)
    row_profile = wide.mean(axis=1)
    assert float(row_profile.std()) > 0.01, "STI bands invisible at 10 nm/px"

    # 4. gate count actually changes the picture
    two = FinfetLayout(**{**lay.__dict__, "gate_count": 2})
    one = FinfetLayout(**{**lay.__dict__, "gate_count": 1})
    assert float(two.evaluate(x, y, 1.0).mean()) > float(one.evaluate(x, y, 1.0).mean())

    # 5. broken fins remove material and stay position-hashed.
    #    Evaluated over the full 1 um field: a 256 nm window spans only a
    #    handful of lattice sites, so "no defect landed here" is a perfectly
    #    likely outcome there and would make this test flaky rather than wrong.
    broken = FinfetLayout(**{**lay.__dict__, "broken_fin_rate": 0.4})
    b1 = broken.evaluate(xf, yf, 1.0)
    assert np.array_equal(b1, broken.evaluate(*G.make_grid((0.0, 0.0), 1000,
                                                           1.0, 0.0), 1.0))
    assert not np.array_equal(b1, fine)
    assert float(b1.mean()) < float(fine.mean())

    json.dumps(lay.to_dict())

    print("layouts/finfet.py self-test OK")
    print(f"  fin pitch / width      : {lay.fin_pitch:.1f} / {lay.fin_width:.1f} nm")
    print(f"  gate pitch / width     : {lay.gate_pitch:.1f} / {lay.gate_width:.1f} nm"
          f"  x{lay.gate_count}")
    print(f"  STI period / width     : {lay.sti_period:.0f} / {lay.sti_width:.0f} nm")
    print(f"  px per fin pitch @10nm : {lay.fin_pitch/10.0:.1f}")
    print(f"  1nm vs 10nm agreement  : r = {corr:.3f}")
    print(f"  STI row-profile std    : {float(row_profile.std()):.4f}")


if __name__ == "__main__":
    _self_test()
