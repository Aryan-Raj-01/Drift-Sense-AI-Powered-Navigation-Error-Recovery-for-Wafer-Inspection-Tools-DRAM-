"""The generator core: plan a sample, then render it.

Everything in this package exists to make these two functions possible:

    plan_sample(seed, index, cfg) -> Plan     cheap, deterministic, no pixels
    render(plan)                  -> (ref, search)   pure function of the plan

The split is the single most consequential design decision in the project.

**Planning produces the ground truth without rendering anything.**  Choosing the
die, the frames, the target and the imaging parameters costs about 3 ms; the
pixels cost about 650 ms.  So a 100 000-sample "dataset" is a 25 MB manifest
that takes five minutes to produce, instead of 100 GB of PNGs that take a day.
Training regenerates the pixels inside the DataLoader from the seed.

**Rendering is a pure function.**  Given a :class:`Plan` -- and nothing else, no
global RNG, no ambient state -- the images are byte-identical on any machine,
in any worker, in any order.  That is what makes the multiprocessing in
:mod:`driftsense.cli.generate` safe and what makes a dataset reproducible from
``(seed, config_hash)``.

**Solvability is decided at plan time.**  The planner will not emit a sample
whose target has no locally unique feature; if the search window offers no
landmark with a unique signature it marks the target with a sub-resolution
defect instead and labels the sample ``hard``.  A sample with no correct answer
is not a hard training example, it is a corrupt one -- see
:mod:`driftsense.layouts.landmarks`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from driftsense import geometry as G
from driftsense.config import IMG_N, REF_PX_NM, SEARCH_PX_NM, GeneratorConfig
from driftsense.layouts import defects as defects_mod
from driftsense.layouts.base import Layout, sample_layout
from driftsense.layouts.defects import Defect
from driftsense.layouts.landmarks import Landmark, LandmarkField
from driftsense.optics.acquisition import (
    AcquisitionPair,
    FrameGeometry,
    build_reference_frame,
    sample_acquisition,
)
from driftsense.optics.chain import CaptureParams, sample_pair
from driftsense.rng import SeedBook

#: Difficulty tiers, in the order used for reporting.
DIFFICULTIES: Tuple[str, ...] = ("easy", "medium", "hard")

#: A landmark bigger than this (nm) makes the sample "easy" rather than "medium".
EASY_LANDMARK_NM = 300.0


@dataclass(frozen=True)
class Plan:
    """Everything needed to render one sample and to label it.

    Attributes:
        index: Position in the dataset.
        seed: The sample's root seed; the plan is a pure function of it.
        config_hash: Fingerprint of the configuration that produced it.
        style: ``"dram"`` or ``"finfet"``.
        difficulty: ``"easy"``, ``"medium"`` or ``"hard"``.
        layout: The die architecture.
        landmark_field: The die's landmark population.
        acquisition: Reference and search frame geometries.
        target: The landmark the reference is centred on, if any.
        defect: The marking defect, present exactly when ``target`` is absent.
        ref_capture: Reference imaging parameters.
        search_capture: Search imaging parameters.
        n_landmarks: Landmarks visible in the search frame (distractor count).
        n_unique: How many of them had a unique signature.
    """

    index: int
    seed: int
    config_hash: str
    style: str
    difficulty: str
    layout: Layout
    landmark_field: LandmarkField
    acquisition: AcquisitionPair
    target: Optional[Landmark]
    defect: Optional[Defect]
    ref_capture: CaptureParams
    search_capture: CaptureParams
    n_landmarks: int = 0
    n_unique: int = 0

    # -- derived ------------------------------------------------------------- #
    @property
    def reference(self) -> FrameGeometry:
        """The reference frame geometry."""
        return self.acquisition.reference

    @property
    def search(self) -> FrameGeometry:
        """The search frame geometry."""
        return self.acquisition.search

    def ground_truth(self) -> Dict[str, Any]:
        """Label dictionary in search-image pixel coordinates.

        The centre comes from :meth:`FrameGeometry.locate`, which inverts the
        *actual* sampling including jitter, drift and scan distortion -- not
        from the nominal rigid map.  ``label_correction_px`` records how far
        apart those two answers were, so the dataset knows its own label noise
        instead of hiding it.
        """
        gx, gy = self.acquisition.target_pixel()
        rel = self.acquisition.relative_rotation
        quad = G.footprint_quad((gx, gy), self.acquisition.footprint_px, rel)
        x0, y0, x1, y1 = G.quad_bbox(quad)
        return {
            "gt_x": round(gx, 4),
            "gt_y": round(gy, 4),
            "bbox": [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)],
            "quad": [[round(float(a), 4), round(float(b), 4)] for a, b in quad],
            "footprint_px": round(self.acquisition.footprint_px, 4),
            "rel_rotation_deg": round(math.degrees(rel), 4),
            "scale_ratio": round(self.acquisition.scale_ratio, 5),
            "label_correction_px": round(self.acquisition.label_correction_px(), 4),
        }


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def _choose_style(rng: np.random.Generator, cfg: GeneratorConfig) -> str:
    """Pick an architecture from the configured set."""
    styles = cfg.styles
    return styles[int(rng.integers(len(styles)))]


def _usable_targets(field_: LandmarkField, population: List[Landmark],
                    search: FrameGeometry, margin_px: float) -> List[Landmark]:
    """Landmarks that are both signature-unique and safely inside the frame.

    The margin keeps the whole reference footprint on-screen: a target 20 px
    from the edge would put half of the reference field of view outside the
    search image, and no algorithm could match what is not there.
    """
    return [lm for lm in field_.unique_candidates(population)
            if search.contains_nm((lm.x, lm.y), margin_px=margin_px)]


def plan_sample(seed: int, index: int = 0,
                cfg: Optional[GeneratorConfig] = None) -> Plan:
    """Plan one sample: geometry, target and imaging parameters, no pixels.

    Args:
        seed: The sample's root seed, from
            :func:`driftsense.rng.sample_seeds`.
        index: Position in the dataset, carried into the manifest.
        cfg: Generator configuration; defaults are used when omitted.

    Returns:
        A fully determined :class:`Plan`.
    """
    cfg = cfg or GeneratorConfig()
    book = SeedBook(seed)
    plan_rng = book.stream("plan")
    layout_rng = book.stream("layout")
    acq_rng = book.stream("acquisition")

    style = _choose_style(plan_rng, cfg)
    layout = sample_layout(style, layout_rng, cfg)
    die_seed = int(layout_rng.integers(0, 2 ** 62))
    field_ = LandmarkField.sample(layout_rng, cfg.landmarks, die_seed)

    search, ref_px, ref_theta = sample_acquisition(
        acq_rng, cfg.acquisition, IMG_N, REF_PX_NM, SEARCH_PX_NM)

    # Half the reference footprint, plus the configured margin, in search pixels.
    margin_px = 0.5 * (IMG_N * ref_px / search.px_nm) + cfg.acquisition.edge_margin_px

    population = field_.in_frame(search)
    candidates = _usable_targets(field_, population, search, margin_px)

    force_hard = plan_rng.random() < cfg.hard_fraction
    target: Optional[Landmark] = None
    defect: Optional[Defect] = None

    if candidates and not force_hard:
        target = candidates[int(plan_rng.integers(len(candidates)))]
        # The tool aims at the landmark but does not land perfectly on it --
        # that navigation error is the whole premise of the problem.
        center = (target.x + float(plan_rng.normal(0.0, 18.0)),
                  target.y + float(plan_rng.normal(0.0, 18.0)))
        difficulty = "easy" if target.size_nm > EASY_LANDMARK_NM else "medium"
    else:
        # Hard: plain periodic array, marked only by one process defect.
        half = search.half_fov_nm
        reach = max(0.0, half - margin_px * search.px_nm)
        center = (search.center_nm[0] + float(plan_rng.uniform(-reach, reach)),
                  search.center_nm[1] + float(plan_rng.uniform(-reach, reach)))
        defect = defects_mod.sample_defect(plan_rng, cfg.defects, center,
                                           layout.lattice())
        center = (defect.x, defect.y)
        difficulty = "hard"

    reference = build_reference_frame(acq_rng, cfg.acquisition, center,
                                      IMG_N, ref_px, ref_theta)
    ref_capture, search_capture = sample_pair(book, cfg, cfg.noise_boost)

    return Plan(
        index=int(index),
        seed=int(seed),
        config_hash=cfg.version_hash(),
        style=style,
        difficulty=difficulty,
        layout=layout,
        landmark_field=field_,
        acquisition=AcquisitionPair(reference=reference, search=search),
        target=target,
        defect=defect,
        ref_capture=ref_capture,
        search_capture=search_capture,
        n_landmarks=len(population),
        n_unique=len(candidates),
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def render_frame(plan: Plan, frame: FrameGeometry, capture: CaptureParams,
                 rng: np.random.Generator, quantised: bool = True) -> np.ndarray:
    """Render one frame of a planned sample.

    Args:
        plan: The sample plan.
        frame: Which frame to render (reference or search geometry).
        capture: That frame's imaging parameters.
        rng: That frame's optics stream.
        quantised: Return ``uint8`` rather than ``float32``.

    Returns:
        The rendered frame.
    """
    x, y = frame.grid()
    aa = frame.px_nm

    material = plan.layout.evaluate(x, y, aa)
    plan.landmark_field.render(material, x, y, frame, aa)
    if plan.defect is not None:
        defects_mod.render(material, x, y, frame, plan.defect, plan.layout, aa)

    return capture.apply(material, rng, quantised=quantised)


def render(plan: Plan, quantised: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Render both frames of a planned sample.

    The two frames draw from different RNG streams, so their noise, charging and
    shading are independent -- the brief's first mandatory requirement, enforced
    structurally rather than by convention.

    Args:
        plan: The sample plan.
        quantised: Return ``uint8`` rather than ``float32``.

    Returns:
        ``(reference, search)``, each ``(1000, 1000)``.
    """
    book = SeedBook(plan.seed)
    ref = render_frame(plan, plan.reference, plan.ref_capture,
                       book.fresh("render.reference"), quantised)
    search = render_frame(plan, plan.search, plan.search_capture,
                          book.fresh("render.search"), quantised)
    return ref, search


def render_sample(seed: int, index: int = 0,
                  cfg: Optional[GeneratorConfig] = None
                  ) -> Tuple[np.ndarray, np.ndarray, Plan]:
    """Plan and render in one call.

    Args:
        seed: The sample's root seed.
        index: Dataset position.
        cfg: Generator configuration.

    Returns:
        ``(reference, search, plan)``.
    """
    plan = plan_sample(seed, index, cfg)
    ref, search = render(plan)
    return ref, search, plan


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_pair(plan: Plan, ref: np.ndarray, search: np.ndarray) -> float:
    """Correlate the demagnified reference against the search image at the label.

    This is the dataset's own sanity check.  It downsamples the reference by the
    recorded scale ratio, undoes the recorded rotation, and correlates the
    result with the search image at the recorded centre.  Values around 0.5-0.9
    are healthy: high enough to prove the label points at the right place, low
    enough to prove the two frames are genuinely independent captures rather
    than copies.  Below ~0.2 means the label is wrong.

    Args:
        plan: The sample plan.
        ref: Rendered reference frame.
        search: Rendered search frame.

    Returns:
        Zero-mean normalised cross-correlation at the ground-truth location.
    """
    gt = plan.ground_truth()
    factor = max(1, int(round(gt["scale_ratio"])))
    small = G.box_downsample(ref.astype(np.float32), factor)

    rot = gt["rel_rotation_deg"]
    if abs(rot) > 0.2:
        try:
            import cv2

            k = small.shape[0]
            m = cv2.getRotationMatrix2D((k / 2.0, k / 2.0), rot, 1.0)
            small = cv2.warpAffine(small, m, (k, k),
                                   borderMode=cv2.BORDER_REPLICATE)
        except Exception:  # pragma: no cover - cv2-less fallback
            pass

    k = small.shape[0]
    x0 = int(round(gt["gt_x"] - k / 2.0))
    y0 = int(round(gt["gt_y"] - k / 2.0))
    x0 = max(0, min(search.shape[1] - k, x0))
    y0 = max(0, min(search.shape[0] - k, y0))
    crop = search[y0:y0 + k, x0:x0 + k].astype(np.float32)

    m = max(2, k // 6)                       # ignore the border: rotation eats it
    a = small[m:-m, m:-m].ravel()
    b = crop[m:-m, m:-m].ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum())) + 1e-9
    return float((a * b).sum() / denom)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.pipeline``."""
    import time

    from driftsense.config import preset
    from driftsense.rng import sample_seeds

    cfg = GeneratorConfig()
    seeds = sample_seeds(20260803, 24)

    # 1. planning is cheap and deterministic
    t0 = time.time()
    plans = [plan_sample(int(s), i, cfg) for i, s in enumerate(seeds)]
    plan_ms = 1000 * (time.time() - t0) / len(plans)
    again = plan_sample(int(seeds[0]), 0, cfg)
    assert again.ground_truth() == plans[0].ground_truth()
    assert again.style == plans[0].style and again.difficulty == plans[0].difficulty
    assert all(p.config_hash == cfg.version_hash() for p in plans)

    # 2. every plan is well posed: a target or a defect, never neither
    for p in plans:
        assert (p.target is None) != (p.defect is None), "target/defect exclusivity"
        gt = p.ground_truth()
        half = gt["footprint_px"] / 2.0
        assert half < gt["gt_x"] < IMG_N - half, gt["gt_x"]
        assert half < gt["gt_y"] < IMG_N - half, gt["gt_y"]
        assert 95.0 < gt["footprint_px"] < 105.0
        assert p.difficulty in DIFFICULTIES

    # 3. difficulty mix roughly follows hard_fraction
    hard = sum(p.difficulty == "hard" for p in plans) / len(plans)
    assert 0.05 < hard < 0.60, hard

    # 4. render is a pure function of the plan
    t0 = time.time()
    ref, search = render(plans[0])
    render_s = time.time() - t0
    r2, s2 = render(plans[0])
    assert np.array_equal(ref, r2) and np.array_equal(search, s2)
    assert ref.shape == (IMG_N, IMG_N) and ref.dtype == np.uint8

    # 5. THE end-to-end correctness test: the label must point at the pattern
    zn = []
    for p in plans[:10]:
        a, b = render(p)
        zn.append(verify_pair(p, a, b))
    zn = np.array(zn)
    assert zn.mean() > 0.35, f"labels look wrong: mean ZNCC {zn.mean():.3f}"
    assert zn.min() > 0.15, f"one label is wrong: min ZNCC {zn.min():.3f}"

    # 6. the two frames are independent captures, not copies
    p = plans[1]
    assert p.ref_capture.noise.dose > p.search_capture.noise.dose
    assert p.ref_capture.probe != p.search_capture.probe

    # 7. the label accounts for scan errors rather than inheriting them
    corr = np.array([pl.ground_truth()["label_correction_px"] for pl in plans])
    assert corr.max() > 0.05, "scan errors are not being corrected for"
    assert corr.max() < 8.0, "scan errors are implausibly large"

    # 8. a noisier preset really is harder to verify but still correct
    hts = preset("hidden_test_sim")
    hp = plan_sample(int(seeds[3]), 3, hts)
    ha, hb = render(hp)
    hz = verify_pair(hp, ha, hb)
    assert hz > 0.1, hz
    assert hp.search_capture.noise.dose < p.search_capture.noise.dose

    print("pipeline.py self-test OK")
    print(f"  plan time              : {plan_ms:.2f} ms/sample")
    print(f"  render time            : {render_s:.2f} s/pair")
    print(f"  -> 100k plans          : {100000*plan_ms/1000/60:.1f} min")
    print(f"  -> 100k renders, 8 cpu : {100000*render_s/8/3600:.1f} h")
    print(f"  difficulty mix         : "
          + ", ".join(f"{d}={sum(q.difficulty == d for q in plans)}"
                      for d in DIFFICULTIES))
    print(f"  label ZNCC (10 pairs)  : mean {zn.mean():.3f}, min {zn.min():.3f}")
    print(f"  scan-error correction  : mean {corr.mean():.2f} px, "
          f"max {corr.max():.2f} px")
    print(f"  landmarks per search   : {np.mean([q.n_landmarks for q in plans]):.1f}"
          f" ({np.mean([q.n_unique for q in plans]):.1f} usable)")
    print(f"  noisier preset ZNCC    : {hz:.3f}")


if __name__ == "__main__":
    _self_test()
