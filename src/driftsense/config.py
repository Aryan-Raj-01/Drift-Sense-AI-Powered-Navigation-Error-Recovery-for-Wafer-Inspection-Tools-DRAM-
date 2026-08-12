"""Configuration tree for the Drift-Sense generator.

Design rules enforced by this module
------------------------------------
1. **One config tree, no module-local defaults.**  If a number influences the
   output it lives here.  A module that invents its own default silently breaks
   reproducibility, because the config hash stored in the manifest will not
   change when that default changes.

2. **Frozen dataclasses.**  A worker process must never be able to mutate the
   config it was handed; with `fork` start-up that mutation would be invisible
   in the parent and would differ between workers.

3. **Ranges, not values.**  Almost every physical parameter is a
   :class:`Range` that is *drawn per sample*.  Domain randomisation is the
   default state of the system, not an optional augmentation pass.

4. **The config is hashed.**  :meth:`GeneratorConfig.version_hash` gives a
   12-hex-character fingerprint of the whole tree plus the schema version.  It
   goes into every manifest.  Reproducibility means "seed + config hash", not
   "seed" -- widening one range changes the data a given seed produces, and the
   hash is what makes that visible instead of silent.

Physical constants (`IMG_N`, `REF_PX_NM`, `SEARCH_PX_NM`) are module-level and
NOT configurable: they are fixed by the problem statement, and code that assumes
a 10x ratio is spread throughout the package.

Usage
-----
    from driftsense.config import GeneratorConfig

    cfg = GeneratorConfig()                          # defaults
    cfg = GeneratorConfig.from_yaml("configs/hidden_test_sim.yaml")
    cfg = cfg.override({"hard_fraction": 0.5,
                        "search_optics": {"dose_e_per_px": [8.0, 60.0, True]}})
    print(cfg.version_hash())
"""

import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# --------------------------------------------------------------------------- #
# Fixed constants (problem statement, not tunable)
# --------------------------------------------------------------------------- #
SCHEMA_VERSION: str = "1.0.0"
IMG_N: int = 1000               #: both frames are 1000 x 1000 px
REF_PX_NM: float = 1.0          #: "100x" capture: 1 nm/px  -> 1 um FOV
SEARCH_PX_NM: float = 10.0      #: "10x" capture: 10 nm/px  -> 10 um FOV
ZOOM: float = SEARCH_PX_NM / REF_PX_NM   #: nominal 10.0
REF_FOOTPRINT_PX: float = IMG_N / ZOOM   #: nominal 100 px inside the search image


# --------------------------------------------------------------------------- #
# Primitive range types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Range:
    """A continuous parameter range drawn once per sample.

    Args:
        lo: Inclusive lower bound.
        hi: Inclusive upper bound.
        log: If ``True``, draw log-uniformly.  Use this for any quantity that
            spans more than one order of magnitude -- electron dose above all.
            A uniform draw over ``[8, 700]`` electrons/px spends 95 % of its
            samples in the easy, low-noise regime; a log-uniform draw spreads
            them evenly across SNR decades, which is what makes the model
            robust to the noisier hidden test set.

    In YAML a range may be written as ``[lo, hi]`` or ``[lo, hi, true]``.
    """

    lo: float
    hi: float
    log: bool = False

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            raise ValueError(f"Range hi < lo: {self.lo} .. {self.hi}")
        if self.log and self.lo <= 0.0:
            raise ValueError(f"log Range needs lo > 0, got {self.lo}")

    def draw(self, rng: np.random.Generator) -> float:
        """Draw a single value."""
        if self.lo == self.hi:
            return float(self.lo)
        if self.log:
            return float(np.exp(rng.uniform(np.log(self.lo), np.log(self.hi))))
        return float(rng.uniform(self.lo, self.hi))

    def draw_many(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` values as a float64 array."""
        if self.log:
            return np.exp(rng.uniform(np.log(self.lo), np.log(self.hi), n))
        return rng.uniform(self.lo, self.hi, n)

    @property
    def mid(self) -> float:
        """Geometric midpoint for log ranges, arithmetic otherwise."""
        if self.log:
            return float(np.sqrt(self.lo * self.hi))
        return 0.5 * (self.lo + self.hi)


@dataclass(frozen=True)
class IntRange:
    """An inclusive integer range (e.g. number of gate bars)."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if self.hi < self.lo:
            raise ValueError(f"IntRange hi < lo: {self.lo} .. {self.hi}")

    def draw(self, rng: np.random.Generator) -> int:
        """Draw a single value, both bounds inclusive."""
        return int(rng.integers(self.lo, self.hi + 1))


# --------------------------------------------------------------------------- #
# Layout configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DramLayoutConfig:
    """Procedural DRAM array parameters.

    Pitch bounds are chosen so the array stays RESOLVABLE at 10 nm/px: at least
    ~4 px per pitch in the search frame.  Below that the wide-search image
    aliases to flat grey and the sample becomes ill-posed rather than merely
    hard -- a distinction that matters, because ill-posed samples put a floor
    under the training loss that gradient descent answers by hedging toward the
    image centre.

    Reference for the 6F^2 array geometry and pitch ratios: K. Kim, IEDM 2005;
    IEEE IRDS "More Moore" chapter.
    """

    wl_pitch_nm: Range = Range(70.0, 170.0)
    bl_pitch_ratio: Range = Range(0.75, 1.35)      # relative to word-line pitch
    wl_width_frac: Range = Range(0.34, 0.52)       # fraction of its own pitch
    bl_width_frac: Range = Range(0.34, 0.52)
    contact_diam_frac: Range = Range(0.26, 0.48)   # fraction of min(pitch)
    missing_contact_rate: Range = Range(0.0, 0.030)
    broken_line_rate: Range = Range(0.0, 0.020)
    level_bg: Range = Range(0.14, 0.28)
    level_wordline: Range = Range(0.42, 0.58)
    level_bitline: Range = Range(0.58, 0.78)
    level_contact: Range = Range(0.80, 0.97)


@dataclass(frozen=True)
class FinfetLayoutConfig:
    """Procedural FinFET array parameters.

    Fin pitch / contacted gate pitch ratios follow Hisamoto et al., IEEE TED
    2000 and Auth et al., VLSI Symp. 2012.  STI regions break the otherwise
    infinite fin array into cells, which is what real logic looks like at 10 um
    field of view and is a large part of why the search image is not uniform.
    """

    fin_pitch_nm: Range = Range(42.0, 95.0)
    gate_pitch_ratio: Range = Range(2.4, 5.0)      # relative to fin pitch
    fin_width_frac: Range = Range(0.26, 0.42)
    gate_width_frac: Range = Range(0.20, 0.34)
    gate_count: IntRange = IntRange(1, 2)          # gate bars per gate period
    epi_diam_frac: Range = Range(0.32, 0.56)       # source/drain epi contacts
    sti_period_nm: Range = Range(600.0, 2200.0)
    sti_width_nm: Range = Range(80.0, 260.0)
    broken_fin_rate: Range = Range(0.0, 0.025)
    level_bg: Range = Range(0.14, 0.28)
    level_fin: Range = Range(0.50, 0.68)
    level_gate: Range = Range(0.70, 0.90)
    level_epi: Range = Range(0.82, 0.98)
    level_sti: Range = Range(0.18, 0.40)


@dataclass(frozen=True)
class ProcessVariationConfig:
    """Line-edge roughness and slow CD drift across the die.

    LER is modelled as a sum of sinusoids with 0.8-3 nm amplitude and
    40-600 nm correlation lengths, which reproduces the low-frequency-dominated
    roughness spectra reported by Constantoudis / Patsis / Gogolides
    (J. Vac. Sci. Technol. B, 2003-2004).  It is deliberately *not* white noise
    on the edge: white roughness averages out under the 10x demagnification,
    long-wavelength roughness does not, and the difference is exactly what a
    matcher sees.
    """

    ler_amp_nm: Range = Range(0.8, 3.0)
    ler_wavelength_nm: Range = Range(40.0, 600.0)
    ler_harmonics: int = 3
    #: LER below this pixel size is skipped (sub-pixel at the search sampling)
    ler_min_px_nm: float = 4.0
    cd_drift_frac: Range = Range(0.0, 0.06)        # slow width drift across die
    cd_drift_period_nm: Range = Range(3000.0, 20000.0)


@dataclass(frozen=True)
class MacroStructureConfig:
    """Power rails / array-block boundaries.

    These coarse bars are the only structure that survives 10x demagnification
    with full contrast, and they are what gives the wide-search image its
    blocky appearance in the Applied Materials examples.
    """

    pitch_nm: Range = Range(1400.0, 3800.0)
    pitch_aspect: Range = Range(0.7, 1.4)          # y pitch relative to x pitch
    width_nm: Range = Range(90.0, 260.0)
    level: Range = Range(0.62, 0.95)
    probability: float = 0.85


@dataclass(frozen=True)
class LandmarkConfig:
    """The unique features a navigation target is actually placed on.

    Landmarks live on a position-hashed jittered lattice over the (infinite)
    die, so any window can be queried independently and always yields the same
    answer -- the reference frame and the search frame agree without sharing
    state.

    ``types`` x 4 orientations x size buckets x polarity forms the *signature*
    space.  The planner requires the target's signature to be unique inside its
    own search window.  Without that check a large fraction of samples have no
    well-defined answer, and with only one landmark type the model degenerates
    into a shape detector.
    """

    types: Tuple[str, ...] = ("pad", "cross", "tee", "ell", "ring",
                              "block", "hbar", "vbar")
    spacing_nm: Range = Range(950.0, 1800.0)
    density: Range = Range(0.35, 0.75)             # occupancy of lattice cells
    size_nm: Range = Range(170.0, 430.0)
    size_bucket_nm: float = 90.0                   # granularity of the signature
    bright_probability: float = 0.55


@dataclass(frozen=True)
class DefectConfig:
    """Hard-mode markers: a single process defect, 2-4 px at search resolution.

    Hard samples are made hard by shrinking the unique feature, never by
    removing uniqueness.  A sample with no unique feature has no correct
    answer, and training on it teaches the model to hedge.
    """

    types: Tuple[str, ...] = ("missing_via", "extra_via", "bridge", "line_break")
    size_nm: Range = Range(14.0, 34.0)


# --------------------------------------------------------------------------- #
# Acquisition geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AcquisitionConfig:
    """Stage and scan errors.

    Every quantity here warps the SAMPLING GRID, never the finished raster.
    Rotating an image with ``warpAffine`` leaves interpolation blur, ringing and
    black corners, all three of which correlate with the rotation label and are
    therefore shortcut cues; rotating the grid has none of those and keeps the
    ground truth exact in closed form.

    ``field_distortion_px`` is the scan-field distortion / vibration term that
    Task 4 calls "local elastic deformation".  It is applied as a smooth,
    low-frequency warp of where the beam lands -- bounded at ~1 px -- because
    silicon is rigid and does not deform locally.  An unbounded elastic warp of
    the rendered image silently invalidates the centre label by several pixels.
    """

    #: absolute stage rotation of the wide-search frame (deg, Gaussian sigma)
    search_rotation_deg_sigma: float = 0.35
    #: rotation of the reference RELATIVE to the search frame (deg, sigma)
    relative_rotation_deg_sigma: float = 1.6
    #: fractional magnification error -> footprint is never exactly 100 px
    ref_mag_error_sigma: float = 0.010
    search_mag_error_sigma: float = 0.004
    #: AR(1) per-row scan jitter amplitude, in pixels (half-normal sigma)
    ref_jitter_px_sigma: float = 0.45
    search_jitter_px_sigma: float = 0.55
    ar1_coefficient: float = 0.85
    #: intra-frame thermal drift accumulated top row -> bottom row (nm, sigma)
    ref_drift_nm_sigma: float = 1.2
    search_drift_nm_sigma: float = 6.0
    #: smooth scan-field distortion, peak amplitude in pixels
    field_distortion_px: Range = Range(0.0, 1.2)
    field_distortion_cells: IntRange = IntRange(2, 5)
    #: half-extent of the die region sampled for frame centres (nm)
    die_extent_nm: float = 1.5e5
    #: keep the reference footprint at least this many px from the search edge
    edge_margin_px: float = 60.0


# --------------------------------------------------------------------------- #
# SEM imaging chain
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OpticsConfig:
    """Per-frame SEM imaging parameters.

    The reference and search frames get SEPARATE instances of this class and
    separate RNG streams.  Sharing either one leaks a shortcut: if both frames
    share a blur kernel or a noise seed, the network can infer correspondence
    from noise autocorrelation instead of from structure, and that shortcut
    does not exist on the hidden test set.

    Order of application is fixed and physical:
        PSF -> edge effect -> charging/non-uniformity -> video gain/gamma
        -> shot noise -> read noise -> stripe noise -> quantisation.
    Blurring *after* adding noise would make the noise spectrum encode the
    blur, which is a second shortcut of the same family.

    References: Reimer (1998) for SE yield vs. surface tilt (edge brightening)
    and Poisson emission statistics; Goldstein et al. (2018) for topographic
    contrast and charging; Cazaux, Scanning 26 (2004) for charging dynamics;
    Bunday et al., Proc. SPIE 5375 (2004) for the CD-SEM dose/noise trade-off.
    """

    #: probe size; astigmatism makes it elliptical
    psf_sigma_px: Range = Range(0.55, 1.30)
    astigmatism_ratio: Range = Range(1.0, 1.8)
    astigmatism_angle_deg: Range = Range(0.0, 180.0)
    motion_blur_px: Range = Range(0.0, 2.5)
    motion_blur_probability: float = 0.25
    #: secondary-electron edge brightening
    edge_gain: Range = Range(0.25, 0.85)
    edge_sigma_px: Range = Range(0.8, 2.2)
    #: low-frequency multiplicative charging field
    charging: Range = Range(0.05, 0.22)
    charging_cells: IntRange = IntRange(4, 9)
    #: detector gain non-uniformity and illumination gradient
    detector_nonuniformity: Range = Range(0.0, 0.06)
    illumination_gradient: Range = Range(0.0, 0.12)
    #: video amplifier
    contrast: Range = Range(0.45, 0.78)
    brightness: Range = Range(0.08, 0.20)
    gamma: Range = Range(0.85, 1.20)
    #: electrons per pixel -- log-uniform, spans the SNR decades
    dose_e_per_px: Range = Range(22.0, 110.0, True)
    read_noise: Range = Range(0.018, 0.055)
    stripe_noise: Range = Range(0.004, 0.022)
    #: SE vs BSE detector settings can flip apparent polarity
    invert_probability: float = 0.06

    @classmethod
    def reference_default(cls) -> "OpticsConfig":
        """High-dose, well-focused capture: the tool dwells on the target."""
        return cls(
            psf_sigma_px=Range(1.10, 2.60),
            astigmatism_ratio=Range(1.0, 1.5),
            motion_blur_px=Range(0.0, 1.5),
            motion_blur_probability=0.15,
            charging=Range(0.02, 0.12),
            dose_e_per_px=Range(120.0, 700.0, True),
            read_noise=Range(0.006, 0.022),
            stripe_noise=Range(0.002, 0.010),
        )

    @classmethod
    def search_default(cls) -> "OpticsConfig":
        """Low-dose wide scan: 100x the area in the same frame time."""
        return cls()


# --------------------------------------------------------------------------- #
# Output / runtime
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OutputConfig:
    """Where samples go and in what form."""

    #: "manifest" writes labels only (25 MB for 100k); "png"/"npz" write pixels
    format: str = "manifest"
    shard_size: int = 1000          #: files per directory -- ext4 dies above ~10k
    png_compress_level: int = 1     #: 1 is ~4x faster than 6 and ~8 % larger
    write_parquet: bool = True
    write_csv: bool = True          #: labels.csv mirror, as the brief requires

    def __post_init__(self) -> None:
        if self.format not in ("manifest", "png", "npz"):
            raise ValueError(f"unknown output format: {self.format!r}")
        if self.shard_size < 1:
            raise ValueError("shard_size must be >= 1")


@dataclass(frozen=True)
class RuntimeConfig:
    """Multiprocessing behaviour.

    ``blas_threads=1`` is not a micro-optimisation: numpy's BLAS spawns one
    thread per core *inside every worker*, so an 8-worker pool on 8 cores
    creates 64 threads that fight for cache.  Measured slowdown is ~3x.
    """

    workers: int = 0                #: 0 -> os.cpu_count() - 1
    chunk_size: int = 8
    blas_threads: int = 1
    start_method: str = "fork"      #: "spawn" on Windows/macOS


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeneratorConfig:
    """The complete generator configuration.

    Attributes:
        styles: Which architectures to draw from.  Both are judged equally, so
            the default mixes them.
        hard_fraction: Fraction of samples with no landmark at all -- pure
            periodic array marked only by a single sub-resolution defect.  The
            hidden test set is stated to include such regions.
        noise_boost: Multiplier applied to the SEARCH frame's noise at draw
            time.  Set > 1 to simulate the noisier evaluation set.
    """

    dram: DramLayoutConfig = field(default_factory=DramLayoutConfig)
    finfet: FinfetLayoutConfig = field(default_factory=FinfetLayoutConfig)
    process: ProcessVariationConfig = field(default_factory=ProcessVariationConfig)
    macro: MacroStructureConfig = field(default_factory=MacroStructureConfig)
    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    defects: DefectConfig = field(default_factory=DefectConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    ref_optics: OpticsConfig = field(default_factory=OpticsConfig.reference_default)
    search_optics: OpticsConfig = field(default_factory=OpticsConfig.search_default)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    styles: Tuple[str, ...] = ("dram", "finfet")
    hard_fraction: float = 0.25
    noise_boost: float = 1.0
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for s in self.styles:
            if s not in ("dram", "finfet"):
                raise ValueError(f"unknown style: {s!r}")
        if not self.styles:
            raise ValueError("styles must not be empty")
        if not 0.0 <= self.hard_fraction <= 1.0:
            raise ValueError("hard_fraction must be in [0, 1]")
        if self.noise_boost <= 0.0:
            raise ValueError("noise_boost must be > 0")

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict view of the whole tree (JSON/YAML serialisable)."""
        return asdict(self)

    def override(self, overrides: Mapping[str, Any]) -> "GeneratorConfig":
        """Return a new config with a nested mapping applied on top.

        Unknown keys raise :class:`ValueError` rather than being ignored --
        a silently dropped override is a reproducibility bug that surfaces
        three days later as "why is my hard set not hard".

        Ranges may be given as ``[lo, hi]``, ``[lo, hi, log]`` or a mapping.
        """
        return _merge(self, overrides, path="")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GeneratorConfig":
        """Build from a mapping, filling anything absent with defaults."""
        return cls().override(data)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "GeneratorConfig":
        """Load a YAML file of overrides on top of the defaults."""
        import yaml  # local import: yaml is only needed for file-based configs

        data = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"{path}: top level must be a mapping")
        return cls.from_dict(data)

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Write the fully resolved config (all defaults made explicit)."""
        import yaml

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        header = (f"# Drift-Sense resolved config\n"
                  f"# schema {self.schema_version}  hash {self.version_hash()}\n")
        p.write_text(header + yaml.safe_dump(self.to_dict(), sort_keys=True))

    # -- identity ---------------------------------------------------------- #
    #: Sections excluded from the version hash: they change how the data is
    #: produced and stored, never what it contains.  Including them would make
    #: ``--workers 8`` report a config mismatch against ``--workers 2``, and a
    #: hash that fires on operational choices is a hash people learn to ignore.
    HASH_EXCLUDED: Tuple[str, ...] = ("runtime", "output")

    def canonical_json(self) -> str:
        """Stable, sorted JSON used as the input to :meth:`version_hash`.

        Only the fields that affect pixel content take part; see
        :data:`HASH_EXCLUDED`.
        """
        data = {k: v for k, v in self.to_dict().items()
                if k not in self.HASH_EXCLUDED}
        return json.dumps(data, sort_keys=True, separators=(",", ":"),
                          default=float)

    def version_hash(self) -> str:
        """12-hex-char fingerprint of schema + every configured value.

        Store this next to the seed.  "Reproducible from the seed" is only true
        for a fixed config; this is what makes a config change visible.
        """
        h = hashlib.sha256()
        h.update(self.schema_version.encode())
        h.update(self.canonical_json().encode())
        return h.hexdigest()[:12]

    def describe(self) -> str:
        """Human-readable one-screen summary for logs and the manifest header."""
        return (
            f"Drift-Sense config  schema={self.schema_version} "
            f"hash={self.version_hash()}\n"
            f"  styles          : {', '.join(self.styles)}\n"
            f"  hard_fraction   : {self.hard_fraction:.2f}\n"
            f"  noise_boost     : {self.noise_boost:.2f}\n"
            f"  ref dose  e/px  : {self.ref_optics.dose_e_per_px.lo:.0f}"
            f" .. {self.ref_optics.dose_e_per_px.hi:.0f}"
            f" ({'log' if self.ref_optics.dose_e_per_px.log else 'uniform'})\n"
            f"  srch dose e/px  : {self.search_optics.dose_e_per_px.lo:.0f}"
            f" .. {self.search_optics.dose_e_per_px.hi:.0f}"
            f" ({'log' if self.search_optics.dose_e_per_px.log else 'uniform'})\n"
            f"  rel rotation    : sigma {self.acquisition.relative_rotation_deg_sigma:.2f} deg\n"
            f"  output          : {self.output.format}"
        )


# --------------------------------------------------------------------------- #
# Recursive override machinery
# --------------------------------------------------------------------------- #
def _coerce_range(current: Union[Range, IntRange], value: Any,
                  path: str) -> Union[Range, IntRange]:
    """Accept ``[lo, hi]``, ``[lo, hi, log]`` or a mapping for a range field."""
    if isinstance(value, (Range, IntRange)):
        return value
    if isinstance(value, Mapping):
        return replace(current, **dict(value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        vals = list(value)
        if isinstance(current, IntRange):
            if len(vals) != 2:
                raise ValueError(f"{path}: IntRange needs [lo, hi], got {vals!r}")
            return IntRange(int(vals[0]), int(vals[1]))
        if len(vals) == 2:
            return Range(float(vals[0]), float(vals[1]))
        if len(vals) == 3:
            return Range(float(vals[0]), float(vals[1]), bool(vals[2]))
        raise ValueError(f"{path}: Range needs [lo, hi] or [lo, hi, log]")
    raise ValueError(f"{path}: cannot read {value!r} as a range")


def _merge(obj: Any, overrides: Mapping[str, Any], path: str) -> Any:
    """Recursively apply a nested mapping to a frozen dataclass instance."""
    if not is_dataclass(obj):
        raise TypeError(f"{path}: not a dataclass")
    known = {f.name for f in fields(obj)}
    kwargs: Dict[str, Any] = {}
    for key, value in overrides.items():
        here = f"{path}.{key}" if path else key
        if key not in known:
            near = ", ".join(sorted(known))
            raise ValueError(f"unknown config key {here!r}; expected one of: {near}")
        current = getattr(obj, key)
        if isinstance(current, (Range, IntRange)):
            kwargs[key] = _coerce_range(current, value, here)
        elif is_dataclass(current):
            if not isinstance(value, Mapping):
                raise ValueError(f"{here}: expected a mapping, got {type(value).__name__}")
            kwargs[key] = _merge(current, value, here)
        elif isinstance(current, tuple):
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise ValueError(f"{here}: expected a list, got {value!r}")
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return replace(obj, **kwargs)


# --------------------------------------------------------------------------- #
# Preset configurations
# --------------------------------------------------------------------------- #
def preset(name: str) -> GeneratorConfig:
    """Named presets used by the CLI and the eval scripts.

    Args:
        name: One of ``"default"``, ``"easy"``, ``"hidden_test_sim"``,
            ``"dram_only"``, ``"finfet_only"``.

    Returns:
        A resolved :class:`GeneratorConfig`.
    """
    base = GeneratorConfig()
    if name == "default":
        return base
    if name == "easy":
        # Sanity-check set: bright landmarks, low noise, little rotation.
        return base.override({
            "hard_fraction": 0.0,
            "acquisition": {"relative_rotation_deg_sigma": 0.3},
            "search_optics": {"dose_e_per_px": [200.0, 700.0, True],
                              "read_noise": [0.004, 0.012]},
        })
    if name == "hidden_test_sim":
        # "MORE noisy than the training examples", more periodic hard cases.
        return base.override({
            "hard_fraction": 0.50,
            "noise_boost": 1.8,
            "search_optics": {"dose_e_per_px": [8.0, 60.0, True],
                              "read_noise": [0.03, 0.09],
                              "stripe_noise": [0.01, 0.04],
                              "charging": [0.10, 0.30]},
        })
    if name == "dram_only":
        return base.override({"styles": ["dram"]})
    if name == "finfet_only":
        return base.override({"styles": ["finfet"]})
    raise ValueError(f"unknown preset: {name!r}")


PRESETS: Tuple[str, ...] = ("default", "easy", "hidden_test_sim",
                            "dram_only", "finfet_only")


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Compile-and-prove-itself check; run ``python -m driftsense.config``."""
    import tempfile

    rng = np.random.default_rng(0)

    # 1. ranges
    r = Range(1.0, 1000.0, log=True)
    v = r.draw_many(np.random.default_rng(1), 20000)
    assert 1.0 <= v.min() and v.max() <= 1000.0
    # log-uniform: half the samples below the geometric mean
    frac = float((v < r.mid).mean())
    assert 0.47 < frac < 0.53, frac
    lin = Range(1.0, 1000.0).draw_many(np.random.default_rng(1), 20000)
    assert float((lin < 31.6).mean()) < 0.05        # uniform hides the low-SNR tail
    assert IntRange(1, 3).draw(rng) in (1, 2, 3)
    for bad in (lambda: Range(5.0, 1.0), lambda: Range(0.0, 1.0, True),
                lambda: IntRange(3, 1)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("range validation did not fire")

    # 2. defaults + hash stability
    a, b = GeneratorConfig(), GeneratorConfig()
    assert a.version_hash() == b.version_hash()
    assert len(a.version_hash()) == 12
    c = a.override({"hard_fraction": 0.5})
    assert c.version_hash() != a.version_hash()
    # operational settings must NOT change the hash: the same data, produced
    # with a different worker count or output format, is the same data
    for op in ({"runtime": {"workers": 8}}, {"output": {"format": "png"}},
               {"runtime": {"chunk_size": 32}}):
        assert a.override(op).version_hash() == a.version_hash(), op
    # ... but anything that touches pixel content must
    for real in ({"noise_boost": 1.5}, {"styles": ["dram"]},
                 {"search_optics": {"dose_e_per_px": [5.0, 50.0, True]}},
                 {"acquisition": {"relative_rotation_deg_sigma": 3.0}}):
        assert a.override(real).version_hash() != a.version_hash(), real
    assert a.hard_fraction == 0.25, "override mutated the original (not frozen!)"

    # 3. nested overrides, all three range spellings
    d = a.override({
        "search_optics": {"dose_e_per_px": [5.0, 50.0, True],
                          "gamma": {"lo": 0.9, "hi": 1.1}},
        "acquisition": {"field_distortion_cells": [1, 4]},
        "styles": ["dram"],
    })
    assert d.search_optics.dose_e_per_px == Range(5.0, 50.0, True)
    assert d.search_optics.gamma == Range(0.9, 1.1)
    assert d.acquisition.field_distortion_cells == IntRange(1, 4)
    assert d.styles == ("dram",)
    assert d.ref_optics == a.ref_optics, "override leaked across siblings"

    # 4. unknown keys are rejected, not ignored
    for bad in ({"hard_frac": 0.5}, {"search_optics": {"dosee": 1}}):
        try:
            a.override(bad)
        except ValueError as exc:
            assert "unknown config key" in str(exc)
        else:
            raise AssertionError("unknown key was silently accepted")
    try:
        a.override({"styles": ["dram", "gaafet"]})
    except ValueError:
        pass
    else:
        raise AssertionError("style validation did not fire")

    # 5. YAML round-trip preserves the hash
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cfg.yaml"
        d.save_yaml(p)
        assert GeneratorConfig.from_yaml(p).version_hash() == d.version_hash()

    # 6. presets
    hashes = {n: preset(n).version_hash() for n in PRESETS}
    assert len(set(hashes.values())) == len(PRESETS), "presets collide"
    hts = preset("hidden_test_sim")
    assert hts.search_optics.dose_e_per_px.lo < a.search_optics.dose_e_per_px.lo
    assert hts.hard_fraction > a.hard_fraction

    print("config.py self-test OK")
    print(f"  default hash        : {a.version_hash()}")
    for n in PRESETS:
        print(f"  preset {n:16s}: {hashes[n]}")
    print()
    print(preset("hidden_test_sim").describe())


if __name__ == "__main__":
    _self_test()
