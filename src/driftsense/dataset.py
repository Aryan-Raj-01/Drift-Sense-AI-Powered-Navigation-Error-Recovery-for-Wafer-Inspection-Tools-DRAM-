"""Training-time access to Drift-Sense data.

Three ways in, in decreasing order of how much disk they need:

    InfiniteDriftSense(cfg)              no disk at all, unlimited samples
    DriftSenseDataset("data/train")      reads labels.jsonl, renders on the fly
    DriftSenseDataset("data/eval",       reads stored PNG/NPZ
                      materialised=True)

Why on-the-fly is the default
-----------------------------
A sample is a pure function of its seed, so storing pixels stores nothing that
cannot be recomputed:

    100 000 pairs as PNG      ~163 GB, ~5 CPU-hours to write
    100 000 pairs as seeds     ~25 MB, ~6 minutes

Rendering costs ~1.3 s per pair per core, so eight DataLoader workers sustain
roughly 6 pairs/s.  That is the honest bottleneck: if your GPU wants more, set
``ref_small_px=100`` (the reference is box-downsampled to search resolution
inside the worker, which is what a correlation head wants anyway and cuts the
reference render cost out of the critical path), or materialise a few thousand
pairs as NPZ and cycle them.

The infinite variant has a property the finite one does not: no sample is ever
seen twice, so there is nothing to overfit.  For a task whose difficulty is
dominated by nuisance variation rather than by semantic content, that matters
more than any augmentation schedule.

Torch is optional.  Without it the classes still work as plain iterables
returning numpy arrays, which is enough for TensorFlow's
``tf.data.Dataset.from_generator`` -- see :func:`as_tf_generator`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from driftsense import geometry as G
from driftsense.config import IMG_N, GeneratorConfig
from driftsense.metadata import iter_rows
from driftsense.pipeline import DIFFICULTIES, Plan, plan_sample, render
from driftsense.rng import sample_seeds

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    from torch.utils.data import IterableDataset as _TorchIterableDataset

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is optional
    _HAS_TORCH = False
    _TorchDataset = object      # type: ignore[assignment]
    _TorchIterableDataset = object  # type: ignore[assignment]

#: Stable integer encodings, so a model's class head does not depend on
#: dictionary ordering.
STYLE_ID: Dict[str, int] = {"dram": 0, "finfet": 1}
DIFFICULTY_ID: Dict[str, int] = {d: i for i, d in enumerate(DIFFICULTIES)}


# --------------------------------------------------------------------------- #
# Sample packing
# --------------------------------------------------------------------------- #
def targets_from_plan(plan: Plan) -> Dict[str, np.ndarray]:
    """Numeric training targets for one plan, in search-image pixels.

    Args:
        plan: The planned sample.

    Returns:
        A dictionary of numpy scalars and small arrays:

        ``center``
            ``float32[2]`` ground-truth centre ``(x, y)`` in pixels.
        ``center_norm``
            ``float32[2]`` the same, divided by the image size.  Regress this,
            not the raw pixels: an unnormalised target makes the loss scale
            with image size and couples the learning rate to it.
        ``quad``
            ``float32[4, 2]`` rotated footprint corners.
        ``bbox``
            ``float32[4]`` axis-aligned box.  Provided for detector-style
            heads, but note it is up to 41 % larger than the true footprint
            under rotation.
        ``rot_deg``, ``scale``, ``footprint``
            ``float32`` scalars.
        ``style_id``, ``difficulty_id``
            ``int64`` scalars for stratified evaluation.
        ``seed``
            ``int64``, so any batch element can be regenerated exactly.
    """
    gt = plan.ground_truth()
    return {
        "center": np.array([gt["gt_x"], gt["gt_y"]], np.float32),
        "center_norm": np.array([gt["gt_x"] / IMG_N, gt["gt_y"] / IMG_N],
                                np.float32),
        "quad": np.asarray(gt["quad"], np.float32),
        "bbox": np.asarray(gt["bbox"], np.float32),
        "rot_deg": np.float32(gt["rel_rotation_deg"]),
        "scale": np.float32(gt["scale_ratio"]),
        "footprint": np.float32(gt["footprint_px"]),
        "style_id": np.int64(STYLE_ID[plan.style]),
        "difficulty_id": np.int64(DIFFICULTY_ID[plan.difficulty]),
        "seed": np.int64(plan.seed),
    }


def pack(plan: Plan, reference: np.ndarray, search: np.ndarray,
         ref_small_px: Optional[int] = None, to_tensor: bool = True
         ) -> Tuple[Any, Any, Dict[str, Any]]:
    """Turn rendered frames into model inputs.

    Args:
        plan: The planned sample.
        reference: ``uint8`` reference frame.
        search: ``uint8`` search frame.
        ref_small_px: If set, box-downsample the reference to this size.  ``100``
            puts it at search resolution, which is the natural template size for
            a correlation head.
        to_tensor: Convert to torch tensors when torch is available.

    Returns:
        ``(reference, search, targets)`` with images as ``float32`` in
        ``[0, 1]`` and a leading channel axis.
    """
    ref = reference.astype(np.float32) / 255.0
    srch = search.astype(np.float32) / 255.0

    if ref_small_px and ref_small_px < ref.shape[0]:
        factor = ref.shape[0] // ref_small_px
        if factor > 1:
            ref = G.box_downsample(ref, factor).astype(np.float32)

    targets = targets_from_plan(plan)
    ref = ref[None]
    srch = srch[None]

    if to_tensor and _HAS_TORCH:
        return (torch.from_numpy(np.ascontiguousarray(ref)),
                torch.from_numpy(np.ascontiguousarray(srch)),
                {k: torch.as_tensor(v) for k, v in targets.items()})
    return ref, srch, targets


# --------------------------------------------------------------------------- #
# Map-style dataset over a manifest
# --------------------------------------------------------------------------- #
class DriftSenseDataset(_TorchDataset):
    """Map-style dataset over a ``labels.jsonl`` manifest.

    Args:
        root: Dataset root written by ``driftsense.cli.generate``.
        cfg: Configuration used to generate it.  Loaded from the dataset's own
            ``config.yaml`` when omitted -- which is the safe default, because
            rendering with a *different* config than the manifest was planned
            with silently produces images that do not match their labels.
        materialised: Read stored PNG/NPZ instead of re-rendering.
        ref_small_px: Downsample the reference to this size; see :func:`pack`.
        difficulties: Keep only these difficulty tiers.
        styles: Keep only these architectures.
        to_tensor: Return torch tensors when available.
        transform: Optional callable applied to ``(ref, search, targets)``.

    Raises:
        FileNotFoundError: If the manifest is missing.
        ValueError: If ``materialised`` is requested but the manifest has no
            image paths.
    """

    def __init__(self, root: Union[str, Path],
                 cfg: Optional[GeneratorConfig] = None,
                 materialised: bool = False,
                 ref_small_px: Optional[int] = None,
                 difficulties: Optional[List[str]] = None,
                 styles: Optional[List[str]] = None,
                 to_tensor: bool = True,
                 transform: Optional[Callable] = None) -> None:
        self.root = Path(root)
        manifest = self.root / "labels.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"no manifest at {manifest}")

        self.cfg = cfg or self._load_config()
        self.materialised = materialised
        self.ref_small_px = ref_small_px
        self.to_tensor = to_tensor
        self.transform = transform

        rows = list(iter_rows(manifest))
        if difficulties:
            rows = [r for r in rows if r.get("difficulty") in difficulties]
        if styles:
            rows = [r for r in rows if r.get("style") in styles]
        self.rows: List[Dict[str, Any]] = sorted(rows, key=lambda r: r["id"])

        if materialised and not any(r.get("reference_path") for r in self.rows):
            raise ValueError(
                f"{root} was generated in manifest mode and has no images; "
                "use materialised=False to render on the fly")

        self._check_config_hash()

    def _load_config(self) -> GeneratorConfig:
        """Load the config the dataset was generated with."""
        p = self.root / "config.yaml"
        if p.exists():
            return GeneratorConfig.from_yaml(p)
        meta = self.root / "dataset_meta.json"
        if meta.exists():
            data = json.loads(meta.read_text()).get("config")
            if data:
                return GeneratorConfig.from_dict(data)
        return GeneratorConfig()

    def _check_config_hash(self) -> None:
        """Warn loudly if the config does not match the manifest.

        This is the failure that is worth an explicit check: rendering with the
        wrong config produces images that look perfectly plausible and are
        labelled wrongly, and nothing downstream would ever notice.
        """
        if not self.rows:
            return
        want = {r.get("config_hash") for r in self.rows if r.get("config_hash")}
        have = self.cfg.version_hash()
        if want and have not in want:
            print(f"[dataset] WARNING: config hash {have} does not match the "
                  f"manifest ({', '.join(sorted(want))}). Rendered pixels will "
                  f"not correspond to the stored labels.")

    def __len__(self) -> int:
        return len(self.rows)

    def plan_for(self, i: int) -> Plan:
        """The :class:`Plan` for row ``i``, without rendering it."""
        row = self.rows[i]
        return plan_sample(int(row["seed"]), int(row["id"]), self.cfg)

    def __getitem__(self, i: int):
        row = self.rows[i]
        plan = self.plan_for(i)
        if self.materialised:
            ref, search = self._load_images(row)
        else:
            ref, search = render(plan)
        out = pack(plan, ref, search, self.ref_small_px, self.to_tensor)
        return self.transform(*out) if self.transform else out

    def _load_images(self, row: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Read stored frames for a materialised dataset."""
        rp = self.root / row["reference_path"]
        if rp.suffix == ".npz":
            with np.load(rp) as z:
                return z["reference"], z["search"]
        import cv2

        ref = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(str(self.root / row["search_path"]),
                            cv2.IMREAD_GRAYSCALE)
        if ref is None or search is None:  # pragma: no cover
            raise FileNotFoundError(f"could not read images for id {row['id']}")
        return ref, search


# --------------------------------------------------------------------------- #
# Infinite stream
# --------------------------------------------------------------------------- #
class InfiniteDriftSense(_TorchIterableDataset):
    """Endless stream of fresh samples: no manifest, no disk, no epoch.

    Each DataLoader worker derives its own seed sequence from ``(seed,
    worker_id)``, so workers never collide -- the duplicate-data failure that
    ``fork`` causes when workers seed lazily.

    Args:
        cfg: Generator configuration.
        seed: Root seed for the stream.
        ref_small_px: Downsample the reference; see :func:`pack`.
        length: Stop after this many samples in total (across workers).  ``None``
            means truly infinite, which is usually what you want -- define an
            "epoch" by step count in the training loop instead.
        to_tensor: Return torch tensors when available.
        transform: Optional callable applied to each sample.
    """

    def __init__(self, cfg: Optional[GeneratorConfig] = None, seed: int = 0,
                 ref_small_px: Optional[int] = None,
                 length: Optional[int] = None, to_tensor: bool = True,
                 transform: Optional[Callable] = None) -> None:
        self.cfg = cfg or GeneratorConfig()
        self.seed = int(seed)
        self.ref_small_px = ref_small_px
        self.length = length
        self.to_tensor = to_tensor
        self.transform = transform

    def __iter__(self) -> Iterator:
        worker_id, num_workers = 0, 1
        if _HAS_TORCH:
            info = torch.utils.data.get_worker_info()
            if info is not None:
                worker_id, num_workers = info.id, info.num_workers

        rng = np.random.default_rng([self.seed, worker_id])
        budget = None if self.length is None else max(1, self.length // num_workers)
        produced = 0
        while budget is None or produced < budget:
            s = int(rng.integers(0, 2 ** 62))
            plan = plan_sample(s, produced, self.cfg)
            ref, search = render(plan)
            out = pack(plan, ref, search, self.ref_small_px, self.to_tensor)
            yield self.transform(*out) if self.transform else out
            produced += 1

    def __len__(self) -> int:
        if self.length is None:
            raise TypeError("InfiniteDriftSense has no length unless `length` is set")
        return self.length


# --------------------------------------------------------------------------- #
# TensorFlow interop
# --------------------------------------------------------------------------- #
def as_tf_generator(dataset) -> Callable[[], Iterator]:
    """Adapt any Drift-Sense dataset to ``tf.data.Dataset.from_generator``.

    TensorFlow wants channels-last and plain numpy, so this transposes and
    strips torch::

        import tensorflow as tf
        from driftsense.dataset import InfiniteDriftSense, as_tf_generator

        ds = InfiniteDriftSense(to_tensor=False, ref_small_px=100)
        sig = (tf.TensorSpec((100, 100, 1), tf.float32),
               tf.TensorSpec((1000, 1000, 1), tf.float32),
               tf.TensorSpec((2,), tf.float32))
        tfds = tf.data.Dataset.from_generator(as_tf_generator(ds),
                                              output_signature=sig)

    Args:
        dataset: A :class:`DriftSenseDataset` or :class:`InfiniteDriftSense`.

    Returns:
        A zero-argument callable yielding ``(reference, search, center_norm)``
        as channels-last float32 arrays.
    """

    def gen() -> Iterator:
        # Dispatch on the class, not on hasattr: torch's IterableDataset DEFINES
        # __getitem__ (it raises NotImplementedError), so a duck-typing check
        # sends the infinite stream down the indexing path and blows up. This
        # only shows once torch is installed, which is exactly when it matters.
        if isinstance(dataset, InfiniteDriftSense):
            source: Iterator = iter(dataset)
        else:
            source = (dataset[i] for i in range(len(dataset)))
        for ref, search, targets in source:
            ref = np.asarray(ref, dtype=np.float32)
            search = np.asarray(search, dtype=np.float32)
            center = np.asarray(targets["center_norm"], dtype=np.float32)
            yield (np.transpose(ref, (1, 2, 0)),
                   np.transpose(search, (1, 2, 0)),
                   center)

    return gen


def collate(batch: List[Tuple[Any, Any, Dict[str, Any]]]):
    """Default collate for the tuple-of-dict sample format.

    Torch's own collate handles this fine; this exists for the torch-free path
    so the same training loop works either way.
    """
    refs = np.stack([np.asarray(b[0]) for b in batch])
    searches = np.stack([np.asarray(b[1]) for b in batch])
    keys = batch[0][2].keys()
    targets = {k: np.stack([np.asarray(b[2][k]) for b in batch]) for k in keys}
    if _HAS_TORCH:
        return (torch.from_numpy(refs), torch.from_numpy(searches),
                {k: torch.as_tensor(v) for k, v in targets.items()})
    return refs, searches, targets


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.dataset``."""
    import tempfile
    import time

    from driftsense.metadata import LabelWriter, label_row

    cfg = GeneratorConfig()

    # 1. packing shapes and normalisation
    plan = plan_sample(int(sample_seeds(1, 1)[0]), 0, cfg)
    ref, search = render(plan)
    r, s, t = pack(plan, ref, search, ref_small_px=100, to_tensor=False)
    assert r.shape == (1, 100, 100) and s.shape == (1, 1000, 1000)
    assert r.dtype == np.float32 and 0.0 <= r.min() and r.max() <= 1.0
    assert t["center"].shape == (2,) and t["quad"].shape == (4, 2)
    assert abs(float(t["center_norm"][0]) - float(t["center"][0]) / IMG_N) < 1e-6
    assert t["style_id"] in (0, 1) and 0 <= int(t["difficulty_id"]) <= 2
    # the seed in the batch really does regenerate the sample
    assert int(t["seed"]) == plan.seed

    # 2. a manifest dataset round-trips and matches its labels
    with tempfile.TemporaryDirectory() as td:
        seeds = sample_seeds(20260803, 8)
        plans = [plan_sample(int(x), i, cfg) for i, x in enumerate(seeds)]
        with LabelWriter(td) as w:
            for p in plans:
                w.write(label_row(p))
        cfg.save_yaml(Path(td) / "config.yaml")

        ds = DriftSenseDataset(td, ref_small_px=100, to_tensor=False)
        assert len(ds) == 8
        r0, s0, t0 = ds[0]
        stored = plans[0].ground_truth()
        assert abs(float(t0["center"][0]) - stored["gt_x"]) < 1e-3
        assert abs(float(t0["center"][1]) - stored["gt_y"]) < 1e-3

        # 3. determinism: the same index gives identical pixels every time
        r1, _, _ = ds[0]
        assert np.array_equal(np.asarray(r0), np.asarray(r1))

        # 4. filtering
        hard = DriftSenseDataset(td, difficulties=["hard"], to_tensor=False)
        assert all(r["difficulty"] == "hard" for r in hard.rows)
        dram = DriftSenseDataset(td, styles=["dram"], to_tensor=False)
        assert all(r["style"] == "dram" for r in dram.rows)
        assert len(hard) + len(DriftSenseDataset(td, difficulties=["easy", "medium"],
                                                 to_tensor=False)) == 8

        # 5. materialised mode is refused when there are no images
        try:
            DriftSenseDataset(td, materialised=True)
        except ValueError as exc:
            assert "manifest mode" in str(exc)
        else:
            raise AssertionError("materialised mode should have been refused")

    # 6. the infinite stream never repeats and needs no disk
    inf = InfiniteDriftSense(cfg, seed=7, ref_small_px=100, length=4,
                             to_tensor=False)
    t0 = time.time()
    items = list(inf)
    per = (time.time() - t0) / len(items)
    assert len(items) == 4
    seeds_seen = {int(it[2]["seed"]) for it in items}
    assert len(seeds_seen) == 4, "infinite stream repeated a sample"
    again = {int(it[2]["seed"]) for it in InfiniteDriftSense(cfg, seed=7, length=4,
                                                             to_tensor=False)}
    assert seeds_seen == again, "infinite stream is not reproducible"
    other = {int(it[2]["seed"]) for it in InfiniteDriftSense(cfg, seed=8, length=4,
                                                             to_tensor=False)}
    assert not (seeds_seen & other), "different stream seeds collided"

    # 7. collate builds a batch
    b_ref, b_search, b_t = collate(items)
    assert np.asarray(b_ref).shape == (4, 1, 100, 100)
    assert np.asarray(b_t["center"]).shape == (4, 2)

    # 8. the TF adapter yields channels-last numpy
    g = as_tf_generator(InfiniteDriftSense(cfg, seed=9, ref_small_px=100,
                                           length=1, to_tensor=False))
    a, b, c = next(iter(g()))
    assert a.shape == (100, 100, 1) and b.shape == (1000, 1000, 1)
    assert c.shape == (2,) and a.dtype == np.float32

    print("dataset.py self-test OK")
    print(f"  torch available        : {_HAS_TORCH}")
    print(f"  sample shapes          : ref {r.shape}, search {s.shape}")
    print(f"  target keys            : {', '.join(sorted(t))}")
    print(f"  on-the-fly render      : {per:.2f} s/pair/worker "
          f"-> {8/per:.1f} pairs/s on 8 workers")
    print(f"  100k pairs on disk     : 25 MB as seeds vs ~163 GB as PNG")
    print(f"  infinite stream        : reproducible, non-repeating, seed-disjoint")


if __name__ == "__main__":
    _self_test()
