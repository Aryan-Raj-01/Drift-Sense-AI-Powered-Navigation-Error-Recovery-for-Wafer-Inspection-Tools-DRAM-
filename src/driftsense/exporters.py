"""Exporters to third-party training formats.

A note on the brief's "directly usable by PyTorch, TensorFlow and Ultralytics
without conversion" requirement: that is not satisfiable as stated, because the
three want mutually incompatible things on disk.  Ultralytics wants a fixed
``images/`` + ``labels/`` tree with one ``.txt`` per image holding normalised
``class cx cy w h``; TensorFlow wants ``tfrecord`` shards of serialised
``tf.Example``; PyTorch wants neither and is happiest with a ``Dataset`` object.
Writing all three as the primary format means three copies of 163 GB.

So the repository keeps one canonical manifest and converts on demand.  PyTorch
and TensorFlow need no conversion at all -- they read the manifest directly
through :mod:`driftsense.dataset` -- and this module materialises the other two.

**A warning about the Ultralytics path.**  This is not an object-detection
problem.  There is no class to detect: which region of the search image is
correct is defined *by the other image*, and every candidate region looks
identical.  A YOLO model given only the search image is being asked to guess.
The export exists because the brief asks for it and because a detector can be a
useful region proposer, but a detector alone cannot solve the task, and a
detector's mAP on this data is not a meaningful score.  Use
:mod:`driftsense.eval.metrics` instead.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from driftsense.config import IMG_N, GeneratorConfig
from driftsense.metadata import iter_rows
from driftsense.pipeline import plan_sample, render


def _require_images(rows: Sequence[Dict[str, Any]], root: Path) -> None:
    """Fail early and clearly when an export needs pixels that do not exist."""
    if not any(r.get("search_path") for r in rows):
        raise ValueError(
            f"{root} has no rendered images (manifest mode). Re-generate with "
            f"--format png, or use export_* with render=True.")


# --------------------------------------------------------------------------- #
# Ultralytics / YOLO
# --------------------------------------------------------------------------- #
def export_yolo(root: Union[str, Path], out: Union[str, Path],
                val_fraction: float = 0.1, render_missing: bool = True,
                cfg: Optional[GeneratorConfig] = None,
                limit: Optional[int] = None) -> Path:
    """Write an Ultralytics-format dataset.

    The label is the reference footprint's axis-aligned bounding box inside the
    search image, as a single class ``target``.  Only the search image is
    written as the "image"; the reference is copied alongside it so a
    two-stream model can still find it.

    Args:
        root: Drift-Sense dataset root.
        out: Destination directory.
        val_fraction: Fraction of samples placed in ``val``.
        render_missing: Render pixels for manifest-mode datasets.
        cfg: Config to render with; read from the dataset when omitted.
        limit: Export at most this many samples.

    Returns:
        Path to the written ``data.yaml``.
    """
    import cv2

    root, out = Path(root), Path(out)
    rows = sorted(iter_rows(root / "labels.jsonl"), key=lambda r: r["id"])
    if limit:
        rows = rows[:limit]
    if not render_missing:
        _require_images(rows, root)

    cfg = cfg or _config_for(root)
    n_val = int(len(rows) * val_fraction)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "reference" / split).mkdir(parents=True, exist_ok=True)

    for k, row in enumerate(rows):
        split = "val" if k < n_val else "train"
        stem = f"{int(row['id']):07d}"

        if row.get("search_path"):
            search_src = root / row["search_path"]
            ref_src = root / row["reference_path"]
            shutil.copyfile(search_src, out / "images" / split / f"{stem}.png")
            shutil.copyfile(ref_src, out / "reference" / split / f"{stem}.png")
        else:
            plan = plan_sample(int(row["seed"]), int(row["id"]), cfg)
            ref, search = render(plan)
            cv2.imwrite(str(out / "images" / split / f"{stem}.png"), search)
            cv2.imwrite(str(out / "reference" / split / f"{stem}.png"), ref)

        x0, y0 = row["bbox_x0"], row["bbox_y0"]
        x1, y1 = row["bbox_x1"], row["bbox_y1"]
        cx = 0.5 * (x0 + x1) / IMG_N
        cy = 0.5 * (y0 + y1) / IMG_N
        w = (x1 - x0) / IMG_N
        h = (y1 - y0) / IMG_N
        (out / "labels" / split / f"{stem}.txt").write_text(
            f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    data_yaml = out / "data.yaml"
    data_yaml.write_text(
        f"# Drift-Sense -> Ultralytics export\n"
        f"# NOTE: the reference image for sample N is reference/<split>/N.png.\n"
        f"# A single-stream detector cannot solve this task; see the module\n"
        f"# docstring in driftsense/exporters.py.\n"
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: [target]\n")
    return data_yaml


# --------------------------------------------------------------------------- #
# TensorFlow
# --------------------------------------------------------------------------- #
def export_tfrecord(root: Union[str, Path], out: Union[str, Path],
                    shard_size: int = 512, cfg: Optional[GeneratorConfig] = None,
                    limit: Optional[int] = None) -> List[Path]:
    """Write ``tf.Example`` shards.

    Images are stored PNG-encoded inside the record rather than raw, which keeps
    a 512-sample shard around 800 MB instead of 1 GB and lets TensorFlow decode
    them lazily on the input thread.

    Args:
        root: Drift-Sense dataset root.
        out: Destination directory.
        shard_size: Samples per ``.tfrecord`` file.
        cfg: Config to render with.
        limit: Export at most this many samples.

    Returns:
        Paths of the written shards.

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    import tensorflow as tf

    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(iter_rows(root / "labels.jsonl"), key=lambda r: r["id"])
    if limit:
        rows = rows[:limit]
    cfg = cfg or _config_for(root)

    def _bytes(v: bytes) -> "tf.train.Feature":
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

    def _floats(v: Iterable[float]) -> "tf.train.Feature":
        return tf.train.Feature(float_list=tf.train.FloatList(
            value=[float(x) for x in v]))

    def _int(v: int) -> "tf.train.Feature":
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))

    written: List[Path] = []
    for start in range(0, len(rows), shard_size):
        chunk = rows[start:start + shard_size]
        path = out / f"driftsense-{start // shard_size:05d}.tfrecord"
        with tf.io.TFRecordWriter(str(path)) as writer:
            for row in chunk:
                plan = plan_sample(int(row["seed"]), int(row["id"]), cfg)
                ref, search = render(plan)
                ex = tf.train.Example(features=tf.train.Features(feature={
                    "reference": _bytes(tf.io.encode_png(ref[..., None]).numpy()),
                    "search": _bytes(tf.io.encode_png(search[..., None]).numpy()),
                    "center": _floats([row["gt_x"], row["gt_y"]]),
                    "center_norm": _floats([row["gt_x"] / IMG_N,
                                            row["gt_y"] / IMG_N]),
                    "quad": _floats([row[f"quad_{a}{i}"]
                                     for i in range(4) for a in ("x", "y")]),
                    "rot_deg": _floats([row["rel_rotation_deg"]]),
                    "scale": _floats([row["scale_ratio"]]),
                    "seed": _int(row["seed"]),
                    "style": _bytes(row["style"].encode()),
                    "difficulty": _bytes(row["difficulty"].encode()),
                }))
                writer.write(ex.SerializeToString())
        written.append(path)
    return written


# --------------------------------------------------------------------------- #
# WebDataset
# --------------------------------------------------------------------------- #
def export_webdataset(root: Union[str, Path], out: Union[str, Path],
                      shard_size: int = 512,
                      cfg: Optional[GeneratorConfig] = None,
                      limit: Optional[int] = None) -> List[Path]:
    """Write plain ``.tar`` shards in WebDataset layout.

    This is the format to reach for if you do decide to materialise a large
    dataset: sequential reads, no per-file metadata storm, and it streams
    straight from object storage.  200 000 loose PNGs in a directory tree is the
    thing that makes a network filesystem fall over.

    Args:
        root: Drift-Sense dataset root.
        out: Destination directory.
        shard_size: Samples per shard.
        cfg: Config to render with.
        limit: Export at most this many samples.

    Returns:
        Paths of the written shards.
    """
    import io
    import tarfile

    import cv2

    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(iter_rows(root / "labels.jsonl"), key=lambda r: r["id"])
    if limit:
        rows = rows[:limit]
    cfg = cfg or _config_for(root)

    written: List[Path] = []
    for start in range(0, len(rows), shard_size):
        chunk = rows[start:start + shard_size]
        path = out / f"driftsense-{start // shard_size:05d}.tar"
        with tarfile.open(path, "w") as tar:
            for row in chunk:
                plan = plan_sample(int(row["seed"]), int(row["id"]), cfg)
                ref, search = render(plan)
                key = f"{int(row['id']):07d}"
                for name, img in (("reference.png", ref), ("search.png", search)):
                    ok, buf = cv2.imencode(".png", img)
                    if not ok:  # pragma: no cover
                        raise RuntimeError("PNG encode failed")
                    _add(tar, f"{key}.{name}", buf.tobytes())
                _add(tar, f"{key}.json", json.dumps(row).encode())
        written.append(path)
    return written


def _add(tar, name: str, payload: bytes) -> None:
    """Append an in-memory member to an open tarfile."""
    import io
    import tarfile

    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tar.addfile(info, io.BytesIO(payload))


def _config_for(root: Path) -> GeneratorConfig:
    """Load the config a dataset was generated with."""
    p = root / "config.yaml"
    if p.exists():
        return GeneratorConfig.from_yaml(p)
    return GeneratorConfig()


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.exporters``."""
    import tempfile

    from driftsense.metadata import LabelWriter, label_row
    from driftsense.rng import sample_seeds

    cfg = GeneratorConfig()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ds"
        seeds = sample_seeds(20260803, 4)
        plans = [plan_sample(int(s), i, cfg) for i, s in enumerate(seeds)]
        with LabelWriter(root) as w:
            for p in plans:
                w.write(label_row(p))
        cfg.save_yaml(root / "config.yaml")

        # 1. YOLO export renders, writes both streams and a valid data.yaml
        yolo = Path(td) / "yolo"
        data_yaml = export_yolo(root, yolo, val_fraction=0.25)
        assert data_yaml.exists()
        imgs = sorted((yolo / "images" / "train").glob("*.png"))
        labs = sorted((yolo / "labels" / "train").glob("*.txt"))
        refs = sorted((yolo / "reference" / "train").glob("*.png"))
        assert len(imgs) == len(labs) == len(refs) == 3
        assert len(list((yolo / "images" / "val").glob("*.png"))) == 1

        # 2. the YOLO box round-trips back to the stored bbox
        row = plans[1].ground_truth()
        text = (yolo / "labels" / "train" / "0000001.txt").read_text().split()
        cls, cx, cy, w_, h_ = text[0], *map(float, text[1:])
        assert cls == "0"
        assert abs(cx * IMG_N - 0.5 * (row["bbox"][0] + row["bbox"][2])) < 1e-2
        assert abs(cy * IMG_N - 0.5 * (row["bbox"][1] + row["bbox"][3])) < 1e-2
        assert 0.0 < w_ < 0.2 and 0.0 < h_ < 0.2      # ~100/1000 of the frame
        # the bbox is bigger than the footprint under rotation, as documented
        assert w_ * IMG_N >= row["footprint_px"] - 1e-6

        # 3. WebDataset shards contain three members per sample, in order
        import tarfile

        wds = export_webdataset(root, Path(td) / "wds", shard_size=3)
        assert len(wds) == 2
        with tarfile.open(wds[0]) as tar:
            names = tar.getnames()
            assert len(names) == 9
            assert "0000000.reference.png" in names
            assert "0000000.search.png" in names
            meta = json.loads(tar.extractfile("0000000.json").read())
            assert meta["seed"] == plans[0].seed

        # 4. manifest-mode datasets are refused when render_missing is off
        rows = list(iter_rows(root / "labels.jsonl"))
        try:
            _require_images(rows, root)
        except ValueError as exc:
            assert "manifest mode" in str(exc)
        else:
            raise AssertionError("should have refused a pixel-free dataset")

    try:
        import tensorflow  # noqa: F401

        tf_available = True
    except Exception:
        tf_available = False

    print("exporters.py self-test OK")
    print(f"  yolo export            : images+labels+reference, data.yaml OK")
    print(f"  yolo bbox round-trip   : < 0.01 px")
    print(f"  webdataset shards      : 2 shards, 3 members/sample")
    print(f"  tfrecord               : "
          f"{'available' if tf_available else 'skipped (tensorflow not installed)'}")
    print(f"  note                   : YOLO alone cannot solve this task -- "
          f"see module docstring")


if __name__ == "__main__":
    _self_test()
