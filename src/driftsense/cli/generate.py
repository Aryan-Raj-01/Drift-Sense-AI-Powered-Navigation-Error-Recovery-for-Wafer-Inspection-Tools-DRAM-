"""``generate.py`` -- produce a Drift-Sense dataset.

    python -m driftsense.cli.generate --num-images 100000 --out data/train

Read this before generating 100 000 pairs
-----------------------------------------
100 000 pairs of 1000x1000 8-bit images is ~200 GB raw and ~100 GB as PNG, and
writing them takes longer than computing them.  You do not need them on disk:
every sample is a pure function of a 64-bit seed, so the default ``manifest``
format writes labels only -- about 25 MB for 100k -- and
:class:`driftsense.dataset.DriftSenseDataset` regenerates the pixels inside the
DataLoader workers.

    manifest mode : ~6 minutes,   25 MB, 100 000 samples
    png mode      : ~5 CPU-hours, ~100 GB, same 100 000 samples

Materialise a few thousand pairs for evaluation and eyeballing; stream the rest.

Concurrency notes
-----------------
* Seeds are derived in the parent, up front, by
  :func:`driftsense.rng.sample_seeds`.  With ``fork`` every worker inherits the
  parent's RNG state, so a pool that seeded lazily would emit N identical copies
  of the dataset and the images would look perfectly fine.
* Only the parent writes ``labels.jsonl``.  Workers return rows; interleaved
  writes from several processes would corrupt lines.
* ``OMP_NUM_THREADS=1`` is set before numpy is imported.  Otherwise each of 8
  workers starts 8 BLAS threads and the 64 of them fight over cache -- measured
  at roughly 3x slower than the single-threaded pool.
* Images are sharded 1000 per directory.  200 000 files in one directory makes
  ext4 lookups, ``ls``, ``rsync`` and git all pathological.
* ``--resume`` restarts from the ids already in the manifest, not from a row
  count: a pool finishes out of order, so after a crash the manifest can hold
  ``{0..999, 1200..1400}`` and a count-based resume would skip the gap forever.
"""

from __future__ import annotations

import os

# Must precede the numpy import: BLAS reads these when it loads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Iterable, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402

from driftsense.config import PRESETS, GeneratorConfig, preset  # noqa: E402
from driftsense.metadata import (  # noqa: E402
    LabelWriter,
    completed_ids,
    export,
    label_row,
    summarise,
)
from driftsense.pipeline import plan_sample, render, verify_pair  # noqa: E402
from driftsense.rng import resolve_workers, sample_seeds  # noqa: E402

try:
    import cv2

    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False

try:
    from PIL import Image

    _HAS_PIL = True
except Exception:  # pragma: no cover
    _HAS_PIL = False

#: Bytes per PNG pair, measured on this generator: ~1.65 MB (noise does not
#: compress). Used only to warn the user before they fill a disk.
BYTES_PER_PNG_PAIR = 1.65e6

_CFG: Dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def _init_worker(cfg: GeneratorConfig, options: Dict[str, Any]) -> None:
    """Pool initialiser: stash the config in the worker's globals."""
    global _CFG
    _CFG = {"cfg": cfg, **options}


def _shard_dir(root: Path, index: int, shard_size: int) -> Path:
    """Directory for a sample, sharded to keep directories small."""
    return root / "images" / f"{index // shard_size:05d}"


def _write_png(img: np.ndarray, path: Path, compress_level: int) -> None:
    """Write an 8-bit PNG as fast as the available library allows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _HAS_CV2:
        cv2.imwrite(str(path), img,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), int(compress_level)])
    elif _HAS_PIL:  # pragma: no cover
        Image.fromarray(img).save(path, compress_level=compress_level)
    else:  # pragma: no cover
        raise RuntimeError("need opencv-python or Pillow to write PNGs")


def _job(task: Tuple[int, int]) -> Dict[str, Any]:
    """Render (or merely plan) one sample and return its manifest row.

    Args:
        task: ``(index, seed)``.

    Returns:
        The label row, with image paths filled in when pixels were written.
    """
    index, seed = task
    cfg: GeneratorConfig = _CFG["cfg"]
    fmt: str = _CFG["format"]
    root = Path(_CFG["out"])
    shard_size: int = _CFG["shard_size"]

    plan = plan_sample(seed, index, cfg)

    if fmt == "manifest":
        return label_row(plan)

    ref, search = render(plan)
    ref_path = search_path = None

    if fmt == "png":
        d = _shard_dir(root, index, shard_size)
        rp = d / f"{index:07d}_ref.png"
        sp = d / f"{index:07d}_search.png"
        _write_png(ref, rp, _CFG["png_compress_level"])
        _write_png(search, sp, _CFG["png_compress_level"])
        ref_path = str(rp.relative_to(root))
        search_path = str(sp.relative_to(root))
    elif fmt == "npz":
        d = _shard_dir(root, index, shard_size)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{index:07d}.npz"
        np.savez_compressed(p, reference=ref, search=search)
        ref_path = search_path = str(p.relative_to(root))

    row = label_row(plan, ref_path, search_path)
    if _CFG["verify"]:
        row["verify_zncc"] = round(verify_pair(plan, ref, search), 4)
    return row


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def _progress(done: int, total: int, t0: float, extra: str = "") -> None:
    """Single-line progress with a throughput-based ETA."""
    elapsed = time.time() - t0
    rate = done / max(elapsed, 1e-9)
    eta = (total - done) / max(rate, 1e-9)
    sys.stdout.write(
        f"\r  {done}/{total}  {rate:7.2f}/s  elapsed {elapsed/60:6.1f}m  "
        f"eta {eta/60:6.1f}m {extra}   ")
    sys.stdout.flush()


def generate(cfg: GeneratorConfig, out: Path, num_images: int,
             root_seed: int, fmt: str, workers: int,
             resume: bool = False, start_index: int = 0,
             verify: bool = False, progress_every: int = 25) -> Dict[str, Any]:
    """Generate a dataset.

    Args:
        cfg: Generator configuration.
        out: Dataset root directory.
        num_images: Number of image pairs.
        root_seed: Dataset root seed.
        fmt: ``"manifest"``, ``"png"`` or ``"npz"``.
        workers: Process count; ``0`` means cores minus one.
        resume: Skip ids already present in ``labels.jsonl``.
        start_index: First sample index, for extending a dataset.
        verify: Compute the ZNCC label check for every rendered pair.
        progress_every: Progress print interval, in samples.

    Returns:
        The dataset summary produced by :func:`driftsense.metadata.summarise`.
    """
    out.mkdir(parents=True, exist_ok=True)
    n_workers = resolve_workers(workers)

    seeds = sample_seeds(root_seed, num_images, offset=start_index)
    tasks: List[Tuple[int, int]] = [
        (start_index + i, int(s)) for i, s in enumerate(seeds)]

    done_ids = completed_ids(out / "labels.jsonl") if resume else set()
    if done_ids:
        before = len(tasks)
        tasks = [t for t in tasks if t[0] not in done_ids]
        print(f"  resuming: {len(done_ids)} already done, "
              f"{len(tasks)} of {before} remaining")

    options = {
        "format": fmt,
        "out": str(out),
        "shard_size": cfg.output.shard_size,
        "png_compress_level": cfg.output.png_compress_level,
        "verify": bool(verify),
    }

    # Record what produced this dataset, before doing any work: a run that dies
    # halfway still leaves a readable provenance file.
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join(sys.argv),
        "num_images": num_images,
        "start_index": start_index,
        "root_seed": root_seed,
        "format": fmt,
        "workers": n_workers,
        "config_hash": cfg.version_hash(),
        "config": cfg.to_dict(),
    }
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2, default=float))
    cfg.save_yaml(out / "config.yaml")

    total = len(tasks)
    if total == 0:
        print("  nothing to do")
        return summarise(out)

    t0 = time.time()
    written = 0
    zncc: List[float] = []

    with LabelWriter(out, append=bool(done_ids)) as writer:
        if n_workers > 1:
            method = cfg.runtime.start_method
            if sys.platform == "win32" and method == "fork":
                method = "spawn"
            ctx = mp.get_context(method)
            with ctx.Pool(n_workers, initializer=_init_worker,
                          initargs=(cfg, options)) as pool:
                for row in pool.imap_unordered(_job, tasks,
                                               chunksize=cfg.runtime.chunk_size):
                    writer.write(row)
                    written += 1
                    if "verify_zncc" in row:
                        zncc.append(row["verify_zncc"])
                    if written % progress_every == 0 or written == total:
                        _progress(written, total, t0)
        else:
            _init_worker(cfg, options)
            for task in tasks:
                row = _job(task)
                writer.write(row)
                written += 1
                if "verify_zncc" in row:
                    zncc.append(row["verify_zncc"])
                if written % progress_every == 0 or written == total:
                    _progress(written, total, t0)

    print()
    export(out, parquet=cfg.output.write_parquet, csv=cfg.output.write_csv)
    summary = summarise(out)

    elapsed = time.time() - t0
    print(f"\n  wrote {written} samples in {elapsed/60:.1f} min "
          f"({written/max(elapsed,1e-9):.2f}/s, {n_workers} workers)")
    print(f"  manifest : {out/'labels.jsonl'}  ({summary['n']} rows total)")
    if cfg.output.write_csv:
        print(f"  csv      : {out/'labels.csv'}")
    print(f"  by style      : {summary['by_style']}")
    print(f"  by difficulty : {summary['by_difficulty']}")
    if zncc:
        arr = np.array(zncc)
        print(f"  label check   : ZNCC mean {arr.mean():.3f}, min {arr.min():.3f}"
              f"  (>0.35 mean is healthy)")
        if arr.mean() < 0.3:
            print("  WARNING: low ZNCC suggests a labelling problem, "
                  "not a hard dataset")
    return summary


def build_config(args: argparse.Namespace) -> GeneratorConfig:
    """Resolve the configuration from preset, YAML file and CLI overrides.

    Precedence, lowest to highest: preset, YAML file, explicit flags.  Flags win
    because they are the most local statement of intent.
    """
    cfg = preset(args.preset)
    if args.config:
        file_cfg = GeneratorConfig.from_yaml(args.config)
        # A YAML file is a complete resolved config, so it replaces the preset.
        cfg = file_cfg

    overrides: Dict[str, Any] = {}
    if args.style != "mixed":
        overrides["styles"] = [args.style]
    if args.hard_frac is not None:
        overrides["hard_fraction"] = args.hard_frac
    if args.noise_boost is not None:
        overrides["noise_boost"] = args.noise_boost
    if args.format != "manifest":
        overrides.setdefault("output", {})["format"] = args.format
    if args.no_csv:
        overrides.setdefault("output", {})["write_csv"] = False
    if args.no_parquet:
        overrides.setdefault("output", {})["write_parquet"] = False
    if args.workers:
        overrides.setdefault("runtime", {})["workers"] = args.workers
    return cfg.override(overrides) if overrides else cfg


def main(argv: Optional[List[str]] = None) -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(
        prog="python -m driftsense.cli.generate",
        description="Generate a Drift-Sense synthetic SEM image-pair dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Concurrency notes")[0])
    ap.add_argument("--num-images", type=int, default=32,
                    help="number of image PAIRS to generate")
    ap.add_argument("--out", default="data/train", help="dataset root directory")
    ap.add_argument("--format", choices=("manifest", "png", "npz"),
                    default="manifest",
                    help="manifest = labels only (recommended for 100k)")
    ap.add_argument("--preset", choices=PRESETS, default="default")
    ap.add_argument("--config", default=None, help="YAML config file")
    ap.add_argument("--style", choices=("dram", "finfet", "mixed"),
                    default="mixed")
    ap.add_argument("--hard-frac", type=float, default=None,
                    help="fraction of defect-only periodic samples")
    ap.add_argument("--noise-boost", type=float, default=None,
                    help=">1 makes the search frame noisier")
    ap.add_argument("--workers", type=int, default=0,
                    help="processes; 0 = cores - 1")
    ap.add_argument("--seed", type=int, default=20260803, help="dataset root seed")
    ap.add_argument("--start-index", type=int, default=0,
                    help="first sample id, for extending a dataset")
    ap.add_argument("--resume", action="store_true",
                    help="skip ids already in labels.jsonl")
    ap.add_argument("--verify", action="store_true",
                    help="ZNCC-check every rendered pair (slower)")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--no-parquet", action="store_true")
    ap.add_argument("--yes", action="store_true",
                    help="skip the disk-space confirmation")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    out = Path(args.out)

    print(cfg.describe())
    print()

    if args.format in ("png", "npz"):
        gb = args.num_images * BYTES_PER_PNG_PAIR / 1e9
        print(f"  [disk] ~{gb:.1f} GB for {args.num_images} pairs in "
              f"{args.format} format")
        if gb > 25 and not args.yes:
            print("  [disk] That is a lot. Consider --format manifest plus "
                  "on-the-fly rendering (driftsense.dataset).")
            reply = input("  continue? [y/N] ").strip().lower()
            if reply not in ("y", "yes"):
                print("  aborted")
                return

    generate(cfg=cfg, out=out, num_images=args.num_images,
             root_seed=args.seed, fmt=args.format,
             workers=args.workers, resume=args.resume,
             start_index=args.start_index, verify=args.verify)


if __name__ == "__main__":
    main()
