"""Deterministic randomness for Drift-Sense.

Why this needs its own module
-----------------------------
Three bugs cost people this kind of project, and all three are randomness bugs:

1. **Duplicate samples across workers.**  With the ``fork`` start method every
   worker inherits the parent's RNG state.  A pool that seeds lazily produces
   ``N`` identical copies of the same data and nobody notices, because the
   images look fine.  :func:`sample_seeds` derives every sample's seed from the
   root seed *up front*, so which worker renders which sample is irrelevant.

2. **Correlated "independent" noise.**  The brief requires the two frames to be
   separate physical captures.  Drawing both from one generator makes their
   noise correlated through the stream position; a network can then match the
   frames on noise statistics rather than structure -- a shortcut that does not
   exist on the hidden test set.  :class:`SeedBook` hands out *named* streams
   that are statistically independent by construction.

3. **Platform-dependent hashing.**  Python's built-in ``hash()`` of a string is
   randomised per process (PYTHONHASHSEED), so any position-hashed procedural
   field built on it changes between runs and between machines.  This module
   uses BLAKE2b, which is stable everywhere, forever.

Everything here is built on :class:`numpy.random.SeedSequence`, whose whole
purpose is turning one root seed into many independent, reproducible streams.

Usage
-----
    from driftsense.rng import SeedBook, sample_seeds, position_rng

    seeds = sample_seeds(root_seed=20260803, n=100_000)   # the whole dataset
    book = SeedBook(seeds[0])
    layout_rng = book.stream("layout")
    ref_rng = book.stream("optics.reference")
    search_rng = book.stream("optics.search")
"""

from __future__ import annotations

import hashlib
import os
import sys
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

#: Streams whose names are used across the package.  Listed for discoverability
#: and to catch typos in review -- ``stream()`` accepts any name.
STANDARD_STREAMS: Tuple[str, ...] = (
    "plan",                # style, difficulty, target choice
    "layout",              # die structural parameters
    "acquisition",         # rotation, magnification error, jitter, drift
    "optics.reference",    # reference frame imaging chain
    "optics.search",       # search frame imaging chain
)

_MAX_SEED = 1 << 62


def label_key(label: str) -> int:
    """Map a stream name to a stable 63-bit integer.

    Uses BLAKE2b rather than :func:`hash`, which is salted per process and would
    make every run produce different streams for the same name.

    Args:
        label: Stream name, e.g. ``"optics.search"``.

    Returns:
        A non-negative integer < 2**63, identical on every platform and run.
    """
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & (_MAX_SEED - 1)


def derive(seed: int, *labels: str) -> np.random.Generator:
    """Build a generator for ``seed`` under a chain of names.

    ``derive(s, "optics", "search")`` and ``derive(s, "optics", "reference")``
    are independent streams; both are reproducible from ``s`` alone.

    Args:
        seed: Root seed for this sample.
        *labels: Zero or more names identifying the stream.

    Returns:
        A fresh :class:`numpy.random.Generator` (PCG64).
    """
    entropy = [int(seed) % _MAX_SEED] + [label_key(l) for l in labels]
    return np.random.default_rng(np.random.SeedSequence(entropy))


class SeedBook:
    """Named RNG streams for one sample, derived from one seed.

    The book caches generators, so ``book.stream("layout")`` called twice
    returns the *same* generator and continues its sequence -- which is what
    you want inside one module, and why modules must not share a stream name.

    Args:
        seed: The sample's root seed.

    Attributes:
        seed: The root seed, echoed into the manifest.
    """

    __slots__ = ("seed", "_cache")

    def __init__(self, seed: int) -> None:
        self.seed = int(seed) % _MAX_SEED
        self._cache: Dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> np.random.Generator:
        """Return (and cache) the generator for a named stream."""
        gen = self._cache.get(name)
        if gen is None:
            gen = derive(self.seed, name)
            self._cache[name] = gen
        return gen

    def fresh(self, name: str) -> np.random.Generator:
        """Return an uncached generator -- always at sequence position zero.

        Use this where the caller must not depend on how many draws some other
        function made first, e.g. when re-rendering one frame in isolation.
        """
        return derive(self.seed, name)

    def subseed(self, name: str) -> int:
        """A derived integer seed, for passing to code that wants a plain int."""
        return int(self.stream(name).integers(0, _MAX_SEED))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"SeedBook(seed={self.seed}, streams={sorted(self._cache)})"


def sample_seeds(root_seed: int, n: int, offset: int = 0) -> np.ndarray:
    """Deterministic seeds for ``n`` samples.

    Generating seeds up front, in the parent process, is what makes the dataset
    independent of worker count, chunk size and scheduling order.  Regenerating
    sample 47 tomorrow on a different machine with a different pool size gives
    byte-identical output.

    Args:
        root_seed: The dataset's root seed.
        n: How many seeds to produce.
        offset: Index of the first sample.  ``sample_seeds(s, 10, offset=90)``
            returns exactly the last 10 of ``sample_seeds(s, 100)``, which lets
            you extend a dataset without regenerating it.

    Returns:
        ``int64`` array of shape ``(n,)``.
    """
    if n < 0 or offset < 0:
        raise ValueError("n and offset must be >= 0")
    ss = np.random.SeedSequence([int(root_seed) % _MAX_SEED, label_key("dataset")])
    children = ss.spawn(offset + n)[offset:]
    return np.array(
        [int(c.generate_state(2, dtype=np.uint64).astype(object).sum()) % _MAX_SEED
         for c in children],
        dtype=np.int64,
    )


def position_rng(die_seed: int, i: int, j: int, kind: str = "landmark"
                 ) -> np.random.Generator:
    """Generator for lattice cell ``(i, j)`` of a procedural field.

    This is what makes the landmark field *position-hashed*: any window of the
    die can be queried independently and always yields the same landmarks, so
    the reference frame and the search frame agree without sharing state or
    rendering a common canvas.  Cell indices may be negative.

    Args:
        die_seed: Seed identifying this die.
        i: Lattice column index (may be negative).
        j: Lattice row index (may be negative).
        kind: Field name, so several fields can share a lattice without
            correlating.

    Returns:
        A generator specific to that cell.
    """
    entropy = [int(die_seed) % _MAX_SEED,
               label_key(kind),
               (int(i) & 0xFFFFFFFF),
               (int(j) & 0xFFFFFFFF)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def apply_thread_env(n_threads: int = 1) -> bool:
    """Pin BLAS/OpenMP thread counts, for use in worker processes.

    Numpy's BLAS starts one thread per core *inside every worker*; an 8-worker
    pool on 8 cores creates 64 threads competing for cache, and the measured
    cost is roughly 3x.

    These variables are read when numpy imports its BLAS, so setting them after
    ``import numpy`` is too late.  Call this at the very top of an entry-point
    script, before any numpy import.  The return value tells you whether you
    made it in time.

    Args:
        n_threads: Threads per worker.  Use 1 with a process pool.

    Returns:
        ``True`` if the variables were set before numpy was imported.
    """
    in_time = "numpy" not in sys.modules
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(n_threads))
    return in_time


def resolve_workers(requested: int) -> int:
    """Turn a configured worker count into a concrete one.

    Args:
        requested: ``0`` means "all cores but one" -- leaving a core free keeps
            the machine usable during a multi-hour run and avoids the
            oversubscription cliff.

    Returns:
        At least 1.
    """
    if requested > 0:
        return int(requested)
    return max(1, (os.cpu_count() or 2) - 1)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Run ``python -m driftsense.rng``."""
    # 1. determinism
    assert label_key("optics.search") == label_key("optics.search")
    assert derive(7, "a").random() == derive(7, "a").random()
    assert derive(7, "a").random() != derive(7, "b").random()
    assert derive(7, "a").random() != derive(8, "a").random()

    # 2. stream independence: cross-correlation of two "independent" captures
    n = 20000
    a = derive(99, "optics.reference").normal(size=n)
    b = derive(99, "optics.search").normal(size=n)
    corr = abs(float(np.corrcoef(a, b)[0, 1]))
    assert corr < 4.0 / np.sqrt(n), f"streams correlated: r={corr:.4f}"

    # 3. SeedBook caching semantics
    book = SeedBook(123)
    assert book.stream("layout") is book.stream("layout")
    assert book.fresh("layout") is not book.stream("layout")
    x = SeedBook(123).stream("layout").random()
    y = SeedBook(123).stream("layout").random()
    assert x == y
    assert SeedBook(123).subseed("plan") == SeedBook(123).subseed("plan")

    # 4. sample_seeds: unique, reproducible, offset-consistent
    s1 = sample_seeds(20260803, 50000)
    assert len(np.unique(s1)) == len(s1), "duplicate sample seeds"
    assert np.array_equal(s1, sample_seeds(20260803, 50000))
    assert np.array_equal(sample_seeds(20260803, 10, offset=90),
                          sample_seeds(20260803, 100)[90:100])
    assert not np.array_equal(s1[:100], sample_seeds(20260804, 100))
    assert (s1 >= 0).all()

    # 5. the fork bug this module exists to prevent: seeds are assigned by
    #    index, so any partition across workers reproduces the same dataset
    seeds = sample_seeds(1, 1000)
    for nw in (1, 3, 8):
        merged = np.concatenate([seeds[w::nw] for w in range(nw)])
        assert len(np.unique(merged)) == 1000

    # 6. position hashing: stable, negative indices fine, cells decorrelated
    assert position_rng(5, -3, 7).random() == position_rng(5, -3, 7).random()
    assert position_rng(5, -3, 7).random() != position_rng(5, -3, 8).random()
    assert position_rng(5, 1, 1, "landmark").random() != \
           position_rng(5, 1, 1, "defect").random()
    grid = np.array([[position_rng(11, i, j).random() for j in range(60)]
                     for i in range(60)])
    assert 0.45 < grid.mean() < 0.55, grid.mean()
    # neighbouring cells must not be correlated, or landmarks would cluster
    r = abs(float(np.corrcoef(grid[:, :-1].ravel(), grid[:, 1:].ravel())[0, 1]))
    assert r < 0.05, f"adjacent cells correlated: r={r:.3f}"

    # 7. plumbing
    assert resolve_workers(4) == 4 and resolve_workers(0) >= 1
    apply_thread_env(1)
    assert os.environ["OMP_NUM_THREADS"] == "1"

    print("rng.py self-test OK")
    print(f"  streams               : {', '.join(STANDARD_STREAMS)}")
    print(f"  cross-stream |r|      : {corr:.5f} (n={n})")
    print(f"  50k seeds, duplicates : 0")
    print(f"  first 3 seeds (root=20260803): {s1[:3].tolist()}")
    print(f"  workers (0 -> auto)   : {resolve_workers(0)}")


if __name__ == "__main__":
    _self_test()
