"""Stratified train/validation/test splitting with train tool coverage.

* Primary split: 80/10/10, stratified by ``domain x topology x
  chain_length_bucket``.
* Tool coverage: every tool appearing in validation/test must appear at least
  once in train; otherwise the split is *re-drawn* with a new sub-seed (we never
  silently move samples). If coverage still fails after ``max_resamples``, we
  raise and list the offending rare tools.
* Outputs: ``train.jsonl``, ``validation.jsonl``, ``test_node.jsonl``,
  ``test_chain.jsonl``, ``test_all.jsonl``, and ``split_manifest.json``.

Both training modes read the *same* manifest; they never re-split.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from taskbench_sft.config import SplitConfig
from taskbench_sft.logging_utils import get_logger
from taskbench_sft.schema import GoldSample, Topology

logger = get_logger(__name__)


class CoverageError(RuntimeError):
    """Raised when train tool coverage cannot be satisfied by resampling."""


def _stratum_key(sample: GoldSample, stratify_by: List[str]) -> Tuple[str, ...]:
    parts = []
    for field in stratify_by:
        if field == "domain":
            parts.append(sample.domain)
        elif field == "topology":
            parts.append(sample.topology.value)
        elif field == "chain_length_bucket":
            parts.append(str(sample.chain_length_bucket))
        else:
            raise ValueError(f"Unknown stratify field: {field}")
    return tuple(parts)


def _split_indices(
    n: int, train_frac: float, val_frac: float, rng: random.Random
) -> Tuple[List[int], List[int], List[int]]:
    """Shuffle indices [0, n) and split into train/val/test by fraction."""
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    # Guard rounding so all three are representable for small strata.
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


def _draw_split(
    samples: List[GoldSample], cfg: SplitConfig, seed: int
) -> Tuple[List[GoldSample], List[GoldSample], List[GoldSample]]:
    """Draw one stratified split with the given seed."""
    strata: Dict[Tuple[str, ...], List[GoldSample]] = defaultdict(list)
    for s in samples:
        strata[_stratum_key(s, cfg.stratify_by)].append(s)

    train: List[GoldSample] = []
    val: List[GoldSample] = []
    test: List[GoldSample] = []
    for key in sorted(strata.keys()):
        bucket = sorted(strata[key], key=lambda x: x.id)  # deterministic base order
        rng = random.Random(f"{seed}|{'|'.join(key)}")
        tr, va, te = _split_indices(len(bucket), cfg.train_frac, cfg.validation_frac, rng)
        train.extend(bucket[i] for i in tr)
        val.extend(bucket[i] for i in va)
        test.extend(bucket[i] for i in te)
    return train, val, test


def _coverage_violations(
    train: List[GoldSample], heldout: List[GoldSample]
) -> Dict[str, List[str]]:
    """Return {tool_id: [example sample ids]} for tools in heldout but not train."""
    train_tools = set()
    for s in train:
        train_tools.update(s.node_names)
    missing: Dict[str, List[str]] = defaultdict(list)
    for s in heldout:
        for tool in s.node_names:
            if tool not in train_tools:
                if len(missing[tool]) < 5:
                    missing[tool].append(s.id)
    return dict(missing)


def make_split(
    samples: List[GoldSample], cfg: SplitConfig
) -> Tuple[List[GoldSample], List[GoldSample], List[GoldSample], int]:
    """Draw a stratified split satisfying train tool coverage.

    Returns ``(train, val, test, used_seed)``.
    """
    usable = [s for s in samples if s.is_usable and s.topology in (Topology.SINGLE, Topology.CHAIN)]
    logger.info("Splitting %d usable samples (single+chain)", len(usable))

    last_missing: Dict[str, List[str]] = {}
    for attempt in range(cfg.max_resamples):
        seed = cfg.seed + attempt
        train, val, test = _draw_split(usable, cfg, seed)
        missing = _coverage_violations(train, val + test)
        if not missing:
            if attempt > 0:
                logger.info("Tool coverage satisfied after %d resample(s) (seed=%d)", attempt, seed)
            return train, val, test, seed
        last_missing = missing
        logger.warning(
            "Attempt %d (seed=%d): %d tools in val/test missing from train; resampling",
            attempt, seed, len(missing),
        )
    raise CoverageError(
        "Could not satisfy train tool coverage after "
        f"{cfg.max_resamples} resamples. Rare tools (tool -> example heldout ids): "
        f"{json.dumps(last_missing, indent=2)}"
    )


def _write_jsonl(path: Path, samples: List[GoldSample]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(path, "w", encoding="utf-8") as f:
        for s in sorted(samples, key=lambda x: x.id):
            line = json.dumps(s.to_record(), ensure_ascii=False)
            f.write(line + "\n")
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def write_split(
    train: List[GoldSample],
    val: List[GoldSample],
    test: List[GoldSample],
    cfg: SplitConfig,
    used_seed: int,
    extra_manifest: Optional[Dict] = None,
) -> Dict:
    """Write all split files + ``split_manifest.json`` and return the manifest."""
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    test_node = [s for s in test if s.topology == Topology.SINGLE]
    test_chain = [s for s in test if s.topology == Topology.CHAIN]

    hashes = {
        "train.jsonl": _write_jsonl(out / "train.jsonl", train),
        "validation.jsonl": _write_jsonl(out / "validation.jsonl", val),
        "test_node.jsonl": _write_jsonl(out / "test_node.jsonl", test_node),
        "test_chain.jsonl": _write_jsonl(out / "test_chain.jsonl", test_chain),
        "test_all.jsonl": _write_jsonl(out / "test_all.jsonl", test),
    }

    def _counts(items: List[GoldSample]) -> Dict[str, int]:
        from collections import Counter

        topo = Counter(s.topology.value for s in items)
        bucket = Counter(s.chain_length_bucket for s in items)
        domain = Counter(s.domain for s in items)
        return {
            "total": len(items),
            "by_topology": dict(topo),
            "by_bucket": dict(bucket),
            "by_domain": dict(domain),
        }

    manifest = {
        "config": cfg.model_dump(),
        "requested_seed": cfg.seed,
        "used_seed": used_seed,
        "splits": {
            "train": _counts(train),
            "validation": _counts(val),
            "test_node": _counts(test_node),
            "test_chain": _counts(test_chain),
            "test_all": _counts(test),
        },
        "file_sha256": hashes,
        "train_sample_ids": sorted(s.id for s in train),
        "validation_sample_ids": sorted(s.id for s in val),
        "test_sample_ids": sorted(s.id for s in test),
    }
    if extra_manifest:
        manifest.update(extra_manifest)

    manifest_str = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
    manifest_hash = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()
    manifest["manifest_sha256"] = manifest_hash
    with open(out / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)

    logger.info(
        "Wrote split to %s: train=%d val=%d test=%d (node=%d chain=%d)",
        out, len(train), len(val), len(test), len(test_node), len(test_chain),
    )
    return manifest


def make_smoke_split(
    samples: List[GoldSample],
    cfg: SplitConfig,
    n_train: int = 24,
    n_val: int = 6,
    n_test_node: int = 4,
    n_test_chain: int = 4,
) -> Tuple[List[GoldSample], List[GoldSample], List[GoldSample], int]:
    """Subsample a tiny Node+Chain split from the proper stratified split.

    Sampling is done *within* the real train/val/test partitions, so there is no
    leakage across splits — the smoke split is a deterministic subset of the full
    split for the same seed.
    """
    train, val, test, used_seed = make_split(samples, cfg)

    def pick(pool: List[GoldSample], n: int, topo: str) -> List[GoldSample]:
        items = sorted([s for s in pool if s.topology.value == topo], key=lambda x: x.id)
        rng = random.Random(f"smoke|{cfg.seed}|{topo}|{n}")
        rng.shuffle(items)
        return items[:n]

    n_node_tr = max(1, n_train // 2)
    n_chain_tr = max(1, n_train - n_node_tr)
    n_node_val = max(1, n_val // 2)
    n_chain_val = max(1, n_val - n_node_val)

    s_train = pick(train, n_node_tr, "single") + pick(train, n_chain_tr, "chain")
    s_val = pick(val, n_node_val, "single") + pick(val, n_chain_val, "chain")
    s_test = pick(test, n_test_node, "single") + pick(test, n_test_chain, "chain")
    logger.info(
        "Smoke split: train=%d (node=%d chain=%d) val=%d test=%d (node=%d chain=%d)",
        len(s_train), n_node_tr, n_chain_tr, len(s_val),
        len(s_test), n_test_node, n_test_chain,
    )
    return s_train, s_val, s_test, used_seed


def load_split_file(path: str | Path) -> List[GoldSample]:
    """Load a split JSONL file back into :class:`GoldSample` objects."""
    samples: List[GoldSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(GoldSample.from_record(json.loads(line)))
    return samples
