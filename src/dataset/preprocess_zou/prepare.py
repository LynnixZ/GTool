"""Orchestrate data loading + topology annotation across domains."""
from __future__ import annotations

from typing import Dict, List, Tuple

from taskbench_sft.config import DataConfig
from taskbench_sft.data.loader import load_domain_samples, load_tool_catalog
from taskbench_sft.data.topology import annotate_all
from taskbench_sft.logging_utils import get_logger
from taskbench_sft.schema import GoldSample, ToolCatalog

logger = get_logger(__name__)


def load_catalogs(cfg: DataConfig) -> Dict[str, ToolCatalog]:
    """Load the tool catalog for each configured domain."""
    return {d: load_tool_catalog(d, cfg.raw_dir) for d in cfg.domains}


def mark_off_catalog(samples: List[GoldSample], catalog: ToolCatalog) -> int:
    """Mark samples whose gold node names are not all in the catalog.

    Returns the number of newly-excluded samples. Topology exclusions are left
    untouched (we only further restrict currently-usable samples).
    """
    n = 0
    for s in samples:
        if not s.is_usable:
            continue
        off = [name for name in s.node_names if name not in catalog]
        if off:
            s.is_usable = False
            s.exclusion_reason = "off_catalog_gold"
            s.trajectory = None
            n += 1
    return n


def load_all_samples(cfg: DataConfig) -> List[GoldSample]:
    """Load + topology-annotate samples for all domains (all topologies kept).

    If ``require_catalog_faithful_gold`` is set, samples with off-catalog gold
    node names are additionally excluded (and logged).
    """
    samples: List[GoldSample] = []
    for domain in cfg.domains:
        domain_samples = annotate_all(load_domain_samples(domain, cfg.raw_dir))
        if cfg.require_catalog_faithful_gold:
            catalog = load_tool_catalog(domain, cfg.raw_dir)
            n_off = mark_off_catalog(domain_samples, catalog)
            if n_off:
                logger.info("Domain %s: excluded %d samples with off-catalog gold", domain, n_off)
        samples.extend(domain_samples)
    return samples


def usable_samples(cfg: DataConfig) -> List[GoldSample]:
    """Return only samples in the included topologies that passed validation."""
    keep = set(cfg.include_topologies)
    out = [
        s
        for s in load_all_samples(cfg)
        if s.is_usable and s.topology.value in keep
    ]
    logger.info("Usable samples (topologies=%s): %d", cfg.include_topologies, len(out))
    return out


def load_samples_and_catalogs(
    cfg: DataConfig,
) -> Tuple[List[GoldSample], Dict[str, ToolCatalog]]:
    return load_all_samples(cfg), load_catalogs(cfg)
