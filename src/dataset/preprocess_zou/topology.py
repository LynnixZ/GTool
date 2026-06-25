"""Topology analysis: recover execution-ordered trajectories from gold links.

For this study we keep only ``single`` and ``chain`` samples. ``chain`` samples
must form a *simple, connected path*; anything that is a DAG (branching/merging)
or disconnected is excluded and the reason is recorded — we never silently
"fix" the gold graph.

Trajectory recovery works on node **indices** (not names) so that a chain which
legitimately repeats a tool is handled correctly and never de-duplicated.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from taskbench_sft.logging_utils import get_logger
from taskbench_sft.schema import (
    DependencyType,
    GoldSample,
    Topology,
)

logger = get_logger(__name__)

_NODE_TAG = re.compile(r"<node-(\d+)>")


def _resource_edges(sample: GoldSample) -> List[Tuple[int, int]]:
    """Index edges (j -> i) derived from ``<node-j>`` arguments (resource type).

    This mirrors the official evaluator's link reconstruction for the resource
    dependency type.
    """
    edges: List[Tuple[int, int]] = []
    for i, node in enumerate(sample.task_nodes):
        for arg in node.arguments:
            if isinstance(arg, dict):
                arg_val = " ".join(str(v) for v in arg.values())
            elif isinstance(arg, list):
                arg_val = " ".join(str(v) for v in arg)
            else:
                arg_val = str(arg)
            for m in _NODE_TAG.finditer(arg_val):
                j = int(m.group(1))
                if j != i and 0 <= j < len(sample.task_nodes):
                    edges.append((j, i))
    return edges


def _link_name_edges(sample: GoldSample) -> Optional[List[Tuple[int, int]]]:
    """Index edges from the explicit gold ``task_links`` (name-based).

    The shipped TaskBench ``data.json`` provides clean, explicit ``task_links``
    for every domain, so we treat them as the authoritative dependency edges for
    trajectory recovery (see README "Data normalization decisions").

    Returns ``None`` if a link endpoint name maps ambiguously to multiple node
    indices (repeated tool names) and we cannot resolve it deterministically — in
    that case the sample is excluded rather than guessed.
    """
    names = sample.node_names
    edges: List[Tuple[int, int]] = []
    for link in sample.task_links:
        src_idxs = [k for k, n in enumerate(names) if n == link.source]
        tgt_idxs = [k for k, n in enumerate(names) if n == link.target]
        if not src_idxs or not tgt_idxs:
            return None
        if len(src_idxs) > 1 or len(tgt_idxs) > 1:
            return None  # ambiguous: repeated tool name as a link endpoint
        edges.append((src_idxs[0], tgt_idxs[0]))
    return edges


def recover_index_edges(sample: GoldSample) -> Optional[List[Tuple[int, int]]]:
    """Recover directed index edges for a sample from the gold ``task_links``.

    ``task_links`` is the authoritative, explicit dependency source in the
    shipped data for all three domains. (The ``<node-j>`` argument
    reconstruction used by the official *resource* evaluator targets the
    post-``format_data.py`` form, which the shipped ``data.json`` is not in — its
    resource arguments use ``<output_of_ToolName>`` tags instead.)
    """
    return _link_name_edges(sample)


def topological_order(n: int, edges: List[Tuple[int, int]]) -> Optional[List[int]]:
    """Kahn topological sort over ``n`` indexed nodes; ``None`` if a cycle exists.

    Ties (multiple zero in-degree nodes) are broken by smallest index for
    determinism. A unique order is *not* required here — chain validation is done
    separately by :func:`is_simple_path`.
    """
    indeg = [0] * n
    adj: List[List[int]] = [[] for _ in range(n)]
    for s, t in edges:
        adj[s].append(t)
        indeg[t] += 1
    frontier = sorted([i for i in range(n) if indeg[i] == 0])
    order: List[int] = []
    while frontier:
        node = frontier.pop(0)
        order.append(node)
        for nxt in sorted(adj[node]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                frontier.append(nxt)
        frontier.sort()
    if len(order) != n:
        return None  # cycle
    return order


def is_simple_path(n: int, edges: List[Tuple[int, int]]) -> bool:
    """True iff the directed graph is a single connected simple path over n nodes."""
    if n == 1:
        return len(edges) == 0
    if len(edges) != n - 1:
        return False
    indeg = [0] * n
    outdeg = [0] * n
    undirected: List[List[int]] = [[] for _ in range(n)]
    for s, t in edges:
        outdeg[s] += 1
        indeg[t] += 1
        undirected[s].append(t)
        undirected[t].append(s)
    # Path property: every node in/out degree <= 1.
    if any(d > 1 for d in indeg) or any(d > 1 for d in outdeg):
        return False
    # Connectivity: a single weakly-connected component.
    seen = set([0])
    stack = [0]
    while stack:
        cur = stack.pop()
        for nb in undirected[cur]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == n


def annotate_sample(sample: GoldSample) -> GoldSample:
    """Populate ``trajectory`` / ``is_usable`` / ``exclusion_reason`` in place.

    Rules:
    * ``single`` with exactly one node -> usable, trajectory = [node].
    * ``chain`` forming a simple connected path -> usable, trajectory = topo order.
    * ``dag`` -> excluded (this study covers node + chain only).
    * disconnected / branching / cyclic / ambiguous -> excluded with a reason.
    """
    n = len(sample.task_nodes)
    names = sample.node_names

    if sample.topology == Topology.DAG:
        sample.is_usable = False
        sample.exclusion_reason = "dag_excluded"
        sample.trajectory = None
        return sample

    if sample.topology == Topology.SINGLE:
        if n != 1:
            sample.is_usable = False
            sample.exclusion_reason = f"single_with_{n}_nodes"
            sample.trajectory = None
        else:
            sample.is_usable = True
            sample.trajectory = [names[0]]
        return sample

    # chain
    if n == 0:
        sample.is_usable = False
        sample.exclusion_reason = "empty_chain"
        sample.trajectory = None
        return sample

    edges = recover_index_edges(sample)
    if edges is None:
        sample.is_usable = False
        sample.exclusion_reason = "ambiguous_repeated_names"
        sample.trajectory = None
        return sample

    if not is_simple_path(n, edges):
        sample.is_usable = False
        sample.exclusion_reason = "not_simple_connected_path"
        sample.trajectory = None
        return sample

    order = topological_order(n, edges)
    if order is None:
        sample.is_usable = False
        sample.exclusion_reason = "cyclic_graph"
        sample.trajectory = None
        return sample

    sample.trajectory = [names[i] for i in order]
    sample.is_usable = True
    return sample


def annotate_all(samples: List[GoldSample]) -> List[GoldSample]:
    """Annotate every sample and log an exclusion summary."""
    from collections import Counter

    reasons: Counter = Counter()
    for s in samples:
        annotate_sample(s)
        if not s.is_usable:
            reasons[s.exclusion_reason] += 1
    if reasons:
        logger.info("Topology exclusions: %s", dict(reasons))
    return samples
