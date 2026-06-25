"""Load and normalize the official TaskBench data into typed objects.

The shipped ``data.json`` stores several fields as JSON-encoded strings. We parse
them into the canonical schema used by the official ``evaluate.py`` /
``inference.py`` (``user_request`` / ``task_steps`` / ``task_nodes`` /
``task_links``). We do **not** alter the gold content.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from taskbench_sft.logging_utils import get_logger
from taskbench_sft.schema import (
    DOMAIN_DEPENDENCY,
    DependencyType,
    GoldSample,
    TaskLink,
    TaskNode,
    Tool,
    ToolCatalog,
    Topology,
    chain_length_bucket,
)

logger = get_logger(__name__)


def _maybe_json(value: Any) -> Any:
    """The official data stores nested structures as JSON strings; decode them."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tool_catalog(domain: str, raw_dir: str | Path) -> ToolCatalog:
    """Load ``tool_desc.json`` for a domain into a :class:`ToolCatalog`."""
    dependency_type = DOMAIN_DEPENDENCY[domain]
    path = Path(raw_dir) / domain / "tool_desc.json"
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    tools: List[Tool] = []
    for node in raw["nodes"]:
        tools.append(
            Tool(
                id=node["id"],
                desc=node.get("desc", ""),
                input_type=node.get("input-type"),
                output_type=node.get("output-type"),
                parameters=node.get("parameters"),
            )
        )
    logger.info("Loaded %d tools for domain %s (%s)", len(tools), domain, dependency_type.value)
    return ToolCatalog(domain=domain, dependency_type=dependency_type, tools=tools)


class GoldParseError(ValueError):
    """Raised when an official gold record cannot be parsed into the schema."""


def _parse_nodes(raw_nodes: Any) -> List[TaskNode]:
    parsed = _maybe_json(raw_nodes)
    # A handful of official records encode a single node as a bare dict; wrap it
    # only when it is unambiguously one node (has a "task" key). Anything else
    # non-list is treated as unparseable gold and the sample is excluded.
    if isinstance(parsed, dict):
        if "task" in parsed:
            parsed = [parsed]
        else:
            raise GoldParseError(f"tool_nodes is a non-node dict: {str(parsed)[:80]}")
    if not isinstance(parsed, list):
        raise GoldParseError(f"tool_nodes is not a list: {type(parsed).__name__}")
    nodes: List[TaskNode] = []
    for n in parsed:
        if not isinstance(n, dict) or "task" not in n:
            raise GoldParseError(f"node is not a {{task, arguments}} dict: {str(n)[:80]}")
        args = n.get("arguments", [])
        if not isinstance(args, list):
            args = [args]
        nodes.append(TaskNode(task=n["task"], arguments=list(args)))
    return nodes


def _parse_links(raw_links: Any) -> List[TaskLink]:
    parsed = _maybe_json(raw_links)
    if not isinstance(parsed, list):
        raise GoldParseError(f"tool_links is not a list: {type(parsed).__name__}")
    links: List[TaskLink] = []
    for l in parsed:
        if not isinstance(l, dict) or "source" not in l or "target" not in l:
            raise GoldParseError(f"link is not a {{source, target}} dict: {str(l)[:80]}")
        links.append(TaskLink(source=l["source"], target=l["target"]))
    return links


def iter_raw_samples(domain: str, raw_dir: str | Path) -> Iterator[Dict[str, Any]]:
    """Yield raw JSON dicts from a domain's ``data.json`` (JSONL)."""
    path = Path(raw_dir) / domain / "data.json"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_domain_samples(domain: str, raw_dir: str | Path) -> List[GoldSample]:
    """Load all gold samples for a domain, normalized to :class:`GoldSample`.

    Topology validation (chain recovery, DAG / disconnected exclusion) and the
    ``trajectory`` field are populated later by
    :func:`taskbench_sft.data.topology.annotate_sample`.
    """
    dependency_type = DOMAIN_DEPENDENCY[domain]
    samples: List[GoldSample] = []
    n_parse_errors = 0
    for raw in iter_raw_samples(domain, raw_dir):
        topology = Topology(raw["type"])
        n_tools = raw.get("n_tools", 0)
        try:
            task_nodes = _parse_nodes(raw.get("tool_nodes", raw.get("task_nodes", "[]")))
            task_links = _parse_links(raw.get("tool_links", raw.get("task_links", "[]")))
            task_steps = _maybe_json(raw.get("tool_steps", raw.get("task_steps", "[]")))
            if not isinstance(task_steps, list):
                raise GoldParseError("tool_steps is not a list")
        except GoldParseError as exc:
            n_parse_errors += 1
            logger.warning("Excluding unparseable gold #id %s (%s): %s", raw.get("id"), domain, exc)
            samples.append(
                GoldSample(
                    id=str(raw["id"]),
                    domain=domain,
                    dependency_type=dependency_type,
                    topology=topology,
                    n_tools=n_tools or 0,
                    user_request=raw.get("instruction", raw.get("user_request", "")),
                    task_steps=[],
                    task_nodes=[],
                    task_links=[],
                    seed=raw.get("seed"),
                    chain_length_bucket=chain_length_bucket(topology, n_tools or 0),
                    is_usable=False,
                    exclusion_reason="unparseable_gold",
                )
            )
            continue
        if not n_tools:
            n_tools = len(task_nodes)
        user_request = raw.get("instruction", raw.get("user_request", ""))
        sample = GoldSample(
            id=str(raw["id"]),
            domain=domain,
            dependency_type=dependency_type,
            topology=topology,
            n_tools=n_tools,
            user_request=user_request,
            task_steps=list(task_steps),
            task_nodes=task_nodes,
            task_links=task_links,
            seed=raw.get("seed"),
            chain_length_bucket=chain_length_bucket(topology, n_tools),
        )
        samples.append(sample)
    logger.info(
        "Loaded %d raw gold samples for domain %s (%d unparseable excluded)",
        len(samples), domain, n_parse_errors,
    )
    return samples


def data_file_hashes(domains: List[str], raw_dir: str | Path) -> Dict[str, str]:
    """SHA-256 of every official file used, for the reproducibility manifest."""
    hashes: Dict[str, str] = {}
    for domain in domains:
        for fname in ["data.json", "tool_desc.json", "graph_desc.json", "user_requests.json"]:
            p = Path(raw_dir) / domain / fname
            if p.exists():
                hashes[f"{domain}/{fname}"] = file_sha256(p)
    return hashes
