"""Self-contained stratified split over GTool's *bundled filtered subset*.

This reproduces the ``taskbench_sft`` splitting logic
(``src/dataset/preprocess_zou/split.py`` + ``topology.py``) but runs directly on
GTool's bundled ``dataset/<domain>/data.json`` (the filtered subset:
huggingface=3630 / multimedia=2981 / dailylife=2787) with **no taskbench_sft
dependency**. We split the subset, not full TaskBench, so no graph rebuild is
needed — GTool's own ``python -m src.dataset.preprocess.<domain>`` already builds
the graphs for exactly these samples.

Faithful to taskbench_sft:
* 80/10/10, stratified by ``topology x chain_length_bucket`` (per-domain: each
  domain is split independently, so ``domain`` would be a constant no-op in the
  stratify key and is intentionally omitted — call this once per ``--domains``).
* Per-stratum deterministic shuffle: ``random.Random(f"{seed}|{key}")``.
* Train tool coverage: every tool in val/test must appear in train, else the
  whole split is re-drawn with ``seed+attempt`` (up to ``max_resamples``).
* single + chain only; chains must be simple connected paths (others excluded).
* ``trajectory`` = topological order of the chain (the gold tool order).

Note: GTool's bundled subset is entirely ``chain`` type, so ``test_node.jsonl``
will normally be empty here — that is expected, not a bug.

Output (into ``--out_dir``): train.jsonl / validation.jsonl / test_node.jsonl /
test_chain.jsonl / test_all.jsonl / split_manifest.json — the exact format
``src/dataset/zou_split.py`` reads.

Usage:
    python -m src.dataset.preprocess_zou.split_subset \
        --raw_root dataset --out_dir artifacts/splits_subset
"""
import os
import re
import json
import random
import hashlib
import argparse
from collections import defaultdict, Counter

# GTool dataset dir <-> taskbench domain label (kept identical to the zou format
# so zou_split.ZouSplitDataset works unchanged).
DIR_TO_DOMAIN = {
    "huggingface": "data_huggingface",
    "multimedia": "data_multimedia",
    "dailylife": "data_dailylifeapis",
}
DOMAIN_DEPENDENCY = {
    "data_huggingface": "resource",
    "data_multimedia": "resource",
    "data_dailylifeapis": "temporal",
}

# chain-length buckets (taskbench_sft.schema.chain_length_bucket)
BUCKET_NODE = "node"


def chain_length_bucket(topology, n_tools):
    if topology == "single" or n_tools <= 1:
        return BUCKET_NODE
    if n_tools == 2:
        return "chain_length_2"
    if n_tools == 3:
        return "chain_length_3"
    return "chain_length_4_plus"


# ----------------------------------------------------- topology (from topology.py)
def _link_name_edges(node_names, task_links):
    """Directed index edges from name-based task_links; None if ambiguous."""
    edges = []
    for link in task_links:
        src_idxs = [k for k, n in enumerate(node_names) if n == link["source"]]
        tgt_idxs = [k for k, n in enumerate(node_names) if n == link["target"]]
        if not src_idxs or not tgt_idxs:
            return None
        if len(src_idxs) > 1 or len(tgt_idxs) > 1:
            return None  # repeated tool name as endpoint -> ambiguous
        edges.append((src_idxs[0], tgt_idxs[0]))
    return edges


def _topological_order(n, edges):
    indeg = [0] * n
    adj = [[] for _ in range(n)]
    for s, t in edges:
        adj[s].append(t)
        indeg[t] += 1
    frontier = sorted([i for i in range(n) if indeg[i] == 0])
    order = []
    while frontier:
        node = frontier.pop(0)
        order.append(node)
        for nxt in sorted(adj[node]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                frontier.append(nxt)
        frontier.sort()
    return order if len(order) == n else None


def _is_simple_path(n, edges):
    if n == 1:
        return len(edges) == 0
    if len(edges) != n - 1:
        return False
    indeg = [0] * n
    outdeg = [0] * n
    undirected = [[] for _ in range(n)]
    for s, t in edges:
        outdeg[s] += 1
        indeg[t] += 1
        undirected[s].append(t)
        undirected[t].append(s)
    if any(d > 1 for d in indeg) or any(d > 1 for d in outdeg):
        return False
    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nb in undirected[cur]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == n


def _annotate(sample):
    """Set trajectory / is_usable / exclusion_reason (mirrors topology.annotate_sample)."""
    names = sample["node_names"]
    n = len(names)
    topo = sample["topology"]

    if topo == "dag":
        sample.update(is_usable=False, exclusion_reason="dag_excluded", trajectory=None)
        return sample
    if topo == "single":
        if n != 1:
            sample.update(is_usable=False, exclusion_reason=f"single_with_{n}_nodes", trajectory=None)
        else:
            sample.update(is_usable=True, exclusion_reason=None, trajectory=[names[0]])
        return sample
    # chain
    if n == 0:
        sample.update(is_usable=False, exclusion_reason="empty_chain", trajectory=None)
        return sample
    edges = _link_name_edges(names, sample["task_links"])
    if edges is None:
        sample.update(is_usable=False, exclusion_reason="ambiguous_repeated_names", trajectory=None)
        return sample
    if not _is_simple_path(n, edges):
        sample.update(is_usable=False, exclusion_reason="not_simple_connected_path", trajectory=None)
        return sample
    order = _topological_order(n, edges)
    if order is None:
        sample.update(is_usable=False, exclusion_reason="cyclic_graph", trajectory=None)
        return sample
    sample.update(is_usable=True, exclusion_reason=None, trajectory=[names[i] for i in order])
    return sample


# ----------------------------------------------------------------- load samples
def load_subset(raw_root, domains=None):
    """domains: optional iterable of GTool dir names (huggingface/multimedia/dailylife)
    to restrict to; None = all."""
    samples = []
    reasons = Counter()
    for dir_, domain in DIR_TO_DOMAIN.items():
        if domains is not None and dir_ not in domains:
            continue
        path = os.path.join(raw_root, dir_, "data.json")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                task_nodes = raw.get("task_nodes", []) or []
                task_links = raw.get("task_links", []) or []
                n_tools = raw.get("n_tools") or len(task_nodes)
                topology = raw.get("type", "chain")
                node_names = [n.get("task") for n in task_nodes]
                s = {
                    "id": str(raw["id"]),
                    "domain": domain,
                    "dependency_type": DOMAIN_DEPENDENCY[domain],
                    "topology": topology,
                    "n_tools": n_tools,
                    "user_request": raw.get("user_request", raw.get("instruction", "")),
                    "task_steps": raw.get("task_steps", raw.get("tool_steps", [])) or [],
                    "task_nodes": task_nodes,
                    "task_links": task_links,
                    "node_names": node_names,
                    "chain_length_bucket": chain_length_bucket(topology, n_tools),
                }
                _annotate(s)
                if not s["is_usable"]:
                    reasons[s["exclusion_reason"]] += 1
                samples.append(s)
    if reasons:
        print("Topology exclusions:", dict(reasons))
    return samples


# --------------------------------------------------------------- stratified split
def _split_indices(n, train_frac, val_frac, rng):
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = min(int(round(n * train_frac)), n)
    n_val = min(int(round(n * val_frac)), n - n_train)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def _draw_split(samples, seed, train_frac, val_frac, stratify_by):
    strata = defaultdict(list)
    for s in samples:
        key = tuple(str(s[f]) for f in stratify_by)
        strata[key].append(s)
    train, val, test = [], [], []
    for key in sorted(strata.keys()):
        bucket = sorted(strata[key], key=lambda x: x["id"])
        rng = random.Random(f"{seed}|{'|'.join(key)}")
        tr, va, te = _split_indices(len(bucket), train_frac, val_frac, rng)
        train += [bucket[i] for i in tr]
        val += [bucket[i] for i in va]
        test += [bucket[i] for i in te]
    return train, val, test


def _coverage_violations(train, heldout):
    train_tools = set()
    for s in train:
        train_tools.update(s["node_names"])
    missing = defaultdict(list)
    for s in heldout:
        for tool in s["node_names"]:
            if tool not in train_tools and len(missing[tool]) < 5:
                missing[tool].append(s["id"])
    return dict(missing)


def make_split(samples, seed, train_frac, val_frac, stratify_by, max_resamples,
               skip_coverage=False):
    usable = [s for s in samples if s["is_usable"] and s["topology"] in ("single", "chain")]
    print(f"Splitting {len(usable)} usable samples (single+chain)")
    if skip_coverage:
        # Smoke / tiny splits can't satisfy train tool coverage; just draw once.
        train, val, test = _draw_split(usable, seed, train_frac, val_frac, stratify_by)
        return train, val, test, seed
    last_missing = {}
    for attempt in range(max_resamples):
        cur_seed = seed + attempt
        train, val, test = _draw_split(usable, cur_seed, train_frac, val_frac, stratify_by)
        missing = _coverage_violations(train, val + test)
        if not missing:
            if attempt:
                print(f"Tool coverage satisfied after {attempt} resample(s) (seed={cur_seed})")
            return train, val, test, cur_seed
        last_missing = missing
        print(f"Attempt {attempt} (seed={cur_seed}): {len(missing)} tools in val/test missing from train; resampling")
    raise RuntimeError(f"Could not satisfy train tool coverage after {max_resamples} resamples. "
                       f"Rare tools: {json.dumps(last_missing, indent=2)}")


def make_split_gnn4plan(samples, seed, test_ids, train_cap, test_cap=0):
    """GNN4Plan / GRAFT / GTool-aligned split (faithful to GNN4TaskPlan's split_data.py +
    finetunellm/main.py):

    * test = the FIXED chain ids in split_ids.json (same test samples the papers report on),
      chain-only.
    * train/val candidates = the single+chain usable pool MINUS the test ids; shuffled with
      `seed`, capped at `train_cap` (GNN4Plan=3000), then split 85/15 into train/val.
    * NO tool-coverage resampling (GNN4Plan doesn't do it; we only WARN).
    """
    usable = [s for s in samples if s["is_usable"] and s["topology"] in ("single", "chain")]
    test_id_set = set(str(i) for i in test_ids)
    test = [s for s in usable if s["id"] in test_id_set and s["topology"] == "chain"]
    found = {s["id"] for s in test}
    missing = test_id_set - found
    if missing:
        print(f"GNN4Plan split: {len(missing)}/{len(test_id_set)} test ids not found as usable "
              f"chains (dropped from test): e.g. {sorted(missing)[:5]}")
    if test_cap:  # smoke only: shrink the fixed test set so a smoke run is fast
        test = sorted(test, key=lambda x: x["id"])[:test_cap]
    pool = sorted((s for s in usable if s["id"] not in test_id_set), key=lambda x: x["id"])
    random.Random(seed).shuffle(pool)
    if train_cap:
        pool = pool[:train_cap]
    n_train = int(round(0.85 * len(pool)))   # GNN4Plan finetunellm/main.py: 0.85
    train, val = pool[:n_train], pool[n_train:]
    cov = _coverage_violations(train, val + test)
    if cov:
        print(f"GNN4Plan split: {len(cov)} tools in val/test missing from train "
              f"(NOT resampled -- faithful to GNN4Plan)")
    print(f"GNN4Plan split: train={len(train)} val={len(val)} test={len(test)} "
          f"(pool={len(pool)}, cap={train_cap}, seed={seed})")
    return train, val, test, seed


# --------------------------------------------------------------------- write out
_RECORD_FIELDS = ["id", "domain", "dependency_type", "topology", "n_tools",
                  "user_request", "task_steps", "task_nodes", "task_links",
                  "trajectory", "chain_length_bucket", "is_usable", "exclusion_reason"]


def _record(s):
    return {k: s.get(k) for k in _RECORD_FIELDS}


def _write_jsonl(path, samples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h = hashlib.sha256()
    with open(path, "w", encoding="utf-8") as f:
        for s in sorted(samples, key=lambda x: x["id"]):
            line = json.dumps(_record(s), ensure_ascii=False)
            f.write(line + "\n")
            h.update(line.encode("utf-8"))
    return h.hexdigest()


def write_split(train, val, test, out_dir, used_seed, cfg):
    os.makedirs(out_dir, exist_ok=True)
    test_node = [s for s in test if s["topology"] == "single"]
    test_chain = [s for s in test if s["topology"] == "chain"]
    hashes = {
        "train.jsonl": _write_jsonl(os.path.join(out_dir, "train.jsonl"), train),
        "validation.jsonl": _write_jsonl(os.path.join(out_dir, "validation.jsonl"), val),
        "test_node.jsonl": _write_jsonl(os.path.join(out_dir, "test_node.jsonl"), test_node),
        "test_chain.jsonl": _write_jsonl(os.path.join(out_dir, "test_chain.jsonl"), test_chain),
        "test_all.jsonl": _write_jsonl(os.path.join(out_dir, "test_all.jsonl"), test),
    }

    def counts(items):
        return {
            "total": len(items),
            "by_topology": dict(Counter(s["topology"] for s in items)),
            "by_bucket": dict(Counter(s["chain_length_bucket"] for s in items)),
            "by_domain": dict(Counter(s["domain"] for s in items)),
        }

    manifest = {
        "config": cfg,
        "requested_seed": cfg["seed"],
        "used_seed": used_seed,
        "source": "gtool_bundled_subset",
        "splits": {
            "train": counts(train), "validation": counts(val),
            "test_node": counts(test_node), "test_chain": counts(test_chain),
            "test_all": counts(test),
        },
        "file_sha256": hashes,
    }
    with open(os.path.join(out_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Wrote split to {out_dir}: train={len(train)} val={len(val)} "
          f"test={len(test)} (node={len(test_node)} chain={len(test_chain)})")
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", type=str, default="dataset",
                   help="Root with GTool's bundled huggingface/ multimedia/ dailylife/ dirs.")
    p.add_argument("--out_dir", type=str, default="artifacts/splits_subset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.8)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--max_resamples", type=int, default=50)
    p.add_argument("--limit_per_domain", type=int, default=0,
                   help="Keep only the first N usable samples per domain (0=all). For smoke tests.")
    p.add_argument("--skip_coverage", action="store_true",
                   help="Skip the train-tool-coverage guarantee (needed for tiny smoke splits).")
    p.add_argument("--domains", type=str, default="",
                   help="Comma-separated GTool dir names to restrict to (e.g. 'huggingface'). Empty=all.")
    p.add_argument("--mode", type=str, default="gnn4plan", choices=["gnn4plan", "stratified"],
                   help="gnn4plan (default): GNN4TaskPlan's fixed split_ids test + capped 85/15 pool. "
                        "stratified: the old zou topology x chain_length_bucket split.")
    p.add_argument("--train_cap", type=int, default=3000,
                   help="gnn4plan: cap the shuffled single+chain train pool (GNN4Plan uses 3000; 0=no cap).")
    p.add_argument("--test_cap", type=int, default=0,
                   help="gnn4plan SMOKE ONLY: shrink the fixed test set to this many (0=full). Not for real runs.")
    args = p.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()] or None
    samples = load_subset(args.raw_root, domains=domains)

    if args.limit_per_domain:
        kept, cnt = [], defaultdict(int)
        for s in samples:
            if not (s["is_usable"] and s["topology"] in ("single", "chain")):
                continue
            if cnt[s["domain"]] < args.limit_per_domain:
                kept.append(s)
                cnt[s["domain"]] += 1
        print(f"limit_per_domain={args.limit_per_domain}: kept {dict(cnt)}")
        samples = kept

    if args.mode == "gnn4plan":
        # test = the FIXED chains from each domain's split_ids.json (GNN4TaskPlan).
        test_ids = []
        for dir_ in (domains or list(DIR_TO_DOMAIN.keys())):
            sp = os.path.join(args.raw_root, dir_, "split_ids.json")
            if not os.path.exists(sp):
                raise FileNotFoundError(
                    f"--mode gnn4plan but {sp} missing -- run scripts/download_gnn4plan.sh first "
                    f"(and point --raw_root at the vendored dir, e.g. dataset_gnn4plan).")
            test_ids += json.load(open(sp, "r", encoding="utf-8"))["test_ids"]["chain"]
        train, val, test, used_seed = make_split_gnn4plan(
            samples, args.seed, test_ids, args.train_cap, test_cap=args.test_cap)
        cfg = {"mode": "gnn4plan", "seed": args.seed, "train_cap": args.train_cap,
               "test_cap": args.test_cap, "test": "fixed split_ids.json chains",
               "train_val_split": "85/15", "out_dir": args.out_dir, "raw_root": args.raw_root}
    else:
        stratify_by = ["topology", "chain_length_bucket"]
        train, val, test, used_seed = make_split(
            samples, args.seed, args.train_frac, args.val_frac, stratify_by,
            args.max_resamples, skip_coverage=args.skip_coverage)
        cfg = {"mode": "stratified", "train_frac": args.train_frac, "validation_frac": args.val_frac,
               "test_frac": round(1 - args.train_frac - args.val_frac, 6),
               "seed": args.seed, "stratify_by": stratify_by, "max_resamples": args.max_resamples,
               "out_dir": args.out_dir, "raw_root": args.raw_root}
    write_split(train, val, test, args.out_dir, used_seed, cfg)


if __name__ == "__main__":
    main()
