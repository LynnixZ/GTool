"""Build GTool graphs from vendored GNN4TaskPlan data (`dataset_gnn4plan/<domain>/`).

GNN4TaskPlan ships `data.json` (TaskBench-format: user_request/task_nodes/task_links/type),
`tool_desc.json`, `graph_desc.json`, `user_requests.json`, `split_ids.json` -- but NOT the
`node_desc.json` GTool's bundled preprocess expects. So this builder derives node_desc from
`tool_desc.json` (lowercased id -> desc) and otherwise follows GTool's graph logic exactly:
predefined tool graph (`graph_desc.json`) + a per-sample request super-node, SBERT features.

Encode-once: the tool-node descriptions, edge types and topology are identical across samples,
so they're encoded a single time; per sample only that sample's request is encoded and
concatenated (bit-identical to encoding them together). Writes `nodes.csv`, `edges.csv`,
`graphs/{i}.pt` next to the data (i = line index in `data.json`, matching ZouSplitDataset).

    python -m src.dataset.preprocess_gnn4plan --root dataset_gnn4plan --domains huggingface
    python -m src.dataset.preprocess_gnn4plan --root dataset_gnn4plan      # all three
"""
import os
import json
import argparse

import torch
import pandas as pd
from tqdm import tqdm
from torch_geometric.data.data import Data

from src.utils.lm_modeling import load_model, load_text2embedding

MODEL_NAME = "sbert"
ALL_DOMAINS = ["huggingface", "multimedia", "dailylife"]


def _textualize_graph(graph_desc):
    """Same as GTool: node/edge tables from graph_desc.json links, edge_attr='precedes'."""
    nodes, edges = {}, []
    for e in graph_desc["links"]:
        src = e["source"].lower().strip()
        dst = e["target"].lower().strip()
        if src not in nodes:
            nodes[src] = len(nodes)
        if dst not in nodes:
            nodes[dst] = len(nodes)
        edges.append({"src": nodes[src], "edge_attr": "precedes", "dst": nodes[dst]})
    return (pd.DataFrame(nodes.items(), columns=["node_attr", "node_id"]),
            pd.DataFrame(edges))


def build_domain(root, domain, model, tokenizer, device, text2embedding):
    path = os.path.join(root, domain)
    raw = open(os.path.join(path, "data.json"), "r", encoding="utf-8").readlines()
    graph_desc = json.load(open(os.path.join(path, "graph_desc.json"), "r", encoding="utf-8"))
    nodes, edges = _textualize_graph(graph_desc)
    edges.to_csv(f"{path}/edges.csv", index=False, columns=["src", "edge_attr", "dst"])
    nodes.to_csv(f"{path}/nodes.csv", index=False, columns=["node_id", "node_attr"])

    td = json.load(open(os.path.join(path, "tool_desc.json"), "r", encoding="utf-8"))
    node_desc = {n["id"].lower().strip(): n.get("desc", "") for n in td["nodes"]}
    node_desc_list = [node_desc[n] for n in nodes["node_attr"]]

    # encode the constant parts ONCE
    node_list = nodes.node_attr.tolist()
    edge_list = edges.edge_attr.tolist() + ["precedes"] * len(node_list)
    node_embeds = text2embedding(model, tokenizer, device, node_desc_list)
    e = text2embedding(model, tokenizer, device, edge_list)
    super_node_edge_index = torch.LongTensor(
        [[i for i in range(len(node_list))], [len(node_list)] * len(node_list)])
    edge_index = torch.hstack((torch.LongTensor([edges.src, edges.dst]), super_node_edge_index))

    os.makedirs(f"{path}/graphs", exist_ok=True)
    for i in tqdm(range(len(raw)), desc=domain):
        rec = json.loads(raw[i])
        request = rec.get("user_request", rec.get("instruction", ""))  # GNN4TaskPlan uses user_request
        req_embed = text2embedding(model, tokenizer, device, [request])
        x = torch.cat([node_embeds, req_embed], dim=0)
        data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(node_list) + 1)
        torch.save(data, f"{path}/graphs/{i}.pt")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="dataset_gnn4plan", help="Vendored GNN4TaskPlan data root.")
    p.add_argument("--domains", default=",".join(ALL_DOMAINS), help="Comma-separated dirs to build.")
    args = p.parse_args()

    model, tokenizer, device = load_model[MODEL_NAME]()
    text2embedding = load_text2embedding[MODEL_NAME]
    for d in args.domains.split(","):
        d = d.strip()
        if not d:
            continue
        print(f"[gnn4plan] building graphs for {d} under {args.root}/")
        build_domain(args.root, d, model, tokenizer, device, text2embedding)


if __name__ == "__main__":
    main()
