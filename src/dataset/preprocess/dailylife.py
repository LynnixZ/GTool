import os
import torch
import pandas as pd
import json
from tqdm import tqdm
from torch_geometric.data.data import Data
from random import random
from src.dataset.preprocess.generate_split import generate_split
from src.utils.lm_modeling import load_model, load_text2embedding
import random

model_name = 'sbert'
path = "dataset/dailylife/"
raw_data = open(path + '/data.json', "r")
raw_data = raw_data.readlines()
with open(path + "/node_desc.json", "r")as f:
    node_desc = json.load(f)

def _textualize_graph(graph):
    nodes = {}
    edges = []
    for e in graph["links"]:
        src, edeg_attr, dst = e["source"], "precedes" , e["target"]
        src = src.lower().strip()
        dst = dst.lower().strip()
        if src not in nodes:
            nodes[src] = len(nodes)
        if dst not in nodes:
            nodes[dst] = len(nodes)
        edges.append({'src': nodes[src], 'edge_attr': edeg_attr.lower().strip(), 'dst': nodes[dst], })

    nodes = pd.DataFrame(nodes.items(), columns=['node_attr', 'node_id'])
    edges = pd.DataFrame(edges)
    return nodes, edges


def _encode_graph(model, tokenizer, device, text2embedding):
    print('Encoding graphs...')
    nodes = pd.read_csv(f'{path}/nodes.csv')
    edges = pd.read_csv(f'{path}/edges.csv')
    os.makedirs(f'{path}/graphs', exist_ok=True)
    node_desc_list = []
    for n in nodes['node_attr']:
        node_desc_list.append(node_desc[n])
    print(node_desc_list)

    # The tool-node features, edge features and topology are IDENTICAL for every sample
    # -> encode them ONCE. Per sample only the request super-node embedding differs, so we
    # encode just that one text and concatenate. Output .pt is bit-identical to encoding
    # node_desc_list + [request] together (SBERT encodes each text independently).
    node_list = nodes.node_attr.tolist()
    edge_list = edges.edge_attr.tolist() + ['precedes'] * len(node_list)
    node_embeds = text2embedding(model, tokenizer, device, node_desc_list)
    e = text2embedding(model, tokenizer, device, edge_list)
    super_node_edge_index = torch.LongTensor([[i for i in range(len(node_list))], [len(node_list)] * len(node_list)])
    edge_index = torch.hstack((torch.LongTensor([edges.src, edges.dst]), super_node_edge_index))

    for i in tqdm(range(len(raw_data))):
        req_embed = text2embedding(model, tokenizer, device, [json.loads(raw_data[i])["user_request"]])
        x = torch.cat([node_embeds, req_embed], dim=0)
        data = Data(x=x, edge_index=edge_index, edge_attr=e, num_nodes=len(node_list) + 1)
        torch.save(data, f'{path}/graphs/{i}.pt')


i = 0
with open(path + 'graph_desc.json', "r") as f:
    node_info = json.load(f)
nodes, edges = _textualize_graph(node_info)
edges.to_csv(f'{path}/edges.csv', index=False, columns=['src', 'edge_attr', 'dst'])
nodes.to_csv(f'{path}/nodes.csv', index=False, columns=['node_id', 'node_attr'])

model, tokenizer, device = load_model[model_name]()
text2embedding = load_text2embedding[model_name]
_encode_graph(model, tokenizer, device, text2embedding)
generate_split(len(raw_data), f'{path}/split')
