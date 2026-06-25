import torch
from random import random
from torch_geometric.utils import negative_sampling
def batch_mask(edge_index, num_nodes, num_graphs, prob=0.1):
    neg_edge_index = negative_sampling(edge_index)

    raw_src = edge_index[0, :]
    raw_dst = edge_index[1, :]
    # skip the adj of supernode
    super_nodes = [i for i in range(num_nodes // num_graphs - 1, num_nodes, num_nodes // num_graphs)]

    src = []
    dst = []
    mask_src = []
    mask_dst = []
    for i in range(raw_src.shape[0]):
        
        if random() < prob and (raw_src[i] not in super_nodes and raw_dst[i] not in super_nodes):
            mask_src.append(raw_src[i])
            mask_dst.append(raw_dst[i]) 
        else:
            src.append(raw_src[i])
            dst.append(raw_dst[i])

    return torch.LongTensor([src, dst]), torch.LongTensor([mask_src, mask_dst]) , neg_edge_index
    