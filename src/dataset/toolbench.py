import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import json

model_name = 'sbert'
path = 'dataset/toolbench'
path_nodes = f'{path}/nodes'
path_edges = f'{path}/edges'
path_graphs = f'{path}/graphs'


class ToolbenchDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.prompt = None
        self.graph = None
        with open(path + '/data.json', "r")as f:
            self.raw_data = f.readlines()
    def __len__(self):
        """Return the len of the dataset."""
        return len(self.raw_data)

    def __getitem__(self, index):
        data = json.loads(self.raw_data[index])
        question = f'{data["user_request"]}'
        graph = torch.load(f'{path}/graphs/{index}.pt', weights_only=False)
        nodes = pd.read_csv(f'{path}/nodes.csv')
        desc = "and a list of tools:\n " + nodes.to_csv(index=False)
        label = ''
        for i, t in enumerate(data["task_nodes"]):
            label +=f"Tool{i + 1}: " +  t["task"] + "\n"
        return {
            'id': index,
            'question': question,
            'label': label,
            'graph': graph,
            'desc': desc,
        }

    def get_idx_split(self):

        # Load the saved indices
        with open(f'{path}/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]
        with open(f'{path}/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]

        return {'train': train_indices, 'val': val_indices, 'test': test_indices}



if __name__ == '__main__':

    dataset = ToolbenchDataset()
    print(len(dataset))
    data = dataset[0]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
