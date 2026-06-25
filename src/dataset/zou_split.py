"""Dataset that feeds GTool from the *zou* (stratified) split.

The split is produced by ``taskbench_sft`` (mirrored under
``src/dataset/preprocess_zou``). Each line of ``train.jsonl`` /
``validation.jsonl`` / ``test_*.jsonl`` is a ``GoldSample.to_record()`` dict
keyed by the original TaskBench ``id`` and carrying its ``domain`` and an
execution-ordered ``trajectory`` (the gold tool order).

Everything *downstream* of the split stays exactly as in GTool: we reuse the
per-domain graphs / ``nodes.csv`` that GTool's own preprocessing
(``python -m src.dataset.preprocess.<domain>``) writes under ``dataset/<domain>/``.
This class only changes *which* samples land in train/val/test and *how* they
are matched to a graph — it does not re-encode anything.

Prerequisite: run GTool preprocessing for every domain you split over, e.g.::

    python -m src.dataset.preprocess.huggingface
    python -m src.dataset.preprocess.multimedia
    python -m src.dataset.preprocess.dailylife
"""
import os
import json

import pandas as pd
import torch
from torch.utils.data import Dataset

# zou domain name -> GTool dataset directory under ``raw_root``.
DOMAIN_TO_DIR = {
    "data_huggingface": "huggingface",
    "data_multimedia": "multimedia",
    "data_dailylifeapis": "dailylife",
}

SPLIT_FILES = {
    "train": "train.jsonl",
    "val": "validation.jsonl",
    # 'test' filename is chosen at runtime via ``test_split`` (test_all/node/chain).
}


class ZouSplitDataset(Dataset):
    def __init__(self, split_dir, raw_root="dataset", test_split="test_all",
                 load_train_val=True):
        super().__init__()
        if not split_dir:
            raise ValueError("ZouSplitDataset requires --split_dir (the preprocess_zou output dir).")
        self.split_dir = split_dir
        self.raw_root = raw_root
        self.prompt = None  # kept for GTool API compatibility (unused)

        self.records = []                 # combined, cross-domain list of records
        self._split = {"train": [], "val": [], "test": []}
        self._id_index = {}               # domain_dir -> {sample_id: line_idx in data.json}
        self._desc_cache = {}             # domain_dir -> desc string (nodes.csv dump)

        files = {}
        if load_train_val:
            files.update(SPLIT_FILES)
        files["test"] = f"{test_split}.jsonl"

        for key, fname in files.items():
            path = os.path.join(split_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._split[key].append(len(self.records))
                    self.records.append(rec)

    # ------------------------------------------------------------------ helpers
    def _domain_dir(self, rec):
        domain = rec["domain"]
        if domain not in DOMAIN_TO_DIR:
            raise KeyError(f"Unknown domain {domain!r}; extend DOMAIN_TO_DIR.")
        return DOMAIN_TO_DIR[domain]

    def _id_map(self, domain_dir):
        """Map a domain's TaskBench id -> positional line index in its data.json.

        GTool saves graphs as ``graphs/{line_index}.pt``, so we recover that
        index by scanning the same ``data.json`` once per domain (cached).
        """
        if domain_dir not in self._id_index:
            mapping = {}
            data_path = os.path.join(self.raw_root, domain_dir, "data.json")
            with open(data_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        mapping[str(json.loads(line)["id"])] = i
            self._id_index[domain_dir] = mapping
        return self._id_index[domain_dir]

    def _desc(self, domain_dir):
        if domain_dir not in self._desc_cache:
            nodes = pd.read_csv(os.path.join(self.raw_root, domain_dir, "nodes.csv"))
            self._desc_cache[domain_dir] = "and a list of tools:\n " + nodes.to_csv(index=False)
        return self._desc_cache[domain_dir]

    # ----------------------------------------------------------------- Dataset
    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        domain_dir = self._domain_dir(rec)
        line_idx = self._id_map(domain_dir)[str(rec["id"])]

        graph = torch.load(os.path.join(self.raw_root, domain_dir, "graphs", f"{line_idx}.pt"))
        desc = self._desc(domain_dir)

        # The split already provides the gold execution order in `trajectory`;
        # fall back to node order only if it is missing (single-node samples).
        traj = rec.get("trajectory") or [n["task"] for n in rec["task_nodes"]]
        label = "".join(f"Tool{k + 1}: {tool}\n" for k, tool in enumerate(traj))

        return {
            "id": index,
            "image_id": rec["id"],
            "domain": rec["domain"],
            "question": rec["user_request"],
            "label": label,
            "graph": graph,
            "desc": desc,
        }

    def get_idx_split(self):
        return {
            "train": self._split["train"],
            "val": self._split["val"],
            "test": self._split["test"],
        }


if __name__ == "__main__":
    import sys

    split_dir = sys.argv[1] if len(sys.argv) > 1 else ""
    ds = ZouSplitDataset(split_dir=split_dir)
    idx = ds.get_idx_split()
    print({k: len(v) for k, v in idx.items()})
    sample = ds[idx["test"][0]]
    for k, v in sample.items():
        print(f"{k}: {str(v)[:120]}")
