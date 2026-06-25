"""Evaluate a trained GTool checkpoint on the *zou* (stratified) split.

Loads the best checkpoint (matched by the same hyper-params used at training
time) and reports metrics. By default it evaluates all three test files
(``test_node``, ``test_chain``, ``test_all``) so you get the per-topology
breakdown the stratified split is designed for; pass ``--test_split`` to
restrict to one.

Example:
    python inference_zou.py \
        --dataset zou_mistral \
        --llm_model_name mistral \
        --split_dir /path/to/artifacts/splits
"""
import gc
from tqdm import tqdm
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader

from src.model.GTool import GTool
from src.dataset.zou_split import ZouSplitDataset
from src.utils.evaluate import eval_grouped, format_grouped
from src.config import parse_args_llama, llama_model_path
from src.utils.ckpt import _reload_best_model
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything


def run_split(model, args, test_split):
    dataset = ZouSplitDataset(split_dir=args.split_dir, raw_root=args.raw_root,
                              test_split=test_split, load_train_val=False)
    idx_split = dataset.get_idx_split()
    test_dataset = [dataset[i] for i in idx_split['test']]
    if len(test_dataset) == 0:
        print(f"[{test_split}] empty -> skipped (the bundled subset is all chains, so test_node is empty).")
        return None
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, shuffle=False, collate_fn=collate_fn)

    path = f'{args.output_dir}/{args.dataset}/llm_model_name_{args.llm_model_name}_gnn_num_layers_{args.gnn_num_layers}_mask_prob_{args.mask_prob}_LLMP_dim_{args.LLMP_dim}_alpha_{args.alpha}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed_{args.seed}_{test_split}.csv'
    print(f'[{test_split}] n={len(test_dataset)} -> {path}')

    progress_bar_test = tqdm(range(len(test_loader)))
    with open(path, "w") as f:
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)

    return eval_grouped(path)


def main(args):
    seed_everything(seed=args.seed)
    print(args)

    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = GTool(args=args)
    model = _reload_best_model(model, args)
    model.eval()

    # 'all' (default unless the user picked a single file) -> report each test file.
    test_splits = ['test_node', 'test_chain', 'test_all'] if args.test_split in ('test_all', 'all') else [args.test_split]

    results = {}
    for ts in test_splits:
        perf = run_split(model, args, ts)
        if perf is not None:
            results[ts] = perf

    print("\n==== Test performance (broken down by gold tool count) ====")
    for ts, perf in results.items():
        print(f"\n[{ts}]")
        print(format_grouped(perf))


if __name__ == "__main__":
    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
