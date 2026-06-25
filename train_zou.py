"""Train GTool on the *zou* (stratified) split.

Same training logic as ``train.py`` — only the data split changes: instead of
GTool's positional ``split/*.txt`` we read the stratified JSONL split produced by
``taskbench_sft`` / ``src.dataset.preprocess_zou`` (selected via ``--split_dir``).

Example:
    python train_zou.py \
        --dataset zou_mistral \
        --llm_model_name mistral \
        --split_dir /path/to/artifacts/splits \
        --raw_root dataset

`--dataset` is only used as the output namespace (``output/<dataset>/``), so use a
distinct tag per run so checkpoints/results don't collide.
"""
import os
import gc
from tqdm import tqdm
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.model.GTool import GTool
from src.dataset.zou_split import ZouSplitDataset
from src.utils.evaluate import eval_grouped, format_grouped
from src.config import parse_args_llama, llama_model_path
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate


def main(args):
    seed_everything(seed=args.seed)
    print(args)

    dataset = ZouSplitDataset(split_dir=args.split_dir, raw_root=args.raw_root,
                              test_split=args.test_split)
    idx_split = dataset.get_idx_split()

    train_dataset = [dataset[i] for i in idx_split['train']]
    val_dataset = [dataset[i] for i in idx_split['val']]
    test_dataset = [dataset[i] for i in idx_split['test']]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, shuffle=False, collate_fn=collate_fn)

    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = GTool(args=args, init_prompt=dataset.prompt)

    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
        betas=(0.9, 0.95)
    )
    trainable_params, all_param = model.print_trainable_params()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, accum_loss = 0., 0.

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)

            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

            if (step + 1) % args.grad_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                accum_loss = 0.

            progress_bar.update(1)

        print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
        val_loss = 0.

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = model(batch)
                val_loss += loss.item()
            val_loss = val_loss / len(val_loader)
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

        if epoch - best_epoch >= args.patience:
            print(f'Early stop at epoch {epoch}')
            break

    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()

    # Final test on the chosen test split (default test_all).
    os.makedirs(f'{args.output_dir}/{args.dataset}', exist_ok=True)
    path = f'{args.output_dir}/{args.dataset}/llm_model_name_{args.llm_model_name}_gnn_num_layers_{args.gnn_num_layers}_mask_prob_{args.mask_prob}_LLMP_dim_{args.LLMP_dim}_alpha_{args.alpha}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed_{args.seed}_{args.test_split}.csv'
    print(f'path: {path}')

    model = _reload_best_model(model, args)
    model.eval()
    progress_bar_test = tqdm(range(len(test_loader)))
    with open(path, "w") as f:
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                output = model.inference(batch)
                df = pd.DataFrame(output)
                for _, row in df.iterrows():
                    f.write(json.dumps(dict(row)) + "\n")
            progress_bar_test.update(1)

    performance = eval_grouped(path)
    print(f'\nTest [{args.test_split}] performance (broken down by gold tool count):')
    print(format_grouped(performance))


if __name__ == "__main__":
    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
