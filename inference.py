import gc
from tqdm import tqdm
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
import time
from src.model.GTool import GTool
from src.dataset import load_dataset
from src.utils.evaluate import eval
from src.config import parse_args_llama, llama_model_path
from src.utils.ckpt import _reload_best_model
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything

def main(args):

    seed_everything(seed=args.seed)
    print(args)

    dataset = load_dataset[args.dataset]()
    idx_split = dataset.get_idx_split()

    test_dataset = [dataset[i] for i in idx_split['test']]

    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, shuffle=False, collate_fn=collate_fn)

    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = GTool(args=args)

    path = f'{args.output_dir}/{args.dataset}/llm_model_name_{args.llm_model_name}_gnn_num_layers_{args.gnn_num_layers}_mask_prob_{args.mask_prob}_LLMP_dim_{args.LLMP_dim}_alpha_{args.alpha}_patience_{args.patience}_num_epochs_{args.num_epochs}_seed_{args.seed}.csv'
    print(f'path: {path}')
    print(len(test_dataset))
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

    performance = eval(path)
    print(f'Test performance {performance}')


if __name__ == "__main__":

    args = parse_args_llama()
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    gc.collect()
