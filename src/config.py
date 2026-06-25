import argparse

llama_model_path = {
    "llama": "meta-llama/Llama-2-7b-hf",
    "vicuna": "lmsys/vicuna-7b-v1.5",
    "qwen3": "Qwen/Qwen3-14B",
    # models we actually train/test on the zou (stratified) split
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen3-8b": "Qwen/Qwen3-8B",
    # tiny models for smoke tests (fit easily on a single 4090/24G)
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",          # smallest Qwen3 (there is no 0.5B in Qwen3)
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
}

def csv_list(string):
    return string.split(',')

def parse_args_llama():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", type=str, default='huggingface')
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--patience", type=float, default=2)

    # Model Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_steps", type=int, default=2)
    
    # Learning Rate Scheduler
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=float, default=1)

    # Inference
    parser.add_argument("--eval_batch_size", type=int, default=8)

    # LLM related
    parser.add_argument("--llm_model_name", type=str, default='llama')
    parser.add_argument("--llm_model_path", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='output')

    # Zou (stratified) split — produced by taskbench_sft / src.dataset.preprocess_zou.
    # Point this at the directory containing train.jsonl / validation.jsonl /
    # test_node.jsonl / test_chain.jsonl / test_all.jsonl / split_manifest.json.
    parser.add_argument("--split_dir", type=str, default='')
    # Root holding GTool's per-domain preprocessed graphs (graphs/, nodes.csv, data.json).
    parser.add_argument("--raw_root", type=str, default='dataset')
    # Which test file to evaluate on: test_all | test_node | test_chain.
    parser.add_argument("--test_split", type=str, default='test_all')

    parser.add_argument("--max_txt_len", type=int, default=3072)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    # GNN related
    parser.add_argument("--gnn_model_name", type=str, default='gt')
    parser.add_argument("--gnn_num_layers", type=int, default=3)
    parser.add_argument("--gnn_in_dim", type=int, default=1024)
    parser.add_argument("--gnn_hidden_dim", type=int, default=1024)
    parser.add_argument("--gnn_num_heads", type=int, default=4)
    parser.add_argument("--gnn_dropout", type=float, default=0.0)
    
    
    parser.add_argument("--mask_prob", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--LLMP_dim", type=int, default=4)


    args = parser.parse_args()
    return args
