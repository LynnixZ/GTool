# 实验设置（Experiment Setup）

本文件记录本次基于 **GTool**（*Graph Enhanced Tool Planning with Large Language Model*, arXiv:2508.12725）的工具规划实验的完整设置：数据、切分、建图、模型、训练方法与评测。代码使用方法见 [README.md](README.md) §8。

---

## 1. 任务

**工具规划（tool planning）**：给定一个自然语言用户请求和一组候选工具，模型输出完成该请求所需工具的**有序序列**（`Tool1: ... → Tool2: ... → ...`），既要选对工具（节点），也要排对依赖顺序（边）。

---

## 2. 数据

### 2.1 数据来源与范围

- 数据集：**TaskBench** 三个域 —— `huggingface`、`multimedia`、`dailylife`。
- 使用 GTool 仓库自带的**过滤子集**（不使用全量 TaskBench）：

  | 域 | 样本数 | 拓扑类型 | 依赖类型 |
  |---|---|---|---|
  | huggingface | 3630 | 全部 chain | resource |
  | multimedia | 2981 | 全部 chain | resource |
  | dailylife | 2787 | 全部 chain | temporal |
  | **合计** | **9398** | | |

- **关键特征：子集全部为 `chain` 类型，不含单工具（`node`）样本。** 因此工具数最少为 2，分层与分桶里都不会出现 `tool=1`。

### 2.2 切分（stratified split）

- 切分逻辑：复刻自 `taskbench_sft` 的分层切分，自包含实现于 [src/dataset/preprocess_zou/split_subset.py](src/dataset/preprocess_zou/split_subset.py)（零外部依赖，直接跑在 GTool 子集上）。
- 比例：**80 / 10 / 10**（train / val / test）。
- 分层维度：`domain × topology × chain_length_bucket`（本子集 topology 恒为 chain，等效于 `domain × chain_length_bucket`）。
- 切分种子：**seed = 42**；每个 stratum 用 `random.Random(f"{seed}|{key}")` 做确定性 shuffle。
- **训练集工具覆盖保证**：val/test 中出现的每个工具，必须在 train 中至少出现一次；否则用 `seed+attempt` 重抽（最多 50 次）。
- chain 合法性校验：必须是**简单连通路径**，否则排除（DAG / 不连通 / 成环 / 重名歧义都剔除）。
- gold 顺序 `trajectory`：由 `task_links` 恢复的**拓扑序**。

实测切分结果（seed=42）：

| split | 数量 | 说明 |
|---|---|---|
| train | 7518 | |
| validation | 939 | |
| test_all | 941 | = test_chain |
| test_chain | 941 | |
| test_node | 0 | 子集无单工具样本，预期为空 |

> 已校验：9398 条 usable 样本的 id 与子集 `data.json` 行号 **0 缺失**匹配。

### 2.3 图构建（沿用 GTool 逻辑）

- 由 `python -m src.dataset.preprocess.<domain>` 生成（每个域一次）。
- **工具依赖图（拓扑）**：来自各域 `graph_desc.json`，是预定义的领域工具依赖图（如 huggingface 为 23 节点 / 225 边），**对所有样本相同，不从样本统计、无训练/测试泄漏**。
- **节点/边特征**：用 SBERT（`sentence-transformers/all-roberta-large-v1`，1024 维）编码节点描述、边类型与用户请求。
- **super-node（请求节点）**：连接所有工具节点，特征 = 该样本用户请求的 SBERT 嵌入；其 GNN 输出即图级表示。
- 边属性统一标记为 `precedes`。
- 每条样本的图存为 `graphs/{line_index}.pt`；[src/dataset/zou_split.py](src/dataset/zou_split.py) 通过 `id → data.json 行号` 把切分样本对应到图。

---

## 3. 模型

### 3.1 主干 LLM（本次实验）

| 别名（`--llm_model_name`） | HF 路径 | 备注 |
|---|---|---|
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.3` | gated（需 HF_TOKEN） |
| `qwen3-8b` | `Qwen/Qwen3-8B` | 非 gated |
| `vicuna` | `lmsys/vicuna-7b-v1.5` | 非 gated；Llama-2 底座，走 `[INST]` 格式 |

（仓库另含 `llama`=Llama-2-7b、`qwen3`=Qwen3-14B、以及烟测用的 `qwen3-0.6b` / `qwen2.5-0.5b` 备用。）

### 3.2 GTool 架构

- **LLM 全程冻结**，仅训练：GNN 编码器 + 两个可学习标记 token（`graph_token_embeds`、`node_token_embeds`）。
- **GNN**：GraphTransformer（`gt`，基于 TransformerConv），**3 层**，hidden=1024，heads=4，in_dim=1024（对齐 SBERT）。
- **图 token 注入**：图级表示投影到 LLM 词嵌入空间，以 soft token 形式拼进 prompt（`[BOS] <graph_token> 图向量 <graph_token> [工具列表 + 用户请求] [/INST] [答案]`）。
- **按模型族适配 prompt 格式**（[src/model/GTool.py](src/model/GTool.py) 的 `PROMPT_FORMATS`，llama/vicuna 保持原样）：

  | 模型族 | BOS / 用户结束 / EOS | tokenizer | pad |
  |---|---|---|---|
  | llama / vicuna | `<s>[INST]` / `[/INST]` / `</s>` | use_fast=False | `<pad>`(id=0) |
  | mistral | `<s>[INST]` / `[/INST]` / `</s>` | use_fast=True | 复用 eos |
  | qwen | `<\|im_start\|>user\n` / `<\|im_end\|>\n<\|im_start\|>assistant\n` / `<\|im_end\|>` | use_fast=True | tokenizer 自带 |

---

## 4. 训练方法

- **损失**：`loss = LM_loss + α · EARE_loss`
  - `LM_loss`：对 gold 工具序列的自回归监督损失。
  - `EARE_loss`（论文中的 MDPL，缺失依赖预测）：随机 mask 一部分边，用两个节点的 GNN 向量构造 prompt 让 LLM 回答 yes/no（是否有边），正样本=被 mask 的真实边、负样本=负采样边。
- **训练超参**（[src/config.py](src/config.py) 默认值）：

  | 参数 | 值 |
  |---|---|
  | optimizer | AdamW (betas=0.9/0.95) |
  | lr / weight_decay | 1e-5 / 0.05 |
  | warmup / 调度 | warmup_epochs=1，cosine（含梯度裁剪 0.1） |
  | num_epochs | 10 |
  | patience（early stop） | 2（按 val loss） |
  | batch_size / grad_steps | 4 / 2 |
  | eval_batch_size | 8 |
  | max_txt_len / max_new_tokens | 3072 / 64 |
  | mask_prob（EARE 边 mask 比例） | 0.1 |
  | α（EARE 权重） | 0.1 |
  | LLMP_dim（EARE 每批正/负边采样数） | 4 |
  | 训练 seed | 0（注意：与切分 seed=42 相互独立） |

- **模型选择**：按验证集 loss 保存最优 checkpoint，early stop（patience=2）。
- 训练入口：[train_zou.py](train_zou.py)（逻辑同 GTool `train.py`，只替换切分）。

---

## 5. 评测

- 入口：[inference_zou.py](inference_zou.py)；指标实现 [src/utils/evaluate.py](src/utils/evaluate.py)。
- **三个指标**：
  - **node_f1**：工具选择 F1（选对哪些工具）。
  - **edge_f1**：相邻工具对 F1（依赖/顺序是否正确）。
  - **NED**：基于 Levenshtein 的归一化编辑距离（`1 - ratio`，越低越好）。
- **按 gold 工具数量分桶报告**（`eval_grouped`）：工具数 = gold label 中 `Tool i:` 的行数。
  - 分桶：**`tool=2` / `tool=3` / `tool>=4`**，外加 **`overall`**（每桶给 count + 三个指标）。
  - 在 `test_chain` / `test_all` 上分别评测（`test_node` 为空自动跳过）。

  输出示例：
  ```
  [test_all]
  group        count   node_f1   edge_f1       ned
  tool=2         312    0.xxxx    0.xxxx    0.xxxx
  tool=3         289    0.xxxx    0.xxxx    0.xxxx
  tool>=4        340    0.xxxx    0.xxxx    0.xxxx
  overall        941    0.xxxx    0.xxxx    0.xxxx
  ```

---

## 6. 复现流程

```bash
# 1) 建图（GTool 原生预处理，每个域一次）
python -m src.dataset.preprocess.huggingface
python -m src.dataset.preprocess.multimedia
python -m src.dataset.preprocess.dailylife

# 2) 分层切分（子集，seed=42）
python -m src.dataset.preprocess_zou.split_subset \
    --raw_root dataset --out_dir artifacts/splits_subset

# 3) 训练（Qwen3 把 mistral 换成 qwen3-8b、tag 换成 zou_qwen3_8b）
python train_zou.py --dataset zou_mistral --llm_model_name mistral \
    --split_dir artifacts/splits_subset --raw_root dataset

# 4) 测试（分桶 + overall）
python inference_zou.py --dataset zou_mistral --llm_model_name mistral \
    --split_dir artifacts/splits_subset --raw_root dataset
```

---

## 7. 注意事项

1. **硬件 / 显存**：[src/model/GTool.py](src/model/GTool.py) 用 `device_map="auto"`，自适应任意 GPU 数/显存（原写死的 2×80G 已移除）。冻结 7B/8B + 小 GNN，单张 A100 默认参数即可；24G（4090）跑 7B/8B 建议 `--batch_size 1~2`、`--max_txt_len 1536`。架构选 Ampere/Ada，避开 Blackwell。
2. **多 GPU**：单个 run 无数据并行（无 DDP），多卡只会把一个模型切片摊开（model-parallel），对能塞进单卡的 7B/8B 不加速。要用多卡请用 [run_grid.sh](run_grid.sh)：每卡跑一个独立模型（任务级并行），模型多于卡时自动排队。
3. **SBERT 本地化**：`all-roberta-large-v1` 默认 `local_files_only=True`，需提前下载到本地缓存。
4. **Qwen3 模板**：当前为能跑通的 baseline chat 模板，若要进一步提点可细化。
5. **种子区分**：切分 seed=42（固定，保证可复现）与训练 seed=0（`--seed`）是两个独立种子。
6. **checkpoint 匹配**：测试时超参（含 `--dataset` tag）必须与训练一致，否则找不到 `..._checkpoint_best.pth`。
