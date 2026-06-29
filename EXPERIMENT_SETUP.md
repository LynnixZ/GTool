# 实验设置（Experiment Setup）

本文件记录本次基于 **GTool**（*Graph Enhanced Tool Planning with Large Language Model*, arXiv:2508.12725）的工具规划实验的完整设置：数据、切分、建图、模型、训练方法与评测。代码使用方法见 [README.md](README.md) §8。

---

## 1. 任务

**工具规划（tool planning）**：给定一个自然语言用户请求和一组候选工具，模型输出完成该请求所需工具的**有序序列**（`Tool1: ... → Tool2: ... → ...`），既要选对工具（节点），也要排对依赖顺序（边）。

> **协议：per-domain（按域独立）。** GTool 论文按域训练（`train.py --dataset <单个域>`），本实验也对每个域**独立**做「切分 → 训练 → 测试」，三个域互不混合——这也是与姊妹实验 `taskbench_sft`（同为 per-domain）对比的 baseline。**3 模型 × 3 域 = 9 个独立 run。**

---

## 2. 数据

### 2.1 数据来源与范围

- 数据集：**GNN4TaskPlan** 提供的三个域 —— `huggingface`、`multimedia`、`dailylife`（GNN4Plan / GRAFT / GTool 都基准在这同一份数据上，便于 1:1 对比）。
- 通过 [scripts/download_gnn4plan.sh](scripts/download_gnn4plan.sh) 从 [WxxShirley/GNN4TaskPlan](https://github.com/WxxShirley/GNN4TaskPlan) 下载到 `dataset_gnn4plan/<域>/`（含 `data.json`、`tool_desc.json`、`graph_desc.json`、`split_ids.json`）。
- 每个域含 single + chain 样本（DAG / 不连通 / 成环 / 重名歧义会在拓扑校验时剔除）。

> 不再使用 GTool 仓库自带的过滤子集（那份是全 chain 的 9398 条）；改用 GNN4TaskPlan 的数据 + 固定测试集，和论文严格可比。

### 2.2 切分（GNN4plan split，per-domain）

忠实复刻 GNN4TaskPlan / GNN4Plan / GRAFT / GTool 的设定，实现于 [src/dataset/preprocess_zou/split_subset.py](src/dataset/preprocess_zou/split_subset.py) 的 `make_split_gnn4plan`（`--mode gnn4plan`）：

- **test = 固定的 `split_ids.json` chains**（每域约 500，chain-only）——和论文报告的**同一批测试样本**。
- **train/val = single+chain 的 usable 池 减去 test**，用 `seed=42` shuffle，**截断到 3000**（`train_cap`），再按 **85 / 15** 切 train/val。
- **不做工具覆盖重抽**（GNN4Plan 不做，只 WARN）。
- 按域独立（每个域读自己的 `split_ids.json`），产物在 `artifacts/splits_gnn4plan/<域>/`。
- gold 顺序 `trajectory`：由 `task_links` 恢复的拓扑序。

实测各域切分（seed=42；huggingface 已本地验证）：

| 域 | test（固定 chains） | train | val | pool |
|---|---|---|---|---|
| huggingface | 498（500−2 无效） | 2550 | 450 | 3000（cap） |
| multimedia | ~split_ids | 2550 | 450 | 3000（cap） |
| dailylife | ~split_ids | 2550 | 450 | 3000（cap） |

> test 为 chain-only，故 `test_node` 恒为空、`test_chain == test_all`；train 含 single+chain。多/日域的具体 test 数取决于各自 `split_ids.json` 与拓扑校验（跑一次即知）。

### 2.3 图构建（沿用 GTool 逻辑，从 GNN4TaskPlan 数据）

- 由 `python -m src.dataset.preprocess_gnn4plan --root dataset_gnn4plan --domains <域>` 生成（GNN4TaskPlan 不带 `node_desc.json`，从 `tool_desc.json` 派生；其余与 GTool 一致）。
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

  输出示例（per-domain，以 huggingface test=363 为例）：
  ```
  [test_all]
  group        count   node_f1   edge_f1       ned
  tool=2          42    0.xxxx    0.xxxx    0.xxxx
  tool=3         136    0.xxxx    0.xxxx    0.xxxx
  tool>=4        185    0.xxxx    0.xxxx    0.xxxx
  overall        363    0.xxxx    0.xxxx    0.xxxx
  ```

---

## 6. 复现流程

先在联网节点下载 GNN4TaskPlan 数据，再按域跑。`run_experiment.sh` 把「建图 → GNN4plan 按域切分 → 训练 → 测试」串起来（幂等，重复跑只补缺失）；每个 `(模型, 域)` 用独立 `TAG=g4p_<模型>_<域>`、`output/<TAG>/`、`artifacts/splits_gnn4plan/<域>/`，train/test 同参数 → checkpoint 路径对得上。

```bash
# 0) 一次性：下载 GNN4TaskPlan 数据（联网节点）
bash scripts/download_gnn4plan.sh                # -> dataset_gnn4plan/<域>/

# 1) 一个模型 × 三个域（串行；首次建图 + 切分）
bash run_experiment.sh mistral                   # 换 qwen3-8b / vicuna 即可

# 只跑单个 (模型, 域)
ONLY_DOMAIN=huggingface bash run_experiment.sh mistral

# 烟测（gnn4plan tiny：train_cap=30 test_cap=8，1 epoch；默认三域）
bash run_experiment.sh --smoke
SMOKE_DOMAIN=huggingface bash run_experiment.sh --smoke   # 单域最快

# 多卡并行：3 模型 × 3 域 = 9 个 run 铺到各卡（一卡一 run）
bash run_grid.sh vicuna mistral qwen3-8b
```

> 底层等价：`python train_zou.py --dataset g4p_<模型>_<域> --llm_model_name <模型> --split_dir artifacts/splits_gnn4plan/<域> --raw_root dataset_gnn4plan`（+ 同参数 `inference_zou.py`）。

### 6.1 迁移（跨域）实验

在某个**源域**训练好后，用同一 checkpoint 测**全部 3 个域**的 test 集（含自身），衡量跨域泛化。可行的原因：GNN 作用在 SBERT 特征上、与具体工具无关；LLM 在 prompt 里看到的是**目标域**的工具列表，所以源域模型能在目标域图上规划。

```bash
# 先正常训练源域（如 huggingface），再：
bash transfer_eval.sh mistral huggingface        # hf 训练 → 测 hf / mm / dl
```

[transfer_eval.sh](transfer_eval.sh) 复用源域 checkpoint（`output/g4p_<模型>_huggingface/`），对每个目标域调 `inference_zou.py --eval_tag <目标域>`，结果写成 `..._evalon_<目标域>_test_all.csv`，互不覆盖。注意：transfer 时传的训练超参（num_epochs/patience/seed）要和训练一致，否则定位不到 checkpoint。

---

## 7. 注意事项

1. **硬件 / 显存**：[src/model/GTool.py](src/model/GTool.py) 用 `device_map="auto"`，自适应任意 GPU 数/显存（原写死的 2×80G 已移除）。冻结 7B/8B + 小 GNN，单张 A100 默认参数即可；24G（4090）跑 7B/8B 建议 `--batch_size 1~2`、`--max_txt_len 1536`。架构选 Ampere/Ada，避开 Blackwell。
2. **多 GPU**：单个 run 无数据并行（无 DDP），多卡只会把一个模型切片摊开（model-parallel），对能塞进单卡的 7B/8B 不加速。要用多卡请用 [run_grid.sh](run_grid.sh)：工作单元是 `(模型, 域)` pair（3×3=9），每卡跑一个 pair（任务级并行），pair 多于卡时自动排队；它会**先串行建好 graphs + 各域 split** 再 fan-out，避免并发建同一份 split 的竞争。
3. **SBERT 本地化**：`all-roberta-large-v1` 默认 `local_files_only=True`，需提前下载到本地缓存。
4. **Qwen3 模板**：当前为能跑通的 baseline chat 模板，若要进一步提点可细化。
5. **种子区分**：切分 seed=42（固定，保证可复现）与训练 seed=0（`--seed`）是两个独立种子。
6. **checkpoint 匹配**：测试时超参（含 `--dataset` tag）必须与训练一致，否则找不到 `..._checkpoint_best.pth`。
