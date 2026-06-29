# GTool: Graph Enhanced Tool Planning with Large Language Model

本仓库是论文 **《Graph Enhanced Tool Planning with Large Language Model》** 的实现代码。
本文档总结了方法原理、代码结构和完整的使用流程，方便在 TaskBench 等数据集上选择不同 LLM 进行训练与测试。

---

## 1. 方法概览（论文做了什么）

工具规划（tool planning）任务：给定一个用户请求和一组候选工具，模型需要输出**完成该请求所需工具的有序序列**（先调用哪个、后调用哪个）。

传统做法只把工具描述当作纯文本喂给 LLM，忽略了工具之间天然存在的**依赖关系**（A 的输出是 B 的输入）。GTool 的核心思想是：**把工具集合建模成一张图，用 GNN 编码后作为“软提示（soft token）”注入冻结的 LLM**，让 LLM 在感知工具拓扑结构的前提下做规划。整体属于 G-Retriever / GraphToken 这一类“图 → soft prompt → 冻结 LLM”的架构。

### 关键组件

1. **工具图构建（图文本化）**
   - 每个数据集提供 `graph_desc.json`，其中 `links` 描述工具间的先后依赖（`source --precedes--> target`）。
   - 代码把图转成节点表 `nodes.csv` 和边表 `edges.csv`（见 `src/dataset/preprocess/*.py`）。
   - 额外加入一个 **super-node（超级节点）**，与所有工具节点相连，用来聚合出“图级别”表示；该超级节点的初始特征用**当前用户请求的文本嵌入**。

2. **文本嵌入**
   - 节点描述（`node_desc.json`）、边类型、用户请求都用 **SBERT（`sentence-transformers/all-roberta-large-v1`，1024 维）** 编码成向量，作为图的节点/边特征。
   - 编码逻辑见 [src/utils/lm_modeling.py](src/utils/lm_modeling.py)（也支持 contriever / word2vec，默认 sbert）。

3. **GNN 编码器**
   - 默认 `gt`（GraphTransformer），另含 `gcn`、`gat`，见 [src/model/gnn.py](src/model/gnn.py)。
   - 输入维度 1024，输出维度对齐 LLM 词向量维度，super-node 的输出向量即作为**图表示**。

4. **图 token 注入冻结 LLM**
   - LLM **全程冻结**，只训练 GNN 编码器以及几个特殊的可学习 token（`graph_token_embeds`、`node_token_embeds`）。
   - 把图表示投影到 LLM 词嵌入空间，拼成 `[BOS] <graph_token> <图向量> <graph_token> [工具描述+用户请求] [/INST] [答案]` 的形式喂入 LLM。
   - 见 [src/model/GTool.py](src/model/GTool.py) 的 `forward` / `inference`。

5. **EARE 辅助损失（边关系自监督）**
   - 训练时随机 mask 掉一部分边（`mask_prob`），把两个被 mask 节点的 GNN 向量拼成 prompt，让 LLM 回答“这两个节点之间是否有边连接（yes/no）”，正样本用真实被 mask 的边、负样本用负采样的边。
   - 目的是逼 GNN 学到能反映图结构的表示。见 `GTool.EARE_loss` 与 [src/utils/mask.py](src/utils/mask.py)。
   - 总损失：`loss = LM_loss + alpha * eare_loss`（`alpha` 默认 0.1）。

6. **输出与评测**
   - 模型自回归生成 `Tool1: xxx\nTool2: yyy\n...` 形式的工具序列。
   - 评测三个指标（见 [src/utils/evaluate.py](src/utils/evaluate.py)）：
     - **node_f1**：工具选择的 F1（选对了哪些工具）。
     - **edge_f1**：相邻工具对的 F1（顺序/依赖是否正确）。
     - **NED**：基于 Levenshtein 的归一化编辑距离（`1 - ratio`，越小越好）。

---

## 2. 代码结构

```
GTool/
├── train.py                       # 训练 + 训练后自动测试评测的主入口
├── inference.py                   # 仅加载已训练 checkpoint 做测试评测
├── src/
│   ├── config.py                  # 所有超参数 + LLM 模型名→路径映射（要加新模型改这里）
│   ├── model/
│   │   ├── GTool.py               # 核心模型：冻结 LLM + GNN + 图 token + EARE loss
│   │   └── gnn.py                 # GCN / GAT / GraphTransformer
│   ├── dataset/
│   │   ├── __init__.py            # load_dataset 注册表（数据集名→类）
│   │   ├── huggingface.py         # 各数据集的 Dataset 类（运行期读取预处理产物）
│   │   ├── multimedia.py
│   │   ├── dailylife.py
│   │   ├── toole.py
│   │   └── preprocess/            # 预处理脚本：建图、SBERT 编码、切分 train/val/test
│   │       ├── huggingface.py
│   │       ├── multimedia.py
│   │       ├── dailylife.py
│   │       ├── toole.py
│   │       └── generate_split.py  # 6:2:2 随机切分（random_state=42）
│   └── utils/                     # 评测、mask、collate、lr 调度、checkpoint、seed
├── dataset/                       # 原始数据（json），预处理产物也会写到各子目录
├── output/                        # checkpoint(.pth) 与测试结果(.csv) 输出目录
└── requirements.txt
```

### 数据集说明

| 数据集 | 属于 | data.json 样本数 | 是否已接入 `load_dataset` |
|---|---|---|---|
| `huggingface` | **TaskBench** | 3630 | ✅ |
| `multimedia`  | **TaskBench** | 2981 | ✅ |
| `dailylife`   | **TaskBench** | 2787 | ✅ |
| `toole`       | ToolE | （无 data.json） | ✅ |
| `toolbench`   | ToolBench | 298 | ❌ 未在 `__init__.py` 注册，暂不可直接用 |

> **TaskBench 训练/测试主要用前三个数据集：`huggingface` / `multimedia` / `dailylife`。**

---

## 3. 环境准备

```bash
pip install -r requirements.txt
```

关键依赖（仓库自带 `requirements.txt` 的固定版本）：`torch==2.1.0`、`transformers==4.51.3`、`torch-geometric==2.6.1`（含 torch-scatter/sparse/cluster 的 cu121 轮子）、`peft==0.14.0`、`Levenshtein==0.26.1`、`gensim==4.3.3`、`accelerate`、`sentencepiece`。

> ⚠️ 跨节点部署（中/美）请勿用 `requirements.txt`（conda freeze，含本地 `file://` 路径不可迁移）。改用 `requirements-node.txt` + `scripts/prestage_all.sh`，且 **torch 用 `2.2.2+cu121`**（`2.1.0` 已从 cu121 源下架，PyG 配套轮子改为 `pt22cu121`）。详见 [DEPLOY.md](DEPLOY.md)。

需要 **CUDA GPU**（代码大量使用 `cuda:0`、`device_map="auto"` 与 bf16 autocast，CPU 跑不起来）。

> ⚠️ SBERT 模型默认 `local_files_only=True`（见 lm_modeling.py），即要求 `sentence-transformers/all-roberta-large-v1` 已在本地 HF 缓存中。首次使用请先把该模型下载到本地，或临时把该参数改为 `False`。

---

## 4. 使用流程

### 步骤 1：预处理（建图 + SBERT 编码 + 切分）

> ⚠️ 仓库中 `dataset/<name>/` 目前**只有原始 json**，还没有 `nodes.csv` / `edges.csv` / `graphs/*.pt` / `split/`。必须先跑预处理，否则 `train.py` / `inference.py` 找不到文件会报错。

每个数据集跑两步（按 README 原始顺序，第一步生成图与编码，第二步可做自检）：

```bash
# 1) 预处理：建图、用 SBERT 编码节点/边/请求、生成 train/val/test 划分
python -m src.dataset.preprocess.huggingface
# python -m src.dataset.preprocess.multimedia
# python -m src.dataset.preprocess.dailylife
# python -m src.dataset.preprocess.toole

# 2) 自检：实例化 Dataset 打印第 0 条样本与各划分数量
python -m src.dataset.huggingface
# python -m src.dataset.multimedia
# python -m src.dataset.dailylife
# python -m src.dataset.toole
```

产物会写到 `dataset/<name>/` 下：`nodes.csv`、`edges.csv`、`graphs/{idx}.pt`、`split/{train,val,test}_indices.txt`。

### 步骤 2：训练（含训练后自动测试）

```bash
python train.py --dataset huggingface --llm_model_name llama
```

`train.py` 会：训练 → 按验证集 loss 保存最优 checkpoint（带 early stop，`patience` 默认 2）→ 自动加载最优权重在测试集生成预测 → 计算 node_f1 / edge_f1 / NED。

- Checkpoint 保存到 `output/<dataset>/..._checkpoint_best.pth`
- 测试预测保存到 `output/<dataset>/..._seed_<seed>.csv`（文件名编码了全部关键超参）

### 步骤 3：单独测试（复用已有 checkpoint）

```bash
python inference.py --dataset huggingface --llm_model_name llama
```

> 注意：`inference.py` 通过完全相同的超参拼出 checkpoint 路径来加载，所以**测试时传的超参必须和训练时一致**，否则会找不到 `..._checkpoint_best.pth`。

---

## 5. 选择/添加要训练的 LLM

模型名到 HF 路径的映射在 [src/config.py](src/config.py#L3-L7) 的 `llama_model_path`：

```python
llama_model_path = {
    "llama":  "meta-llama/Llama-2-7b-hf",
    "vicuna": "lmsys/vicuna-7b-v1.5",
    "qwen3":  "Qwen/Qwen3-14B",
}
```

**要加新模型**：在这里加一行 `"别名": "HF 仓库或本地路径"`，然后用 `--llm_model_name 别名` 即可。

### ⚠️ 换模型时需要注意（重要）

当前 `GTool.py` 是按 **Llama-2** 的约定写死的，换其它家族的模型前请检查/适配以下几处：

1. **取词嵌入的路径**：`self.model.model.get_input_embeddings()`（GTool.py:59）依赖 `.model.model` 这层结构，Llama 可用；个别架构层级不同，必要时改成 `self.model.get_input_embeddings()`。
2. **特殊 token / 对话模板**：`BOS='<s>[INST]'`、`EOS_USER='[/INST]'`、`EOS='</s>'`、`PAD='<pad>'`（GTool.py:10-13）是 Llama-2 chat 格式。Qwen3 等模型的特殊 token 和 chat template 不同，直接套用会影响效果，建议按目标模型调整。
3. **显存设置**：`max_memory={0:'80GiB', 1:'80GiB'}` + `device_map="auto"`（GTool.py:32-36）假设有 2 张 80G 卡。请按你的硬件改这里（单卡 / 不同显存）。14B 模型显存需求明显高于 7B。

---

## 6. 主要超参（`src/config.py`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | huggingface | 数据集：huggingface / multimedia / dailylife / toole |
| `--llm_model_name` | llama | LLM 别名（对应 `llama_model_path`） |
| `--lr` / `--wd` | 1e-5 / 0.05 | 学习率 / 权重衰减 |
| `--num_epochs` | 10 | 训练轮数 |
| `--patience` | 2 | early stop 容忍轮数 |
| `--batch_size` / `--eval_batch_size` | 4 / 8 | 训练 / 评测 batch |
| `--grad_steps` | 2 | 梯度累积步（用于 lr 调度节奏） |
| `--max_txt_len` / `--max_new_tokens` | 3072 / 64 | 输入截断长度 / 生成长度 |
| `--gnn_model_name` | gt | GNN 类型：gt / gcn / gat |
| `--gnn_num_layers` | 3 | GNN 层数 |
| `--gnn_in_dim` / `--gnn_hidden_dim` | 1024 / 1024 | GNN 维度（与 SBERT 1024 对齐） |
| `--mask_prob` | 0.1 | EARE 中被 mask 的边比例 |
| `--alpha` | 0.1 | EARE 损失权重 |
| `--LLMP_dim` | 4 | EARE 每个 batch 采样的正/负边对数量 |
| `--seed` | 0 | 随机种子 |

---

## 7. 在 TaskBench 上跑一遍的最小命令清单

```bash
# 0. 装依赖（并确保本地有 all-roberta-large-v1）
pip install -r requirements.txt

# 1. 预处理三个 TaskBench 数据集
python -m src.dataset.preprocess.huggingface
python -m src.dataset.preprocess.multimedia
python -m src.dataset.preprocess.dailylife

# 2. 针对某个模型逐数据集训练+测试
python train.py --dataset huggingface --llm_model_name llama
python train.py --dataset multimedia  --llm_model_name llama
python train.py --dataset dailylife   --llm_model_name llama

# 3. 换模型：在 config.py 注册别名后
python train.py --dataset huggingface --llm_model_name qwen3
```

结果（node_f1 / edge_f1 / NED）会在训练结束时打印，并可从 `output/<dataset>/*.csv` 复算（用 `src/utils/evaluate.py` 的 `eval(path)`）。

---

## 8. 本次实验用法（Mistral / Qwen3 / vicuna）

> ⚠️ **协议已更新为 GNN4plan（GNN4TaskPlan 数据 + 固定 `split_ids.json` 测试集），并新增跨域迁移实验。** 权威说明见 [EXPERIMENT_SETUP.md](EXPERIMENT_SETUP.md)（数据/切分/复现/迁移）。最短路径：
> ```bash
> bash scripts/download_gnn4plan.sh          # 下 GNN4TaskPlan 数据 -> dataset_gnn4plan/
> bash run_experiment.sh mistral             # 建图(从GNN4TaskPlan) + gnn4plan切分 + 训 + 测，3 域
> bash transfer_eval.sh mistral huggingface  # 跨域：hf 训练 → 测 hf/mm/dl
> ```
> 本节下方保留的是**早期 zou 分层切分**的说明（数据用仓库自带子集），已被 GNN4plan 取代，仅作历史参考——`run_experiment.sh` / `run_grid.sh` 现在默认 `RAW_ROOT=dataset_gnn4plan` + `--mode gnn4plan`。

---

### （历史）zou 分层切分版本

本小节是早期「用分层切分逻辑」的说明（已被 §8 顶部的 GNN4plan 取代）：**切分用 `preprocess_zou` 的分层逻辑，其余全部沿用 GTool；数据用仓库自带的过滤子集（不碰全量 TaskBench）。**

> **按域训练（per-domain，对齐 GTool 论文协议）**：GTool 论文是逐数据集训练的（`train.py --dataset <单个域>`），所以这里也**每个域独立切分、独立训练、独立测试**——**3 个模型 × 3 个域 = 9 次运行**，互不混合。每个 (model, domain) 用各自的 TAG / `output/<tag>/` / checkpoint / split 目录，绝不串台。

### 8.1 与 GTool 原生流程的区别

- **切分**：不再用 GTool 的 `split/*.txt`（按下标随机 6:2:2），改用分层切分（80/10/10，**在每个域内**按 `topology × chain_length_bucket` 分层 + 训练集工具覆盖保证，seed=42），文件为 `train.jsonl / validation.jsonl / test_node.jsonl / test_chain.jsonl / test_all.jsonl`。**每个域单独切分到 `artifacts/splits_subset/<domain>/`**（域是常量，故不再进 stratify key）。
- **数据范围**：用仓库自带的**过滤子集** `dataset/<domain>/data.json`（huggingface 3630 / multimedia 2981 / dailylife 2787，全部为 `chain` 类型），不重建图、不引入全量 TaskBench。
- **样本匹配**：新数据类 [src/dataset/zou_split.py](src/dataset/zou_split.py) 用 `id → data.json 行号` 找到 GTool 预处理好的图 `.pt`；gold 顺序直接取记录里的 `trajectory`。
- **其余全部不变**：建图、SBERT 编码、GNN、冻结 LLM、graph token、EARE/MDPL 损失、评测指标，都还是 GTool 的逻辑。

> 说明：子集**全是 chain**（没有单工具 `node` 样本），所以分层实际是 `chain_length_bucket`，且切出的 `test_node.jsonl` 为空、`test_chain == test_all`——这是预期行为，不是 bug。

### 8.2 准备数据与切分

```bash
# 1) GTool 原生预处理：为子集建图（SBERT 编码 + super-node），生成 graphs/、nodes.csv 等
python -m src.dataset.preprocess.huggingface
python -m src.dataset.preprocess.multimedia
python -m src.dataset.preprocess.dailylife

# 2) 按域分别跑分层切分（自包含，不依赖 taskbench_sft），每个域产出独立 JSONL 到
#    artifacts/splits_subset/<domain>/。用 --domains 指定单个域。
python -m src.dataset.preprocess_zou.split_subset \
    --raw_root dataset --out_dir artifacts/splits_subset/huggingface --domains huggingface
python -m src.dataset.preprocess_zou.split_subset \
    --raw_root dataset --out_dir artifacts/splits_subset/multimedia  --domains multimedia
python -m src.dataset.preprocess_zou.split_subset \
    --raw_root dataset --out_dir artifacts/splits_subset/dailylife   --domains dailylife
```

> 实际上不必手动跑这步：`run_experiment.sh` / `run_grid.sh` 会在每个域第一次用到时自动建好对应的 `artifacts/splits_subset/<domain>/`（幂等，已存在则跳过）。

[src/dataset/preprocess_zou/split_subset.py](src/dataset/preprocess_zou/split_subset.py) 忠实复刻了 `taskbench_sft` 的切分逻辑（同样的 80/10/10、per-stratum 确定性 shuffle `random.Random(f"{seed}|{key}")`、训练集工具覆盖 resample、chain 简单路径校验、`trajectory` = 拓扑序），但直接读 GTool 子集、零外部依赖，**且按域独立切分**（stratify key = `topology × chain_length_bucket`，域内常量的 `domain` 已移除）。

### 8.3 训练（按域）

**推荐：直接用编排脚本，它会对一个模型自动跑全部 3 个域**（建图 → 切分 → 训练 → 测试，每个域独立 TAG/输出/checkpoint）：

```bash
# 一个模型 × 3 个域（huggingface / multimedia / dailylife）
bash run_experiment.sh mistral            # TAG = zou_mistral_<domain>
bash run_experiment.sh qwen3-8b           # TAG = zou_qwen3_8b_<domain>

# 只跑某一个域
ONLY_DOMAIN=huggingface bash run_experiment.sh mistral

# 全网格：3 模型 × 3 域 = 9 次运行，每 GPU 一次、轮转分发（无 DDP）
bash run_grid.sh vicuna mistral qwen3-8b
GPUS="0 1 2 3" bash run_grid.sh vicuna mistral qwen3-8b
```

底层逐域命令（按域单独训练，对齐论文）：

```bash
# Mistral-7B-Instruct-v0.3，huggingface 域（其余域改 --dataset/--split_dir 即可）
python train_zou.py \
    --dataset zou_mistral_huggingface \
    --llm_model_name mistral \
    --split_dir artifacts/splits_subset/huggingface \
    --raw_root dataset
```

- `--split_dir` 必须指向**该域**的切分目录 `artifacts/splits_subset/<domain>/`。
- `--raw_root dataset` 指向 GTool 自带子集（已由 8.2 第 1 步建好图）。
- `--dataset` 仅作为输出命名空间（`output/<dataset>/`），**必须含域名**（如 `zou_mistral_huggingface`），不同 (模型,域) 用不同 tag，避免 checkpoint/结果互相覆盖。
- 训练逻辑与 `train.py` 完全一致：按验证集 loss 保存最优、early stop、训练后在 `--test_split`（默认 `test_all`）上自动评测。

### 8.4 测试（按域）

```bash
# 必须与训练用同样的 --dataset（含域名）和 --split_dir，才能定位到同一个 checkpoint
python inference_zou.py \
    --dataset zou_mistral_huggingface \
    --llm_model_name mistral \
    --split_dir artifacts/splits_subset/huggingface \
    --raw_root dataset
```

- 默认会在 `test_node` / `test_chain` / `test_all` 上分别评测（空的 `test_node` 会自动跳过）。
- **每个测试集都会按 gold answer 的工具数量分桶报告**：`tool=2 / tool=3 / tool>=4`，再加一行 `overall`，每行给 (count, node_f1, edge_f1, ned)。逻辑见 [src/utils/evaluate.py](src/utils/evaluate.py) 的 `eval_grouped`（工具数 = gold label 里 `Tool i:` 的行数）。`train_zou.py` 训练后的测试也用同样的分桶输出。
- 想只测某一个：加 `--test_split test_chain`。
- 测试时的超参（`llm_model_name / gnn_num_layers / mask_prob / LLMP_dim / alpha / patience / num_epochs / seed / dataset`）必须与训练一致，否则找不到 `..._checkpoint_best.pth`。

  输出示例：
  ```
  [test_all]
  group        count   node_f1   edge_f1       ned
  tool=2         312    0.xxxx    0.xxxx    0.xxxx
  tool=3         289    0.xxxx    0.xxxx    0.xxxx
  tool>=4        340    0.xxxx    0.xxxx    0.xxxx
  overall        941    0.xxxx    0.xxxx    0.xxxx
  ```

### 8.5 模型适配说明（已处理）

GTool 原代码把 Llama-2 的对话标记写死了。为让 Mistral / Qwen3 能跑，[src/model/GTool.py](src/model/GTool.py) 增加了**按模型族选择 prompt 格式**（`PROMPT_FORMATS` + `resolve_prompt_format`），**llama / vicuna 路径保持原样不变**：

| 模型族 | BOS / 用户结束 / EOS | tokenizer | pad |
|---|---|---|---|
| llama / vicuna | `<s>[INST]` / `[/INST]` / `</s>` | `use_fast=False` | `<pad>`(id=0)，与原版一致 |
| mistral | `<s>[INST]` / `[/INST]` / `</s>`（与 Llama-2 同款 INST 格式） | `use_fast=True` | 复用 eos |
| qwen | `<|im_start|>user\n` / `<|im_end|>\n<|im_start|>assistant\n` / `<|im_end|>` | `use_fast=True` | tokenizer 自带 |

> 这是能跑起来的 baseline 模板；Qwen3 的 chat template 较特殊，若要进一步提点可再细化。另外 `GTool.py` 里 `max_memory={0:'80GiB',1:'80GiB'}` 默认假设两张 80G 卡，请按你的硬件改（单卡/小显存）。

### 8.6 zou-split 相关新增文件一览

| 文件 | 作用 |
|---|---|
| [src/dataset/preprocess_zou/split_subset.py](src/dataset/preprocess_zou/split_subset.py) | 在 GTool 子集上跑分层切分（自包含复刻 taskbench_sft 逻辑），产出 JSONL |
| [src/dataset/zou_split.py](src/dataset/zou_split.py) | 读取分层切分 JSONL，按 id 匹配子集的图，输出 GTool 样本格式 |
| [train_zou.py](train_zou.py) | 训练（= `train.py` 逻辑 + zou 切分） |
| [inference_zou.py](inference_zou.py) | 测试（默认对 node/chain/all 分别评测，空集自动跳过） |
| `src/config.py` | 新增 `mistral` / `qwen3-8b` 模型映射，新增 `--split_dir/--raw_root/--test_split` 参数 |
| `src/model/GTool.py` | 新增按模型族的 prompt 格式（llama/vicuna 不变） |

