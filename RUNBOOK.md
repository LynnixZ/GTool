# 跨中美节点跑实验 · Runbook（可移植）

一份通用操作手册:在**中国节点**(单机,可联网,走镜像)和**美国 Slurm 集群**
(登录节点联网 / 计算节点离线)跑训练实验。换一个 repo 也能照搬,只改路径和模型。

核心思想一句话:**把"联网准备"和"离线运行"彻底分开** —— 在能上网的地方把
依赖 + 数据 + 模型全下到共享盘,在跑实验的地方只读缓存、不碰网络。

---

## 0. 为什么要分两段

代码运行时**不知道自己离线**:执行到"下载模型"会真的去连网,离线就**卡住或报错**,
不是"自动跳过"。而且 HuggingFace **默认连用缓存前都要先联网检查更新**(HEAD/ETag),
所以哪怕模型已经在硬盘上,离线节点不设 `HF_HUB_OFFLINE=1` 也会失败。

因此固定为两段:

1. **PART 1 — 联网准备**:在有网的节点下 env + 数据 + 模型到**共享/数据盘**。
2. **PART 2 — 离线运行**:在计算节点设 `HF_HUB_OFFLINE=1`,只读缓存跑。

---

## 1. 环境变量速查表

### 联网准备（PART 1）

| 变量 | 作用 |
|---|---|
| `WORK_DIR` / `HF_HOME` | 数据盘路径 + HF 缓存位置（务必指向共享/大盘） |
| `HF_TOKEN` | 下 gated 模型（Llama/Mistral 等）用 |
| `HF_XET_HIGH_PERFORMANCE=1` | Xet 模型并行下载（新模型，如 Qwen3） |
| `HF_HUB_ENABLE_HF_TRANSFER=1` | 经典 LFS 并行下载（需装 `hf_transfer`） |

### 离线运行（PART 2）

| 变量 | 作用 |
|---|---|
| `HF_HUB_OFFLINE=1` | HF 只读本地缓存，**绝不联网** |
| `TRANSFORMERS_OFFLINE=1` | transformers 同上 |
| `WANDB_MODE=offline` | W&B 写本地文件，事后 `wandb sync` |
| `WORK_DIR` / `HF_HOME` | **必须重新设**（见 §4 坑 2） |

### 中国镜像（PART 1 加在最前）

| 变量 | 值 |
|---|---|
| `HF_ENDPOINT` | `https://hf-mirror.com` |
| `PIP_INDEX_URL` | `https://pypi.tuna.tsinghua.edu.cn/simple` |
| `TORCH_INDEX_URL` | `https://mirror.sjtu.edu.cn/pytorch-wheels/cu121`（见 §3） |

---

## 2. 美国 Slurm 集群（登录节点联网 / 计算节点离线）

要点:

- **登录节点有网、可 SSH**;**计算节点不能 SSH、按节点本地盘对 Slurm 不可见**。
- 一切(**代码 repo、venv、数据、HF 缓存**)放**共享 NFS**(如 `/playpen-shared/<你的目录>/...`),
  **绝不能放 `$HOME`** —— `/home` 是登录节点本地盘,计算节点看不到,`cd $HOME/...` 会直接失败。
- ⚠️ **共享目录名可能 ≠ 登录名**(实测:目录是 `xinyu` 但 `$USER=xinyuzh`)→ 别盲目用
  `$USER` / `%u`,核对真实目录名,改 `WORK_DIR`、sbatch 的 `cd` 与 `--output/--error`。
- 整节点跑(`gpu:8`)内存用 **`--mem=0`**(整节点全部);**别写 `--mem=480G`** —— 超 a100/ada
  物理上限(~472GiB)会被拒。部分卡用具体值(如 `--mem=150G`)。
- 每人有 **GPU 上限**(如 8 卡);选**和你 torch 匹配的 GPU 架构分区**
  (cu121 torch → A100/Ampere ✅;**别用最新 Blackwell**,会 "no kernel image")。

```bash
# ── 登录节点（有网）：一次性准备 ──
export WORK_DIR=/playpen-shared/$USER/tb_work HF_HOME=$WORK_DIR/hf_home
export HF_TOKEN=hf_xxx                       # gated 模型；先在 HF 网页接受许可
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1
tmux new -s prep                             # 下载久，挂着
#  ... 下载 deps + data + 全部模型到 $WORK_DIR ...

# ── 提交离线 job 到计算节点 ──
sbatch job.sbatch                            # #SBATCH 里声明分区/卡数/时长
squeue -u $USER
tail -f $WORK_DIR/logs/*.out
```

`job.sbatch` 里(计算节点离线执行):

```bash
#SBATCH --partition=a100        # Ampere；避开 Blackwell
#SBATCH --gres=gpu:8            # ≤ 个人上限
#SBATCH --output=/playpen-shared/%u/.../logs/%j.out
cd $REPO
source $WORK_DIR/GTool_venv/bin/activate        # GTool 实际由 scripts/job_env.sh 处理
export WORK_DIR=... HF_HOME=...                 # 重新设（计算节点是干净环境）
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
# 用 Slurm 实际分配的卡：
NGPU=$(nvidia-smi -L | wc -l); export GPUS=$(seq -s' ' 0 $((NGPU-1)))
bash run_experiment.sh
```

- **提交用 `sbatch`,不是 `bash`**(`bash` 会在登录节点本地跑;`#SBATCH` 行只有
  `sbatch` 才读)。
- `sbatch` 默认 `--export=ALL` 会带上提交时的环境,但**别依赖它**,在 job 里重设一遍。
- 改卡数:`sbatch --gres=gpu:4 --cpus-per-task=48 --mem=240G job.sbatch`(命令行覆盖
  `#SBATCH`)。
- 离线 job **不需要** `HF_TOKEN` / `WANDB_API_KEY`(离线不认证)。

---

## 3. 中国节点（单机,可联网,走镜像）

中国没有登录/计算之分,但**仍按 PART1 联网 → PART2 离线**跑,既能验证离线管路,
也和美国逻辑一致。

```bash
# PART 1：镜像 + 路径 + 下载
export WORK_DIR=/root/autodl-tmp/tb_work HF_HOME=$WORK_DIR/hf_home
export HF_ENDPOINT=https://hf-mirror.com
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export TORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu121
export HF_XET_HIGH_PERFORMANCE=1 HF_HUB_ENABLE_HF_TRANSFER=1
#  ... 下 deps + data + 模型 ...

# PART 2：离线（同一台机，新 shell 要重设 WORK_DIR/HF_HOME）
export WORK_DIR=/root/autodl-tmp/tb_work HF_HOME=$WORK_DIR/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
bash run_experiment.sh
```

### 烟测：先用 Qwen 0.5B 跑通整条管路（~10 分钟）

正式跑前，先用**最小模型 + 限速开关**把"联网下载 → 离线训练 → 推理 → 评估"整条链
走一遍。目的是**验证管路通不通**，不是真训练 —— 数字没有意义。

```bash
SMOKE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"     # 0.5B，非 gated，秒下秒训

# PART 1（联网，镜像）：只下这一个小模型 + 数据 + 依赖
MODELS="$SMOKE_MODEL" bash scripts/prestage_all.sh

# PART 2（离线）：1 模型 × 1 域 × 4 设置 = 4 个 unit
export WORK_DIR=/root/autodl-tmp/tb_work HF_HOME=$WORK_DIR/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
MODELS="$SMOKE_MODEL" DOMAINS="data_huggingface" \
  MAX_STEPS=10 INFER_LIMIT=8 DELETE_MODELS=0 \
  bash scripts/run_grid.sh
```

限速开关（**只在烟测用，正式跑不要设**）：

- `MAX_STEPS=10` —— 每个 SFT 只训 **10 步**（正式是 5 epoch + early stopping）。
- `INFER_LIMIT=8` —— 每次推理只跑 **8 条**测试样本（正式是整个 test 集）。
- `DOMAINS=data_huggingface` —— 只一个域；`DELETE_MODELS=0` —— 留着模型方便重跑。

通过标准：

- 整条跑完**不报错**（下载 → split → 训练 → 推理 → 出指标）；
- 指标**不全是 0**（说明格式 / 解析 / 评估链路通）；
- SFT 的 `node_f1` / `trajectory_exact_match` **略高于 Base**（10 步也能看出苗头）。

烟测过了，美国全量大概率也能跑通。**注意**：新开 shell 跑 PART 2 时务必重设
`WORK_DIR`/`HF_HOME`（否则退回默认路径，离线预检会找不到缓存 → `model not accessible`）。

---

## 4. 中国镜像经验（重点）

### HF 模型 —— `hf-mirror.com`

- `export HF_ENDPOINT=https://hf-mirror.com`,几乎所有模型都能下。
- **并行加速**两条路,都设上(覆盖不同后端):
  - `HF_XET_HIGH_PERFORMANCE=1` —— 新模型走 Xet 存储（Qwen3 等）。
  - `HF_HUB_ENABLE_HF_TRANSFER=1` —— 经典 LFS 走 `hf_transfer` 并行（需装它）。
- 这俩**默认是关的**,不设就走慢的单流。看到 `hf_transfer is deprecated` 警告无害
  （对 Xet 模型它本就不生效，对经典 LFS 仍有效）。

### pip —— 清华源

- `export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`,小包秒装。

### torch —— 最大的坑

- **别**用官方 `download.pytorch.org`：中国 ~3 MB/s，torch 一坨 2.5GB 能拖十几分钟。
- **别**用清华默认 PyPI 的 `torch`：它解析成**最新版**，捆 **cu13**，在 **CUDA 12.x
  驱动**上 `torch.cuda.is_available()` 是 **False**（"no kernel" / 起不来）。
- **要么** 用 **SJTU 的 cu121 wheel 镜像**（= 官方 cu121 轮子，只是从中国下、快）：
  `export TORCH_INDEX_URL=https://mirror.sjtu.edu.cn/pytorch-wheels/cu121`
- **要么**（更省事）**复用镜像自带的 torch**：很多云镜像（AutoDL 等）base 环境
  已带能用的 CUDA torch。venv 用 `--system-site-packages` 创建即可复用，**完全不下
  torch**。先验证：`python -c "import torch;print(torch.__version__, torch.cuda.is_available())"`
  为 `True` 就直接用。
- 装完务必确认 `torch.cuda.is_available()` 为 **True**（版本无所谓，匹配驱动才重要）。

### GitHub 拉代码 / 数据

- HTTP/2 framing 报错：`git config --global http.version HTTP/1.1`。
- clone/pull 慢或失败：用 ghproxy 之类反代前缀。
- 下 `raw.githubusercontent.com` 上的数据：换 **jsDelivr**
  (`https://cdn.jsdelivr.net/gh/<user>/<repo>@<ref>/<path>`) 通常最稳。

### W&B

- 中国连 wandb.ai 不稳：**离线跑**（`WANDB_MODE=offline`），日志写共享盘，
  事后在有网的机器 `wandb sync <dir>`。

---

## 5. 常见坑 & 排查

| 现象 | 原因 | 解决 |
|---|---|---|
| 离线节点 `model not accessible` / FATAL（但模型已下） | 没设 `HF_HUB_OFFLINE`，或 `HF_HOME` 指错 | 设 `HF_HUB_OFFLINE=1`；`echo $HF_HOME` 核对路径 |
| 计算节点找不到 venv / 路径不存在 | 新环境丢了 `WORK_DIR`，退回了默认值 | job 里**重新 export** `WORK_DIR`/`HF_HOME` |
| 又在重下 2.5GB torch | venv 没复用 base torch | 删 venv，用 `--system-site-packages` 重建 |
| `torch.cuda.is_available()` False | 装了 cu13，但驱动是 12.x | 用 SJTU cu121 镜像，或复用 base torch |
| Slurm job 看不到文件 | 数据在本地盘 | 全部放共享 FS（`/playpen-shared` 等） |
| `cd $HOME/...` 失败 / job 找不到 repo | `$HOME`(/home) 是登录节点本地盘，计算节点不可见 | repo 放共享 NFS，`cd /playpen-shared/.../repo` |
| sbatch 路径找不到（你明明建了目录） | 共享目录名 ≠ `$USER` / `%u`（如 xinyu vs xinyuzh） | 用真实目录名改 `WORK_DIR` / `cd` / `--output` |
| job 被拒 / 内存请求失败 | `--mem=480G` 超节点物理上限(~472GiB) | 用 `--mem=0`(整节点) 或具体值如 `--mem=150G` |
| `OMP_NUM_THREADS Invalid value` | 容器把它设成非法值 | 无害；`export OMP_NUM_THREADS=1` 消除 |
| `torch_dtype deprecated` / torchvision image.so | 第三方库的提示 | 无害，忽略 |

---

## 6. 密钥规范

- `HF_TOKEN` / `WANDB_API_KEY` 只放**仓库外**或 **gitignored** 的文件（`~/env.sh`、
  `run.sh`、`*.secret.sh`），**绝不写进会提交的脚本**。
- 可提交的脚本通过 `source ~/env.sh` 或继承上层 `export` 拿到密钥。
- 离线 job 根本不需要密钥（不认证）；只有 PART1 下 gated 模型、事后 `wandb sync`
  才用到，那都在联网节点做。

---

## 7. 检查清单

### PART 1（联网）

- [ ] `WORK_DIR`/`HF_HOME` 指向共享/大盘
- [ ] 中国：镜像三件套（HF_ENDPOINT / PIP_INDEX_URL / TORCH_INDEX_URL）
- [ ] gated 模型：`HF_TOKEN` + 已接受许可
- [ ] tmux 挂着下；下完核对模型/数据齐全
- [ ] `torch.cuda.is_available()` 为 True

### PART 2（离线）

- [ ] `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline`
- [ ] 重新 export `WORK_DIR`/`HF_HOME`
- [ ] 激活 venv；`GPUS` 按实际卡数
- [ ] 美国：`sbatch`（非 bash）、分区匹配 GPU 架构、共享 FS
