# 部署：中 / 美节点跑 GTool 实验

跟 [RUNBOOK.md](RUNBOOK.md) 同一套思路：**PART1 联网准备**（下依赖+模型到盘）→ **PART2 离线运行**。
下面是 GTool 这个 repo 的具体脚本与命令。配置脚本都在 [scripts/](scripts/)，无任何密钥，可入库。

> 数据已在 repo 内（`dataset/<domain>/`），所以 PART1 只下依赖 + 模型，不下数据。

## 脚本一览

| 文件 | 阶段 | 作用 |
|---|---|---|
| [scripts/setup_china.sh](scripts/setup_china.sh) | PART1 | 中国镜像（hf-mirror / 清华 pip / SJTU cu121 torch）+ 并行下载。**source** |
| [scripts/setup_US.sh](scripts/setup_US.sh) | PART1 | 美国官方源 + Xet/hf_transfer，清掉镜像残留。**source** |
| [scripts/prep_env_china.sh](scripts/prep_env_china.sh) | PART1 | 中国联网环境（路径 + source setup_china）。**source** |
| [scripts/prep_env.sh](scripts/prep_env.sh) | PART1 | 美国联网环境（共享 NFS 路径 + source setup_US）。**source** |
| [scripts/prestage_all.sh](scripts/prestage_all.sh) | PART1 | venv（尽量复用 base torch，否则装 2.1.0）+ PyG 轮子 + `requirements-node.txt` + 下模型（gated 自动检测）。**bash** |
| [scripts/job_env.sh](scripts/job_env.sh) | PART2 | 离线环境（`HF_HUB_OFFLINE` 等）+ 激活 venv + `GPUS`。**source** |
| [scripts/job_unites.sbatch](scripts/job_unites.sbatch) | PART2 | 美国 Slurm 离线作业模板（分区/mem/共享 NFS 坑都注释了）。**sbatch** |
| [run_experiment.sh](run_experiment.sh) | PART2 | 建图→切分→训练→测试（幂等）。`--smoke` 走小模型小数据 |

模型：`Qwen/Qwen2.5-0.5B-Instruct`(烟测) · `sentence-transformers/all-roberta-large-v1`(建图 SBERT，必下) · `Qwen/Qwen3-8B` · `mistralai/Mistral-7B-Instruct-v0.3`(gated)。

---

## 中国节点（单机，镜像）

```bash
# PART 1（联网）
source scripts/prep_env_china.sh        # 路径在 setup_china.sh 顶部，按节点改
export HF_TOKEN=hf_xxx                   # 可选；下 Mistral 才需要（先在 HF 网页接受许可）
bash scripts/prestage_all.sh            # 装依赖 + 下模型

# PART 2（离线，同机新 shell）
source scripts/prep_env_china.sh        # 重设 WORK_DIR/HF_HOME（坑 2）
source scripts/job_env.sh               # 开离线开关 + 激活 venv
bash run_experiment.sh --smoke          # 先烟测；通了再 ↓
bash run_experiment.sh qwen3-8b
bash run_experiment.sh mistral          # 4090(24G) 若 OOM：加 --batch_size 1 --max_txt_len 1536
```

## 美国 Slurm 集群（登录节点联网 / 计算节点离线）

```bash
# ── 登录节点（有网）：PART 1 ──
source scripts/prep_env.sh              # 共享 NFS 路径；若共享目录名≠$USER 先改 WORK_DIR
export HF_TOKEN=hf_xxx                   # 下 Mistral 用
tmux new -s prep                         # 下载久，挂着
bash scripts/prestage_all.sh

# ── 提交离线作业到计算节点：PART 2 ──
# 先把 repo 放到共享 NFS（不是 $HOME），并核对 job_unites.sbatch 里的 REPO / --output 路径
MODEL=mistral  sbatch scripts/job_unites.sbatch
MODEL=qwen3-8b sbatch scripts/job_unites.sbatch
SMOKE=1 MODEL=qwen2.5-0.5b sbatch -p a100 --gres=gpu:1 scripts/job_unites.sbatch   # 烟测
squeue -u $USER
tail -f /playpen-shared/$USER/tb_work/logs/gtool-*.out
```

---

## 关键提醒（详见 RUNBOOK §3/§4/§5、§8）

- **torch 必须 `cuda.is_available()==True`**：`prestage_all.sh` 会校验，False 直接终止（多半是 cu13 装错）。GTool 的 PyG 轮子锁 `pt21cu121`，所以 torch 必须是 2.1.x。
- **GPU**：cu121 支持 A100/Ada(4090)，**避开 Blackwell**。0.5B 烟测单卡足够；7B/8B 在 24G 上需调小 `--batch_size/--max_txt_len`。
- **建图需 GPU**：`preprocess.*`（SBERT 编码）只能在有卡的节点跑，已包在 `run_experiment.sh`/作业里；首次建全部图（一次性），之后复用。
- **美国共享盘**：repo+venv+缓存放 `/playpen-shared`（**不是 $HOME**）；共享目录名可能 ≠ 登录名，注意改 `WORK_DIR`/`cd`/`--output`。
- **GTool 不用 wandb**：无需任何 WANDB 设置。
- 实验设置详情见 [EXPERIMENT_SETUP.md](EXPERIMENT_SETUP.md)。
