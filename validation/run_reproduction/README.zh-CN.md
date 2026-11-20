# runs 复现记录与清理

记录日期：2026-11-20。数据源和原始权重仍在项目 `data/`、`weight/`；本次仅清理 `runs/`，不修改这些输入。

## 保存了什么

- `records/`：各次运行的命令、源数据及抽样记录、环境版本、预测指标、有效训练参数、训练状态、模型变化检查、脚本和报告。
- `source_snapshot.tar.gz`：本次清理前的代码、配置、中英文 README 和回归检查快照。
- `pip-list.json` / `pip-freeze.txt`：实际验证环境，包括继承的依赖。
- `original-pyvenv.cfg`：原隔离环境的基础环境信息。
- `cleanup_manifest.json`：每个删除或压缩文件的路径、大小和处理方式。
- `runs/` 中保留预测、指标、日志、训练参数和运行脚本；大于 5 MiB 的普通文件以 `.gz` 压缩保存，原路径需要解压后才能用于旧脚本或链接。

删除训练产生的 `.safetensors` 和 `optimizer.pt`，删除 `runs/readme_check_20261119/environment/`。这意味着无法直接加载这些训练副本或从旧优化器状态续训；可按下述命令从原始权重重新训练。旧报告中的环境激活路径不再有效。

## 环境

在项目根目录执行 README 中的安装流程：

```bash
conda env create --solver libmamba -f runtime.yml
conda activate nmrreasoner
```

实际历史验证使用 Python 3.11.14 / PyTorch 2.8.0+cu128、vLLM 0.11.0、Transformers 4.56.2、TRL 0.23.0、Unsloth 2025.10.10；GPU 为物理 GPU 3（48 GiB）。原 venv 继承已有 `llama_factory_py311_cuda124` 环境的二进制包；完整新建 Conda 环境此前仅做过依赖解析，不能称为已经实测的等价安装。安装后应与 `pip-list.json` 对照关键版本，尤其是 CUDA/PyTorch、vLLM、RDKit。历史完整版本已保存，包源是否继续提供这些版本不在此次验证范围内。

若本机基础环境仍存在，可按 `original-pyvenv.cfg` 的路径用 `python -m venv --system-site-packages /path/to/new_environment` 重建隔离环境，再在新环境中安装 `runtime.yml` 的 pip 固定依赖及记录中的 RDKit 版本。不要直接修改基础环境。

## 重跑实验

`replay.py` 默认只打印命令；加 `--execute` 才执行。`--output-root` 指定新的输出位置，避免覆盖历史结果；`--gpu` 指定空闲物理 GPU。数据和权重按项目相对位置查找，所有训练和推理命令显式带模型路径。历史 Transformers 小样本检查会显式保留 Transformers 后端，不受当前默认 vLLM 的影响。

```bash
# 先检查待执行命令
python validation/run_reproduction/replay.py full-2.2m

# 最近一轮：SE-200k 权重 → MultiNMR-2.2M 的完整 54,615 条测试集
python validation/run_reproduction/replay.py full-2.2m --gpu 3 --execute

# 正向与逆向 real200k 权重 → 各自 MultiNMR-200k 完整测试集
python validation/run_reproduction/replay.py full-200k --gpu 3 --execute

# 原 README 的小样本推理、复算、束搜索、投票
python validation/run_reproduction/replay.py smoke --gpu 3 --execute

# 修复后的训练、保存和重载推理：每任务前 8 条，1 epoch，batch 1，累积 4，2 个优化步骤
python validation/run_reproduction/replay.py training --gpu 3 --execute

# vLLM 单样本验证
python validation/run_reproduction/replay.py vllm-smoke --gpu 3 --execute
python validation/run_reproduction/replay.py bidirectional-single --gpu 3 --execute
```

小样本准备由脚本自动完成：训练取前 4 条 / 前 8 条，测试取前 3 条，保持原始顺序。历史 JSONL→Arrow 转换样例和文件仍在 `runs/readme_check_20261119/`；源数据位置及行数见 `records/readme_check_20261119/data_manifest.json`。

每次推理自动生成 `metrics.json`，原命令中的 evaluate 会保存独立复算结果。应核对两者一致、预测行数等于实际选择的测试行数，训练应再次核对 max_length=5000、dataset_num_proc=1、两步更新及权重变化。相关原始验证脚本和已完成检查记录见 records 下的 `check_saved_training.py` 与 `fixed_training_checks.json`；检查脚本的路径以历史目录为准，重跑后应改为新的输出目录。重跑命令不保证跨硬件、驱动和软件版本得到逐字节一致的预测或权重。

## 历史阶段的区别

`train_existing_env` 是最初在旧环境中的一步训练；初始 warmup 学习率为 0，不作为权重更新验收。`training` 是发现 TRL 参数未生效前的中间验证，有部分中断；不要把它当作最终结果。`training_fixed` 才是修复后通过验收的两步训练。旧命令、日志、fit.before.py 和原 README 快照保留用于解释差异，重跑入口使用最终修复代码，不重新制造旧版本故障。

旧 `finalize.py`、`monitor.py` 和后台管理 PID 仅供审计，不应再次启动；它们含历史进程编号。需要后台运行时，对新的 replay 命令使用 `nohup` 或独立会话。

## 解压保留的文件

```bash
# 例如：恢复预测文件以便独立复算；-k 保留压缩副本
 gzip -dk runs/se_200k_on_2_2m_20261120/inference/predictions.txt.gz
```

若旧报告链接指向已压缩文件，先按清单解压。大文件解压会重新占用磁盘；已有指标和小文件报告无需解压即可读取。

按运行目录批量恢复预测及 ground_truth（默认只列出文件，加 --execute 才解压）：

```bash
python validation/run_reproduction/restore.py se_200k_on_2_2m_20261120/inference --execute
```

恢复脚本会校验解压内容的 SHA-256，保留 `.gz` 副本并跳过已经存在的目标文件。
