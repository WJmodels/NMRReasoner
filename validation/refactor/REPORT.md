# 重构与真实权重验证

> 历史验证记录：以下结果对应当时的旧数据与权重路径。2026-11-19 起，当前默认示例改用项目内 `data/` 和 `weight/` 的 SE MultiNMR-200k 资源，新增 NCSP 配置。下列历史结果不代表新资源的推理验证；当前用法见项目 README。

验证日期：2026-11-16。使用用户指定的真实数据和完整 checkpoint，在同一 Python 环境、同一张 GPU 上分别运行重构前与重构后的代码。

## 重构内容

- `nmrreasoner/cli.py`：中文命令帮助、常用解码参数、简洁错误与 `--debug`。
- `config.py`：配置优先级、参数校验、可迁移的相对路径和路径展开。
- `assets.py`：Arrow 列检查、任务判断、唯一完整 checkpoint 定位。
- `inference.py` / `backends.py`：实验流程与 Transformers/vLLM 生成后端分离。
- `evaluation.py` / `records.py`：评估、投票输入对齐检查和结果保存。
- `training.py`：训练调用与工作目录管理；`experiments/settings.py` 集中原训练参数。
- `test.py`、`train.py`、`run.py` 和 `workflows.py` 的既有入口保留；新增 `python -m nmrreasoner`。
- README 改为简洁中文指南。运行代码和默认配置已移除旧机器绝对路径；历史报告及日志保留原内容，以免改写历史证据。

路径规则有明确调整：JSON 中相对的 `data/model` 路径基于 JSON 所在目录，命令行路径与输出目录基于当前工作目录。旧的绝对路径配置仍支持；默认配置已改为相对路径。

有效输入的化学算法与生成参数保持不变。新增检查会提前拒绝非法参数、错拼配置项、投票行数不一致，以及不适用的正向位移投票。训练包装器会在结束或异常后恢复工作目录和它设置的四个环境变量。

## 真实输入与环境

- 数据：项目相邻 `test/data/40w_solvent_test_22188`，共 **22,188 条**，选取前 **3 条**做对照。
- 完整权重：`test/weight/depth1_real_393034/outputs/checkpoint-73698/model.safetensors`，约 3.4 GB；实际加载整个模型，未使用模拟模型或替代权重。
- GPU：物理 GPU 3，RTX 4090；`CUDA_VISIBLE_DEVICES=3`，进程内设备 `cuda:0`。
- Python 环境：`qwen3_unsloth`；PyTorch **2.7.0+cu126**、Transformers **4.51.3**、Datasets **3.5.1**、RDKit **2024.09.6**。
- 解码：Transformers、greedy、beam width 1，每条最多新增 5000 tokens，原 Qwen2 提示词。

实际解析路径和首条问题见 [inspect.json](inspect.json)；资产大小和 SHA256 见 [assets.json](assets.json)。

## 检查结果

| 检查 | 结果 |
| --- | --- |
| 原有 + 新增回归检查 | **16 / 16 通过** |
| 提示词/推理工具、评分、投票模块语法树 | 与重构前完全一致 |
| 两种推理后端的生成分支语法树 | 与重构前完全一致 |
| 训练实现 | 将外置参数还原后，语法树完全一致 |
| 真实推理 JSONL / TXT | 重构前后逐字节一致 |
| 生成记录 / 全部 8 个评分指标 | 完全一致 |
| `evaluate` 独立复算 | 与推理评分完全一致 |
| `vote` 候选与 Top-K 指标 | 重构前后完全一致 |
| 完整 answer 标签 / 达到 token 上限 | 3 / 3；0 / 3 |
| 重构前 / 后推理与评分耗时 | 110.13 / 107.92 秒（非性能基准） |

当前三条样本的有效 SMILES 比例为 **66.67%**，准确率为 **0%**，普通评分的平均 Tanimoto 为 **0.213346**。失败预测原样保留。此检查证明本次小样本上重构保持原输出，不代表全量准确率或模型质量。

未重新训练，未实际运行 vLLM，也未运行全部 22,188 条样本。训练和 vLLM 的保留性由代码语法树核对支持，不能替代各自环境中的实测。原 `has_answer` 字段语义保持不变；上表“完整 answer”另行检查了开始和结束标签。

## 复现与证据

激活包含依赖的环境后，在 NMRReasoner 目录执行：

```bash
python test.py inspect
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 python test.py --output runs/refactor_check
python test.py evaluate --data runs/refactor_check/ground_truth \
  --predictions runs/refactor_check/predictions.txt --output runs/refactor_check/rescore.json
python test.py vote --data runs/refactor_check/ground_truth \
  --predictions runs/refactor_check/predictions.jsonl --output runs/refactor_vote
python -m unittest discover -s checks -v
```

- [重构前代码备份](source_before.zip)：修改前源码、配置和说明，不包含权重与数据。
- [源码兼容性核对](source_compatibility.json)、[实测对照](comparison.json)、[回归日志](regression.log)。
- [重构前预测](before/predictions.jsonl)、[重构后预测](after/predictions.jsonl)、[运行记录](after/validation.json)。
- [推理日志](after.log)、[复算指标](rescore.json)、[投票指标](after_vote/consensus/summary_metrics.json)。
