from pathlib import Path
import json, shutil

run = Path(__file__).resolve().parent
root = run.parents[1]
dest = root/'validation/readme_20261119'
dest.mkdir(parents=True, exist_ok=True)
for name in ['environment.json','dependency_checks.json','data_manifest.json','inference_checks.json',
             'inference_commands.json','training_fixed_commands.json','vllm_commands.json',
             'conda-libmamba-resolve.json','environment-resolve.log','environment-import.log',
             'regression.log']:
    if (run/name).exists(): shutil.copy2(run/name,dest/name)
for task in ['se','ncsp']:
    (dest/task).mkdir(exist_ok=True)
    for name in ['fixed_training_checks.json']:
        shutil.copy2(run/task/name,dest/task/name)
    for stage in ['inference','trained_inference_fixed','vllm']:
        src=run/task/stage
        if not src.exists():continue
        target=dest/task/stage;target.mkdir(exist_ok=True)
        for name in ['predictions.jsonl','metrics.json','validation.json']:
            if (src/name).exists():shutil.copy2(src/name,target/name)
for name in ['fixed_se_train.log','fixed_ncsp_train.log','se_vllm.log','ncsp_vllm.log','se_vote.log']:
    if (run/name).exists():shutil.copy2(run/name,dest/name)

checks=json.loads((run/'inference_checks.json').read_text())
training={task:json.loads((run/task/'fixed_training_checks.json').read_text()) for task in ['se','ncsp']}
for c in training.values():
    assert c['max_length']==5000 and c['dataset_num_proc']==1 and c['changed_elements']>0 and c['optimizer_steps']==2
for name in ['inference_commands.json','training_fixed_commands.json','vllm_commands.json']:
    commands=json.loads((run/name).read_text());assert commands and all(c['returncode']==0 for c in commands)
report=f'''# README 本地使用验证

日期：2026-11-19。使用本机下载的 SE / NCSP 数据和完整权重，将 README 中 `/path/to/...` 替换为实际路径执行。模型和数据原件未修改，所有实验输出位于 `runs/readme_check_20261119/`。

## 结论

数据准备、两种任务的 Transformers 推理、评估复算、SE 束搜索和候选投票、两种任务的 vLLM 推理，以及两种任务的小规模训练、完整模型保存和重载推理均已通过。验证过程中发现并修复了环境依赖冲突和训练参数未生效的问题。

这证明当前小规模流程可执行，不代表模型预测质量达标，也不代表完整训练集或测试集已跑完。

## 修复的问题

1. `wandb==0.17.9` 要求 protobuf < 6，与原 `protobuf==6.33.0` 冲突。改为 `protobuf==5.29.5`。
2. 未固定版本的 torchao 会安装到 0.18.0，在 PyTorch 2.8 下导入 Unsloth 时触发 `ScalingType` 导入错误。固定为 `torchao==0.13.0`。
3. TRL 0.23 忽略旧 SFTTrainer 接口中的序列长度、预处理进程参数，实测保存的是 `max_length=1024`、`dataset_num_proc=None`。改用 SFTConfig，并使用 processing_class 传入分词器；重新训练后保存值为 5000 和 1。兼容旧 TRL 的 max_seq_length 字段。
4. 原生成记录使用 `bool(extract_str_answer(...))`，缺少答案时返回的字符串 `F` 仍为真。现按答案标签记录 `has_answer` 和 `answer_complete`，新增相应回归检查。
5. 训练过程中抛出的异常继续向上传播，使 CLI 返回失败码，避免仅打印错误后仍报告成功。

提示词、训练文本模板、学习率及梯度累积等实验参数、化学评分和投票算法未调整。README 增加了全参数微调说明和“有效谱图样本数为 0 时不能把误差 0 解读为准确”的说明；Conda 示例显式选择 libmamba。

## 环境与安装检查

- Python 3.11；PyTorch 2.8.0+cu128；Transformers 4.56.2；TRL 0.23.0。
- Unsloth 2025.10.10；Unsloth Zoo 2025.10.12；vLLM 0.11.0。
- GPU：物理 GPU 3，RTX 4090（48 GiB）；外部设置 CUDA_VISIBLE_DEVICES=3，进程内使用 cuda:0。
- 修正后的 runtime.yml 全部 23 项 pip 依赖版本匹配，requirements.txt 的依赖检查通过。
- `conda env create --dry-run --solver libmamba -f runtime.yml` 解析通过。

**安装验证范围**：未从零下载并创建完整 Conda 环境。实际执行使用新建的隔离 venv，继承已有 PyTorch/CUDA 二进制，在其中安装 runtime.yml 所指定的 pip 依赖；没有升级或替换原 Conda 环境。Conda 层只做了解析验证。未重新下载数据与权重，也未验证外部下载链接。

版本清单：[environment.json](environment.json)；依赖核对：[dependency_checks.json](dependency_checks.json)；Conda 解析：[conda-libmamba-resolve.json](conda-libmamba-resolve.json)。

## 实际运行范围

| 流程 | 输入与检查 | 结果 |
| --- | --- | --- |
| JSONL → Arrow | 按 README 原代码，SE / NCSP 各 4 条训练样本、3 条测试样本 | 通过 |
| inspect | 两套真实测试集及完整权重，各 54,615 条测试数据 | 通过 |
| SE Transformers 推理 | 前 3 条，最多新增 5000 tokens | 通过，3 条均有完整答案标签 |
| NCSP Transformers 推理 | 前 3 条，最多新增 5000 tokens | 通过；1 条达到上限，只有 2 条有完整答案标签 |
| evaluate | 对两种任务分别复算 | 与各自推理指标完全一致 |
| SE / NCSP vLLM 推理 | 各取首条真实测试样本，默认 greedy、最多新增 5000 tokens | 均通过 |
| SE beam search | 3 条样本，beam width 3 | 每条 3 个候选 |
| SE vote | 聚合真实 greedy 与 beam 输出，Top-1 至 Top-10 | 通过 |
| SE 训练 | 前 8 条真实训练样本，1 epoch，batch 1，累积 4 | 2 个优化步骤，保存完整模型并重载推理通过 |
| NCSP 训练 | 前 8 条真实训练样本，1 epoch，batch 1，累积 4 | 2 个优化步骤，保存完整模型并重载推理通过 |
| 回归检查 | 原有检查 + 答案完成状态检查 | 17 / 17 通过 |

训练使用原预热参数 5。只跑 4 条样本时只有 1 个优化步骤，初始预热学习率为 0，因此正式训练验证改为 8 条、2 步，并检查权重实际发生变化。

两种任务保存后的 `max_length=5000`、`dataset_num_proc=1`、学习率 0.0002、梯度累积 4 均已核对。在第一层 q_proj 权重中，SE 有 {training['se']['changed_elements']} 个元素变化，NCSP 有 {training['ncsp']['changed_elements']} 个元素变化。详见 [SE 参数与权重检查](se/fixed_training_checks.json)、[NCSP 参数与权重检查](ncsp/fixed_training_checks.json)。

## 预测结果的边界

SE 的三条 greedy 预测中，1 条为有效 SMILES，准确率为 0。NCSP 在 Transformers 下的三条预测均没有可用于误差统计的有效谱图，`valid_samples_for_error_metrics=0`；其误差字段为 0 是空统计结果，不是零误差。这些输出原样保留，没有更换样本或调整评分规则来改善结果。

vLLM 对同一首条 NCSP 样本生成了有效谱图，有效误差统计样本数为 1，¹³C MAE 为 3.2929、¹H MAE 为 0.2314。同一权重在两个后端的输出存在明显差异，本次没有进一步定位数值或实现层面的原因，也不能据单条样本判断哪个后端整体更准确。请分别比较各后端结果。详见 [vLLM 命令记录](vllm_commands.json)、[NCSP vLLM 指标](ncsp/vllm/metrics.json)。

本次仅验证单卡、小规模样本；未覆盖完整数据集、多卡并行、vLLM 束搜索或所有参数组合。

[SE 原始预测](se/inference/predictions.jsonl) · [NCSP 原始预测](ncsp/inference/predictions.jsonl) · [推理与投票核对](inference_checks.json)。

## 复现记录

精确命令、模型路径、环境和退出码保存在：

- [数据来源和抽样范围](data_manifest.json)
- [推理、复算和投票命令](inference_commands.json)
- [修正后训练与重载命令](training_fixed_commands.json)
- [SE 训练日志](fixed_se_train.log)
- [NCSP 训练日志](fixed_ncsp_train.log)
- [回归日志](regression.log)

已验证环境可以在 NMRReasoner 目录下激活：

```bash
source runs/readme_check_20261119/environment/bin/activate
```

完整模型及优化器状态保存在 `runs/readme_check_20261119/{{se,ncsp}}/training_fixed/`，不随验证报告复制或纳入 Git。
'''
(dest/'REPORT.md').write_text(report)
print(dest/'REPORT.md')
