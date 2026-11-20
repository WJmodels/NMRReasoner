# NMRReasoner 实测记录

> 历史验证记录：以下结果对应当时的旧数据与权重路径。2026-11-19 起，当前默认示例改用项目内 `data/` 和 `weight/` 的 SE MultiNMR-200k 资源，新增 NCSP 配置。下列历史结果不代表新资源的推理验证；当前用法见项目 README。

验证日期：2026-10-20。使用真实 Arrow 数据和指定名称的本地权重，完成加载、生成、导出、评估与投票。

## 输入和环境

- 数据：`/home/wanglab/lln/project/CLReasoner/test/data/40w_solvent_test_22188`，共 22,188 条。
- 实际权重：`/home/wanglab/lln/project/CLReasoner/test/weight/depth1_real_393034/outputs/checkpoint-73698`。
- 本次选择：按 Arrow 顺序的前三条，未修改问题和答案。
- GPU：NVIDIA RTX 4090，`CUDA_VISIBLE_DEVICES=0`。
- PyTorch：2.10.0+cu128；Transformers：5.2.0；后端：transformers。
- 解码：greedy，最多新增 5,000 tokens，原始 Qwen2 提示词模板。
- 推理与评估耗时：60.60 秒（不含单独的投票与回归检查）。

用户给出的 `/home/sunhnayu/...` 路径不存在；本次使用工作区 `test/` 中同名资产，逆合成权重由已有 7z 包解压。无法对不存在的旧路径做逐字节比对。路径映射、数据 SHA256 与模型文件信息见 [assets.json](assets.json)。

## 结果

| 检查项 | 实测结果 |
| --- | --- |
| 完整答案 | 3 / 3 |
| 达到 token 上限 | 0 / 3 |
| 有效预测比例 | 33.33% |
| 原评估口径 accuracy | 0.00% |
| 平均 Tanimoto | 0.321429 |
| 回归检查 | 6 / 6 通过 |
| 原评分器复算 | 8 个指标全部一致 |
| 自一致性投票 | 实际执行，结果见 voting/consensus/summary_metrics.json |

本次 NMR 三条预测中两条包含无效 SMILES，三条都未命中标准答案；这些失败样本原样保留。代码链路可以运行，但当前小样本结果不能证明该权重具有足够预测精度。

核心迁移核对：原指标模块的 56 个函数、投票模块的 28 个函数在语法树层面完全一致；原推理与训练脚本的函数/类无遗漏。详见项目根目录 `migration.json`。没有执行重新训练，也没有验证 vLLM 后端或全部测试集。投票仅以这一次推理的 JSONL 检查流程，并非多模型提升效果的实验。

## 复现

从本项目根目录执行（输出到新的目录，避免覆盖本次记录）：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 /opt/anaconda3/envs/unsloth_py312_260316/bin/python run.py infer \
  --limit 3 --max-tokens 5000 --output runs/reproduce
```

需要全量评估时使用 `--limit 0`。训练和 vLLM 需按 README 安装相应依赖。

## 原始证据

- [预测 JSONL](real_checkpoint/predictions.jsonl)
- [选中样本与参数、耗时、生成长度](real_checkpoint/validation.json)
- [原评分器一致性检查](real_checkpoint/compatibility.json)
- [指标](real_checkpoint/metrics.json)
- [推理日志](inference.log)
- [回归检查日志](regression.log)
- [投票日志](voting.log)
