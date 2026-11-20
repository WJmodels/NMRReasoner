# SE-200k 权重在 MultiNMR-2.2M 测试集上的推理结果

测试集共 54,615 条，全部完成。使用指定的 SE-CoT-1_MultiNMR_200k_weight 完整权重，默认 vLLM，temperature=0、beam_width=1、max_tokens=5000，物理 GPU 3。

| 指标 | 结果 |
| --- | --- |
| 有效结构比例 | 0.901913 |
| 结构准确率 | 0.405347 |
| 平均 Tanimoto | 0.737557 |
| Tanimoto ≥ 0.4 比例 | 0.745839 |
| Tanimoto ≥ 0.6 比例 | 0.580152 |
| Tanimoto ≥ 0.8 比例 | 0.461137 |
| Tanimoto ≥ 0.9 比例 | 0.427428 |

完整答案标签：54,447 / 54,615。

有效率和准确率以全部测试样本为分母；平均 Tanimoto 仅统计预测和参考均可解析的结构。沿用原项目评分逻辑，默认忽略手性。

## README 流程核对

已按中英文 README 的 inspect → infer → evaluate 顺序执行，只替换数据、权重和输出路径，将 --limit 设为文档规定的 0，并选择空闲 GPU 3。未显式指定 --backend，运行记录确认默认使用 vLLM。JSONL/TXT 预测行数均与完整测试集一致，自动指标和独立复算完全一致。回归检查通过。此次检查覆盖该结构推断流程，不代表重新验证训练、数据转换或其他后端。

- [运行命令及退出码](commands.json)
- [完整统计](summary.json)
- [预测结果](inference/predictions.jsonl)
- [原始指标](inference/metrics.json)
- [独立复算](inference/rescore.json)
- [回归日志](regression.log)

运行时 README 快照及哈希保存在本目录，便于核对对应文档版本。
