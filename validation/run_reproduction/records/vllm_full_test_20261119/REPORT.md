# vLLM 完整测试集推理结果

正向 NCSP 和逆向 SE 各运行全部 54,615 条测试样本。使用各自原始完整权重，默认 vLLM 后端，temperature=0，beam_width=1，max_tokens=5000，物理 GPU 3。未筛选或替换样本。

| 任务 | 指标 | 结果 |
| --- | --- | --- |
| 正向 NCSP | 有效误差统计样本数 | 16015 / 54615 |
| 正向 NCSP | 13C MAE | 3.472674 ppm |
| 正向 NCSP | 13C MSE | 39.337505 ppm² |
| 正向 NCSP | 13C RMSD | 4.961663 ppm |
| 正向 NCSP | 1H MAE | 0.229092 ppm |
| 正向 NCSP | 1H MSE | 0.124158 ppm² |
| 正向 NCSP | 1H RMSD | 0.293016 ppm |
| 逆向 SE | 有效结构比例 | 0.878824 |
| 逆向 SE | 结构准确率 | 0.075986 |
| 逆向 SE | 平均 Tanimoto | 0.344503 |

NCSP 误差沿用项目原始评估规则：对位移排序，并在碳、氢峰数均满足筛选条件的样本上汇总；有效样本数为分母，不能视为全部样本均预测有效。SE 有效率与准确率以全部测试样本为分母；平均 Tanimoto 仅在预测和参考结构均可解析的样本上计算，沿用原始 Morgan 指纹（半径 2、1024 bits）和去立体化学设置。

两项均已核对预测行数与测试集行数一致，自动指标与独立 evaluate 复算完全一致。完整指标和答案完成数量见 [summary.json](summary.json)。

- [正向完整指标](ncsp/metrics.json)
- [逆向完整指标](se/metrics.json)
- [正向预测](ncsp/predictions.jsonl)
- [逆向预测](se/predictions.jsonl)
- [执行命令及退出状态](commands.json)
