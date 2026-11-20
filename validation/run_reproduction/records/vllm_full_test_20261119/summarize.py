"""Summarize completed full-test predictions without rerunning generation."""
import json
from pathlib import Path
from nmrreasoner.records import answer_status
root=Path(__file__).resolve().parents[2]
run=Path(__file__).resolve().parent
summary={}
for task in ['ncsp','se']:
 folder=run/task
 report=json.loads((folder/'validation.json').read_text())
 metrics=json.loads((folder/'metrics.json').read_text())
 rescored=json.loads((folder/'rescored_metrics.json').read_text())
 assert report['backend']=='vllm' and report['config']['limit']==0
 assert report['dataset_rows']==report['evaluated_rows']==54615
 assert metrics==rescored
 counts={'prediction_rows':0,'has_answer':0,'answer_complete':0}
 with (folder/'predictions.jsonl').open() as f:
  for line in f:
   candidates=json.loads(line)['text']
   assert len(candidates)==1
   counts['prediction_rows']+=1
   for key,value in answer_status(candidates[0]).items(): counts[key]+=int(value)
 assert counts['prediction_rows']==report['evaluated_rows']
 summary[task]={'model':report['model'],'data':report['data'],'elapsed_seconds':report['elapsed_seconds'],**counts,'metrics':metrics,'independent_rescore_matches':True}
(run/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
n=summary['ncsp']['metrics']; s=summary['se']['metrics']
lines=['# vLLM 完整测试集推理结果','', '正向 NCSP 和逆向 SE 各运行全部 54,615 条测试样本。使用各自原始完整权重，默认 vLLM 后端，temperature=0，beam_width=1，max_tokens=5000，物理 GPU 3。未筛选或替换样本。','', '| 任务 | 指标 | 结果 |','| --- | --- | --- |',f"| 正向 NCSP | 有效误差统计样本数 | {n['valid_samples_for_error_metrics']} / 54615 |"]
for iso in ['13C','1H']:
 for metric,unit in [('mae','ppm'),('mse','ppm²'),('rmsd','ppm')]:
  lines.append(f"| 正向 NCSP | {iso} {metric.upper()} | {n[f'average_{metric}_zero_padding_{iso}_nmr_shift']:.6f} {unit} |")
for key,title in [('prediction_validity_rate','有效结构比例'),('accuracy','结构准确率'),('average_tanimoto','平均 Tanimoto')]:
 lines.append(f'| 逆向 SE | {title} | {s[key]:.6f} |')
lines+=['','NCSP 误差沿用项目原始评估规则：对位移排序，并在碳、氢峰数均满足筛选条件的样本上汇总；有效样本数为分母，不能视为全部样本均预测有效。SE 有效率与准确率以全部测试样本为分母；平均 Tanimoto 仅在预测和参考结构均可解析的样本上计算，沿用原始 Morgan 指纹（半径 2、1024 bits）和去立体化学设置。','', '两项均已核对预测行数与测试集行数一致，自动指标与独立 evaluate 复算完全一致。完整指标和答案完成数量见 [summary.json](summary.json)。','', '- [正向完整指标](ncsp/metrics.json)','- [逆向完整指标](se/metrics.json)','- [正向预测](ncsp/predictions.jsonl)','- [逆向预测](se/predictions.jsonl)','- [执行命令及退出状态](commands.json)','']
(run/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(summary,ensure_ascii=False,indent=2))
