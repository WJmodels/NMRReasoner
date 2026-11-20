"""Execute the README inspect/infer/evaluate flow against the requested assets."""
import hashlib,json,os,shlex,subprocess,sys,time,traceback
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from nmrreasoner.records import answer_status
model=root/'weight/Structure elucidation/SE-CoT-1_MultiNMR_200k_weight'
data=root/'data/Structure elucidation/test/MultiNMR-2.2M/SE_MultiNMR_2.2M_test_arrow'
dest=out/'inference'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
records=[]
def write(path,value):
 tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n'); tmp.replace(path)
def status(state,stage,**extra):
 write(out/'status.json',dict(state=state,stage=stage,pid=os.getpid(),updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),**extra))
def run(stage,args):
 status('running',stage)
 cmd=[sys.executable,*map(str,args)]; start=time.time()
 with (out/f'{stage}.log').open('w') as log:
  result=subprocess.run(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 records.append(dict(stage=stage,command=cmd,shell_command=shlex.join(cmd),returncode=result.returncode,seconds=time.time()-start))
 write(out/'commands.json',records)
 if result.returncode: raise RuntimeError(f'{stage} failed: exit {result.returncode}; see {stage}.log')
 print(stage,'completed',flush=True)
try:
 for name in ['README.md','README.zh-CN.md']:
  (out/(name+'.snapshot')).write_bytes((root/name).read_bytes())
 write(out/'manifest.json',dict(model=str(model),data=str(data),gpu=3,backend='vllm',limit=0,max_tokens=5000,readme_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['README.md','README.zh-CN.md']}))
 run('inspect',['test.py','inspect','--data',data,'--model',model])
 run('infer',['test.py','--data',data,'--model',model,'--limit','0','--max-tokens','5000','--output',dest])
 run('evaluate',['test.py','evaluate','--data',dest/'ground_truth','--predictions',dest/'predictions.txt','--output',dest/'rescore.json'])
 run('regression',['-m','unittest','discover','-s','checks','-v'])
 status('running','verify_and_report')
 report=json.loads((dest/'validation.json').read_text())
 metrics=json.loads((dest/'metrics.json').read_text())
 assert metrics==json.loads((dest/'rescore.json').read_text()),'Automatic metrics and independent evaluation differ'
 assert report['backend']=='vllm' and report['config']['limit']==0
 assert Path(report['model']).resolve()==model.resolve()
 assert Path(report['data']).resolve()==data.resolve()
 counts=dict(prediction_rows=0,has_answer=0,answer_complete=0)
 with (dest/'predictions.jsonl').open() as f:
  for line in f:
   texts=json.loads(line)['text']; assert len(texts)==1
   counts['prediction_rows']+=1
   for k,v in answer_status(texts[0]).items(): counts[k]+=int(v)
 with (dest/'predictions.txt').open() as f: txt_rows=sum(1 for _ in f)
 assert txt_rows==counts['prediction_rows']==report['evaluated_rows']==report['dataset_rows']
 summary=dict(model=str(model),data=str(data),backend=report['backend'],total_samples=report['dataset_rows'],**counts,metrics=metrics,independent_rescore_matches=True,readme_inspect_infer_evaluate_passed=True)
 write(out/'summary.json',summary)
 lines=['# SE-200k 权重在 MultiNMR-2.2M 测试集上的推理结果','',f"测试集共 {report['dataset_rows']:,} 条，全部完成。使用指定的 SE-CoT-1_MultiNMR_200k_weight 完整权重，默认 vLLM，temperature=0、beam_width=1、max_tokens=5000，物理 GPU 3。",'', '| 指标 | 结果 |','| --- | --- |']
 for key,title in [('prediction_validity_rate','有效结构比例'),('accuracy','结构准确率'),('average_tanimoto','平均 Tanimoto'),('accuracy_040','Tanimoto ≥ 0.4 比例'),('accuracy_060','Tanimoto ≥ 0.6 比例'),('accuracy_080','Tanimoto ≥ 0.8 比例'),('accuracy_090','Tanimoto ≥ 0.9 比例')]:
  lines.append(f'| {title} | {metrics[key]:.6f} |')
 lines+=['',f"完整答案标签：{counts['answer_complete']:,} / {counts['prediction_rows']:,}。",'', '有效率和准确率以全部测试样本为分母；平均 Tanimoto 仅统计预测和参考均可解析的结构。沿用原项目评分逻辑，默认忽略手性。','', '## README 流程核对','', '已按中英文 README 的 inspect → infer → evaluate 顺序执行，只替换数据、权重和输出路径，将 --limit 设为文档规定的 0，并选择空闲 GPU 3。未显式指定 --backend，运行记录确认默认使用 vLLM。JSONL/TXT 预测行数均与完整测试集一致，自动指标和独立复算完全一致。回归检查通过。此次检查覆盖该结构推断流程，不代表重新验证训练、数据转换或其他后端。','', '- [运行命令及退出码](commands.json)','- [完整统计](summary.json)','- [预测结果](inference/predictions.jsonl)','- [原始指标](inference/metrics.json)','- [独立复算](inference/rescore.json)','- [回归日志](regression.log)','', '运行时 README 快照及哈希保存在本目录，便于核对对应文档版本。']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
 status('completed','done',report=str(out/'REPORT.md'))
except BaseException as exc:
 status('failed','pipeline',error=str(exc))
 traceback.print_exc(); sys.exit(1)
