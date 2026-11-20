import json, os, subprocess, sys, time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=Path(__file__).resolve().parent
records=[]
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
for task,folder,prefix in [('ncsp','NMR chemical shift prediction','NCSP'),('se','Structure elucidation','SE')]:
 data=root/'data'/folder/'test/MultiNMR-200k'/f'{prefix}_MultiNMR_200k_test_arrow'
 model=root/'weight'/folder/f'{prefix}-CoT-1_MultiNMR_real200k_weight'
 dest=out/task
 for phase,args in [('infer',['test.py','--data',str(data),'--model',str(model),'--limit','1','--max-tokens','5000','--output',str(dest)]),('evaluate',['test.py','evaluate','--data',str(dest/'ground_truth'),'--predictions',str(dest/'predictions.txt'),'--output',str(dest/'rescored_metrics.json')])]:
  cmd=[sys.executable,*args]; start=time.time()
  with (out/f'{task}_{phase}.log').open('w') as log:
   result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
  records.append(dict(task=task,phase=phase,command=cmd,returncode=result.returncode,seconds=time.time()-start))
  (out/'commands.json').write_text(json.dumps(records,indent=2))
  print(task,phase,result.returncode,flush=True)
  if result.returncode: raise SystemExit(result.returncode)
 assert json.loads((dest/'metrics.json').read_text())==json.loads((dest/'rescored_metrics.json').read_text())
 assert json.loads((dest/'validation.json').read_text())['backend']=='vllm'
print('Both default-vLLM runs and independent metric checks passed.',flush=True)
