import json, os, subprocess, sys, time
from pathlib import Path
run=Path(__file__).resolve().parent
root=run.parents[1]
while True:
 phases=[]
 if (run/'commands.json').exists(): phases=json.loads((run/'commands.json').read_text())
 if any(p['returncode'] for p in phases):
  print('FAILED: see commands.json and task logs',flush=True); sys.exit(1)
 if len(phases)==4:
  result=subprocess.run([sys.executable,str(run/'summarize.py')],cwd=root,env=dict(os.environ,PYTHONPATH=str(root)))
  sys.exit(result.returncode)
 active='se' if (run/'se_infer.log').exists() else 'ncsp'
 p=run/f'{active}_infer.log'
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-2000)); tail=f.read().decode(errors='replace').replace('\r','\n').splitlines()
  status={'active_task':active,'completed_phases':len(phases),'last_log':tail[-1] if tail else '', 'updated_at':time.strftime('%Y-%m-%d %H:%M:%S')}
  (run/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
  print(status['updated_at'],active,status['last_log'],flush=True)
 time.sleep(55)
