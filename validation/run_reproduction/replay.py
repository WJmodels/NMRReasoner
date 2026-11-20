"""Replay archived commands into new output directories; default is print-only."""
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD='/media/capios/20TB/lln/project/CLReasoner/NMRReasoner'
PROFILES={
 'smoke':'readme_check_20261119/inference_commands.json',
 'training':'readme_check_20261119/training_fixed_commands.json',
 'vllm-smoke':'readme_check_20261119/vllm_commands.json',
 'bidirectional-single':'vllm_bidirectional_20261119/commands.json',
 'full-200k':'vllm_full_test_20261119/commands.json',
 'full-2.2m':'se_200k_on_2_2m_20261120/commands.json',
}
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('profile',choices=PROFILES)
p.add_argument('--output-root',type=Path,default=ROOT/'runs/reproduced')
p.add_argument('--gpu',default='3')
p.add_argument('--execute',action='store_true')
a=p.parse_args(); dest=a.output_root.resolve()
if dest==ROOT/'runs':p.error('Use a new output directory, not the historical runs directory')
records=json.loads((HERE/'records'/PROFILES[a.profile]).read_text())
def remap(value):
 if value.startswith(OLD+'/runs/'):return str(dest/value[len(OLD+'/runs/'):])
 if value.startswith(OLD+'/'):return str(ROOT/value[len(OLD+'/'):])
 return value
commands=[]
for record in records:
 original=record['command']; cmd=[sys.executable,*map(remap,original[1:])]
 # Preserve the effective historical backend, despite the later default change.
 if len(cmd)>1 and cmd[1]=='test.py' and not any(x in cmd for x in ['evaluate','vote','inspect','--backend']):
  backend='vllm'
  if '--output' in original:
   oldout=original[original.index('--output')+1]
   v=HERE/'records'/oldout.replace(OLD+'/runs/','',1)/'validation.json'
   if v.exists(): backend=json.loads(v.read_text())['backend']
   elif a.profile in ['smoke','training']:backend='transformers'
  cmd+=['--backend',backend]
 commands.append(cmd)
 print('CUDA_VISIBLE_DEVICES='+a.gpu,shlex.join(cmd))
if not a.execute:sys.exit(0)
# Recreate the original first-row subsets used by small-scale checks.
if a.profile in ['smoke','training','vllm-smoke','bidirectional-single']:
 from datasets import load_from_disk
 for task,folder,prefix in [('se','Structure elucidation','SE'),('ncsp','NMR chemical shift prediction','NCSP')]:
  for split,n in [('train',4),('train8',8),('test',3)]:
   training=split.startswith('train')
   source=ROOT/'data'/folder/('train' if training else 'test')/'MultiNMR-200k'/f'{prefix}{"-CoT-1" if training else ""}_MultiNMR_200k_{"train" if training else "test"}_arrow'
   target=dest/'readme_check_20261119'/task/'arrow'/split
   if not target.exists():
    target.parent.mkdir(parents=True,exist_ok=True)
    load_from_disk(str(source)).select(range(n)).save_to_disk(str(target))
logdir=dest/'command_logs'/a.profile;logdir.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
for i,cmd in enumerate(commands):
 with (logdir/f'{i:02d}.log').open('w') as log:
  subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Commands completed; outputs:',dest)
