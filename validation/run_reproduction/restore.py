"""Restore losslessly compressed run artifacts selected by run-relative prefix."""
import argparse,gzip,hashlib,json,shutil
from pathlib import Path
here=Path(__file__).resolve().parent;root=here.parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('prefix',help='For example: se_200k_on_2_2m_20261120/inference')
p.add_argument('--execute',action='store_true',help='Without this, only list files')
a=p.parse_args()
selected=(root/'runs'/a.prefix).resolve()
if root/'runs' not in selected.parents and selected!=root/'runs':p.error('prefix must stay within runs/')
for e in json.loads((here/'cleanup_manifest.json').read_text())['entries']:
 if e['action']!='gzip_preserve_content' or e['state']!='done':continue
 target=root/e['path']
 if target!=selected and selected not in target.parents:continue
 print(e['retained_path'],'->',e['path'])
 if not a.execute or target.exists():continue
 compressed=root/e['retained_path'];tmp=target.with_suffix(target.suffix+'.restoring')
 digest=hashlib.sha256()
 with gzip.open(compressed,'rb') as source,tmp.open('wb') as dest:
  while chunk:=source.read(4*1024**2):digest.update(chunk);dest.write(chunk)
 if digest.hexdigest()!=e['sha256']:
  tmp.unlink();raise RuntimeError(f'Integrity mismatch: {compressed}')
 tmp.replace(target)
