"""One-off audited cleanup of generated run artifacts; never touches input assets."""
import gzip,hashlib,json,os,shutil,time
from pathlib import Path
root=Path(__file__).resolve().parents[2]
runs=root/'runs'; archive=Path(__file__).resolve().parent
manifest=archive/'cleanup_manifest.json'
if manifest.exists():raise SystemExit('Cleanup manifest already exists; refusing to repeat automatically')
# Assert no live process is executing a run script or using the disposable interpreter.
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try: args=(proc/'cmdline').read_bytes().split(b'\0')
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if any(a.decode(errors='replace').startswith(str(runs)+'/') and (a.endswith(b'.py') or a.endswith(b'/python')) for a in args):
  raise RuntimeError(f'Active process uses runs: {proc.name}')
env=runs/'readme_check_20261119/environment'
entries=[]
for f in sorted(runs.rglob('*')):
 if env in f.parents or f==env or not f.is_file() or f.is_symlink():continue
 size=f.stat().st_size
 if f.suffix=='.safetensors' or f.name=='optimizer.pt':action='delete_regenerable_training_binary'
 elif size>5*1024**2 and f.suffix!='.gz':action='gzip_preserve_content'
 else:continue
 entries.append(dict(path=str(f.relative_to(root)),bytes_before=size,action=action,state='planned'))
if env.exists():
 entries.append(dict(path=str(env.relative_to(root)),bytes_before=sum(f.lstat().st_size for f in env.rglob('*') if f.is_file() and not f.is_symlink()),action='delete_disposable_environment',state='planned'))
record=dict(created_at=time.strftime('%Y-%m-%d %H:%M:%S'),entries=entries)
def save():
 tmp=manifest.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(manifest)
save()
for e in entries:
 f=root/e['path']; assert runs in f.parents
 if e['action']=='gzip_preserve_content':
  target=Path(str(f)+'.gz');tmp=Path(str(target)+'.tmp')
  if target.exists():raise RuntimeError(f'Refusing to overwrite {target}')
  digest=hashlib.sha256()
  with f.open('rb') as source,gzip.open(tmp,'wb',compresslevel=6) as dest:
   while chunk:=source.read(4*1024**2):digest.update(chunk);dest.write(chunk)
  check=hashlib.sha256()
  with gzip.open(tmp,'rb') as source:
   while chunk:=source.read(4*1024**2):check.update(chunk)
  assert check.digest()==digest.digest(),f'Compression integrity failed: {f}'
  tmp.replace(target);e.update(sha256=digest.hexdigest(),retained_path=str(target.relative_to(root)),bytes_after=target.stat().st_size)
  f.unlink()
 elif e['action']=='delete_disposable_environment':shutil.rmtree(f)
 else:f.unlink()
 e['state']='done';save();print(e['action'],e['path'],flush=True)
record['completed_at']=time.strftime('%Y-%m-%d %H:%M:%S')
record['bytes_reclaimed_estimate']=sum(e['bytes_before']-e.get('bytes_after',0) for e in entries)
save()
print('Cleanup completed. Bytes reclaimed:',record['bytes_reclaimed_estimate'],flush=True)
