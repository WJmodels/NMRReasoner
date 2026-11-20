"""Wait for the existing pipeline to exit without polling, then write its report."""
import ctypes, errno, json, os, select, signal, subprocess, sys, time
from pathlib import Path
run=Path(__file__).resolve().parent
root=run.parents[1]
pid=int(sys.argv[1])
def status(state, **extra):
 payload={'state':state,'pipeline_pid':pid,'updated_at':time.strftime('%Y-%m-%d %H:%M:%S'),**extra}
 tmp=run/'status.tmp'
 tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 tmp.replace(run/'status.json')
status('running', stages=['NCSP full inference','NCSP evaluation','SE full inference','SE evaluation','summary report'])
try:
 try:
  libc=ctypes.CDLL(None,use_errno=True)
  fd=libc.syscall(434,pid,0)  # Linux x86_64 pidfd_open
  if fd<0:
   error=ctypes.get_errno()
   if error==errno.ESRCH: raise ProcessLookupError()
   raise OSError(error,os.strerror(error))
 except ProcessLookupError:
  fd=None
 if fd is not None:
  select.select([fd],[],[])
  os.close(fd)
 records=json.loads((run/'commands.json').read_text()) if (run/'commands.json').exists() else []
 if len(records)<4 and not any(r['returncode'] for r in records):
  status('running',note='Original process exited before completion; resuming unfinished stages independently.')
  try: os.killpg(2578910,signal.SIGTERM)
  except ProcessLookupError: pass
  time.sleep(3)
  with (run/'recovery.log').open('a') as log:
   result=subprocess.run([sys.executable,'-u',str(run/'reproduce.py')],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  if result.returncode: raise RuntimeError('Resumed pipeline failed; see recovery.log')
  records=json.loads((run/'commands.json').read_text())
 if len(records)!=4 or any(r['returncode']!=0 for r in records):
  raise RuntimeError('Pipeline did not complete all four stages successfully; inspect commands.json and logs.')
 result=subprocess.run([sys.executable,str(run/'summarize.py')],cwd=root,env=dict(os.environ,PYTHONPATH=str(root)))
 if result.returncode:
  raise RuntimeError(f'Report generation failed with exit code {result.returncode}')
 status('completed',report=str(run/'REPORT.md'),summary=str(run/'summary.json'))
except Exception as exc:
 status('failed',error=str(exc))
 raise
