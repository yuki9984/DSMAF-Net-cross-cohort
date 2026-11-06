"""Shared binding and file operations for a single immutable worker campaign."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,time,contextlib
import extension_settings as S

class CampaignBusy(OSError):pass

def now():return datetime.now(timezone.utc).isoformat()
def binding():return {'plan_sha256':S.sha(S.PLAN),'source_manifest_sha256':S.sha(S.BUNDLE/'source_manifest.json')}
def progress(phase,**values):
 value={'phase':phase,'utc':now(),'planned_new_fits':30,**values}
 S.atomic_json(S.RESULTS/'worker_progress.json',value)
 line=json.dumps(value,ensure_ascii=False)
 with (S.RESULTS/'console.log').open('a',encoding='utf-8') as stream:stream.write(line+'\n')
 print(line,flush=True)

def bound_report(path,status):
 report=S.read(path)
 if report.get('status')!=status or any(report.get(k)!=v for k,v in binding().items()):
  raise RuntimeError('Stale or invalid stage report: '+str(path))
 return report

def bind_campaign():
 S.RESULTS.mkdir(parents=True,exist_ok=True)
 marker=S.RESULTS/'campaign_binding.json'
 if marker.exists():
  if S.read(marker)!=binding():raise RuntimeError('Campaign code or plan changed; keep this campaign and use a versioned folder')
 else:S.atomic_json(marker,binding())
 return binding()

@contextlib.contextmanager
def campaign_lock():
 S.RESULTS.mkdir(parents=True,exist_ok=True)
 with (S.RESULTS/'campaign.lock').open('a+b') as lock:
  lock.seek(0,2)
  if lock.tell()==0:lock.write(b'1');lock.flush()
  lock.seek(0)
  if os.name=='nt':
   import msvcrt
   try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
   except OSError as error:raise CampaignBusy('Another campaign is running') from error
  else:
   import fcntl
   try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
   except OSError as error:raise CampaignBusy('Another campaign is running') from error
  try:yield
  finally:
   lock.seek(0)
   if os.name=='nt':msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
   else:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def retain_failed(path,kind):
 original=Path(path);root=S.RESULTS.resolve()
 try:relative=original.absolute().relative_to(S.RESULTS.absolute())
 except ValueError:raise RuntimeError('Retention source outside campaign outputs')
 source=original.resolve()
 if original.is_symlink() or source!=(root/relative).absolute():raise RuntimeError('Aliased retention path refused')
 target=(root/'failed_attempts'/kind/(source.name+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))).resolve()
 if not source.is_relative_to(root) or source==root or not target.is_relative_to(root) or source.is_symlink():
  raise RuntimeError('Refusing an unverified retention path')
 target.parent.mkdir(parents=True,exist_ok=True);source.rename(target)
 progress('FAILED_ATTEMPT_RETAINED',source=str(source),retained=str(target))
 return target

def exclusive_npz(path,**arrays):
 import numpy as np
 converted={}
 for name,value in arrays.items():
  item=np.asarray(value)
  if item.dtype.kind=='O':
   if not all(isinstance(x,(str,np.str_)) for x in item.ravel()):raise ValueError('Object arrays are forbidden: '+name)
   item=item.astype(str)
  converted[name]=item
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():raise FileExistsError('Completed predictions cannot be overwritten: '+str(path))
 temporary=path.with_name(path.name+'.tmp')
 if temporary.exists():retain_failed(temporary,'inference')
 with temporary.open('xb') as stream:np.savez_compressed(stream,**converted)
 # Only the campaign controller may create prediction files during this stage.
 if path.exists():raise FileExistsError('Concurrent prediction writer detected')
 temporary.rename(path)


def run_logged(arguments,label):
 import subprocess,sys
 logs=S.RESULTS/'logs';logs.mkdir(parents=True,exist_ok=True)
 log=logs/(label+'.log')
 progress('STAGE_STARTED',stage=label,log=str(log))
 with log.open('a',encoding='utf-8') as stream:
  stream.write('\nATTEMPT '+now()+'\n');stream.flush()
  process=subprocess.Popen([sys.executable,'-B','-u',*map(str,arguments)],cwd=S.BUNDLE,stdout=stream,stderr=subprocess.STDOUT)
  try:
   last=time.monotonic()
   while process.poll() is None:
    time.sleep(2)
    if time.monotonic()-last>=30:
     with log.open('rb') as tail:
      tail.seek(max(0,log.stat().st_size-4096));lines=tail.read().decode('utf-8',errors='replace').splitlines()
     progress('STAGE_RUNNING',stage=label,last_log_line=lines[-1][-300:] if lines else '',log=str(log))
     last=time.monotonic()
  finally:
   if process.poll() is None:
    process.terminate()
    try:process.wait(timeout=60)
    except subprocess.TimeoutExpired:process.kill();process.wait()
 if process.returncode:
  raise RuntimeError('Stage failed; keep '+str(log))
 return log
