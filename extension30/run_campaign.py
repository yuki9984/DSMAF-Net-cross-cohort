"""Worker orchestration: check, qualify, trial, train, evaluate, analyze, return."""
from pathlib import Path
import argparse,ctypes,json,os,shutil,sys,traceback,zipfile
import extension_settings as S
from campaign_support import binding,bind_campaign,campaign_lock,bound_report,progress,run_logged,CampaignBusy

def configure(a):
 selected=S.locations()
 for key,value in [('original_project',a.original_project),('pad_images',a.pad_images),('external_images',a.external_images),('initialization_file',a.initialization_file)]:
  if value:selected[key]=str(Path(value).resolve())
 if selected:S.atomic_json(S.BUNDLE/'worker_locations.json',selected)
 temp=S.BUNDLE/'tmp';temp.mkdir(exist_ok=True)
 os.environ.update(TEMP=str(temp),TMP=str(temp),PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',PYTHONUNBUFFERED='1',CUDA_CACHE_PATH=str(S.BUNDLE/'cache/cuda'))
 (S.BUNDLE/'cache/cuda').mkdir(parents=True,exist_ok=True)

def stage_states():
 result={}
 for name,path,status in [('inputs','worker_input_audit.json','ALL_WORKER_INPUTS_PASS'),('qualification','gpu_preflight.json','ALL_MODEL_PREFLIGHT_PASS'),('trials','engineering_trials.json','ALL_TWO_ENGINEERING_TRIALS_PASS'),('training','training_complete.json','ALL_30_FITS_FROZEN'),('evaluation','evaluation_complete.json','ALL_60_EVALUATIONS_PASS'),('analysis','analysis/analysis_complete.json','ALL_EXTENSION_STATISTICS_PASS')]:
  try:bound_report(S.RESULTS/path,status);result[name]='PASS'
  except Exception:result[name]='PENDING_OR_FAILED'
 return result

def pack_results():
 S.RESULTS.mkdir(parents=True,exist_ok=True);states=stage_states();complete=all(v=='PASS' for v in states.values())
 if complete:
  try:
   from run_matrix import freeze_training
   from analyze_extension import verify_complete
   freeze_training();verify_complete(S.RESULTS/'analysis','ALL_EXTENSION_STATISTICS_PASS')
   states['final_integrity']='PASS'
  except Exception as error:
   states['final_integrity']='FAILED';complete=False
   (S.RESULTS/'final_integrity_error.txt').write_text(str(error),encoding='utf-8')
 counts={'training':0,'evaluation':0}
 if states['training']=='PASS':counts['training']=len(S.read(S.RESULTS/'training_complete.json')['runs'])
 if states['evaluation']=='PASS':counts['evaluation']=len(S.read(S.RESULTS/'evaluation_complete.json')['files'])
 S.atomic_json(S.RESULTS/'campaign_status.json',{**binding(),'status':'COMPLETE' if complete else 'PARTIAL','stage_states':states,'completed_formal_fits':counts['training'],'completed_evaluations':counts['evaluation'],'checkpoint_tensors_in_return_archive':False,'keep_all_weights_on_worker':True})
 returned=S.BUNDLE/'results_return';returned.mkdir(exist_ok=True)
 target=returned/('PAA30_complete_results.zip' if complete else 'PAA30_partial_results.zip')
 temporary=target.with_name(target.name+'.tmp')
 with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
  for folder,prefix in [(S.RESULTS,'PAA30_RESULTS/outputs'),(S.BUNDLE/'materials','PAA30_RESULTS/materials')]:
   for p in sorted(folder.rglob('*')):
    if p.is_file() and (p.suffix.lower() in ['.json','.csv','.npz','.txt','.log'] or (folder==S.MATERIALS and p.name=='external_encoding.py')) and p.name!='campaign.lock':archive.write(p,prefix+'/'+p.relative_to(folder).as_posix())
  for p in S.BUNDLE.iterdir():
   if p.is_file() and p.suffix.lower() in ['.py','.json','.txt','.ps1','.cmd'] and p.name!='worker_locations.json':archive.write(p,'PAA30_RESULTS/protocol/'+p.name)
  for p in (S.BUNDLE/'references').iterdir():
   if p.is_file():archive.write(p,'PAA30_RESULTS/protocol/references/'+p.name)
 with zipfile.ZipFile(temporary) as archive:
  if archive.testzip() is not None:raise RuntimeError('Return archive CRC failure')
 temporary.replace(target)
 target.with_name(target.name+'.sha256.txt').write_text(S.sha(target)+'  '+target.name+'\n',encoding='ascii')
 return target,states

def require_space():
 finished=len(list((S.RESULTS/'fits').glob('fold_*/*/DONE.json')))
 required=(2*1024**3)+(30-min(30,finished))*200*1024**2
 if shutil.disk_usage(S.BUNDLE).free<required:raise RuntimeError(f'Need {required/1024**3:.1f} GiB free; keep original weights and use a folder with enough free space on this computer')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--stage',choices=['all','check','qualify','trials','train','evaluate','analyze','pack'],default='all')
 p.add_argument('--original-project',type=Path);p.add_argument('--pad-images',type=Path);p.add_argument('--external-images',type=Path);p.add_argument('--initialization-file',type=Path)
 p.add_argument('--cpu-check',action='store_true');p.add_argument('--metadata-only',action='store_true')
 a=p.parse_args();configure(a);S.verify_manifest(materials=True)
 if (a.cpu_check or a.metadata_only) and a.stage!='check':p.error('CPU/metadata flags are for input checking only')
 if a.metadata_only and not a.cpu_check:p.error('--metadata-only requires --cpu-check')
 if a.stage=='check' and a.cpu_check:
  from worker_inputs import check_worker
  print(json.dumps(check_worker(allow_cpu=True,check_image_files=not a.metadata_only),indent=2));return
 with campaign_lock():
  bind_campaign()
  keep_awake=False
  try:
   if a.stage in ['all','check','qualify','trials','train','evaluate']:
    from worker_inputs import check_worker
    check_worker();progress('WORKER_INPUTS_VERIFIED')
   if a.stage=='check':return
   if os.name=='nt' and a.stage in ['all','qualify','trials','train','evaluate']:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);keep_awake=True
   if a.stage in ['all','qualify','trials','train']:require_space()
   if a.stage in ['all','qualify']:
    try:bound_report(S.RESULTS/'gpu_preflight.json','ALL_MODEL_PREFLIGHT_PASS')
    except Exception:run_logged([S.BUNDLE/'preflight_models.py','--device','cuda','--qualify-training'],'gpu_qualification')
   if a.stage in ['all','trials']:run_logged([S.BUNDLE/'engineering_trials.py'],'engineering_trials')
   if a.stage in ['all','train']:
    require_space()
    from run_matrix import run_training
    run_training();progress('ALL_30_TRAINING_FITS_FROZEN',completed_formal_fits=30)
   if a.stage in ['all','evaluate']:
    from evaluate_extension import evaluate_matrix
    evaluate_matrix();progress('ALL_60_EVALUATIONS_VERIFIED',completed_evaluations=60)
   if a.stage in ['all','analyze']:
    from analyze_extension import analyze_extension
    analyze_extension();progress('ALL_EXTENSION_STATISTICS_VERIFIED')
   target,states=pack_results();progress('RETURN_ARCHIVE_READY',archive=str(target),archive_sha256=S.sha(target),stage_states=states)
  finally:
   if keep_awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
 try:main()
 except CampaignBusy:
  print('Another campaign is already running. Keep the original window open; no duplicate job started.',flush=True)
  raise SystemExit(2)
 except KeyboardInterrupt:
  progress('INTERRUPTED_BY_OPERATOR')
  try:pack_results()
  except Exception:pass
  raise SystemExit(130)
 except Exception:
  S.RESULTS.mkdir(parents=True,exist_ok=True)
  error=traceback.format_exc();(S.RESULTS/'STOPPED_error.txt').write_text(error,encoding='utf-8')
  progress('STOPPED_FOR_REVIEW',error=error)
  try:
   path,_=pack_results();print('PARTIAL RETURN ARCHIVE:',path,flush=True)
  except Exception:pass
  raise
