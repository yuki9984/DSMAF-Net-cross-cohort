"""Run only the approved two extensions in the original qualified laptop project."""
from pathlib import Path
import argparse,os,sys,json,time,subprocess,shutil,zipfile
from datetime import datetime,timezone
BUNDLE=Path(__file__).resolve().parent

def parse():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',default=str(BUNDLE.parent));p.add_argument('--weights-root');p.add_argument('--pad-images');p.add_argument('--output');p.add_argument('--stage',choices=['check','all','interventions','training','evaluate','analyze','pack'],default='all');return p.parse_args()

def one_fit_audit(d,fold,arm,seed):
 from followup_settings import read,sha256,PLAN,BUNDLE
 c=read(d/'config.json');h=read(d/'history.json');s=read(d/'summary.json')
 for key,value in {'fold':fold,'arm':arm,'seed':seed,'loss':'ordinary_cross_entropy','oversample_scc':False,'plan_sha256':sha256(PLAN),'source_manifest_sha256':sha256(BUNDLE/'source_manifest.json'),'selection_metric':'balanced_accuracy','outer_test_used_for_selection':False}.items():assert c.get(key)==value,(d,key)
 scores=[z['valid']['bac'] for z in h['epochs']];first=scores.index(max(scores))+1;assert h['best_epoch']==s['best_epoch']==first and len(scores)==s['epochs_completed']
 return {'status':'PASS','fold':fold,'arm':arm,'seed':seed,'checkpoint_sha256':sha256(d/'best_valid.pth'),'config_sha256':sha256(d/'config.json'),'history_sha256':sha256(d/'history.json'),'summary_sha256':sha256(d/'summary.json'),'training_seconds':s['training_seconds']}

def run_training(args):
 from followup_settings import RESULTS,ARMS,SEEDS,cv_run,read,atomic_json,freeze_training
 from gpu_tasks import setup,progress
 setup(args);done=[];started=time.monotonic()
 for fold in range(5):
  for seed in SEEDS:
   for arm in ARMS:
    d=cv_run(fold,arm,seed);complete=d/'DONE.json'
    if (d/'summary.json').exists():
     rec=one_fit_audit(d,fold,arm,seed)
     if complete.exists():assert read(complete)==rec,'Completed fit changed'
     else:atomic_json(complete,rec)
     done.append(rec);progress('TRAINING45_RESUME_PASS',completed=len(done),planned=45,fold=fold,arm=arm,seed=seed);continue
    if d.exists():
     target=RESULTS/'interrupted_training'/f'{d.parent.name}_{d.name}_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")}'
     assert d.resolve().is_relative_to((RESULTS/'recipe45').resolve()) and target.resolve().is_relative_to(RESULTS.resolve()) and not target.exists()
     target.parent.mkdir(parents=True,exist_ok=True);d.rename(target);progress('PARTIAL_NEW_FIT_RETAINED_RESTART_FROM_SEED',retained=str(target),fold=fold,arm=arm,seed=seed)
    progress('TRAINING45_FIT_STARTED',completed=len(done),planned=45,fold=fold,arm=arm,seed=seed)
    log=RESULTS/'logs'/f'train_fold{fold}_{arm}_seed{seed}.log';log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,DSMAF_PROJECT_ROOT=str(Path(args.project_root).resolve()),DSMAF_FOLLOWUP_OUTPUT=str(RESULTS),PYTHONUNBUFFERED='1')
    with log.open('a',encoding='utf-8') as f:
     f.write('\nNEW_ATTEMPT '+datetime.now(timezone.utc).isoformat()+'\n');f.flush();r=subprocess.run([sys.executable,'-B','-u',str(BUNDLE/'train_recipe.py'),'--fold',str(fold),'--arm',arm,'--seed',str(seed)],cwd=Path(args.project_root),env=env,stdout=f,stderr=subprocess.STDOUT)
    assert r.returncode==0,'Training stopped; preserve log '+str(log)
    rec=one_fit_audit(d,fold,arm,seed);atomic_json(complete,rec);done.append(rec)
    progress('TRAINING45_FIT_PASS',completed=len(done),planned=45,fold=fold,arm=arm,seed=seed,fit_seconds=rec['training_seconds'],elapsed_seconds=time.monotonic()-started,completed_fit_mean_seconds=sum(x['training_seconds'] for x in done)/len(done))
 records=freeze_training();progress('ALL45_CHECKPOINTS_FROZEN',completed=45,planned=45,outer_test_evaluated=False,external_evaluated=False)
 return records

def pack_results(label):
 from followup_settings import RESULTS,read,atomic_json,sha256
 status={name:(read(RESULTS/path).get('status') if (RESULTS/path).exists() else 'PENDING') for name,path in [('interventions','interventions/complete.json'),('training45','recipe45/training_complete.json'),('evaluation45','recipe45/evaluation_complete.json'),('intervention_analysis','analysis/interventions/complete.json'),('recipe_analysis','analysis/recipe45/complete.json')]}
 records=[]
 for p in sorted((RESULTS/'recipe45').glob('fold_*/*/best_valid.pth')):records.append({'path':p.relative_to(RESULTS).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)})
 atomic_json(RESULTS/'checkpoint_inventory45.json',{'new_best_weights':len(records),'files':records,'keep_locally':True,'tensors_in_return_archive':False})
 atomic_json(RESULTS/'campaign_status.json',{'stage_status':status,'all_complete':all(v=='PASS' for v in status.values()),'source_manifest_sha256':sha256(BUNDLE/'source_manifest.json'),'plan_sha256':sha256(BUNDLE/'analysis_plan.json'),'weights_not_included':True})
 out=BUNDLE/'results_return'/('FOLLOWUP_'+label+'_results.zip');out.parent.mkdir(exist_ok=True)
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(RESULTS.rglob('*')):
   if p.is_file() and p.suffix.lower() in ['.json','.csv','.npz','.log','.txt'] and 'interrupted_training' not in p.relative_to(RESULTS).parts:z.write(p,'FOLLOWUP_RESULTS/'+p.relative_to(RESULTS).as_posix())
  for p in sorted(BUNDLE.iterdir()):
   if p.is_file() and p.suffix in ['.py','.json','.txt']:z.write(p,'FOLLOWUP_RESULTS/protocol/'+p.name)
 with zipfile.ZipFile(out) as z:assert z.testzip() is None
 out.with_name(out.name+'.sha256.txt').write_text(sha256(out)+'  '+out.name+'\n',encoding='ascii')
 return out,status

def main():
 args=parse();os.environ['DSMAF_PROJECT_ROOT']=str(Path(args.project_root).resolve())
 if args.output:os.environ['DSMAF_FOLLOWUP_OUTPUT']=str(Path(args.output).resolve())
 if args.pad_images:os.environ['DSMAF_PAD_IMAGES']=str(Path(args.pad_images).resolve())
 os.environ['PYTHONUNBUFFERED']='1';os.environ['TORCH_HOME']=str(Path(args.project_root).resolve()/'cache')
 from followup_settings import verify_manifest,RESULTS,atomic_json,sha256,read,PLAN
 records,root=verify_manifest();RESULTS.mkdir(parents=True,exist_ok=True)
 if args.stage=='check':
  print(json.dumps({'status':'METADATA_SOURCE_PLAN_PASS','original_checkpoints':len(records),'new_training_fits_planned':45,'new_training_started':False,'GPU_inference_performed':False,'project_root':str(Path(args.project_root).resolve()),'plan_sha256':sha256(PLAN)},indent=2),flush=True);return
 assert RESULTS==(BUNDLE/'outputs').resolve(),'GPU/analysis stages must use the new bundle outputs directory; old experiment directories are protected.'
 with (RESULTS/'campaign.lock').open('a+') as lock:
  if os.name=='nt':
   import msvcrt
   lock.write('1');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
  else:raise RuntimeError('Run GPU stages on the original Windows 4060 laptop.')
  from gpu_tasks import perturb_stage,evaluate_recipe,progress
  binding={'plan_sha256':sha256(PLAN),'source_manifest_sha256':sha256(BUNDLE/'source_manifest.json'),'project_root':str(Path(args.project_root).resolve())};marker=RESULTS/'campaign_started.json'
  if marker.exists():assert read(marker)==binding,'Campaign bindings changed'
  else:atomic_json(marker,binding)
  if args.stage in ['all','training','interventions','evaluate']:
   free=shutil.disk_usage(RESULTS).free;finished=sum(1 for _ in (RESULTS/'recipe45').glob('fold_*/*/DONE.json'));required=(3+9*max(0,45-finished)/45)*1024**3 if args.stage in ['all','training'] else 3*1024**3;assert free>=required,f'Need {required/1024**3:.1f} GiB free for remaining stages; keep all original weights.'
  import ctypes
  ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
  try:
   if args.stage in ['all','interventions']:
    perturb_stage(args)
    from analyze_followup import analyze_interventions
    analyze_interventions();archive,status=pack_results('interventions');progress('INTERVENTIONS_COMPLETE_RETURN_AVAILABLE',archive=str(archive),archive_sha256=sha256(archive),new_training_fits_completed=0)
   if args.stage in ['all','training']:run_training(args)
   if args.stage in ['all','evaluate']:evaluate_recipe(args)
   if args.stage in ['all','analyze']:
    from analyze_followup import analyze_recipe
    analyze_recipe()
   if args.stage in ['all','pack','analyze','evaluate']:
    archive,status=pack_results('complete');finished=all(v=='PASS' for v in status.values());progress('ALL_APPROVED_EXTENSIONS_PASS' if finished else 'PARTIAL_RESULTS_PACKED',archive=str(archive),archive_sha256=sha256(archive),stage_status=status,new_training_fits_completed=45 if status['training45']=='PASS' else 0)
  finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  from followup_settings import RESULTS,atomic_json
  atomic_json(RESULTS/'worker_progress.json',{'phase':'STOPPED_FOR_REVIEW','error':traceback.format_exc(),'utc':datetime.now(timezone.utc).isoformat(),'planned_new_training_fits':45});raise
