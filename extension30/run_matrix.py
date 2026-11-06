"""Freeze and resume exactly 30 fits, retaining interrupted attempts and receipts."""
from pathlib import Path
import argparse,json,subprocess,sys,time
import extension_settings as S
from campaign_support import binding,bound_report,progress,retain_failed,now,run_logged

def check_completed(folder,fold,arm,seed,load_checkpoint=True):
 config=S.read(folder/'config.json');history=S.read(folder/'history.json');summary=S.read(folder/'summary.json')
 expected={'protocol':'paa_latefusion30_20261103','fold':fold,'arm':arm,'seed':seed,'extension_plan_sha256':S.sha(S.PLAN),'extension_manifest_sha256':S.sha(S.BUNDLE/'source_manifest.json'),'information_condition':S.ENCODINGS[arm],'loss':'balanced_softmax','oversample_scc':True,'batch_size':16,'epochs':150,'patience':15,'lr':2e-4,'weight_decay':1e-4,'optimizer':'adamw','scheduler':'cosine','selection_split':'valid','selection_metric':'balanced_accuracy','outer_test_used_for_selection':False,'sampler_generator_seed':seed,'rng_reset_after_model_init':True,'trial':False,'pretrained_weight_sha256':S.read(S.PLAN)['initialization_sha256'],'split_manifest_sha256':S.read(S.PLAN)['fold_manifest_sha256']}
 for key,value in expected.items():
  if config.get(key)!=value:raise RuntimeError('Completed fit mismatch: '+key)
 if not config.get('initial_state_sha256'):raise RuntimeError('Missing initialization identity')
 scores=[row['valid']['bac'] for row in history['epochs']]
 import math
 if not scores or not all(math.isfinite(v) for v in scores) or len(scores)>150 or summary['epochs_completed']!=len(scores):raise RuntimeError('Incomplete or invalid history')
 first=scores.index(max(scores))+1
 if len(scores)<150 and len(scores)-first!=15:raise RuntimeError('Training did not stop at the fixed patience or epoch cap')
 if [r['epoch'] for r in history['epochs']]!=list(range(1,len(scores)+1)):raise RuntimeError('Epoch history is not sequential')
 if history['best_epoch']!=first or summary['best_epoch']!=first or abs(summary['best_valid_bac']-max(scores))>1e-12:raise RuntimeError('Checkpoint selection record changed')
 checkpoint=folder/'best_valid.pth'
 row={'fold':fold,'arm':arm,'seed':seed,'checkpoint_sha256':S.sha(checkpoint),'config_sha256':S.sha(folder/'config.json'),'history_sha256':S.sha(folder/'history.json'),'summary_sha256':S.sha(folder/'summary.json'),'initial_state_sha256':config['initial_state_sha256'],'epochs_completed':len(scores),'best_epoch':first}
 receipt=folder/'DONE.json'
 if receipt.exists():
  if S.read(receipt)!=row:raise RuntimeError('Previously frozen fit changed: '+str(folder))
  return row
 if load_checkpoint:
  import torch
  ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
  if ck['arm']!=arm or ck['seed']!=seed or ck['epoch']!=first or abs(ck['validation_bac']-max(scores))>1e-12:raise RuntimeError('Saved checkpoint differs from selected epoch')
  if json.dumps(ck['config'],sort_keys=True)!=json.dumps(config,sort_keys=True):raise RuntimeError('Checkpoint/config mismatch')
  if not ck['model_state_dict'] or not all(torch.isfinite(t).all() for t in ck['model_state_dict'].values() if t.is_floating_point()):raise RuntimeError('Nonfinite or empty checkpoint')
  del ck
 S.atomic_json(receipt,row)
 return row

def freeze_training():
 rows=[check_completed(S.cv_run(f,a,s),f,a,s) for f,a,s in S.matrix()]
 for fold in range(5):
  for seed in S.SEEDS:
   identities=[r['initial_state_sha256'] for r in rows if r['fold']==fold and r['seed']==seed and r['arm'].startswith('late_age_')]
   if len(identities)!=2 or len(set(identities))!=1:raise RuntimeError('Information conditions used different initializations')
 report={**binding(),'status':'ALL_30_FITS_FROZEN','runs':rows,'new_formal_fits_completed':30,'outer_or_external_evaluated':False}
 path=S.RESULTS/'training_complete.json'
 if path.exists() and S.read(path)!=report:raise RuntimeError('Frozen campaign inventory changed')
 S.atomic_json(path,report);return report

def run_training():
 S.validate_inputs();bound_report(S.RESULTS/'gpu_preflight.json','ALL_MODEL_PREFLIGHT_PASS');bound_report(S.RESULTS/'engineering_trials.json','ALL_TWO_ENGINEERING_TRIALS_PASS')
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no training started')
 S.ensure_pretrained();rows=[]
 for fold,arm,seed in S.matrix():
  folder=S.cv_run(fold,arm,seed);rid=S.run_id(fold,arm,seed)
  attempt_record=S.RESULTS/'attempts'/(rid+'.json')
  if folder.exists() and (folder/'summary.json').exists():
   row=check_completed(folder,fold,arm,seed);rows.append(row)
   progress('COMPLETED_FIT_VERIFIED',completed=len(rows),planned=30,run_id=rid);continue
  if folder.exists():
   if not attempt_record.exists() or any(S.read(attempt_record).get(k)!=v for k,v in binding().items()):raise RuntimeError('Unknown partial directory; refusing to move '+str(folder))
   retain_failed(folder,'training')
  S.atomic_json(attempt_record,{**binding(),'run_id':rid,'utc':now()})
  progress('FORMAL_FIT_STARTED',completed=len(rows),planned=30,run_id=rid)
  started=time.monotonic()
  run_logged([S.BUNDLE/'train_extension.py','--arm',arm,'--fold',str(fold),'--seed',str(seed)],rid)
  row=check_completed(folder,fold,arm,seed);rows.append(row)
  progress('FORMAL_FIT_COMPLETE',completed=len(rows),planned=30,run_id=rid,seconds=time.monotonic()-started)
 return freeze_training()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train',action='store_true');a=p.parse_args()
 if a.train:print(json.dumps(run_training(),indent=2))
 else:
  r=S.validate_inputs();print(json.dumps({k:v for k,v in r.items() if k!='checks'},indent=2))
if __name__=='__main__':main()
