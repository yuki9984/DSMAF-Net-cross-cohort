"""Verify a worker return using reviewed local code; never execute returned code."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import argparse,csv,json,shutil,zipfile
import numpy as np
import extension_settings as S
from campaign_support import binding

def extract_return(archive,destination):
 destination=Path(destination)
 if destination.exists():raise FileExistsError('Use a new receiver directory')
 with zipfile.ZipFile(archive) as z:
  infos=z.infolist()
  if sum(i.file_size for i in infos)>4*1024**3:raise ValueError('Return archive exceeds the expected size limit')
  names=set()
  for info in infos:
   name=info.filename;parts=name.split('/')
   if '\\' in name or ':' in name or name.startswith('/') or any(p in ('.','..') for p in parts):raise ValueError('Unsafe ZIP path')
   if len(parts)<2 or parts[0]!='PAA30_RESULTS' or parts[1] not in ['outputs','materials','protocol']:raise ValueError('Unexpected return layout')
   if name.casefold() in names:raise ValueError('Duplicate case-insensitive ZIP path')
   names.add(name.casefold())
   if ((info.external_attr>>16)&0o170000)==0o120000 or info.flag_bits&1:raise ValueError('Links and encrypted members are not allowed')
   if not info.is_dir() and Path(name).suffix.lower() not in ['.json','.csv','.npz','.txt','.log','.py','.ps1','.cmd','.md']:raise ValueError('Unexpected return file type')
  destination.mkdir(parents=True,exist_ok=False);root=destination.resolve()
  for info in infos:
   path=root.joinpath(*PurePosixPath(info.filename).parts[1:])
   if not path.resolve().is_relative_to(root):raise ValueError('ZIP escaped receiver root')
   if info.is_dir():path.mkdir(parents=True,exist_ok=True);continue
   path.parent.mkdir(parents=True,exist_ok=True)
   with z.open(info) as source,path.open('xb') as target:shutil.copyfileobj(source,target)
 return destination

def compare_csv(worker,local):
 def load(path):
  with Path(path).open(encoding='utf-8-sig',newline='') as f:
   return {(r['type'],r['comparison'],r['metric'],r['target']):r for r in csv.DictReader(f)}
 a=load(worker);b=load(local)
 if set(a)!=set(b):raise RuntimeError('Returned summary rows differ')
 maximum=0.
 for key in a:
  for name in ['estimate','ci95_low','ci95_high','seed_sd']:
   delta=abs(float(a[key][name])-float(b[key][name]));maximum=max(maximum,delta)
   if delta>1e-10:raise RuntimeError('Returned summary numeric mismatch: '+str(key))
 return maximum

def verify_received(destination):
 dest=Path(destination);protocol=dest/'protocol'
 if S.sha(protocol/'source_manifest.json')!=S.sha(S.BUNDLE/'source_manifest.json') or S.sha(protocol/'analysis_plan.json')!=S.sha(S.PLAN):raise RuntimeError('Worker used a different release')
 for row in S.read(S.BUNDLE/'source_manifest.json')['materials']:
  relative=Path(row['path']).relative_to('materials')
  if S.sha(dest/'materials'/relative)!=row['sha256']:raise RuntimeError('Returned fixed material changed')
 old=S.RESULTS;S.RESULTS=dest/'outputs'
 try:
  from campaign_support import bound_report
  status=S.read(S.RESULTS/'campaign_status.json')
  if status.get('status')!='COMPLETE' or status.get('completed_formal_fits')!=30 or status.get('completed_evaluations')!=60:raise RuntimeError('This is a partial or incomplete campaign; inspect progress and logs before analysis')
  if any(status.get(k)!=v for k,v in binding().items()):raise RuntimeError('Return campaign binding differs')
  inventory=bound_report(S.RESULTS/'training_complete.json','ALL_30_FITS_FROZEN')
  for row in inventory['runs']:
   folder=S.cv_run(row['fold'],row['arm'],row['seed'])
   for field,filename in [('config_sha256','config.json'),('history_sha256','history.json'),('summary_sha256','summary.json')]:
    if S.sha(folder/filename)!=row[field]:raise RuntimeError('Returned training record changed')
   if S.read(folder/'DONE.json')!=row:raise RuntimeError('Worker checkpoint receipt differs from inventory')
  from analyze_extension import assemble_new_predictions,save_analysis,verify_complete
  verify_complete(S.RESULTS/'analysis','ALL_EXTENSION_STATISTICS_PASS')
  recomputed=dest/'receiver_recomputed_analysis'
  report=save_analysis(assemble_new_predictions(),recomputed,S.read(S.RESULTS/'analysis/analysis_complete.json')['pairs'],True)
  csv_delta=compare_csv(S.RESULTS/'analysis/summary_intervals.csv',recomputed/'summary_intervals.csv')
  maximum=0.
  with np.load(S.RESULTS/'analysis/bootstrap_arrays.npz',allow_pickle=False) as a,np.load(recomputed/'bootstrap_arrays.npz',allow_pickle=False) as b:
   if not np.array_equal(a['names'],b['names']):raise RuntimeError('Bootstrap column identities differ')
   for key in ['point_internal','point_external','draws_internal','draws_external','class_fold_weights','loss_fold_weights']:
    delta=float(np.max(np.abs(a[key]-b[key])));maximum=max(maximum,delta)
    if delta>1e-10:raise RuntimeError('Returned bootstrap array mismatch: '+key)
  output={'status':'RETURNED_PREDICTIONS_AND_STATISTICS_INDEPENDENTLY_RECOMPUTED','formal_fit_records_verified':30,'evaluations_verified':60,'paired_seeds':3,'bootstrap_repeats':2000,'maximum_summary_difference':csv_delta,'maximum_array_difference':maximum,'neural_training_repeated':False,'neural_inference_repeated':False,'checkpoint_tensors_present':False,'checkpoint_scope':'Worker-recorded hashes and metadata verified; receiver did not revalidate omitted checkpoint tensors','manuscript_result_insertion_ready_for_author_review':True}
  S.atomic_json(dest/'RETURN_REVIEW.json',output);return output
 finally:S.RESULTS=old

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--destination',type=Path)
 a=p.parse_args();dest=a.destination or S.BUNDLE/'received'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
 extract_return(a.archive,dest);print(json.dumps(verify_received(dest),indent=2))
if __name__=='__main__':main()
