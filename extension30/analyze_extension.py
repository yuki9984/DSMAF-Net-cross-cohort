"""Assemble verified new/archived predictions and produce audit-ready tables."""
from pathlib import Path
import argparse,csv,json
import numpy as np
import pandas as pd
import extension_settings as S
from campaign_support import binding,bound_report,exclusive_npz
from analysis_core import calculate_statistics

PAIRS=[('late_age_recorded','late_age_only'),('late_fusion','late_age_recorded')]

def write_csv(path,rows):
 rows=list(rows)
 if not rows:raise ValueError('Empty results table')
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
 with tmp.open('w',encoding='utf-8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 tmp.replace(path)

def archived_arrays():
 from worker_inputs import verify_material_arrays
 return verify_material_arrays()

def assemble_new_predictions():
 a=archived_arrays();evaluation=bound_report(S.RESULTS/'evaluation_complete.json','ALL_60_EVALUATIONS_PASS')
 if len(evaluation['files'])!=60:raise RuntimeError('Expected 60 evaluated model/cohort artifacts')
 import evaluate_extension as E
 records=E.frozen_records();lookup={(r['fold'],r['arm'],r['seed'],r['cohort']):r for r in evaluation['files']}
 if len(lookup)!=60:raise RuntimeError('Duplicate evaluation records')
 newi=np.empty((len(S.ARMS),3,2298,6),dtype=float);newe=np.empty((len(S.ARMS),3,5,1701,6),dtype=float)
 internal_index=pd.Index(a['internal_images'].astype(str));coverage=np.zeros((len(S.ARMS),3,2298),dtype=bool)
 for method,arm in enumerate(S.ARMS):
  for seed_index,seed in enumerate(S.SEEDS):
   for fold in range(5):
    config=S.read(S.cv_run(fold,arm,seed)/'config.json');rec=records[(fold,arm,seed)]
    for cohort in ['internal','external']:
     row=lookup[(fold,arm,seed,cohort)];path=S.prediction_file(cohort,fold,arm,seed)
     if S.sha(path)!=row['prediction_sha256']:raise RuntimeError('Evaluated predictions changed')
     frame,meta,labels,patients=E.load_evaluation_inputs(config,cohort)
     with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
     E.validate_prediction_arrays(arrays,frame,labels,patients,E.input_fingerprint(frame,meta,labels,patients),rec,S.sha(S.cv_run(fold,arm,seed)/'config.json'))
     if cohort=='internal':
      idx=internal_index.get_indexer(frame.img_id.astype(str))
      if (idx<0).any() or coverage[method,seed_index,idx].any() or not np.all(a['internal_folds'][idx]==fold):raise RuntimeError('Invalid OOF coverage')
      if not np.array_equal(a['internal_labels'][idx],labels) or not np.array_equal(a['internal_patients'][idx].astype(str),patients.astype(str)):raise RuntimeError('OOF label/patient identity changed')
      newi[method,seed_index,idx]=arrays['probabilities'];coverage[method,seed_index,idx]=True
     else:
      if not np.array_equal(a['external_images'].astype(str),frame.img_id.astype(str).to_numpy()) or not np.array_equal(a['external_labels'],labels):raise RuntimeError('External reference order changed')
      newe[method,seed_index,fold]=arrays['probabilities']
 if not coverage.all():raise RuntimeError('Missing unique OOF predictions')
 a['prob_internal']=np.concatenate([a['prob_internal'],newi],axis=0)
 a['prob_external']=np.concatenate([a['prob_external'],newe],axis=0)
 a['arms']=np.array(a['arms'].astype(str).tolist()+list(S.ARMS))
 return a

def save_analysis(a,out,pairs,final):
 out=Path(out)
 if out.exists() and any(out.iterdir()):raise FileExistsError('Analysis output must be new or verified by its existing completion receipt')
 out.mkdir(parents=True,exist_ok=True)
 plan=S.read(S.PLAN);arms=a['arms'].astype(str).tolist();seeds=a['seeds'].astype(int).tolist()
 repeats=plan['bootstrap_repeats'];seed=plan['bootstrap_seed']
 if final and (arms!=['full','image_only','late_fusion',*S.ARMS] or seeds!=list(S.SEEDS) or repeats!=2000):raise RuntimeError('Incomplete final analysis design')
 result=calculate_statistics(arms,seeds,a['internal_labels'],a['internal_patients'],a['internal_folds'],a['prob_internal'],a['external_labels'],a['external_patients'],a['prob_external'],pairs,repeats,seed)
 write_csv(out/'summary_intervals.csv',result['summary']);write_csv(out/'metrics_and_contrasts_by_seed.csv',result['by_seed'])
 arrays=result['arrays']
 exclusive_npz(out/'bootstrap_arrays.npz',names=np.array(arrays['names']),point_internal=arrays['point_internal'],point_external=arrays['point_external'],draws_internal=arrays['draws_internal'],draws_external=arrays['draws_external'],class_fold_weights=result['class_fold_weights'],loss_fold_weights=result['loss_fold_weights'])
 text=['Exploratory paired three-seed extension. Intervals condition on fitted models; no retraining uncertainty or causal effects are estimated.']
 if final:
  for key in ['late_age_recorded-minus-late_age_only','late_fusion-minus-late_age_recorded']:
   for target in ['internal','external']:
    row=next(r for r in result['summary'] if r['comparison']==key and r['target']==target and r['metric']=='BACC')
    values=[r['value'] for r in result['by_seed'] if r['comparison']==key and r['target']==target and r['metric']=='BACC']
    text.append(f"{key}, {target}: mean BACC difference {row['estimate']:+.2f} percentage points (95% conditional interval {row['ci95_low']:+.2f} to {row['ci95_high']:+.2f}); three seed differences {', '.join(f'{v:+.2f}' for v in values)}.")
  text.append('The B/C/D comparisons use the same ordinary late-fusion graph. D is the archived full-information late-fusion control. No isolated age effect is estimated. This follow-up was designed after viewing the earlier DSMAF extension and external cohort; report all seeds and any disagreement.')
 else:text.append('CONTROLS-ONLY SOFTWARE VALIDATION: these are archived model results, not new experiments.')
 (out/'results_for_manuscript_WORKING.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
 files=[{'path':p.name,'sha256':S.sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()]
 report={**binding(),'status':'ALL_EXTENSION_STATISTICS_PASS' if final else 'ARCHIVED_CONTROLS_STATISTICS_PASS','methods':arms,'paired_seeds':seeds,'pairs':pairs,'bootstrap_repeats':repeats,'bootstrap_seed':seed,'rejected_internal_draws':arrays['rejected_internal'],'rejected_external_draws':arrays['rejected_external'],'prediction_ensemble':False,'internal_images':len(a['internal_labels']),'external_images':len(a['external_labels']),'intervals':'Descriptive unadjusted patient-cluster percentile intervals conditional on the fixed fitted models','class_fold_weights':result['class_fold_weights'].tolist(),'loss_fold_weights':result['loss_fold_weights'].tolist(),'files':files,'new_neural_work_performed_by_this_analysis':False,'input_inventories':{'controls_sha256':S.sha(S.MATERIALS/'controls.npz'),**({'training_sha256':S.sha(S.RESULTS/'training_complete.json'),'evaluation_sha256':S.sha(S.RESULTS/'evaluation_complete.json')} if final else {})}}
 S.atomic_json(out/'analysis_complete.json',report);return report

def verify_complete(out,status):
 out=Path(out);report=bound_report(out/'analysis_complete.json',status)
 S.verify_manifest(materials=True)
 if report['input_inventories']['controls_sha256']!=S.sha(S.MATERIALS/'controls.npz'):raise RuntimeError('Control inputs changed')
 if status=='ALL_EXTENSION_STATISTICS_PASS':
  for name,file in [('training_sha256','training_complete.json'),('evaluation_sha256','evaluation_complete.json')]:
   if report['input_inventories'][name]!=S.sha(S.RESULTS/file):raise RuntimeError('Analysis input inventory changed')
  for row in S.read(S.RESULTS/'evaluation_complete.json')['files']:
   if S.sha(S.RESULTS/row['prediction_file'])!=row['prediction_sha256']:raise RuntimeError('Input prediction artifact changed')
 for rec in report['files']:
  if S.sha(out/rec['path'])!=rec['sha256']:raise RuntimeError('Completed analysis changed')
 return report

def analyze_extension():
 out=S.RESULTS/'analysis'
 if (out/'analysis_complete.json').exists():return verify_complete(out,'ALL_EXTENSION_STATISTICS_PASS')
 if out.exists():
  from campaign_support import retain_failed
  retain_failed(out,'analysis')
 return save_analysis(assemble_new_predictions(),out,PAIRS,True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--controls-only',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
 if a.controls_only:
  if a.output is None:p.error('--controls-only requires a dedicated --output folder')
  report=save_analysis(archived_arrays(),a.output,[('full','image_only'),('full','late_fusion'),('late_fusion','image_only')],False)
 else:report=analyze_extension()
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
