"""Analyze the complete fixed-model MILK run; requires all 135 prediction receipts."""
import argparse,pathlib
import numpy as np
import milk_common as C
import milk_statistics as M
import run_milk as E
PAIRS=[('dsmaf_age_recorded','dsmaf_age_only'),('full','dsmaf_age_recorded'),
       ('late_age_recorded','late_age_only'),('late_fusion','late_age_recorded'),
       ('late_fusion','image_only'),('full','image_only'),('metablock_cv','image_only'),
       ('rg_dermnet_adapted_cv','image_only'),('full','late_fusion'),('dsmaf_age_only','image_only')]
def analyze():
 C.setup().verify_manifest(materials=True)
 completed=C.read(C.ROOT/'evaluation/evaluation_complete.json')
 if completed['status']!='ALL_135_MILK_EVALUATIONS_VERIFIED' or len(completed['files'])!=135:raise ValueError('Complete model set is required')
 run=C.read(C.ROOT/'evaluation/run_lock.json');digest=C.json_hash(run)
 if completed['run_sha256']!=digest:raise ValueError('Run identity mismatch')
 if run['source_binding']!=E.source_binding():raise ValueError('Evaluation code changed')
 cases,idx,lock=C.check_cohort()
 if C.sha(C.ROOT/'prepared/cohort_lock.json')!=run['cohort_lock_sha256']:raise ValueError('Cohort lock changed')
 y=np.array([int(r['label']) for r in cases]);ids=np.array([r['isic_id'] for r in cases]);lesions=np.array([r['lesion_id'] for r in cases])
 with np.load(C.BUNDLE/'materials/controls.npz',allow_pickle=False) as z:
  cw,iw=M.fold_weights(z['internal_labels'],z['internal_folds'])
 records={(r['fold'],r['arm'],r['seed']):r for r in run['records']}
 effective=np.empty((len(C.ARMS),len(C.SEEDS),len(y),3))
 fold_rows=[]
 for ai,arm in enumerate(C.ARMS):
  for si,seed in enumerate(C.SEEDS):
   probs=[]
   for fold in range(5):
    r=records[(fold,arm,seed)]
    binding=C.json_hash(dict(run=digest,checkpoint=r['checkpoint_sha256'],config=C.json_hash(r['config'])))
    receipt=E.existing_prediction(r,ids,y,lesions,binding)
    if receipt is None:raise ValueError('Missing receipt')
    path=C.ROOT/receipt['path']
    with np.load(path,allow_pickle=False) as z:
     p=z['probabilities'].copy();probs.append(p)
    values=M.metric_vector(y,M.response(y,p))
    fold_rows.extend(dict(arm=arm,seed=seed,fold=fold,metric=metric,value=float(value)) for metric,value in zip(M.METRICS,values))
   effective[ai,si]=M.collapse_folds(y,np.stack(probs),cw,iw)
 target=C.ROOT/'evaluation/analysis'
 if target.exists():raise FileExistsError('Preserve previous analysis; inspect it before regenerating')
 target.mkdir()
 C.write_csv(target/'all_135_model_metrics.csv',fold_rows)
 np.savez_compressed(target/'effective_responses.npz',arms=np.array(C.ARMS),seeds=np.array(C.SEEDS),labels=y,lesion_ids=lesions,responses=effective,class_fold_weights=cw,loss_fold_weights=iw)
 subsets={'all_eligible':np.ones(len(y),bool),'histopathology_only':np.array([r['diagnosis_confirm_type']=='histopathology' for r in cases])}
 subset_reports={}
 for name,mask in subsets.items():
  if set(y[mask])!=set(range(6)):
   subset_reports[name]=dict(status='NOT_ESTIMATED_MISSING_CLASS',support=np.bincount(y[mask],minlength=6).tolist());continue
  result=M.summarize(list(C.ARMS),list(C.SEEDS),y[mask],lesions[mask],effective[:,:,mask],PAIRS,repeats=2000,seed=20261104)
  for row in result['summary']+result['by_seed']:row['subset']=name
  C.write_csv(target/(name+'_summary.csv'),result['summary']);C.write_csv(target/(name+'_all_seeds.csv'),result['by_seed'])
  np.savez_compressed(target/(name+'_resamples.npz'),arms=np.array(C.ARMS),metrics=np.array(M.METRICS),seed_metrics=result['level_seed_metrics'],bootstrap=result['level_bootstrap_draws'])
  subset_reports[name]=dict(status='COMPLETE',images=int(mask.sum()),support=np.bincount(y[mask],minlength=6).tolist())
 C.save(target/'analysis_receipt.json',dict(status='COMPLETE',run_sha256=digest,model_count=135,subsets=subset_reports,
       bootstrap_repeats=2000,bootstrap_seed=20261104,unit='lesion_not_patient',class_stratified=True,probability_ensemble=False,
       conditioning='fixed fitted models and fixed released cohort, not new-training uncertainty; no multiplicity adjustment',
       source_sha256=C.sha(pathlib.Path(__file__)),files={p.name:C.sha(p) for p in target.iterdir() if p.is_file()}))
 print('Complete saved-prediction analysis:',target)
if __name__=='__main__':analyze()

