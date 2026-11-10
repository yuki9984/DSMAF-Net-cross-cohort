"""Verify a completed FOLLOWUP_RESULTS archive without weights or executing its code."""
from pathlib import Path,PurePosixPath
import argparse,csv,hashlib,json,re,stat,sys,zipfile
import numpy as np,pandas as pd
from receiver_statistics import *
P=Path(__file__).resolve().parent;BUNDLE=P/'inputs'
PLAN_HASH='da91d6e46808af2c2fd8ff892555d53bcbc5d321fc0890c74de6641f83733e38'
SOURCE_HASH='b39432c83ebba8871c4d49ad6e2d20098cf78121e4a194f17048c6796c0df67f'
ARMS=('full','image_only','late_fusion');SEEDS=tuple(range(20261013,20261018));NEW_SEEDS=SEEDS[:3]
def require(ok,message):
 if not ok:raise ValueError(message)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def name(r):return f"fold_{r['fold']:02d}_{r['arm']}_seed{r['seed']}"
def key(r):return r['fold'],r['arm'],r['seed']
def npz(p):
 with zipfile.ZipFile(p) as z:require(sum(i.file_size for i in z.infolist())<64*1024**2,'Unexpected NPZ allocation size')
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def safe_extract(archive,destination):
 destination=Path(destination);require(not destination.exists(),'Use a new empty extraction directory')
 with zipfile.ZipFile(archive) as z:
  infos=z.infolist();require(len(infos)<=4000 and sum(i.file_size for i in infos)<=4*1024**3,'Unexpected return size');seen=set()
  for i in infos:
   n=i.filename;p=PurePosixPath(n)
   require(i.orig_filename==n,'Noncanonical raw archive path')
   require(not p.is_absolute() and p.parts and p.parts[0]=='FOLLOWUP_RESULTS' and '..' not in p.parts and '\\' not in n and ':' not in n,'Unsafe archive path')
   require(n.casefold() not in seen,'Duplicate or case-colliding archive member');seen.add(n.casefold())
   require(not stat.S_ISLNK(i.external_attr>>16),'Symlink member is unsupported')
   require(i.file_size<512*1024**2,'Unexpected single-member size')
   for part in p.parts:require(part.rstrip(' .')==part and not re.fullmatch(r'(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?',part),'Unsafe Windows path component')
   require(i.is_dir() or p.suffix.lower() in ['.json','.csv','.npz','.log','.txt','.py'],'Unexpected return file type')
  require(z.testzip() is None,'ZIP CRC check failed')
  destination.mkdir(parents=True)
  for i in infos:
   target=destination/Path(*PurePosixPath(i.filename).parts)
   if i.is_dir():target.mkdir(parents=True,exist_ok=True)
   else:
    target.parent.mkdir(parents=True,exist_ok=True)
    with z.open(i) as src,target.open('wb') as dst:
     import shutil
     shutil.copyfileobj(src,dst)
 return destination/'FOLLOWUP_RESULTS'

def original_inputs():
 require(sha(BUNDLE/'analysis_plan.json')==PLAN_HASH and sha(BUNDLE/'source_manifest.json')==SOURCE_HASH,'Pinned plan/source package changed')
 for e in read(P/'INPUT_MANIFEST.json')['files']:require(sha(P/e['path'])==e['sha256'],'Projected input changed: '+e['path'])
 plan=read(BUNDLE/'analysis_plan.json');require(sha(BUNDLE/'baseline_probabilities.npz')==plan['baseline_projection_sha256'],'Original prediction projection changed')
 a=npz(BUNDLE/'baseline_probabilities.npz');require(a['arms'].astype(str).tolist()==list(ARMS) and a['seeds'].tolist()==list(SEEDS),'Baseline arm/seed order changed')
 prior=BUNDLE/'prior_campaign/inputs';old=read(prior/'cv/training_complete.json')['runs'];require(len(old)==75 and len({key(r) for r in old})==75,'Original model inventory incomplete')
 manifest=read(prior/'splits/fold_manifest.json');frames={};trainframes={}
 for fold in range(5):
  for role,dest in [('test',frames),('train',trainframes)]:
   info=manifest['folds'][fold]['roles'][role];file=prior/'splits'/info['file'];require(sha(file)==info['sha256'],'Frozen split changed');dest[fold]=pd.read_csv(file)
 external=pd.read_csv(prior/'external_cases.csv');return a,{key(r):r for r in old},frames,trainframes,external

def check_reproduction(check):
 require(isinstance(check,dict) and np.isfinite(check.get('max_abs_probability_error',np.nan)) and 0<=check['max_abs_probability_error']<=1e-4 and check.get('changed_hard_predictions')==0,'Reported GPU reproduction/equivalence gate failed')

def check_permutations(root,frames,external):
 hashes={}
 for cohort,fold,frame in [('internal',f,frames[f]) for f in range(5)]+[('external',0,external)]:
  p=root/'interventions/permutation_indices'/(f'internal_fold_{fold:02d}.npz' if cohort=='internal' else 'external.npz');z=npz(p)
  require(str(z['plan_sha256'])==PLAN_HASH and np.array_equal(z['img_ids'].astype(str),frame.img_id.astype(str)),'Permutation ID/plan mismatch')
  known=np.flatnonzero((frame.gender_FEMALE+frame.gender_MALE).to_numpy()==1) if cohort=='internal' else np.flatnonzero(frame.gender.isin(['FEMININO','MASCULINO']).to_numpy())
  expected=[]
  for repeat in range(100):
   rng=np.random.default_rng(np.random.SeedSequence([2026102301,0 if cohort=='internal' else 1,fold if cohort=='internal' else 0,repeat]));indices=np.arange(len(frame),dtype=np.int32);indices[known]=rng.permutation(known);expected.append(indices)
  expected=np.stack(expected);require(np.array_equal(z['indices'],expected),'Permutation indices fail frozen deterministic/state-preserving rule')
  hashes[cohort,fold]=hashlib.sha256(expected.tobytes()).hexdigest()
 return hashes

def aligned_prediction(z,frame,a,cohort,fold):
 require(np.array_equal(z['img_ids'].astype(str),frame.img_id.astype(str)) and len(set(z['img_ids'].astype(str)))==len(frame),'Prediction image identities/order differ')
 require(np.array_equal(z['patient_ids'].astype(str),frame.patient_id.astype(str)),'Prediction patient mapping differs')
 labels=frame['diagnostic_number' if cohort=='internal' else 'label'].to_numpy(int);require(np.array_equal(z['labels'],labels),'Prediction labels differ')
 if cohort=='internal':
  idx=pd.Index(a['internal_images'].astype(str)).get_indexer(z['img_ids'].astype(str));require(np.all(idx>=0) and np.all(a['internal_folds'][idx]==fold),'Wrong outer-fold cases')
  require(np.array_equal(labels,a['internal_labels'][idx]),'Original OOF labels differ')
 else:idx=np.arange(len(labels));require(np.array_equal(z['img_ids'].astype(str),a['external_images'].astype(str)),'External cohort changed')
 return labels,idx

def intervention_effects(root,a,old,frames,external):
 complete=read(root/'interventions/complete.json');runs=complete['runs']
 require(complete.get('status')=='PASS' and complete.get('plan_sha256')==PLAN_HASH and len(runs)==150,'Incomplete intervention records')
 wanted={(c,f,arm,s) for c in ['internal','external'] for f in range(5) for arm in ARMS for s in SEEDS}
 require({(r['cohort'],*key(r)) for r in runs}==wanted,'Intervention model/cohort inventory differs')
 permutation_hashes=check_permutations(root,frames,external);wc,wi=fold_weights(a)
 effects={cohort:{condition:np.zeros((3,5,len(a[cohort+'_labels']),5)) for condition in ['sex_unknown','sex_recorded_permutation']} for cohort in ['internal','external']}
 paths={c:np.zeros((3,5,100,6)) for c in ['internal','external']};covered=np.zeros((3,5,len(a['internal_labels'])),bool)
 max_error=0
 for rec in runs:
  cohort,fold,arm,seed=rec['cohort'],rec['fold'],rec['arm'],rec['seed'];ai=ARMS.index(arm);si=SEEDS.index(seed);frame=frames[fold] if cohort=='internal' else external
  p=root/'interventions'/cohort/(name(rec)+'.npz');require(sha(p)==rec['artifact_sha256'],'Intervention artifact hash changed');require(read(p.with_suffix('.json'))==rec,'Intervention sidecar differs')
  for k,v in old[key(rec)].items():require(rec.get(k)==v,'Original checkpoint/config binding changed: '+k)
  z=npz(p);lock=dict(plan_sha256=PLAN_HASH,checkpoint_sha256=old[key(rec)]['checkpoint_sha256'],source_manifest_sha256=SOURCE_HASH,cohort=cohort,model=name(rec),permutation_indices_sha256=permutation_hashes[cohort,fold if cohort=='internal' else 0])
  require(json.loads(str(z['provenance']))==lock,'Intervention provenance differs')
  checks=json.loads(str(z['checks']));require(all(rec.get(k)==v for k,v in checks.items()),'Embedded reproduction checks differ')
  check_reproduction(checks['observed_direct_vs_original'])
  if arm=='image_only':require(checks['metadata_invariant_by_construction'] is True,'Image-only invariance not recorded');check_reproduction(checks['image_only_unknown_spot_check'])
  else:
   check_reproduction(checks['cached_vs_direct']);check_reproduction(checks['cached_vs_original'])
   require(set(checks['perturbed_direct_spot_checks'])=={'sex_unknown','sex_recorded_perm_000','sex_recorded_perm_050','sex_recorded_perm_099'},'GPU cache spot checks incomplete')
   for check in checks['perturbed_direct_spot_checks'].values():check_reproduction(check)
  y,idx=aligned_prediction(z,frame,a,cohort,fold);baseline=a['prob_internal'][ai,si,idx] if cohort=='internal' else a['prob_external'][ai,si,fold]
  observed=softmax(z['logits_observed']);require(observed.shape==baseline.shape,'Observed shape differs');error=float(np.abs(observed-baseline).max());max_error=max(error,max_error)
  require(error<=1e-4 and np.array_equal(observed.argmax(-1),baseline.argmax(-1)),'Observed predictions do not reproduce original models')
  reference=case_values(y,baseline)
  if arm=='image_only':deltas={c:np.zeros((len(y),5)) for c in effects[cohort]};repeated_correct=np.zeros((100,len(y)))
  else:
   unknown=softmax(z['logits_sex_unknown']);permuted=softmax(z['logits_sex_recorded_perm']);require(unknown.shape==(len(y),6) and permuted.shape==(100,len(y),6),'Intervention logits shape differs')
   ud=case_values(y,unknown)-reference;pdiff=case_values(y,permuted)-reference[None]
   deltas={'sex_unknown':np.column_stack([ud,unknown.argmax(-1)!=baseline.argmax(-1)]),'sex_recorded_permutation':np.column_stack([pdiff.mean(0),(permuted.argmax(-1)!=baseline.argmax(-1)).mean(0)])};repeated_correct=pdiff[:,:,0]
  if cohort=='internal':require(not covered[ai,si,idx].any(),'Duplicate OOF intervention cases');covered[ai,si,idx]=True
  for condition,v in deltas.items():
   if cohort=='internal':effects[cohort][condition][ai,si,idx]=v
   else:effects[cohort][condition][ai,si,:,:2]+=v[:,:2]*wc[fold,y,None];effects[cohort][condition][ai,si,:,2:]+=v[:,2:]*wi[fold]
  for c in range(6):
   mask=y==c;require(mask.any(),'Empty expected diagnostic class')
   paths[cohort][ai,si,:,c]+=repeated_correct[:,mask].sum(1)/np.bincount(a['internal_labels'],minlength=6)[c] if cohort=='internal' else repeated_correct[:,mask].mean(1)*wc[fold,c]
 require(covered.all(),'OOF intervention coverage incomplete')
 return effects,paths,max_error

def validate_selection_history(r,h,s):
 scores=np.array([e['valid']['bac'] for e in h['epochs']],float);require(len(scores)>0 and len(scores)<=150 and np.isfinite(scores).all() and np.all((scores>=0)&(scores<=1)),'Invalid selection history')
 require([e['epoch'] for e in h['epochs']]==list(range(1,len(scores)+1)),'Selection epochs nonsequential')
 first=int(scores.argmax())+1;require(first==r['best_epoch']==h['best_epoch']==s['best_epoch'] and s['epochs_completed']==len(scores),'First-maximum selection rule failed')
 require(len(scores)==150 or len(scores)-first==15,'Early-stopping history incompatible')
 require(abs(float(scores.max())-s['best_valid_bac'])<1e-12 and abs(float(scores.max())-h['best_valid_bac'])<1e-12,'Best selection score inconsistent')
 require(h.get('external_evaluated') is False and s.get('external_evaluated') is False and s.get('plan_sha256')==PLAN_HASH,'External selection boundary differs')

def validate_recipe_config(c,previous):
 excluded={'protocol','plan_sha256','source_manifest_sha256','loss','oversample_scc','supplemental_trainer_sha256'}
 for k,v in previous.items():
  if k not in excluded:require(c.get(k)==v,'Unapproved common training setting changed: '+k)
 for k,v in {'protocol':'posthoc_recipe45_crossentropy_natural_20261023','plan_sha256':PLAN_HASH,'source_manifest_sha256':SOURCE_HASH,'loss':'ordinary_cross_entropy','oversample_scc':False,'supplemental_trainer_sha256':sha(BUNDLE/'train_recipe.py')}.items():require(c.get(k)==v,'New recipe binding changed: '+k)

def new_recipe_arrays(root,a,old,frames,external):
 training=root/'recipe45/training_complete.json';tc=read(training);runs=tc['runs'];require(tc.get('status')=='PASS' and tc.get('run_count')==45 and tc.get('plan_sha256')==PLAN_HASH and tc.get('outer_test_evaluated') is False and tc.get('external_evaluated') is False,'45 training fits not frozen')
 wanted={(f,arm,s) for f in range(5) for arm in ARMS for s in NEW_SEEDS};require(len(runs)==45 and {key(r) for r in runs}==wanted,'45-fit inventory differs')
 inventory=read(root/'checkpoint_inventory45.json');require(inventory.get('new_best_weights')==45 and len(inventory['files'])==45,'New weight hash inventory incomplete');weight_files={r['path']:r for r in inventory['files']}
 for r in runs:
  d=root/'recipe45'/f"fold_{r['fold']:02d}"/f"{r['arm']}_seed{r['seed']}";c=read(d/'config.json');h=read(d/'history.json');s=read(d/'summary.json');previous=read(BUNDLE/'prior_campaign/inputs/cv'/f"fold_{r['fold']:02d}"/f"{r['arm']}_seed{r['seed']}"/'config.json')
  for file,obj in [('config',c),('history',h),('summary',s)]:require(sha(d/(file+'.json'))==r[file+'_sha256'],'New training record hash changed')
  validate_recipe_config(c,previous)
  validate_selection_history(r,h,s)
  inv=weight_files.get(f"recipe45/fold_{r['fold']:02d}/{r['arm']}_seed{r['seed']}/best_valid.pth");require(inv and inv['sha256']==r['checkpoint_sha256'] and inv['bytes']>0,'New retained-weight hash binding differs')
  done=read(d/'DONE.json');require(all(done.get(k)==r[k] for k in ['fold','arm','seed','checkpoint_sha256','config_sha256','history_sha256','summary_sha256']),'DONE training binding differs')
 ec=read(root/'recipe45/evaluation_complete.json');er=ec['runs'];require(ec.get('status')=='PASS' and ec.get('models')==45 and len(er)==90 and ec.get('training_complete_sha256')==sha(training),'Evaluation not bound to all frozen fits')
 expected={(c,*k) for c in ['internal','external'] for k in wanted};require({(r['cohort'],*key(r)) for r in er}==expected,'Evaluation records incomplete')
 im=np.full((3,3,len(a['internal_labels']),6),np.nan);ex=np.full((3,3,5,len(a['external_labels']),6),np.nan);bykey={key(r):r for r in runs}
 for rec in er:
  cohort=rec['cohort'];f,arm,seed=key(rec);ai=ARMS.index(arm);si=NEW_SEEDS.index(seed);p=root/'recipe45/evaluation'/cohort/(name(rec)+'.npz')
  require(sha(p)==rec['artifact_sha256'] and read(p.with_suffix('.json'))==rec,'New prediction binding differs');z=npz(p)
  lock={**bykey[key(rec)],'plan_sha256':PLAN_HASH,'training_complete_sha256':sha(training),'cohort':cohort};require(json.loads(str(z['provenance']))==lock,'New prediction provenance differs')
  y,idx=aligned_prediction(z,frames[f] if cohort=='internal' else external,a,cohort,f);prob=softmax(z['logits']);require(prob.shape==(len(y),6),'New six-class logits shape differs')
  if cohort=='internal':require(np.isnan(im[ai,si,idx]).all(),'New OOF duplicate');im[ai,si,idx]=prob
  else:require(np.isnan(ex[ai,si,f]).all(),'New external duplicate');ex[ai,si,f]=prob
 require(np.isfinite(im).all() and np.isfinite(ex).all(),'New recipe predictions incomplete');return im,ex

def compare_csv(produced,path,keys,values):
 ours=pd.DataFrame(produced).set_index(keys).sort_index();worker=pd.read_csv(path).set_index(keys).sort_index()
 require(ours.index.is_unique and worker.index.is_unique and ours.index.equals(worker.index),'Summary row identities differ: '+path.name)
 x=ours[values].to_numpy(float);y=worker[values].to_numpy(float);require(np.isfinite(y).all() and np.allclose(x,y,rtol=0,atol=1e-8),'Recomputed summary differs: '+path.name)
 return dict(file=path.as_posix(),rows=len(ours),maximum_absolute_error=float(np.abs(x-y).max(initial=0)))

def compare_draws(ours,path):
 other=npz(path)
 for k,v in ours.items():
  if k=='names':require(list(other[k].astype(str))==v,'Bootstrap endpoint ordering differs')
  else:require(np.allclose(np.asarray(v),other[k],rtol=0,atol=1e-8),'Bootstrap draws differ: '+k)

def run(root,out):
 a,old,frames,trainframes,external=original_inputs()
 # Source snapshot is bound to the frozen manifest; excluded worker logs are not replay inputs.
 for e in read(BUNDLE/'source_manifest.json')['files']:
  p=P/'neural_source'/e['path']
  if p.is_file():require(sha(p)==e['sha256'],'Pinned source changed: '+e['path'])
 status=read(root/'campaign_status.json');require(status.get('all_complete') is True and status.get('source_manifest_sha256')==SOURCE_HASH and status.get('plan_sha256')==PLAN_HASH,'Full return not complete or not bound to the approved plan')
 require(status['stage_status']==dict.fromkeys(['interventions','training45','evaluation45','intervention_analysis','recipe_analysis'],'PASS'),'Incomplete or unexpected campaign stages')
 runtime=read(root/'input_audit/runtime.json');require(runtime.get('status')=='PASS' and all(runtime.get(k)==v for k,v in [('python','3.11.9'),('torch','2.6.0+cu124'),('torchvision','0.21.0+cu124'),('cuda','12.4')]) and '4060' in runtime.get('gpu',''),'Worker runtime differs')
 verification=read(root/'input_audit/input_verification.json');require(verification.get('status')=='PASS' and verification.get('weights')==75 and verification.get('PAD_images')==2298 and verification.get('external_images')==1701,'Reported original input audit incomplete')
 deltas,trajectories,max_error=intervention_effects(root,a,old,frames,external);ip,ep=new_recipe_arrays(root,a,old,frames,external)
 rows,paired,intervals,draws=recipe_statistics(a,ip,ep);reports=[];folder=root/'analysis/recipe45'
 reports.append(compare_csv(rows,folder/'all_paired_three_seed_metrics.csv',['recipe','cohort','arm','seed'],[m[0] for m in METRICS]));reports.append(compare_csv(paired,folder/'all_seed_paired_contrasts.csv',['recipe','comparison','seed','target'],[m[0] for m in METRICS]));reports.append(compare_csv(intervals,folder/'all_recipe_contrasts_and_interactions.csv',['key','target'],['estimate','ci95_low','ci95_high']));compare_draws(draws,folder/'paired_cluster_draws.npz')
 for label,data in [('recipe_seed_metrics',rows),('recipe_seed_contrasts',paired),('recipe_intervals',intervals)]:pd.DataFrame(data).to_csv(out/(label+'.csv'),index=False)
 seedrows=[];effects=[{},{}];summary=[];repetitions=[];specs=METRICS+[('prediction_flip_percent',4,'mean',100)]
 for ci,cohort in enumerate(['internal','external']):
  y=a[cohort+'_labels']
  for condition,v in deltas[cohort].items():
   for ai,arm in enumerate(ARMS):
    for si,seed in enumerate(SEEDS):seedrows.append(dict(cohort=cohort,condition=condition,arm=arm,seed=seed,**metric_row(v[ai,si],y,True)))
    for metric,col,kind,scale in specs:effects[ci][condition+'|'+arm+'|'+metric]=(v[ai,:,:,col].mean(0),kind,scale)
  values=trajectories[cohort].mean(-1)*100
  for ai,arm in enumerate(ARMS):
   perseed=values[ai].mean(1);perrepeat=values[ai].mean(0);sd=float(perrepeat.std(ddof=1))
   summary.append(dict(cohort=cohort,arm=arm,mean_delta_bacc_pp=float(perseed.mean()),trained_seed_SD_pp=float(perseed.std(ddof=1)),permutation_SD_of_five_seed_means_pp=sd,MCSE_pp=sd/10,repetitions=100,trained_seeds=5))
   for si,seed in enumerate(SEEDS):
    for repeat,value in enumerate(values[ai,si]):repetitions.append(dict(cohort=cohort,arm=arm,seed=seed,repeat=repeat,delta_bacc_pp=float(value)))
 intervals,draws=paired_intervals(*effects,a);folder=root/'analysis/interventions'
 reports.append(compare_csv(seedrows,folder/'all_seed_intervention_deltas.csv',['cohort','condition','arm','seed'],[m[0] for m in specs]));reports.append(compare_csv(intervals,folder/'all_intervention_patient_cluster_intervals.csv',['key','target'],['estimate','ci95_low','ci95_high']));reports.append(compare_csv(summary,folder/'permutation_seed_SD_permutation_SD_and_MCSE.csv',['cohort','arm'],['mean_delta_bacc_pp','trained_seed_SD_pp','permutation_SD_of_five_seed_means_pp','MCSE_pp','repetitions','trained_seeds']));reports.append(compare_csv(repetitions,folder/'all_repetition_BACC_deltas.csv',['cohort','arm','seed','repeat'],['delta_bacc_pp']));compare_draws(draws,folder/'paired_patient_cluster_draws.npz')
 for label,data in [('intervention_seed_metrics',seedrows),('intervention_intervals',intervals),('permutation_summary',summary),('permutation_repetitions',repetitions)]:pd.DataFrame(data).to_csv(out/(label+'.csv'),index=False)
 return dict(status='SAVED_PREDICTIONS_AND_RECORDS_VERIFIED',model_cohort_intervention_records=150,new_training_records=45,new_model_cohort_prediction_records=90,observed_probability_max_error=max_error,summary_checks=reports,bootstrap_draws='Both 2000-draw paired interval sets reproduced',new_weight_tensor_contents='Not returned or loaded; hash inventories/config/history/prediction bindings checked only',GPU_claim_scope='Reported source-pinned worker gates checked; no independent neural rerun',no_new_training=True,no_neural_inference=True,returned_code_executed=False,submission_ready=False)

def main():
 out=P.parent/'verification_output/followup';out.mkdir(parents=True,exist_ok=True)
 report=run(P/'returned',out)
 report['archive_scope']='Projected saved outputs and records; no private worker logs, raw images, or new weight tensors.'
 write(out/'verification_report.json',report)
 print('FOLLOWUP_SAVED_LOGITS_RECORDS_AND_PAIRED_INTERVALS_PASS',flush=True)
if __name__=='__main__':main()
