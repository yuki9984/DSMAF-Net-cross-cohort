"""Analysis of all predeclared results; never selects seeds or checkpoints."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from followup_settings import *
from inference_core import probabilities
from stat_core import responses,summarize_effect,intervals
PAIRS=[('full','image_only'),('late_fusion','image_only'),('full','late_fusion')]
METRICS=[('bacc',0,'bacc',100),*[(f'recall_{c}',0,f'class_{i}',100) for i,c in enumerate(LABELS)],('sek_to_mel',1,'class_5',100),('nll',2,'mean',1),('brier',3,'mean',1)]

def weights(a):
 n=np.stack([np.bincount(a['internal_labels'][a['internal_folds']==f],minlength=6) for f in range(5)]);return n/n.sum(0),n.sum(1)/n.sum()

def recipe_values(a,ip,ep):
 wc,wi=weights(a);yi=a['internal_labels'];ye=a['external_labels'];vi=responses(yi,ip);ve=np.zeros(ep.shape[:2]+(len(ye),4))
 for f in range(5):
  z=responses(ye,ep[:,:,f]);ve[...,:2]+=z[...,:2]*wc[f,ye][None,None,:,None];ve[...,2:]+=z[...,2:]*wi[f]
 return vi,ve

def load_new_arrays(a):
 records=freeze_training();complete=read(RESULTS/'recipe45/evaluation_complete.json');assert complete['status']=='PASS' and complete['models']==45
 ip=np.full((3,3,len(a['internal_labels']),6),np.nan);ep=np.full((3,3,5,len(a['external_labels']),6),np.nan)
 for r in records:
  ai=ARMS.index(r['arm']);si=SEEDS.index(r['seed'])
  for cohort in ['internal','external']:
   p=RESULTS/'recipe45/evaluation'/cohort/(model_name(r)+'.npz');side=read(p.with_suffix('.json'));assert side['artifact_sha256']==sha256(p) and side['checkpoint_sha256']==r['checkpoint_sha256']
   with np.load(p,allow_pickle=False) as z:
    ids=z['img_ids'].astype(str);labels=z['labels'];patients=z['patient_ids'].astype(str);probs=probabilities(z['logits'])
   if cohort=='internal':
    idx=pd.Index(a['internal_images'].astype(str)).get_indexer(ids);assert (idx>=0).all() and np.array_equal(labels,a['internal_labels'][idx]) and np.array_equal(patients,a['internal_patients'][idx].astype(str));assert np.all(a['internal_folds'][idx]==r['fold']);ip[ai,si,idx]=probs
   else:
    assert np.array_equal(ids,a['external_images'].astype(str)) and np.array_equal(labels,a['external_labels']) and np.array_equal(patients,a['external_patients'].astype(str));ep[ai,si,r['fold']]=probs
 assert np.isfinite(ip).all() and np.isfinite(ep).all();return ip,ep

def analyze_recipe():
 a=original_arrays();newi,newe=load_new_arrays(a);oldi=a['prob_internal'][:,:3];olde=a['prob_external'][:,:3];values={r:recipe_values(a,i,e) for r,i,e in [('original',oldi,olde),('crossentropy_natural',newi,newe)]}
 effects=[{},{}];metrics=[]
 for recipe,(vi,ve) in values.items():
  for cohort,v,y in [('internal',vi,a['internal_labels']),('external',ve,a['external_labels'])]:
   for ai,arm in enumerate(ARMS):
    for si,seed in enumerate(SEEDS):metrics.append({'recipe':recipe,'cohort':cohort,'arm':arm,'seed':seed,**{name:summarize_effect(v[ai,si,:,col],y,kind)*scale for name,col,kind,scale in METRICS}})
  for first,second in PAIRS:
   ai,bi=ARMS.index(first),ARMS.index(second);name=first+'-minus-'+second
   for metric,col,kind,scale in METRICS:
    key=recipe+'|'+name+'|'+metric
    for cohort,v in enumerate([vi,ve]):effects[cohort][key]=((v[ai,:,:,col]-v[bi,:,:,col]).mean(0),kind,scale)
 # Recipe interactions are paired at case level and use the very same bootstrap draws.
 keys=list(effects[0])
 for key in keys:
  if key.startswith('original|'):
   other=key.replace('original|','crossentropy_natural|',1);newkey=key.replace('original|','recipe_difference|',1)
   for effects_for_cohort in effects:
    x,kind,scale=effects_for_cohort[other];old=effects_for_cohort[key][0];effects_for_cohort[newkey]=(x-old,kind,scale)
 rows,draws=intervals(*effects,a['internal_labels'],a['internal_patients'],a['external_labels'],a['external_patients'])
 out=RESULTS/'analysis/recipe45';out.mkdir(parents=True,exist_ok=True);pd.DataFrame(metrics).to_csv(out/'all_paired_three_seed_metrics.csv',index=False);pd.DataFrame(rows).to_csv(out/'all_recipe_contrasts_and_interactions.csv',index=False)
 metricframe=pd.DataFrame(metrics);metricframe.groupby(['recipe','cohort','arm'])[[m[0] for m in METRICS]].agg(['mean','std']).to_csv(out/'three_seed_means_and_sample_SDs.csv')
 perseed=[]
 for recipe in ['original','crossentropy_natural']:
  for first,second in PAIRS:
   for seed in SEEDS:
    pairs={}
    for cohort in ['internal','external']:
     z=metricframe[(metricframe.recipe==recipe)&(metricframe.cohort==cohort)&(metricframe.seed==seed)].set_index('arm');pairs[cohort]={metric:float(z.loc[first,metric]-z.loc[second,metric]) for metric,_,_,_ in METRICS}
    for target in ['internal','external','external_minus_internal']:
     point=pairs[target] if target in pairs else {m:pairs['external'][m]-pairs['internal'][m] for m,_,_,_ in METRICS};perseed.append({'recipe':recipe,'comparison':first+'-minus-'+second,'seed':seed,'target':target,**point})
 pd.DataFrame(perseed).to_csv(out/'all_seed_paired_contrasts.csv',index=False)
 atomic_npz(out/'paired_cluster_draws.npz',**draws)
 atomic_json(out/'complete.json',{'status':'PASS','training_models':45,'seed_metrics_rows':len(metrics),'contrast_interval_rows':len(rows),'paired_seeds':list(SEEDS),'original_five_seed_results_unchanged':True,'post_hoc':True,'conditional_fixed_models':True,'new_recipe_does_not_identify_loss_and_sampling_separately':True,'rejected_draws':[draws['rejected_internal'],draws['rejected_external']]})
 print('RECIPE45_ALL_CONTRASTS_PASS',len(rows),flush=True)

def analyze_interventions():
 a=original_arrays();wc,wi=weights(a);yi=a['internal_labels'];ye=a['external_labels'];globalcounts=np.bincount(yi,minlength=6)
 complete=read(RESULTS/'interventions/complete.json');assert complete['status']=='PASS' and len(complete['runs'])==150
 assert {(r['cohort'],r['fold'],r['arm'],r['seed']) for r in complete['runs']}=={(cohort,f,arm,s) for cohort in ['internal','external'] for f in range(5) for arm in ARMS for s in ALL_SEEDS}
 conditions=['sex_unknown','sex_recorded_permutation'];deltas={cohort:{c:np.zeros((3,5,len(y),5)) for c in conditions} for cohort,y in [('internal',yi),('external',ye)]};paths={cohort:np.zeros((3,5,100,6)) for cohort in ['internal','external']};covered=np.zeros((3,5,len(yi)),dtype=bool)
 for rec in complete['runs']:
  cohort=rec['cohort'];arm=rec['arm'];ai=ARMS.index(arm);si=ALL_SEEDS.index(rec['seed']);fold=rec['fold'];file=RESULTS/'interventions'/cohort/(model_name(rec)+'.npz');assert sha256(file)==rec['artifact_sha256']
  with np.load(file,allow_pickle=False) as z:
   ids=z['img_ids'].astype(str);y=z['labels'];patients=z['patient_ids'].astype(str)
   if cohort=='internal':
    idx=pd.Index(a['internal_images'].astype(str)).get_indexer(ids);assert (idx>=0).all() and np.array_equal(y,yi[idx]);assert np.array_equal(patients,a['internal_patients'][idx].astype(str));assert not covered[ai,si,idx].any();covered[ai,si,idx]=True;base=a['prob_internal'][ai,si,idx]
   else:
    idx=np.arange(len(ye));assert np.array_equal(ids,a['external_images'].astype(str)) and np.array_equal(y,ye);assert np.array_equal(patients,a['external_patients'].astype(str));base=a['prob_external'][ai,si,fold]
   baseline=responses(y,base);bpred=base.argmax(1)
   if arm=='image_only':
    changes={c:np.zeros((len(y),5)) for c in conditions};permcorrect=np.zeros((100,len(y)))
   else:
    pu=probabilities(z['logits_sex_unknown']);pp=probabilities(z['logits_sex_recorded_perm']);assert pp.shape==(100,len(y),6)
    du=responses(y,pu)-baseline;dp=responses(y,pp)-baseline[None,:,:];changes={'sex_unknown':np.column_stack((du,(pu.argmax(1)!=bpred).astype(float))),'sex_recorded_permutation':np.column_stack((dp.mean(0),(pp.argmax(-1)!=bpred[None,:]).mean(0)))};permcorrect=dp[:,:,0]
   for c in conditions:
    v=changes[c]
    if cohort=='internal':deltas[cohort][c][ai,si,idx]=v
    else:
     deltas[cohort][c][ai,si,:,:2]+=v[:,:2]*wc[fold,y,None];deltas[cohort][c][ai,si,:,2:]+=v[:,2:]*wi[fold]
   for c in range(6):
    mask=y==c;assert mask.any()
    contribution=permcorrect[:,mask].sum(1)/globalcounts[c] if cohort=='internal' else permcorrect[:,mask].mean(1)*wc[fold,c]
    paths[cohort][ai,si,:,c]+=contribution
 assert covered.all();effects=[{},{}];rows=[]
 meta_metrics=METRICS+[('prediction_flip_percent',4,'mean',100)]
 for ci,(cohort,y) in enumerate([('internal',yi),('external',ye)]):
  for condition in conditions:
   v=deltas[cohort][condition]
   for ai,arm in enumerate(ARMS):
    for si,seed in enumerate(ALL_SEEDS):rows.append({'cohort':cohort,'condition':condition,'arm':arm,'seed':seed,**{metric:summarize_effect(v[ai,si,:,col],y,kind)*scale for metric,col,kind,scale in meta_metrics}})
    for metric,col,kind,scale in meta_metrics:effects[ci][condition+'|'+arm+'|'+metric]=(v[ai,:,:,col].mean(0),kind,scale)
 intervalrows,draws=intervals(*effects,yi,a['internal_patients'],ye,a['external_patients']);out=RESULTS/'analysis/interventions';out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(out/'all_seed_intervention_deltas.csv',index=False);pd.DataFrame(intervalrows).to_csv(out/'all_intervention_patient_cluster_intervals.csv',index=False);atomic_npz(out/'paired_patient_cluster_draws.npz',**draws)
 summaries=[];rep_rows=[]
 for cohort in ['internal','external']:
  trajectory=paths[cohort].mean(-1)*100
  for ai,arm in enumerate(ARMS):
   perseed=trajectory[ai].mean(1);perrepeat=trajectory[ai].mean(0)
   summaries.append({'cohort':cohort,'arm':arm,'mean_delta_bacc_pp':float(perseed.mean()),'trained_seed_SD_pp':float(perseed.std(ddof=1)),'permutation_SD_of_five_seed_means_pp':float(perrepeat.std(ddof=1)),'MCSE_pp':float(perrepeat.std(ddof=1)/10),'repetitions':100,'trained_seeds':5})
   for si,seed in enumerate(ALL_SEEDS):
    assert abs(perseed[si]-next(r['bacc'] for r in rows if r['cohort']==cohort and r['condition']=='sex_recorded_permutation' and r['arm']==arm and r['seed']==seed))<1e-10
    for repeat,value in enumerate(trajectory[ai,si]):rep_rows.append({'cohort':cohort,'arm':arm,'seed':seed,'repeat':repeat,'delta_bacc_pp':float(value)})
 pd.DataFrame(summaries).to_csv(out/'permutation_seed_SD_permutation_SD_and_MCSE.csv',index=False);pd.DataFrame(rep_rows).to_csv(out/'all_repetition_BACC_deltas.csv',index=False)
 atomic_json(out/'complete.json',{'status':'PASS','model_cohort_records':150,'seed_delta_rows':len(rows),'patient_interval_rows':len(intervalrows),'repetition_rows':len(rep_rows),'post_hoc':True,'original_75_model_collection':True,'image_only_zero_effects_by_construction_and_neural_spot_check':True,'patient_intervals_condition_on_fixed_models_and_repetitions':True,'internal_permutations':'within held-out outer folds','external_permutations':'across all recorded cases, shared across fold models','rejected_draws':[draws['rejected_internal'],draws['rejected_external']]})
 print('INTERVENTION_ALL_RESULTS_PASS',len(intervalrows),flush=True)

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['interventions','recipe45','all'],default='all');a=ap.parse_args()
 if a.stage in ['interventions','all']:analyze_interventions()
 if a.stage in ['recipe45','all']:analyze_recipe()
