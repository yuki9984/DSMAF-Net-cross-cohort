"""Post hoc evaluation restriction for multiple ages under a released patient ID."""
from pathlib import Path
import argparse,csv,hashlib,json
from collections import defaultdict
import numpy as np
import pandas as pd
ARMS=['full','image_only','late_fusion']
PAIRS=[('full','image_only'),('late_fusion','image_only'),('full','late_fusion')]
LABELS=['ACK','BCC','MEL','NEV','SCC','SEK']
SEEDS=list(range(20261013,20261018))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def calculate(root,output):
 root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 paths=[root/'data/PAD_metadata.csv',root/'internal75/pooled_predictions.npz',root/'matched_cohort/probabilities.npz']
 raw=pd.read_csv(paths[0],dtype={'patient_id':str,'img_id':str},keep_default_na=False)
 raw['age']=pd.to_numeric(raw.age,errors='raise');assert np.isfinite(raw.age).all()
 assert len(raw)==2298 and raw.img_id.is_unique and raw.patient_id.nunique()==1373
 grouped=raw.groupby('patient_id').age
 varying=sorted(grouped.nunique()[lambda s:s>1].index.tolist())
 # Independently derive the rule with CSV parsing and exact numerical age sets.
 age_sets=defaultdict(set);raw_rows=list(csv.DictReader(paths[0].open(encoding='utf-8-sig',newline='')))
 for r in raw_rows:age_sets[r['patient_id']].add(float(r['age']))
 assert varying==sorted(p for p,a in age_sets.items() if len(a)>1)
 z=np.load(paths[1],allow_pickle=False);images=z['img_ids'].astype(str);patients=z['patient_ids'].astype(str);y=z['labels'];folds=z['outer_folds']
 index=raw.set_index('img_id').loc[images]
 assert np.array_equal(index.patient_id.to_numpy(),patients)
 assert np.array_equal(index.diagnostic.map({c:i for i,c in enumerate(LABELS)}),y)
 keep=~np.isin(patients,varying)
 assert len(varying)==90 and int((~keep).sum())==316 and int(keep.sum())==1982 and len(set(patients[keep]))==1283
 with np.load(paths[2],allow_pickle=False) as other:
  for a,b in [('img_ids','internal_images'),('patient_ids','internal_patients'),('labels','internal_labels'),('outer_folds','internal_folds')]:assert np.array_equal(z[a],other[b])
  assert other['seeds'].tolist()==SEEDS
  for arm in ARMS:
   ai=other['arms'].tolist().index(arm)
   assert np.array_equal(z['probs_'+arm],other['prob_internal'][ai])
 hits={}
 for arm in ARMS:
  p=z['probs_'+arm];assert p.shape==(5,2298,6) and np.isfinite(p).all() and p.min()>=0 and np.max(abs(p.sum(-1)-1))<1e-12
  hits[arm]=(p.argmax(-1)==y).astype(float)
 mask=pd.DataFrame({'img_id':images,'patient_id':patients,'outer_fold':folds,'diagnosis':[LABELS[v] for v in y],'raw_age':index.age.to_numpy(),'age_varies_within_released_patient':~keep,'retained_for_evaluation':keep})
 mask.to_csv(output/'evaluation_mask.csv',index=False)
 excluded=[]
 for pid in varying:
  rr=raw[raw.patient_id.eq(pid)]
  excluded.append({'patient_id':pid,'images':len(rr),'distinct_ages':';'.join(str(a) for a in sorted(age_sets[pid])),'age_range_years':max(age_sets[pid])-min(age_sets[pid])})
 pd.DataFrame(excluded).to_csv(output/'excluded_patient_ids.csv',index=False)
 counts=[]
 for subset,take in [('all_cases',np.ones(len(y),bool)),('restricted',keep),('excluded',~keep)]:
  for ci,label in enumerate(LABELS):
   sel=take&(y==ci);counts.append({'subset':subset,'diagnosis':label,'images':int(sel.sum()),'patients_with_this_diagnosis':len(set(patients[sel]))})
  counts.append({'subset':subset,'diagnosis':'All classes','images':int(take.sum()),'patients_with_this_diagnosis':len(set(patients[take]))})
 pd.DataFrame(counts).to_csv(output/'class_support.csv',index=False)
 foldcounts=[]
 for f in range(5):
  for subset,take in [('all_cases',np.ones(len(y),bool)),('restricted',keep),('excluded',~keep)]:
   sel=take&(folds==f);foldcounts.append({'fold':f,'subset':subset,'images':int(sel.sum()),'patients':len(set(patients[sel]))})
 pd.DataFrame(foldcounts).to_csv(output/'fold_support.csv',index=False)
 seedrows=[]
 for subset,take in [('all_cases',np.ones(len(y),bool)),('restricted',keep)]:
  for arm in ARMS:
   for si,seed in enumerate(SEEDS):
    recalls=np.array([hits[arm][si,take&(y==c)].mean()*100 for c in range(6)])
    seedrows.append({'subset':subset,'arm':arm,'seed':seed,'bacc_percent':float(recalls.mean()),**{f'recall_{LABELS[c]}':float(recalls[c]) for c in range(6)}})
 df=pd.DataFrame(seedrows);df.to_csv(output/'model_seed_metrics.csv',index=False)
 seedcontrasts=[];contrasts=[]
 yr=y[keep];pr=patients[keep];patient,inv=np.unique(pr,return_inverse=True);n=len(patient)
 delta=np.stack([(hits[a]-hits[b]).mean(0)[keep] for a,b in PAIRS],axis=1)
 count=np.zeros((n,6),dtype=np.int64);np.add.at(count,(inv,yr),1)
 sums=np.zeros((n,6,len(PAIRS)))
 for j in range(len(PAIRS)):np.add.at(sums[:,:,j],(inv,yr),delta[:,j])
 rng=np.random.default_rng(20261019);draws=[];weights=[];rejected=0
 while len(draws)<2000:
  selected=rng.integers(0,n,size=n);den=count[selected].sum(0)
  if (den==0).any():rejected+=1;continue
  draws.append((sums[selected].sum(0)/den[:,None]).mean(0)*100)
  weights.append(np.bincount(selected,minlength=n))
 draws=np.stack(draws);weights=np.stack(weights)
 # Independent multinomial-count contraction of the same patient draws.
 den=weights@count
 independent=np.stack([((weights@sums[:,:,j])/den).mean(1)*100 for j in range(len(PAIRS))],axis=1)
 maxerror=float(np.max(abs(draws-independent)));assert maxerror<1e-10
 for j,(a,b) in enumerate(PAIRS):
  v={}
  for subset in ['all_cases','restricted']:
   aa=df[df.subset.eq(subset)&df.arm.eq(a)].sort_values('seed').bacc_percent.to_numpy()
   bb=df[df.subset.eq(subset)&df.arm.eq(b)].sort_values('seed').bacc_percent.to_numpy()
   d=aa-bb;v[subset]=d
   for seed,value in zip(SEEDS,d):seedcontrasts.append({'subset':subset,'contrast':a+'-minus-'+b,'seed':seed,'bacc_difference_pp':float(value)})
  lo,hi=np.quantile(draws[:,j],[.025,.975])
  contrasts.append({'contrast':a+'-minus-'+b,'all_cases_mean_pp':float(v['all_cases'].mean()),'restricted_mean_pp':float(v['restricted'].mean()),'restricted_seed_sd_pp':float(v['restricted'].std(ddof=1)),'restricted_ci95_low_pp':float(lo),'restricted_ci95_high_pp':float(hi)})
 pd.DataFrame(seedcontrasts).to_csv(output/'seed_contrasts.csv',index=False)
 pd.DataFrame(contrasts).to_csv(output/'paired_contrasts.csv',index=False)
 np.savez_compressed(output/'bootstrap_draws.npz',draws_pp=draws,pairs=np.array([a+'-minus-'+b for a,b in PAIRS]),seed=np.array(20261019),repeats=np.array(2000))
 # Cross-check unfiltered results against the previously archived per-seed metrics.
 expected=pd.read_csv(root/'internal75/expected_metrics_per_seed.csv')
 joined=df[df.subset.eq('all_cases')].merge(expected,on=['arm','seed'])
 baseline_error=float(np.max(abs(joined.bacc_percent-100*joined.bacc)))
 assert len(joined)==15 and baseline_error<1e-10
 report={'status':'PASS','post_hoc':True,'original_images':2298,'original_patients':1373,'excluded_age_variable_patients':90,'excluded_images':316,'retained_patients':1283,'retained_images':1982,'all_five_seeds':SEEDS,'fitted_models':75,'recipe':'Original balanced-softmax plus SCC oversampling','class_order':LABELS,'contrasts':contrasts,'bootstrap_repeats':2000,'bootstrap_seed':20261019,'rejected_missing_class_draws':rejected,'independent_bootstrap_implementation_max_error_pp':maxerror,'original_per_seed_metric_max_error_pp':baseline_error,'source_age_rule_independently_matched':True,'pooled_and_matched_model_probabilities_identical':True,'all_inputs_unchanged':True,'training_records_removed':False,'new_training':False,'new_inference':False,'probability_ensemble':False,'interval_scope':'Paired patient-cluster percentile intervals, unadjusted and conditional on the fixed models; no model/seed resampling.','interpretation_scope':'Restricts evaluation only; does not remove prior training/selection influence. Multiple ages are not adjudicated errors. Case composition changes; not causal evidence.','source_hashes':{p.relative_to(root).as_posix():sha(p) for p in paths}}
 write(output/'report.json',report)
 print(json.dumps(report,indent=2),flush=True)
 return report
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();calculate(args.root,args.output)
