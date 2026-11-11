"""Post hoc equal-fold aggregation of fixed models; probability vectors are not ensembled."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
S=ROOT/'results_replay/statistical_replay'
ARMS=['image_only','late_fusion','full','metablock_cv','rg_dermnet_adapted_cv','dsmaf_age_only','dsmaf_age_recorded','late_age_only','late_age_recorded']
SEEDS=[20261013,20261014,20261015]
PAIRS=[('dsmaf_age_recorded','dsmaf_age_only'),('full','dsmaf_age_recorded'),('late_age_recorded','late_age_only'),('late_fusion','late_age_recorded')]+[(a,'image_only') for a in ['full','late_fusion','metablock_cv','rg_dermnet_adapted_cv']]
METRICS=['BACC']+['recall_'+c for c in ['ACK','BCC','MEL','NEV','SCC','SEK']]+['NLL','Brier']
REPEATS=2000;BOOTSTRAP_SEED=2026111101
spec=importlib.util.spec_from_file_location('fixed_model_milk_statistics',S/'MILK10k/milk_statistics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def read_json(p):return json.loads(p.read_text(encoding='utf-8'))
def read_csv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
 with p.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def bootstrap(y,values,stratified):
 # Every method, seed, metric and aggregation uses the same draws in this cohort.
 rng=np.random.default_rng(BOOTSTRAP_SEED);mean=values.mean(2);draws=np.empty((REPEATS,2,len(ARMS),9));n=len(y)
 for start in range(0,REPEATS,64):
  count=min(64,REPEATS-start);counts=np.zeros((count,n))
  for b in range(count):
   while True:
    idx=m.stratified_indices(y,rng) if stratified else rng.choice(n,size=n,replace=True)
    if len(np.unique(y[idx]))==6:break
   counts[b]=np.bincount(idx,minlength=n)
  rec=[]
  for c in range(6):
   mask=y==c;den=counts[:,mask].sum(1)
   rec.append(np.einsum('bn,wan->bwa',counts[:,mask],mean[:,:,mask,0])/den[:,None,None])
  rec=np.stack(rec,-1);draws[start:start+count,:,:,0]=rec.mean(-1)*100;draws[start:start+count,:,:,1:7]=rec*100
  for j in range(2):draws[start:start+count,:,:,7+j]=np.einsum('bn,wan->bwa',counts,mean[:,:,:,1+j])/n
 return draws

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--record-reference',action='store_true');a=ap.parse_args();out=a.output.resolve()
 if out.exists():raise FileExistsError('Use a new output directory.')
 out.mkdir(parents=True);arrays={};summary=[];seeds_out=[];changes=[];direction=[];source_rows=[]
 plans=[read_json(S/d/'plan.json') for d in ['original_and_PAA60','late_fusion_followup']]
 loaded=[]
 for d in ['original_and_PAA60','late_fusion_followup']:
  p=S/d/'matched.npz';source_rows.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  with np.load(p,allow_pickle=False) as z:loaded.append({k:z[k] for k in z.files})
 for k in ['yi','ye','folds','gi','ge']:np.testing.assert_array_equal(loaded[0][k],loaded[1][k])
 for arm in set(plans[0]['methods'])&set(plans[1]['methods']):
  for key in ['pi','pe']:np.testing.assert_allclose(loaded[0][key][plans[0]['methods'].index(arm)],loaded[1][key][plans[1]['methods'].index(arm)],atol=1e-12,rtol=0)
 y=loaded[0]['ye'];assert len(np.unique(loaded[0]['ge']))==len(y),'DermatoNet requires unique released patient identifiers here'
 pe=np.stack([loaded[0]['pe'][plans[0]['methods'].index(arm)] if arm in plans[0]['methods'] else loaded[1]['pe'][plans[1]['methods'].index(arm)] for arm in ARMS])
 cw,iw=m.fold_weights(loaded[0]['yi'],loaded[0]['folds'])
 cohorts=[('DermatoNet',y,pe,False)]
 probabilities=[];labels=None;ids=None
 for arm in ARMS:
  ss=[]
  for seed in SEEDS:
   folds=[]
   for fold in range(5):
    p=ROOT/'study_inputs/milk/predictions'/f'fold_{fold:02d}_{arm}_seed{seed}.npz'
    source_rows.append({'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    with np.load(p,allow_pickle=False) as z:
     if labels is None:labels=z['labels'];ids=z['lesion_ids']
     np.testing.assert_array_equal(z['labels'],labels);np.testing.assert_array_equal(z['lesion_ids'],ids);folds.append(z['probabilities'])
   ss.append(folds)
  probabilities.append(ss)
 assert len(labels)==4474 and len(set(ids))==4474
 cohorts.append(('MILK10k',labels,np.asarray(probabilities),True))
 for cohort,y,p,stratified in cohorts:
  values=np.stack([np.array([[m.collapse_folds(y,p[i,j],weights,losses) for j in range(3)] for i in range(9)]) for weights,losses in [(cw,iw),(np.full((5,6),.2),np.full(5,.2))]])
  levels=np.array([[[m.metric_vector(y,v) for v in method] for method in vv] for vv in values])
  # Match existing published weighted point estimates before considering equal weights.
  refs=read_csv(S/'MILK10k/all_eligible_summary.csv') if cohort=='MILK10k' else read_csv(S/'original_and_PAA60/summary_intervals.csv')+read_csv(S/'late_fusion_followup/summary_intervals.csv')
  for i,arm in enumerate(ARMS):
   for mi,metric in enumerate(METRICS):
    lookup_metric='recall_'+str(mi-1) if cohort=='DermatoNet' and 0<mi<7 else metric
    reference=[r for r in refs if r['type']=='model_level' and r['comparison']==arm and r['metric']==lookup_metric and (cohort=='MILK10k' or r['target']=='external')]
    assert reference,(cohort,arm,metric)
    assert max(abs(float(r['estimate'])-levels[0,i,:,mi].mean()) for r in reference)<2e-9,(cohort,arm,metric)
  draws=bootstrap(y,values,stratified);arrays[cohort+'_seed_levels']=levels;arrays[cohort+'_draws']=draws
  for wi,weighting in enumerate(['fixed_PAD_support','equal_fold']):
   specs=[('model_level',arm,levels[wi,i],draws[:,wi,i]) for i,arm in enumerate(ARMS)]
   for left,right in PAIRS:
    li=ARMS.index(left);ri=ARMS.index(right);specs.append(('paired_contrast',left+'-minus-'+right,levels[wi,li]-levels[wi,ri],draws[:,wi,li]-draws[:,wi,ri]))
   for kind,name,sv,bv in specs:
    lo,hi=np.quantile(bv,[.025,.975],axis=0)
    for k,metric in enumerate(METRICS):
     summary.append({'cohort':cohort,'weighting':weighting,'type':kind,'comparison':name,'metric':metric,'estimate':float(sv[:,k].mean()),'ci95_low':float(lo[k]),'ci95_high':float(hi[k]),'seed_sd':float(sv[:,k].std(ddof=1)),'resamples':REPEATS,'interval_scope':'conditional_fitted_models_unadjusted','cluster_unit':'lesion_not_patient' if stratified else 'patient'})
     for seed,value in zip(SEEDS,sv[:,k]):seeds_out.append({'cohort':cohort,'weighting':weighting,'type':kind,'comparison':name,'metric':metric,'seed':seed,'value':float(value)})
  for i,arm in enumerate(ARMS):
   for si,seed in enumerate(SEEDS):changes.append({'cohort':cohort,'arm':arm,'seed':seed,'fixed_BACC':float(levels[0,i,si,0]),'equal_BACC':float(levels[1,i,si,0]),'equal_minus_fixed':float(levels[1,i,si,0]-levels[0,i,si,0])})
  for left,right in [PAIRS[0],PAIRS[2]]:
   li=ARMS.index(left);ri=ARMS.index(right)
   for si,seed in enumerate(SEEDS):direction.append({'cohort':cohort,'design':'DSMAF-Net' if left.startswith('dsmaf') else 'Late fusion','seed':seed,'fixed_C_minus_B':float(levels[0,li,si,0]-levels[0,ri,si,0]),'equal_C_minus_B':float(levels[1,li,si,0]-levels[1,ri,si,0])})
 write_csv(out/'summary.csv',summary);write_csv(out/'all_seeds.csv',seeds_out);write_csv(out/'weighting_changes.csv',changes);write_csv(out/'recording_status_directions.csv',direction);np.savez_compressed(out/'resamples.npz',**arrays)
 expected=Path(__file__).resolve().parent/'expected'
 if expected.exists() and not a.record_reference:
  ref=read_csv(expected/'summary.csv');assert len(ref)==len(summary)
  for x,z in zip(ref,summary):
   for k in ['estimate','ci95_low','ci95_high','seed_sd']:assert abs(float(x[k])-z[k])<2e-9
  with np.load(expected/'resamples.npz',allow_pickle=False) as z:
   for k,v in arrays.items():np.testing.assert_allclose(v,z[k],atol=2e-9,rtol=0)
 report={'status':'EQUAL_FOLD_ANALYSIS_PASS','post_hoc':True,'retraining_or_neural_inference':False,'seed':BOOTSTRAP_SEED,'paired_resamples':REPEATS,'arms':ARMS,'seeds':SEEDS,'input_hashes':source_rows,'fixed_weighted_points_match_published':True,'equal_C_minus_B_negative_count':sum(r['equal_C_minus_B']<0 for r in direction),'direction_comparisons':len(direction),'maximum_absolute_BACC_level_change':{c:max(abs(r['equal_minus_fixed']) for r in changes if r['cohort']==c) for c in ['DermatoNet','MILK10k']},'interpretation':'Equal arithmetic means of separately scored fold models; no probability ensembling.'}
 (out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='input_hashes'},indent=2),flush=True)
if __name__=='__main__':main()