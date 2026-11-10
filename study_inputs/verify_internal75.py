"""CPU verification of the completed retrospective internal evaluation."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(y,p):
 pred=p.argmax(1);correct=(pred==y);confidence=p.max(1);out={}
 out['bacc']=float(np.mean([correct[y==i].mean() for i in range(6)]))
 out['accuracy']=float(correct.mean())
 out['nll']=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean())
 out['brier']=float(np.mean(np.sum((p-np.eye(6)[y])**2,axis=1)))
 ece=0.
 for k in range(10):
  take=(confidence>=k/10)&((confidence<(k+1)/10) if k<9 else (confidence<=1))
  if take.any():ece+=take.mean()*abs(correct[take].mean()-confidence[take].mean())
 out['ece_10bin']=float(ece)
 out.update({f'recall_{k}':float(correct[y==k].mean()) for k in range(6)})
 return out
def run():
 for row in read(ROOT/'MANIFEST.json')['files']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
 I=ROOT/'internal75';prov=read(I/'provenance.json')
 z=np.load(I/'pooled_predictions.npz',allow_pickle=False);y=z['labels'];ids=z['patient_ids'];images=z['img_ids']
 assert len(y)==len(set(images))==2298 and len(set(ids))==1373
 models=['full','image_only','late_fusion'];seeds=list(range(20261013,20261018))
 frames={};training_checks=0
 for fold in range(5):
  frames[fold]={r:pd.read_csv(I/f'splits/fold_{fold:02d}_{r}.csv') for r in ['train','valid','test']}
  for a,b in [('train','valid'),('train','test'),('valid','test')]:
   for column in ['img_id','patient_id','image_sha256']:
    assert not set(frames[fold][a][column])&set(frames[fold][b][column]),(fold,column)
  for role,df in frames[fold].items():
   assert df.img_id.is_unique
  age=frames[fold]['train'].age
  counts=np.bincount(frames[fold]['train'].diagnostic_number,minlength=6).tolist()
  for a in models:
   for seed in seeds:
    d=I/f'training_records/fold_{fold:02d}/{a}_seed{seed}'
    c,h,s=read(d/'config.json'),read(d/'history.json'),read(d/'summary.json')
    assert c['plan_sha256']==prov['original_locked_plan_sha256'] and c['source_manifest_sha256']==prov['original_worker_source_manifest_sha256']
    assert (c['fold'],c['arm'],c['seed'])==(fold,a,seed) and c['batch_size']==16 and c['epochs']==150 and c['patience']==15
    assert c['outer_test_used_for_selection'] is False and c['selection_split']=='valid'
    assert c['loss']=='balanced_softmax' and c['oversample_scc'] and c['lr']==2e-4 and c['weight_decay']==1e-4
    assert c['pretrained_weight_sha256']=='11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'
    assert c['train_class_counts']==counts
    for k,v in {'mean':age.mean(),'std':age.std(ddof=0),'median':age.median()}.items():assert abs(c['metadata_age_train_stats'][k]-v)<1e-12
    scores=np.array([e['valid']['bac'] for e in h['epochs']])
    best=int(scores.argmax())+1
    assert best==h['best_epoch']==s['best_epoch'] and len(scores)==s['epochs_completed']
    assert len(scores)==150 or len(scores)-best==15
    assert abs(scores.max()-s['best_valid_bac'])<1e-12
    training_checks+=1
 pool=pd.concat([frames[f]['test'].assign(outer_fold=f) for f in range(5)]).sort_values('img_id').reset_index(drop=True)
 assert pool.img_id.tolist()==images.tolist() and pool.patient_id.tolist()==ids.tolist() and np.array_equal(pool.diagnostic_number,y)
 assert pool.groupby('patient_id').outer_fold.nunique().max()==1 and pool.groupby('image_sha256').outer_fold.nunique().max()==1
 expected=pd.read_csv(I/'expected_metrics_per_seed.csv').set_index(['arm','seed'])
 errors=[];correct={};rows=[]
 for a in models:
  p=z['probs_'+a];assert p.shape==(5,2298,6) and np.isfinite(p).all() and p.min()>=0 and np.max(abs(p.sum(2)-1))<1e-12
  correct[a]=(p.argmax(2)==y).astype(float)
  for k,seed in enumerate(seeds):
   v=metrics(y,p[k]);rows.append({'arm':a,'seed':seed,**v})
   errors.extend(abs(v[n]-expected.loc[(a,seed),n]) for n in v)
 assert max(errors)<1e-12
 patients,inv=np.unique(ids.astype(str),return_inverse=True);n=len(patients)
 count=np.zeros((n,6));np.add.at(count,(inv,y),1)
 rng=np.random.default_rng(20261019);weights=np.stack([np.bincount(rng.integers(0,n,size=n),minlength=n) for _ in range(2000)])
 den=weights@count;assert (den>0).all()
 contrasts=[]
 for row in read(I/'expected_intervals.json')['contrasts']:
  other=row['contrast'].split('full-minus-')[1];delta=(correct['full']-correct[other]).mean(0)
  sums=np.zeros((n,6));np.add.at(sums,(inv,y),delta)
  samples=((weights@sums)/den).mean(1)
  actual={'mean_delta_bacc':float(np.mean([delta[y==k].mean() for k in range(6)])),'ci95_low':float(np.quantile(samples,.025)),'ci95_high':float(np.quantile(samples,.975))}
  assert all(abs(actual[k]-row[k])<1e-12 for k in actual)
  contrasts.append({'contrast':row['contrast'],**actual})
 out=ROOT/'verification_output';out.mkdir(exist_ok=True)
 pd.DataFrame(rows).to_csv(out/'internal75_metrics.csv',index=False)
 report={'status':'PASS','training_records_checked':training_checks,'oof_model_seed_sets':15,'images_per_set':2298,'patients_per_set':1373,'metric_values_compared':len(errors),'maximum_metric_difference':float(max(errors)),'contrasts':contrasts,'checkpoint_binary_files_included':False,'neural_retraining_performed':False,'scope':'Fixed prediction statistics and saved configuration/history records; no independent end-to-end inference verification.'}
 (out/'internal75_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':run()

