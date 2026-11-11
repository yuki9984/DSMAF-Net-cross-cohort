"""Replay the complete MILK statistical analysis from confidential coded model responses."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import milk_statistics as S
R=Path(__file__).resolve().parent
def rows(name):
 with (R/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def compare(left,right,keys,numeric):
 def idx(v):return {tuple(str(r[k]) for k in keys):r for r in v}
 a,b=idx(left),idx(right)
 assert len(a)==len(left)==len(b)==len(right) and set(a)==set(b)
 error=0.
 for key,row in a.items():
  for k,v in row.items():
   if k in numeric:error=max(error,abs(float(v)-float(b[key][k])))
   else:assert str(v)==str(b[key][k]),(key,k,v,b[key][k])
 assert error<1e-10
 return error
def main():
 for name,digest in json.loads((R/'file_hashes.json').read_text(encoding="utf-8")).items():
  assert Path(name).name==name and hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
 plan=json.loads((R/'plan.json').read_text(encoding="utf-8"))
 with np.load(R/'coded_responses.npz',allow_pickle=False) as z:
  a={k:z[k] for k in z.files}
 arms=a['arms'].tolist();seeds=a['seeds'].tolist();y=a['labels'];v=a['per_fold_responses']
 assert v.shape==(9,3,5,4474,3) and len(set(a['lesion_codes']))==4474
 assert np.isfinite(v).all() and set(np.unique(v[:,:,:,:,0])).issubset({0.,1.})
 cw,iw=a['class_fold_weights'],a['loss_fold_weights']
 assert np.allclose(cw.sum(0),1) and np.isclose(iw.sum(),1)
 effective=np.zeros((9,3,len(y),3))
 fold=[]
 for ai,arm in enumerate(arms):
  for si,seed in enumerate(seeds):
   for f in range(5):
    value=v[ai,si,f]
    effective[ai,si,:,0]+=value[:,0]*cw[f,y]
    effective[ai,si,:,1:]+=value[:,1:]*iw[f]
    fold.extend(dict(arm=arm,seed=seed,fold=f,metric=m,value=float(x)) for m,x in zip(S.METRICS,S.metric_vector(y,value)))
 max_fold=compare(fold,rows('all_135_model_metrics.csv'),['arm','seed','fold','metric'],{'value'})
 with np.load(R/'effective_reference.npz',allow_pickle=False) as z:
  np.testing.assert_allclose(effective,z['responses'],atol=1e-10,rtol=0)
 reports={}
 for subset,mask in [('all_eligible',np.ones(len(y),dtype=bool)),('histopathology_only',a['histopathology'])]:
  result=S.summarize(arms,seeds,y[mask],a['lesion_codes'][mask],effective[:,:,mask],plan['pairs'],plan['repeats'],plan['seed'])
  for row in result['summary']+result['by_seed']:row['subset']=subset
  key=['type','comparison','metric','unit','subset']
  e1=compare(result['summary'],rows(subset+'_summary.csv'),key,{'estimate','ci95_low','ci95_high','seed_sd','resamples'})
  e2=compare(result['by_seed'],rows(subset+'_all_seeds.csv'),key+['seed'],{'value'})
  errors=[]
  with np.load(R/(subset+'_resamples.npz'),allow_pickle=False) as z:
   assert z['arms'].tolist()==arms and z['metrics'].tolist()==list(S.METRICS)
   for actual,name in [(result['level_seed_metrics'],'seed_metrics'),(result['level_bootstrap_draws'],'bootstrap')]:
    np.testing.assert_allclose(actual,z[name],atol=1e-10,rtol=0)
    errors.append(float(np.max(np.abs(actual-z[name]))))
  reports[subset]=dict(lesions=int(mask.sum()),paired_draws=plan['repeats'],summary_rows=len(result['summary']),seed_rows=len(result['by_seed']),max_table_error=max(e1,e2),max_array_error=max(errors))
 print(json.dumps(dict(status='PASS',models=135,max_fold_metric_error=max_fold,subsets=reports,scope='Saved-response statistics; no training or inference'),indent=2))
if __name__=='__main__':main()
