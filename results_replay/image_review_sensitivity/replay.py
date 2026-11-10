from pathlib import Path
import argparse,os,sys,json,csv,hashlib,importlib.util
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import numpy as np
R=Path(__file__).resolve().parent
ap=argparse.ArgumentParser(description='Replay the post hoc author-review image-exclusion sensitivity from coded saved outputs.')
ap.add_argument('--core',type=Path,default=R.parent/'statistical_replay/MILK10k')
a=ap.parse_args();C=a.core
plan=json.loads((R/'coded_plan.json').read_text(encoding='utf-8'))
for row in json.loads((C.parent/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'))['files']:
 if row['path'] in ['MILK10k/milk_statistics.py','MILK10k/coded_responses.npz','MILK10k/effective_reference.npz']:
  assert hashlib.sha256((C.parent/row['path']).read_bytes()).hexdigest()==row['sha256']
with np.load(C/'coded_responses.npz',allow_pickle=False) as z:
 arms=z['arms'].tolist();seeds=z['seeds'].tolist();y=z['labels'];codes=z['lesion_codes']
with np.load(C/'effective_reference.npz',allow_pickle=False) as z:v=z['responses']
keep=~np.isin(codes,plan['excluded_lesion_codes'])
assert int(keep.sum())==plan['retained_images'] and int((~keep).sum())==plan['removed_images']
spec=importlib.util.spec_from_file_location('coded_image_review_stats',C/'milk_statistics.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
r=m.summarize(arms,seeds,y[keep],codes[keep],v[:,:,keep,:],plan['pairs'],repeats=plan['repeats'],seed=plan['bootstrap_seed'])
for filename,key,valuekey in [('summary.csv','summary','estimate'),('all_seeds.csv','by_seed','value')]:
 with (R/filename).open(encoding='utf-8',newline='') as f:ref=list(csv.DictReader(f))
 got=r[key]
 assert len(ref)==len(got)
 for x,z in zip(ref,got):
  assert x['comparison']==z['comparison'] and x['metric']==z['metric']
  assert abs(float(x[valuekey])-z[valuekey])<5e-10
  if key=='summary':
   for c in ['ci95_low','ci95_high','seed_sd']:assert abs(float(x[c])-z[c])<5e-10
  else:assert int(x['seed'])==z['seed']
with np.load(R/'resamples.npz',allow_pickle=False) as z:
 for key in ['level_seed_metrics','level_bootstrap_draws']:np.testing.assert_allclose(r[key],z[key],rtol=0,atol=5e-10)
print(json.dumps({'status':'AUTHOR_REVIEW_EXCLUSION_REPLAY_PASS','retained_lesions':int(keep.sum()),'removed_lesions':int((~keep).sum()),'paired_resamples':plan['repeats'],'training_or_inference':False}),flush=True)
