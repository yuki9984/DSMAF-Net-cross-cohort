"""Separate exploratory campaign; no mutation of original protocols or results."""
from pathlib import Path
import os,sys,json,hashlib,importlib.util
from functools import lru_cache
import numpy as np,pandas as pd
BUNDLE=Path(__file__).resolve().parent
PROJECT=Path(os.environ.get('DSMAF_PROJECT_ROOT',BUNDLE.parent)).resolve()
PRIOR=BUNDLE/'prior_campaign';PLAN=BUNDLE/'analysis_plan.json';SPLITS=PRIOR/'inputs/splits'
RESULTS=Path(os.environ.get('DSMAF_FOLLOWUP_OUTPUT',BUNDLE/'outputs')).resolve()
ARMS=('full','image_only','late_fusion');SEEDS=(20261013,20261014,20261015);ALL_SEEDS=tuple(range(20261013,20261018));LABELS=('ACK','BCC','MEL','NEV','SCC','SEK')
sys.path.insert(0,str(PROJECT/'protocol_v2'))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)
def atomic_npz(p,**v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**v)
 tmp.replace(p)
def plan():return read(PLAN)
def load_prior():
 sys.path.insert(0,str(PRIOR));spec=importlib.util.spec_from_file_location('frozen_crosscohort_runner',PRIOR/'run_worker.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);return r
R=load_prior()
def verify_manifest():
 m=read(BUNDLE/'source_manifest.json')
 for rec in m['files']:
  assert sha256(BUNDLE/rec['path'])==rec['sha256'],'Changed followup bundle: '+rec['path']
 assert sha256(PRIOR/'BUNDLE_MANIFEST.json')==plan()['prior_campaign_manifest_sha256']
 assert sha256(BUNDLE/'baseline_probabilities.npz')==plan()['baseline_projection_sha256']
 records,root=R.verify_metadata(PROJECT)
 for rec in read(BUNDLE/'extra_runtime_sources.json')['files']:
  assert sha256((PROJECT if rec['scope']=='project' else root)/rec['path'])==rec['sha256'],'Changed augmentation/training/runtime source: '+rec['path']
 from common import PROJECT as active
 assert active.resolve()==PROJECT,'Select the original worker project root containing protocol_v2.'
 return records,root

def pad_images():
 from common import pad_images as original
 return Path(os.environ['DSMAF_PAD_IMAGES']) if os.environ.get('DSMAF_PAD_IMAGES') else original()
def model_code():
 from common import model_code as original
 return original()
def cv_run(fold,arm,seed):
 assert fold in range(5) and arm in ARMS and seed in SEEDS
 return RESULTS/'recipe45'/f'fold_{fold:02d}'/f'{arm}_seed{seed}'
def frames_for_fold(fold,include_test=False):
 from data_v2 import fit_age_stats
 out={};fm=read(SPLITS/'fold_manifest.json')
 for role in (('train','valid','test') if include_test else ('train','valid')):
  rec=fm['folds'][fold]['roles'][role];p=SPLITS/rec['file'];assert sha256(p)==rec['sha256'];out[role]=pd.read_csv(p)
 return out,fit_age_stats(out['train'])
def ensure_pretrained():
 import torch
 from torchvision.models import ResNet50_Weights
 from urllib.parse import urlparse
 expected='11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'
 target=PROJECT/'cache/hub/checkpoints'/Path(urlparse(ResNet50_Weights.IMAGENET1K_V2.url).path).name
 torch.hub.set_dir(str(target.parent.parent));assert target.is_file() and sha256(target)==expected,'Original pretrained weight required: '+str(target)
 return expected

def training_records():
 rows=[]
 for f in range(5):
  for arm in ARMS:
   for seed in SEEDS:
    d=cv_run(f,arm,seed);c=read(d/'config.json');h=read(d/'history.json');s=read(d/'summary.json')
    for k,v in {'fold':f,'arm':arm,'seed':seed,'plan_sha256':sha256(PLAN),'source_manifest_sha256':sha256(BUNDLE/'source_manifest.json'),'loss':'ordinary_cross_entropy','oversample_scc':False,'selection_split':'valid','selection_metric':'balanced_accuracy','outer_test_used_for_selection':False}.items():assert c.get(k)==v,(d,k)
    scores=[r['valid']['bac'] for r in h['epochs']];best=int(np.argmax(scores))+1
    assert best==h['best_epoch']==s['best_epoch'] and s['epochs_completed']==len(scores)
    assert abs(max(scores)-s['best_valid_bac'])<1e-12
    rows.append({'fold':f,'arm':arm,'seed':seed,'best_epoch':best,'checkpoint_sha256':sha256(d/'best_valid.pth'),'config_sha256':sha256(d/'config.json'),'history_sha256':sha256(d/'history.json'),'summary_sha256':sha256(d/'summary.json')})
 assert len(rows)==45
 return rows

def freeze_training():
 records=training_records();p=RESULTS/'recipe45/training_complete.json';v={'status':'PASS','run_count':45,'runs':records,'plan_sha256':sha256(PLAN),'outer_test_evaluated':False,'external_evaluated':False}
 if p.exists():assert read(p)==v,'Frozen 45-fit inventory differs'
 else:atomic_json(p,v)
 return records

@lru_cache(maxsize=1)
def original_arrays():
 with np.load(BUNDLE/'baseline_probabilities.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
 assert a['arms'].astype(str).tolist()==list(ARMS) and a['seeds'].tolist()==list(ALL_SEEDS)
 return a

def model_name(r):return f"fold_{r['fold']:02d}_{r['arm']}_seed{r['seed']}"
def reference_for(r,cohort,frame):
 a=original_arrays();ai=ARMS.index(r['arm']);si=ALL_SEEDS.index(r['seed'])
 if cohort=='internal':
  idx=pd.Index(a['internal_images'].astype(str)).get_indexer(frame.img_id.astype(str));assert (idx>=0).all();assert np.all(a['internal_folds'][idx]==r['fold']);return a['prob_internal'][ai,si,idx]
 assert frame.img_id.astype(str).tolist()==a['external_images'].astype(str).tolist();return a['prob_external'][ai,si,r['fold']]
