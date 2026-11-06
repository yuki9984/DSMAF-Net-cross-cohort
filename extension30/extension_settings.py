"""Self-contained runtime and immutable experiment settings; no training on import."""
from pathlib import Path
import hashlib,json,os,sys
BUNDLE=Path(__file__).resolve().parent
PROJECT=BUNDLE/'runtime'
PLAN=BUNDLE/'analysis_plan.json'
SPLITS=PROJECT/'protocol_v2_supplemental_20261019/splits'
MATERIALS=BUNDLE/'materials'
RESULTS=BUNDLE/'outputs'
SEEDS=(20261013,20261014,20261015)
ENCODINGS={'late_age_only':'age_only','late_age_recorded':'age_recorded'}
ARMS=tuple(ENCODINGS)
BASE_ARMS=ARMS
LABELS=('ACK','BCC','MEL','NEV','SCC','SEK')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def atomic_json(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
 q.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def locations():return read(BUNDLE/'worker_locations.json') if (BUNDLE/'worker_locations.json').exists() else {}
def original_project():return Path(os.environ.get('DSMAF_ORIGINAL_PROJECT',locations().get('original_project',str(BUNDLE.parent)))).resolve()
def pad_images():
 selected=os.environ.get('DSMAF_PAD_IMAGES') or locations().get('pad_images')
 if selected:return Path(selected).resolve()
 root=original_project()
 for relative in ['teacher_original/DSMAF/PAD-UFES-20/images','teacher_materials/20261014/original/DSMAF/PAD-UFES-20/images']:
  if (root/relative).is_dir():return root/relative
 raise FileNotFoundError('PAD image directory is required; pass --pad-images to run_campaign.py')
def external_images():
 selected=os.environ.get('DSMAF_EXTERNAL_IMAGES') or locations().get('external_images')
 if selected:return Path(selected).resolve()
 root=original_project()
 for relative in ['protocol_v2_followup_20261023/prior_campaign/inputs/external_images','protocol_v2_crosscohort_20261022/inputs/external_images']:
  if (root/relative).is_dir():return root/relative
 raise FileNotFoundError('External image directory is required; pass --external-images')
def initialization_file():
 selected=os.environ.get('DSMAF_IMAGENET_WEIGHTS') or locations().get('initialization_file')
 if selected:return Path(selected).resolve()
 root=original_project()
 for relative in ['cache/hub/checkpoints/resnet50-11ad3fa6.pth','cache/checkpoints/resnet50-11ad3fa6.pth']:
  if (root/relative).is_file():return root/relative
 raise FileNotFoundError('Original resnet50-11ad3fa6.pth is required; pass --initialization-file')
def setup_original():
 sys.path.insert(0,str(PROJECT/'protocol_v2'))
 import common
 common.PROJECT=PROJECT
 common.model_code=lambda:PROJECT/'compute_checks/20261014_transfer/gpu_probe/code'
 common.pad_images=pad_images
 return common

def matrix():return [(f,a,s) for f in range(5) for s in SEEDS for a in ARMS]
def run_id(fold,arm,seed):
 if (fold,arm,seed) not in matrix():raise ValueError('Unplanned extension fit')
 return f'fold_{fold:02d}_{arm}_seed{seed}'
def cv_run(fold,arm,seed):run_id(fold,arm,seed);return RESULTS/'fits'/f'fold_{fold:02d}'/f'{arm}_seed{seed}'
def prediction_file(cohort,fold,arm,seed):
 if cohort not in ['internal','external']:raise ValueError('Unknown evaluation cohort')
 return RESULTS/'predictions'/cohort/(run_id(fold,arm,seed)+'.npz')
def verify_manifest(materials=False):
 m=read(BUNDLE/'source_manifest.json')
 for row in m['files']+(m.get('materials',[]) if materials else []):
  p=(BUNDLE/row['path']).resolve()
  if not p.is_relative_to(BUNDLE.resolve()) or not p.is_file() or sha(p)!=row['sha256']:
   raise RuntimeError('Changed release input: '+row['path'])
 return m

def frames_for_fold(fold,include_test=False):
 import pandas as pd
 setup_original()
 from data_v2 import fit_age_stats
 if fold not in range(5):raise ValueError('Invalid outer fold')
 fm=read(SPLITS/'fold_manifest.json')
 if sha(SPLITS/'fold_manifest.json')!=read(PLAN)['fold_manifest_sha256']:raise RuntimeError('Fold manifest changed')
 out={}
 for role in (['train','valid','test'] if include_test else ['train','valid']):
  row=fm['folds'][fold]['roles'][role];p=SPLITS/row['file']
  if sha(p)!=row['sha256']:raise RuntimeError('Split changed: '+row['file'])
  out[role]=pd.read_csv(p)
 for i,(ra,a) in enumerate(out.items()):
  for rb,b in list(out.items())[i+1:]:
   if set(a.patient_id)&set(b.patient_id) or set(a.img_id)&set(b.img_id):raise RuntimeError('Patient or image overlap: '+ra+'/'+rb)
 return out,fit_age_stats(out['train'])

def ensure_pretrained():
 import torch
 target=initialization_file()
 actual=sha(target)
 if target.name!='resnet50-11ad3fa6.pth' or actual!=read(PLAN)['initialization_sha256']:
  raise RuntimeError('Original ImageNet V2 initialization hash/name mismatch')
 torch.hub.set_dir(str(target.parent.parent))
 return actual

def validate_inputs():
 import numpy as np
 from information_conditions import restrict_metadata
 setup_original()
 from data_v2 import encode_pad_metadata
 verify_manifest();checks=[]
 for fold in range(5):
  frames,stats=frames_for_fold(fold)
  for role,frame in frames.items():
   observed=encode_pad_metadata(frame,stats)
   for arm,condition in ENCODINGS.items():
    limited=restrict_metadata(observed,condition)
    if not np.array_equal(limited[:,:2],observed[:,:2]):raise RuntimeError('Age encoding changed')
    checks.append({'fold':fold,'role':role,'arm':arm,'images':len(frame),'condition':condition})
 return {'status':'ALL_TRAINING_INPUTS_PASS','runs':30,'input_condition_checks':len(checks),'training_started':False,'checks':checks}
