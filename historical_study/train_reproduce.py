"""Portable I/O adapter for the preserved study training loops.

This launcher changes data/output locations and release-integrity checks only.
It does not change the model, loss, sampler, checkpoint rule or training loop.
A successful input check is not evidence of a completed GPU reproduction.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, sys, types
ROOT=Path(__file__).resolve().parent
LABELS=('ACK','BCC','MEL','NEV','SCC','SEK')
ARMS=('full','no_dsdm','shared_q','no_dsdm_shared_q','image_only','late_fusion','late_fusion_cap')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def arguments():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--study-inputs',type=Path,required=True,help='Locally authorized study-input folder; never included in the public code candidate.')
 p.add_argument('--experiment',choices=['primary','cv','alternative'],default='cv')
 p.add_argument('--arm',choices=ARMS,default='full')
 p.add_argument('--seed',type=int,default=20261013)
 p.add_argument('--fold',type=int,choices=range(5),default=0)
 p.add_argument('--pad-images',type=Path)
 p.add_argument('--validate-inputs',action='store_true')
 p.add_argument('--download-initialization',action='store_true')
 return p.parse_args()
def verify_release():
 m=read(ROOT.parent/'MANIFEST.json')
 for rec in m['files']:
  p=ROOT.parent/rec['path']
  if not p.is_file() or sha(p)!=rec['sha256']:raise RuntimeError('Release input changed: '+rec['path'])
 return m
def load_frames(a):
 import pandas as pd
 if a.experiment=='primary':paths={k:a.study_inputs/'neural_setup/original_split'/('pad_'+k+'.csv') for k in ['train','valid']}
 else:paths={k:a.study_inputs/'internal75/splits'/f'fold_{a.fold:02d}_{k}.csv' for k in ['train','valid','test']}
 frames={k:pd.read_csv(p) for k,p in paths.items()}
 for key,frame in frames.items():
  required={'img_id','patient_id','age','gender_FEMALE','gender_MALE','diagnostic_number'}
  if not required.issubset(frame):raise RuntimeError('Missing required columns: '+key)
  if frame.img_id.duplicated().any():raise RuntimeError('Duplicate images: '+key)
  if not set(frame.diagnostic_number).issubset(range(6)):raise RuntimeError('Invalid labels: '+key)
 for i,(r1,f1) in enumerate(frames.items()):
  for r2,f2 in list(frames.items())[i+1:]:
   if set(f1.patient_id)&set(f2.patient_id):raise RuntimeError('Patient overlap: '+r1+'/'+r2)
 if a.pad_images:
  missing=[str(x) for f in frames.values() for x in f.img_id if not (a.pad_images/str(x)).is_file()]
  if missing:raise FileNotFoundError('Missing PAD images: '+str(missing[:3]))
 return frames
def main():
 a=arguments()
 if a.seed not in range(20261013,20261018):raise ValueError('Use one of the five prespecified seeds.')
 if a.experiment!='primary' and a.arm not in ('full','image_only','late_fusion'):raise ValueError('Only the three evaluated model families are allowed for grouped evaluation.')
 if a.experiment=='alternative' and a.seed>20261015:raise ValueError('The alternative recipe used only seeds 20261013--20261015.')
 verify_release();frames=load_frames(a)
 sys.path.insert(0,str(ROOT/'model_sources/protocol_v2'))
 import common,data_v2
 stats=data_v2.fit_age_stats(frames['train'])
 summary={'status':'INPUTS_VERIFIED','experiment':a.experiment,'fold':None if a.experiment=='primary' else a.fold,'arm':a.arm,'seed':a.seed,'samples':{k:len(v) for k,v in frames.items()},'training_age_statistics':stats.to_dict(),'images_checked':a.pad_images is not None,'neural_training_performed':False}
 if a.validate_inputs:print(json.dumps(summary,indent=2));return
 if not a.pad_images:raise ValueError('--pad-images is required for training; download the public release first.')
 import torch
 if not torch.cuda.is_available():raise RuntimeError('The preserved training loops require a CUDA GPU.')
 runtime=ROOT/'.neural_runtime';runtime.mkdir(exist_ok=True)
 model_root=ROOT/'model_sources/compute_checks/20261014_transfer/gpu_probe/code'
 common.PROJECT=runtime
 common.model_code=lambda:model_root
 common.pad_images=lambda:a.pad_images.resolve()
 common.split_dir=lambda:a.study_inputs/'neural_setup/original_split'
 common.output_dir=lambda:ROOT/'runs/primary'
 data_v2.pad_images=common.pad_images;data_v2.split_dir=common.split_dir
 os.environ['TORCH_HOME']=str(runtime/'cache')
 def ensure_pretrained():
  from torchvision.models import ResNet50_Weights
  from urllib.parse import urlparse
  torch.hub.set_dir(str(runtime/'cache/hub'))
  url=ResNet50_Weights.IMAGENET1K_V2.url
  target=Path(torch.hub.get_dir())/'checkpoints'/Path(urlparse(url).path).name
  if not target.exists() and a.download_initialization:
   target.parent.mkdir(parents=True,exist_ok=True);torch.hub.download_url_to_file(url,str(target),progress=True)
  expected='11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca'
  if not target.is_file() or sha(target)!=expected:raise RuntimeError('The original ImageNet V2 initialization is required. Use --download-initialization to fetch it.')
  return expected
 if a.experiment=='primary':source=ROOT/'model_sources/protocol_v2/train.py'
 else:
  module=types.ModuleType('settings' if a.experiment=='cv' else 'followup_settings')
  module.ARMS=('full','image_only','late_fusion')
  module.SEEDS=tuple(range(20261013,20261016))
  module.PLAN=a.study_inputs/'neural_setup'/('locked_plan.json' if a.experiment=='cv' else 'analysis_plan.json')
  module.SPLITS=a.study_inputs/'internal75/splits'
  module.pad_images=common.pad_images
  module.cv_run=lambda f,arm,seed:ROOT/'runs'/a.experiment/f'fold_{f:02d}'/f'{arm}_seed{seed}'
  def frames_for_fold(fold,include_test=False):
   from copy import copy
   current=copy(a);current.fold=fold;f=load_frames(current)
   return {k:v for k,v in f.items() if include_test or k!='test'},data_v2.fit_age_stats(f['train'])
  module.frames_for_fold=frames_for_fold
  module.ensure_pretrained=ensure_pretrained
  module.verify_manifest=verify_release
  sys.modules[module.__name__]=module
  source=ROOT/('model_sources/protocol_v2_supplemental_20261019/train_cv.py' if a.experiment=='cv' else 'followup/inputs/train_recipe.py')
 spec=importlib.util.spec_from_file_location('preserved_study_training',source)
 target=importlib.util.module_from_spec(spec);sys.modules[spec.name]=target;spec.loader.exec_module(target)
 target.pretrained_weight_hash=ensure_pretrained
 target.verify_manifest=verify_release
 # CLI defaults in the preserved loops retain batch size 16, 150 epochs,
 # patience 15, AdamW, lr 2e-4, weight decay 1e-4 and cosine scheduling.
 sys.argv=[str(source),'--arm',a.arm,'--seed',str(a.seed)]
 if a.experiment!='primary':sys.argv+=['--fold',str(a.fold)]
 target.main()
if __name__=='__main__':main()
