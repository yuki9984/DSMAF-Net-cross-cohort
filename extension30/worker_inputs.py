"""Worker discovery and read-only source/image integrity checks."""
from pathlib import Path
import json,platform,sys,time
import numpy as np
import pandas as pd
import extension_settings as S
from campaign_support import binding,progress

def qualify_runtime():
 import torch,torchvision
 actual={'python':platform.python_version(),'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda_build':torch.version.cuda,'platform':platform.system()}
 if (actual['python'],actual['torch'],actual['torchvision'],actual['cuda_build'])!=('3.11.9','2.6.0+cu124','0.21.0+cu124','12.4'):
  raise RuntimeError('Original runtime required; do not silently upgrade or substitute: '+json.dumps(actual))
 if not torch.cuda.is_available() or '4060' not in torch.cuda.get_device_name(0):raise RuntimeError('The original RTX4060 worker is required')
 if platform.system()!='Windows':raise RuntimeError('Use the original Windows worker')
 actual.update(gpu=torch.cuda.get_device_name(0),gpu_bytes=torch.cuda.get_device_properties(0).total_memory,numpy=np.__version__,pandas=pd.__version__)
 return actual

def verify_material_arrays():
 S.verify_manifest(materials=True)
 with np.load(S.MATERIALS/'controls.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
 if a['arms'].astype(str).tolist()!=['full','image_only','late_fusion'] or a['seeds'].tolist()!=list(S.SEEDS):raise RuntimeError('Control method/seed order changed')
 if a['prob_internal'].shape!=(3,3,2298,6) or a['prob_external'].shape!=(3,3,5,1701,6):raise RuntimeError('Control array shape changed')
 for key in ['prob_internal','prob_external']:
  p=a[key]
  if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(-1),1,atol=1e-6,rtol=0):raise RuntimeError('Invalid saved probabilities')
 internal_index=pd.Index(a['internal_images'].astype(str))
 if len(internal_index)!=2298 or not internal_index.is_unique or len(np.unique(a['internal_patients']))!=1373:raise RuntimeError('Internal identities changed')
 for fold in range(5):
  frame,_=S.frames_for_fold(fold,include_test=True);test=frame['test'];idx=internal_index.get_indexer(test.img_id.astype(str))
  if (idx<0).any() or not np.all(a['internal_folds'][idx]==fold):raise RuntimeError('Outer fold assignment changed')
  if not np.array_equal(a['internal_labels'][idx],test.diagnostic_number.to_numpy()) or not np.array_equal(a['internal_patients'][idx].astype(str),test.patient_id.astype(str).to_numpy()):raise RuntimeError('Control/split agreement failed')
 external=pd.read_csv(S.MATERIALS/'external_cases.csv')
 for column,key in [('img_id','external_images'),('patient_id','external_patients')]:
  if not np.array_equal(external[column].astype(str).to_numpy(),a[key].astype(str)):raise RuntimeError('External order or identities changed')
 if not np.array_equal(external.label.to_numpy(),a['external_labels']):raise RuntimeError('External labels changed')
 rows=S.read(S.MATERIALS/'control_configs.json')
 if len(rows)!=45 or {(r['fold'],r['arm'],r['seed']) for r in rows}!={(f,a,s) for f in range(5) for a in ['full','image_only','late_fusion'] for s in S.SEEDS}:raise RuntimeError('Need all 45 unique matched controls')
 source_checks={'v2_model_sha256':S.PROJECT/'protocol_v2/model_v2.py','v2_data_sha256':S.PROJECT/'protocol_v2/data_v2.py','supplemental_trainer_sha256':S.PROJECT/'protocol_v2_supplemental_20261019/train_cv.py','model_source_sha256':S.setup_original().model_code()/'models/hfpa_net.py'}
 for row in rows:
  c=row['config']
  for key,path in source_checks.items():
   if c.get(key)!=S.sha(path):raise RuntimeError('Frozen runtime differs from reused control source: '+key)
  for key,value in {'loss':'balanced_softmax','oversample_scc':True,'batch_size':16,'epochs':150,'patience':15,'lr':2e-4,'weight_decay':1e-4,'selection_metric':'balanced_accuracy','split_manifest_sha256':S.read(S.PLAN)['fold_manifest_sha256']}.items():
   if c.get(key)!=value:raise RuntimeError('Mismatched reused control: '+key)
 return a

def check_images(root,frame,label):
 root=Path(root).resolve();records=[];started=time.monotonic()
 for i,row in enumerate(frame.itertuples(index=False)):
  p=(root/str(row.img_id)).resolve()
  if not p.is_relative_to(root) or not p.is_file() or S.sha(p)!=row.sha256:raise RuntimeError('Missing/changed '+label+' image: '+str(row.img_id))
  records.append(str(row.img_id))
  if (i+1)%500==0:progress('IMAGE_HASH_CHECK',cohort=label,checked=i+1,total=len(frame))
 return {'images':len(records),'seconds':time.monotonic()-started,'all_sha256_match':True}

def check_worker(allow_cpu=False,check_image_files=True):
 S.RESULTS.mkdir(parents=True,exist_ok=True)
 S.validate_inputs();a=verify_material_arrays()
 runtime={'gpu_checked':False} if allow_cpu else qualify_runtime()
 image_reports={}
 if check_image_files:
  image_reports['PAD']=check_images(S.pad_images(),pd.read_csv(S.SPLITS/'pad_image_hashes.csv'),'PAD')
  image_reports['external']=check_images(S.external_images(),pd.read_csv(S.MATERIALS/'external_image_hashes.csv'),'external')
 initialization={'checked':False}
 if not allow_cpu:initialization={'checked':True,'sha256':S.ensure_pretrained()}
 report={**binding(),'status':'ALL_WORKER_INPUTS_PASS' if not allow_cpu and check_image_files else 'CPU_INPUT_AUDIT_PASS','runtime':runtime,'image_hashes':image_reports,'initialization':initialization,'internal_images':2298,'internal_patients':1373,'external_images':1701,'reused_controls':45,'new_training_fits_planned':30,'new_training_started':False}
 S.atomic_json(S.RESULTS/('worker_input_audit.json' if not allow_cpu else 'cpu_input_audit.json'),report)
 return report
