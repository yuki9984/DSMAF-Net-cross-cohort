"""Fixed-checkpoint batch inference with per-file integrity receipts and no TTA."""
from pathlib import Path
import argparse,gc,json,hashlib,time
import numpy as np
import pandas as pd
import extension_settings as S
from campaign_support import binding,bound_report,exclusive_npz,retain_failed,progress

def material_root():return S.MATERIALS

def load_evaluation_inputs(config,cohort):
 S.setup_original()
 from data_v2 import AgeStats,encode_pad_metadata
 from information_conditions import restrict_metadata
 stats=AgeStats(**config['metadata_age_train_stats'])
 if cohort=='internal':
  frames,fitted=S.frames_for_fold(config['fold'],include_test=True)
  if any(abs(getattr(stats,k)-getattr(fitted,k))>1e-10 for k in ['mean','std','median']):raise RuntimeError('Checkpoint age statistics differ from its training split')
  frame=frames['test'];metadata=encode_pad_metadata(frame,stats)
  labels=frame.diagnostic_number.to_numpy(dtype=np.int64);patients=frame.patient_id.astype(str).to_numpy()
 elif cohort=='external':
  frame=pd.read_csv(S.MATERIALS/'external_cases.csv')
  from data_v2 import _age_features
  age_z,missing=_age_features(frame.age,stats)
  sex=frame.gender.fillna('').astype(str).str.strip().str.upper()
  if not sex.isin(['FEMININO','MASCULINO','']).all():raise RuntimeError('Unmapped external sex')
  metadata=np.column_stack((age_z,missing,(sex=='FEMININO').to_numpy(),(sex=='MASCULINO').to_numpy(),(sex=='').to_numpy())).astype(np.float32)
  labels=frame.label.to_numpy(dtype=np.int64);patients=frame.patient_id.astype(str).to_numpy()
 else:raise ValueError('Invalid cohort')
 return frame,restrict_metadata(metadata,config['information_condition']),labels,patients

def input_fingerprint(frame,metadata,labels,patients):
 h=hashlib.sha256()
 for x in [frame.img_id.astype(str).to_numpy(),metadata,labels,patients]:
  if np.asarray(x).dtype.kind in 'USO':h.update(json.dumps(np.asarray(x).astype(str).tolist(),separators=(',',':')).encode())
  else:h.update(np.ascontiguousarray(x).tobytes())
 return h.hexdigest()

def validate_prediction_arrays(arrays,frame,labels,patients,input_sha,record,config_sha):
 p=arrays['probabilities'];logits=arrays['logits']
 if p.shape!=(len(frame),6) or logits.shape!=p.shape or not np.isfinite(logits).all() or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-6,rtol=0):raise RuntimeError('Invalid prediction values')
 for key,value in [('img_id',frame.img_id.astype(str).to_numpy()),('patient_id',patients.astype(str)),('y_true',labels)]:
  if not np.array_equal(arrays[key],value):raise RuntimeError('Prediction case/order mismatch: '+key)
 expected={**binding(),'checkpoint_sha256':record['checkpoint_sha256'],'config_sha256':config_sha,'input_sha256':input_sha}
 for key,value in expected.items():
  if str(arrays[key])!=value:raise RuntimeError('Prediction provenance mismatch: '+key)
 shifted=logits-logits.max(1,keepdims=True);expected_p=np.exp(shifted);expected_p/=expected_p.sum(1,keepdims=True)
 if not np.allclose(p,expected_p,atol=1e-12,rtol=1e-12):raise RuntimeError('Saved probabilities differ from saved logits')

def frozen_records():
 report=bound_report(S.RESULTS/'training_complete.json','ALL_60_FITS_FROZEN')
 actual={(r['fold'],r['arm'],r['seed']) for r in report['runs']}
 if len(report['runs'])!=60 or actual!=set(S.matrix()):raise RuntimeError('Wrong frozen fit inventory')
 return {(r['fold'],r['arm'],r['seed']):r for r in report['runs']}

def existing_prediction(output,frame,meta,labels,patients,record,config_sha):
 receipt=output.with_suffix('.receipt.json')
 if not output.exists():
  if receipt.exists():raise RuntimeError('Receipt exists without predictions')
  return None
 if not receipt.exists():retain_failed(output,'unreceipted_predictions');return None
 rec=S.read(receipt)
 if rec.get('prediction_sha256')!=S.sha(output):raise RuntimeError('Previously completed predictions changed')
 with np.load(output,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
 validate_prediction_arrays(arrays,frame,labels,patients,input_fingerprint(frame,meta,labels,patients),record,config_sha)
 return rec

def load_model(record,config):
 import torch
 import extension_models
 from worker_inputs import qualify_runtime
 qualify_runtime()
 path=S.cv_run(config['fold'],config['arm'],config['seed'])/'best_valid.pth'
 if S.sha(path)!=record['checkpoint_sha256']:raise RuntimeError('Frozen weights changed')
 ck=torch.load(path,map_location='cpu',weights_only=False)
 if json.dumps(ck['config'],sort_keys=True)!=json.dumps(config,sort_keys=True):raise RuntimeError('Checkpoint configuration mismatch')
 model=extension_models.make_model(config['arm'],pretrained=False)
 model.load_state_dict(ck['model_state_dict'],strict=True);del ck
 model.to('cuda:0').eval()
 if hasattr(model,'device'):model.device=torch.device('cuda:0')
 return model

def infer_arrays(model,frame,metadata,labels,image_root):
 import torch
 from torch.utils.data import DataLoader
 sys_common=S.setup_original()
 import sys
 sys.path.insert(0,str(sys_common.model_code()))
 from custom_dataset import meta_img_dataset_test
 from train_hfpa import build_transforms
 root=Path(image_root).resolve();paths=[]
 for name in frame.img_id:
  p=(root/str(name)).resolve()
  if not p.is_relative_to(root) or not p.is_file():raise RuntimeError('Missing/out-of-root evaluation image')
  paths.append(str(p))
 _,transform=build_transforms(224)
 loader=DataLoader(meta_img_dataset_test(paths,metadata,labels.tolist(),transform),batch_size=16,shuffle=False,num_workers=0,pin_memory=True)
 values=[]
 with torch.inference_mode():
  for images,meta,_ in loader:
   with torch.autocast(device_type='cuda',dtype=torch.float16):logits=model(images.to('cuda:0'),meta.to('cuda:0'))
   values.append(logits.float().cpu().numpy())
 logits=np.concatenate(values).astype(np.float64);shifted=logits-logits.max(1,keepdims=True)
 probabilities=np.exp(shifted);probabilities/=probabilities.sum(1,keepdims=True)
 return logits,probabilities

def evaluate_run(fold,arm,seed,cohorts=('internal','external'),records=None):
 records=frozen_records() if records is None else records;record=records[(fold,arm,seed)]
 folder=S.cv_run(fold,arm,seed);config_path=folder/'config.json'
 if S.sha(config_path)!=record['config_sha256']:raise RuntimeError('Frozen training config changed')
 config=S.read(config_path);model=None;outputs=[]
 try:
  for cohort in cohorts:
   frame,meta,labels,patients=load_evaluation_inputs(config,cohort)
   output=S.prediction_file(cohort,fold,arm,seed);config_sha=S.sha(config_path)
   existing=existing_prediction(output,frame,meta,labels,patients,record,config_sha)
   if existing is not None:outputs.append(existing);continue
   if model is None:model=load_model(record,config)
   started=time.monotonic();image_root=S.pad_images() if cohort=='internal' else S.external_images()
   logits,probabilities=infer_arrays(model,frame,meta,labels,image_root)
   arrays={'img_id':frame.img_id.astype(str).to_numpy(),'patient_id':patients,'y_true':labels,'logits':logits,'probabilities':probabilities,**binding(),'checkpoint_sha256':record['checkpoint_sha256'],'config_sha256':config_sha,'input_sha256':input_fingerprint(frame,meta,labels,patients)}
   validate_prediction_arrays(arrays,frame,labels,patients,str(arrays['input_sha256']),record,config_sha)
   exclusive_npz(output,**arrays)
   receipt={**binding(),'status':'PREDICTIONS_VERIFIED','cohort':cohort,'fold':fold,'arm':arm,'seed':seed,'cases':len(frame),'prediction_file':output.relative_to(S.RESULTS).as_posix(),'prediction_sha256':S.sha(output),'checkpoint_sha256':record['checkpoint_sha256'],'config_sha256':config_sha,'input_sha256':str(arrays['input_sha256']),'seconds':time.monotonic()-started,'probability_ensemble':False}
   S.atomic_json(output.with_suffix('.receipt.json'),receipt);outputs.append(receipt)
   progress('FIXED_MODEL_INFERENCE_PASS',run_id=S.run_id(fold,arm,seed),cohort=cohort,cases=len(frame))
 finally:
  if model is not None:
   import torch
   del model;gc.collect();torch.cuda.empty_cache()
 return outputs

def evaluate_matrix():
 S.verify_manifest(materials=True)
 from run_matrix import freeze_training
 freeze_training();records=frozen_records();rows=[]
 for fold,arm,seed in S.matrix():rows.extend(evaluate_run(fold,arm,seed,records=records))
 if len(rows)!=120:raise RuntimeError('Incomplete internal/external prediction inventory')
 report={**binding(),'status':'ALL_120_EVALUATIONS_PASS','files':rows,'new_formal_fits':60,'all_models_frozen_before_evaluation':True}
 S.atomic_json(S.RESULTS/'evaluation_complete.json',report);return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--all',action='store_true');p.add_argument('--arm',choices=S.ARMS);p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--seed',type=int,choices=S.SEEDS);p.add_argument('--cohort',choices=['internal','external'])
 a=p.parse_args()
 if a.all:print(json.dumps(evaluate_matrix(),indent=2))
 else:
  if None in (a.arm,a.fold,a.seed,a.cohort):p.error('Supply --all or --arm --fold --seed --cohort')
  print(json.dumps(evaluate_run(a.fold,a.arm,a.seed,(a.cohort,)),indent=2))
if __name__=='__main__':main()
