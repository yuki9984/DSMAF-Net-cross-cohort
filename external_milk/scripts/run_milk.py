"""CPU-only MILK evaluation. Formal runs require all 135 frozen weights."""
import argparse,gc,json,os,pathlib,platform,sys,time,traceback,uuid
import numpy as np
import milk_common as C

def runtime():
 C.setup()
 import torch,torchvision
 torch.set_num_threads(4)
 try:torch.set_num_interop_threads(1)
 except RuntimeError:pass
 return dict(python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,device='cpu',precision='float32',batch_size=8,threads=4)
def source_binding():
 paths=[pathlib.Path(__file__),C.ROOT/'scripts/milk_common.py',C.ROOT/'scripts/milk_statistics.py',
        C.BUNDLE/'extension_models.py',C.BUNDLE/'source_manifest.json']
 return {str(p.relative_to(C.PROJECT)):C.sha(p) for p in paths}
def create_model(arm):
 from model_v2 import make_model as old
 from extension_models import make_model as extension
 if arm in C.ARMS[:3]:m=old(arm,pretrained=False)
 elif arm in C.ARMS[7:]:m=old('late_fusion',pretrained=False)
 else:m=extension(arm,pretrained=False)
 import torch
 if hasattr(m,'device'):m.device=torch.device('cpu')
 return m.float().cpu().eval()
def normalize(rgb):
 import torch
 x=torch.from_numpy(np.array(rgb,copy=True)).permute(0,3,1,2).contiguous().float().div_(255)
 mean=torch.tensor([.485,.456,.406]).view(1,3,1,1)
 std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
 return (x-mean)/std
def qualify():
 env=runtime()
 import torch
 from torchvision import transforms
 from information_conditions import restrict_metadata
 torch.manual_seed(20261104)
 rng=np.random.default_rng(411)
 rgb=rng.integers(0,256,size=(2,224,224,3),dtype=np.uint8)
 expected=torch.stack([transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])(a) for a in rgb])
 x=normalize(rgb)
 if not torch.equal(x,expected):raise RuntimeError('Cached normalization differs from original')
 full=np.array([[0,0,1,0,0],[0,1,0,0,1]],dtype=np.float32)
 checks=[]
 for arm in C.ARMS:
  m=create_model(arm);meta=torch.from_numpy(restrict_metadata(full,C.CONDITIONS[arm]))
  with torch.inference_mode():
   a=m(x,meta).numpy()
   b=np.concatenate([m(x[i:i+1],meta[i:i+1]).numpy() for i in range(2)])
  if a.shape!=(2,6) or not np.isfinite(a).all() or not np.allclose(a,b,atol=1e-4,rtol=1e-4):raise RuntimeError('Batch-partition consistency failed: '+arm)
  checks.append(dict(arm=arm,finite_forward=True,batch_partition_max_abs=float(np.abs(a-b).max())))
  print('SYNTHETIC MODEL PASS',arm,flush=True)
  del m,a,b;gc.collect()
 report=dict(status='CPU_SYNTHETIC_RUNTIME_QUALIFIED',environment=env,source_binding=source_binding(),checks=checks,
             exact_normalization_match=True,clinical_images_used=False,trained_weights_loaded=False,
             cpu_cuda_parity='NOT_TESTED; CPU float32 is the common precision for all MILK models; no assertion of equivalence to prior CUDA AMP predictions')
 C.save(C.ROOT/'reports/cpu_pipeline_qualification.json',report)
 return report
def validate_arrays(z,ids,labels,lesions,binding):
 n=len(ids)
 if z['logits'].shape!=(n,6) or z['probabilities'].shape!=(n,6):raise ValueError('Prediction shape mismatch')
 logits=z['logits'];p=z['probabilities']
 if not np.isfinite(logits).all() or not np.isfinite(p).all():raise ValueError('Nonfinite output')
 q=np.exp(logits-logits.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
 if not np.allclose(q,p,atol=1e-12,rtol=1e-12):raise ValueError('Probabilities not derived from saved logits')
 for key,value in [('image_ids',ids),('lesion_ids',lesions),('labels',labels)]:
  if not np.array_equal(z[key],value):raise ValueError('Case order mismatch: '+key)
 if str(z['binding_sha256'])!=binding:raise ValueError('Input/model/runtime binding mismatch')
def prediction_paths(r):
 p=C.ROOT/'evaluation/predictions'/(C.task_id(r['fold'],r['arm'],r['seed'])+'.npz')
 return p,p.with_suffix('.receipt.json')
def existing_prediction(r,ids,labels,lesions,binding):
 p,receipt=prediction_paths(r)
 if not p.exists() and not receipt.exists():return None
 if not p.exists() or not receipt.exists():raise RuntimeError('Incomplete saved output retained for inspection: '+str(p))
 rec=C.read(receipt)
 if rec['binding_sha256']!=binding or C.sha(p)!=rec['prediction_sha256']:raise RuntimeError('Changed saved prediction')
 with np.load(p,allow_pickle=False) as z:validate_arrays(z,ids,labels,lesions,binding)
 return rec
def predict(r,cases,indices,run_digest,metadata):
 import torch
 ids=np.array([c['isic_id'] for c in cases]);lesions=np.array([c['lesion_id'] for c in cases]);labels=np.array([int(c['label']) for c in cases])
 binding=C.json_hash(dict(run=run_digest,checkpoint=r['checkpoint_sha256'],config=C.json_hash(r['config'])))
 cached=existing_prediction(r,ids,labels,lesions,binding)
 if cached is not None:return cached
 if C.sha(r['path'])!=r['checkpoint_sha256']:raise RuntimeError('Changed frozen checkpoint')
 ck=torch.load(r['path'],map_location='cpu',weights_only=False) # own frozen training artifacts, SHA checked
 if C.json_hash(ck['config'])!=C.json_hash(r['config']):raise RuntimeError('Checkpoint/config disagreement')
 encoding_report=C.read(C.ROOT/'reports/input_encoding_audit.json')['folds'][r['fold']]
 if ck['config']['metadata_age_train_stats']!=encoding_report['training_age_stats']:raise RuntimeError('Training age statistics changed')
 m=create_model(r['arm']);m.load_state_dict(ck['model_state_dict'],strict=True);del ck
 meta=metadata[f"fold_{r['fold']}_{C.CONDITIONS[r['arm']]}"][indices]
 cache=np.load(C.ROOT/'prepared/candidate_rgb224.npy',allow_pickle=False,mmap_mode='r')
 outputs=[];started=time.monotonic()
 with torch.inference_mode():
  for start in range(0,len(ids),8):
   selected=indices[start:start+8]
   x=normalize(cache[selected]);v=torch.from_numpy(meta[start:start+8].copy())
   logits=m(x,v)
   if logits.shape!=(len(selected),6) or not torch.isfinite(logits).all():raise RuntimeError('Invalid model forward')
   outputs.append(logits.numpy().astype(np.float64))
   if start%800==0:print(C.task_id(r['fold'],r['arm'],r['seed']),start,'/',len(ids),flush=True)
 logits=np.concatenate(outputs);p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
 arrays=dict(image_ids=ids,lesion_ids=lesions,labels=labels,logits=logits,probabilities=p,binding_sha256=np.array(binding))
 validate_arrays(arrays,ids,labels,lesions,binding)
 target,receipt=prediction_paths(r);target.parent.mkdir(parents=True,exist_ok=True)
 temp=target.with_name(target.name+'.partial_'+uuid.uuid4().hex)
 with temp.open('xb') as f:np.savez_compressed(f,**arrays)
 if target.exists():raise FileExistsError('Never overwrite previous predictions')
 temp.rename(target)
 rec=dict(status='VERIFIED',fold=r['fold'],arm=r['arm'],seed=r['seed'],cases=len(ids),binding_sha256=binding,
          prediction_sha256=C.sha(target),seconds=time.monotonic()-started,probability_ensemble=False,
          path=str(target.relative_to(C.ROOT)))
 C.save(receipt,rec);del m,cache;gc.collect()
 return rec
def formal():
 records,missing=C.inventory()
 if missing:raise RuntimeError('Missing frozen inputs; run --check. No partial-model evaluation is permitted.')
 env=runtime()
 qualification=C.read(C.ROOT/'reports/cpu_pipeline_qualification.json')
 if qualification['status']!='CPU_SYNTHETIC_RUNTIME_QUALIFIED' or qualification['environment']!=env or qualification['source_binding']!=source_binding():raise RuntimeError('Requalify the exact runtime/source before outcomes')
 S=C.setup();S.verify_manifest(materials=True)
 cases,indices,lock=C.check_cohort()
 # All hashes are checked before the first new external prediction.
 for r in records:
  if C.sha(r['path'])!=r['checkpoint_sha256']:raise RuntimeError('Frozen checkpoint hash mismatch: '+r['path'])
 run=dict(cohort_lock_sha256=C.sha(C.ROOT/'prepared/cohort_lock.json'),environment=env,source_binding=source_binding(),
          records=[{k:r[k] for k in ['fold','arm','seed','checkpoint_sha256','config','campaign']} for r in records],
          all135_frozen_before_evaluation=True,clinical_outcome_used_for_selection=False)
 runfile=C.ROOT/'evaluation/run_lock.json';digest=C.json_hash(run)
 if runfile.exists():
  if C.json_hash(C.read(runfile))!=digest:raise RuntimeError('Run configuration changed; existing outputs retained')
 else:C.save(runfile,run)
 process_lock=C.ROOT/'evaluation/RUNNING.json'
 with process_lock.open('x',encoding='utf-8') as f:json.dump(dict(pid=os.getpid(),run=digest),f)
 try:
  with np.load(C.ROOT/'prepared/candidate_metadata_inputs.npz',allow_pickle=False) as z:
   if not np.array_equal(z['isic_id'][indices],np.array([c['isic_id'] for c in cases])):raise ValueError('Metadata cache order mismatch')
   outputs=[predict(r,cases,indices,digest,z) for r in records]
  C.save(C.ROOT/'evaluation/evaluation_complete.json',dict(status='ALL_135_MILK_EVALUATIONS_VERIFIED',run_sha256=digest,files=outputs))
 except BaseException:
  C.save(C.ROOT/'evaluation'/('failure_'+uuid.uuid4().hex+'.json'),dict(traceback=traceback.format_exc(),outputs_retained=True))
  raise
 finally:process_lock.unlink()
def check():
 records,missing=C.inventory()
 report=dict(status='WAITING_FOR_FROZEN_WEIGHTS' if missing else 'ALL_WEIGHT_PATHS_PRESENT_HASHES_CHECKED_AT_RUN',
             available_record_count=len(records),planned_records=135,missing=missing,
             cohort_locked=(C.ROOT/'prepared/cohort_lock.json').exists(),
             runtime_qualified=(C.ROOT/'reports/cpu_pipeline_qualification.json').exists(),formal_inference_started=(C.ROOT/'evaluation/run_lock.json').exists())
 C.save(C.ROOT/'reports/run_readiness.json',report);print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
 g.add_argument('--check',action='store_true');g.add_argument('--qualify',action='store_true');g.add_argument('--run',action='store_true')
 a=p.parse_args()
 if a.check:check()
 elif a.qualify:qualify()
 else:formal()
if __name__=='__main__':main()

