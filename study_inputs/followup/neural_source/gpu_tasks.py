"""GPU stages for the approved intervention and 45-fit sensitivity campaign."""
import json,sys,time,gc
from pathlib import Path
import numpy as np,pandas as pd
from followup_settings import *
from inference_core import probabilities,direct_logits,cached_logits,reproduction
from interventions import scenarios

def setup(args):
 records,root=verify_manifest();weights=R.find_weights(PROJECT,args.weights_root,records);pad=R.find_pad(PROJECT,args.pad_images,weights)
 R.input_checks(PROJECT,weights,pad,records,RESULTS/'input_audit');device=R.qualify_runtime(RESULTS/'input_audit')
 import platform
 assert platform.python_version()==plan()['inference']['python'],'Use the original Python 3.11.9 worker environment'
 import os
 os.environ['DSMAF_PAD_IMAGES']=str(pad)
 sys.path.insert(0,str(root));return records,weights,pad,device

def checkpoint_model(path,record,config,device):
 from model_v2 import make_model
 import torch
 ck=R.safe_checkpoint(path);assert ck['arm']==record['arm'] and ck['seed']==record['seed'] and ck['epoch']==record['best_epoch']
 assert R.canonical(ck['config'])==config,'Checkpoint configuration differs'
 assert all(torch.isfinite(v).all().item() for v in ck['model_state_dict'].values() if v.is_floating_point()),'Non-finite checkpoint weights'
 m=make_model(record['arm'],pretrained=False);m.load_state_dict(ck['model_state_dict'],strict=True);del ck
 return m.to(device).eval()

def loader(cache,frame,metadata,indices=None):
 from torch.utils.data import DataLoader,Subset
 ds=cache.dataset(frame,metadata)
 if indices is not None:ds=Subset(ds,list(indices))
 return DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,pin_memory=True)

def progress(stage,**kw):
 from datetime import datetime,timezone
 v={'phase':stage,'utc':datetime.now(timezone.utc).isoformat(),'planned_new_training_fits':45,**kw};atomic_json(RESULTS/'worker_progress.json',v);print(json.dumps(v),flush=True)

def metadata_frame(cohort,fold):
 return R.frame_for_fold(fold) if cohort=='internal' else pd.read_csv(PRIOR/'inputs/external_cases.csv')

def metadata_array(frame,config,cohort):
 from data_v2 import AgeStats,encode_pad_metadata
 from external_encoding import encode_external_metadata
 return (encode_pad_metadata if cohort=='internal' else encode_external_metadata)(frame,AgeStats(**config['metadata_age_train_stats']))

def perturb_stage(args):
 import torch
 from train_hfpa import build_transforms
 records,weights,pad,device=setup(args);_,transform=build_transforms(224);done=[];started=time.monotonic()
 for cohort in ['internal','external']:
  allframe=pd.concat([R.frame_for_fold(f) for f in range(5)]).sort_values('img_id').reset_index(drop=True) if cohort=='internal' else metadata_frame('external',0)
  cache=R.ImageCache(allframe,pad if cohort=='internal' else PRIOR/'inputs/external_images',transform)
  for r in sorted(records,key=lambda z:(z['fold'],z['seed'],ARMS.index(z['arm']))):
   name=model_name(r);frame=metadata_frame(cohort,r['fold']);config=read(R.fit_dir(r)/'config.json');metadata=metadata_array(frame,config,cohort);matrices,indices=scenarios(metadata,cohort,r['fold'])
   indexfile=RESULTS/'interventions/permutation_indices'/(f'internal_fold_{r["fold"]:02d}.npz' if cohort=='internal' else 'external.npz')
   if indexfile.exists():
    with np.load(indexfile,allow_pickle=False) as old:assert np.array_equal(old['indices'],indices) and np.array_equal(old['img_ids'].astype(str),frame.img_id.astype(str))
   else:atomic_npz(indexfile,indices=indices,img_ids=frame.img_id.astype(str).to_numpy(dtype='U'),plan_sha256=np.asarray(sha256(PLAN)))
   lock={'plan_sha256':sha256(PLAN),'checkpoint_sha256':r['checkpoint_sha256'],'source_manifest_sha256':sha256(BUNDLE/'source_manifest.json'),'cohort':cohort,'model':name,'permutation_indices_sha256':__import__('hashlib').sha256(indices.tobytes()).hexdigest()}
   out=RESULTS/'interventions'/cohort/(name+'.npz');side=out.with_suffix('.json')
   if out.exists():
    with np.load(out,allow_pickle=False) as z:assert json.loads(str(z['provenance']))==lock and np.array_equal(z['img_ids'].astype(str),frame.img_id.astype(str));saved=json.loads(str(z['checks']))
    rec={**r,**lock,**saved,'artifact_sha256':sha256(out)}
    if side.exists():assert read(side)==rec,'Saved intervention sidecar differs'
    else:atomic_json(side,rec)
    done.append(rec);progress('INTERVENTION_RESUME_PASS',completed=len(done),planned=150,model=name,cohort=cohort);continue
   assert not side.exists(),'Missing artifact for existing sidecar'
   model=checkpoint_model(weights/R.weight_relative(r),r,config,device);t=time.monotonic();direct=direct_logits(model,loader(cache,frame,metadata),device);reference=reference_for(r,cohort,frame);checks={'observed_direct_vs_original':reproduction(probabilities(direct),reference)}
   probe=np.unique(np.concatenate([np.arange(start,min(start+16,len(frame))) for start in [0,(len(frame)//2//16)*16,((len(frame)-1)//16)*16]]))
   if r['arm']=='image_only':
    altered=direct_logits(model,loader(cache,frame,matrices['sex_unknown'],probe),device);checks['image_only_unknown_spot_check']=reproduction(probabilities(altered),probabilities(direct[probe]));assert not model.use_metadata
    data={'logits_observed':direct};checks['metadata_invariant_by_construction']=True;checks['perturbed_neural_conditions_executed']=1
   else:
    logits=cached_logits(model,loader(cache,frame,metadata),matrices,device);checks['cached_vs_direct']=reproduction(probabilities(logits['observed']),probabilities(direct));checks['cached_vs_original']=reproduction(probabilities(logits['observed']),reference)
    checks['perturbed_direct_spot_checks']={}
    for key in ['sex_unknown','sex_recorded_perm_000','sex_recorded_perm_050','sex_recorded_perm_099']:
     p=probabilities(direct_logits(model,loader(cache,frame,matrices[key],probe),device));checks['perturbed_direct_spot_checks'][key]=reproduction(probabilities(logits[key][probe]),p)
    data={'logits_observed':logits['observed'],'logits_sex_unknown':logits['sex_unknown'],'logits_sex_recorded_perm':np.stack([logits[f'sex_recorded_perm_{k:03d}'] for k in range(100)])};checks['metadata_invariant_by_construction']=False;checks['perturbed_neural_conditions_executed']=101
   checks['seconds']=time.monotonic()-t
   atomic_npz(out,img_ids=frame.img_id.astype(str).to_numpy(dtype='U'),patient_ids=frame.patient_id.astype(str).to_numpy(dtype='U'),labels=frame['diagnostic_number' if cohort=='internal' else 'label'].to_numpy(int),provenance=np.asarray(json.dumps(lock,sort_keys=True)),checks=np.asarray(json.dumps(checks,sort_keys=True)),**data)
   rec={**r,**lock,**checks,'artifact_sha256':sha256(out)};atomic_json(side,rec);done.append(rec)
   del model,direct,data;gc.collect();torch.cuda.empty_cache();progress('INTERVENTION_MODEL_PASS',completed=len(done),planned=150,cohort=cohort,model=name,seconds=checks['seconds'],elapsed_seconds=time.monotonic()-started)
  del cache;gc.collect()
 atomic_json(RESULTS/'interventions/complete.json',{'status':'PASS','model_cohort_records':150,'metadata_model_cohort_records':100,'runs':done,'new_training_fits':0,'plan_sha256':sha256(PLAN)})


def evaluate_recipe(args):
 import torch
 from train_hfpa import build_transforms
 from data_v2 import encode_pad_metadata
 records=freeze_training();_,weights,pad,device=setup(args);_,transform=build_transforms(224);done=[]
 for cohort in ['internal','external']:
  allframe=pd.concat([R.frame_for_fold(f) for f in range(5)]).sort_values('img_id').reset_index(drop=True) if cohort=='internal' else metadata_frame('external',0)
  cache=R.ImageCache(allframe,pad if cohort=='internal' else PRIOR/'inputs/external_images',transform)
  for r in records:
   out=RESULTS/'recipe45/evaluation'/cohort/(model_name(r)+'.npz');side=out.with_suffix('.json');frame=metadata_frame(cohort,r['fold'])
   lock={**r,'plan_sha256':sha256(PLAN),'training_complete_sha256':sha256(RESULTS/'recipe45/training_complete.json'),'cohort':cohort}
   if out.exists():
    with np.load(out,allow_pickle=False) as z:assert json.loads(str(z['provenance']))==lock and np.array_equal(z['img_ids'].astype(str),frame.img_id.astype(str))
    rec={**lock,'artifact_sha256':sha256(out)}
    if side.exists():assert read(side)==rec
    else:atomic_json(side,rec)
   else:
    assert not side.exists();config=read(cv_run(r['fold'],r['arm'],r['seed'])/'config.json');metadata=metadata_array(frame,config,cohort)
    model=checkpoint_model(cv_run(r['fold'],r['arm'],r['seed'])/'best_valid.pth',r,config,device)
    z=direct_logits(model,loader(cache,frame,metadata),device);atomic_npz(out,img_ids=frame.img_id.astype(str).to_numpy(dtype='U'),patient_ids=frame.patient_id.astype(str).to_numpy(dtype='U'),labels=frame['diagnostic_number' if cohort=='internal' else 'label'].to_numpy(int),logits=z,provenance=np.asarray(json.dumps(lock,sort_keys=True)))
    rec={**lock,'artifact_sha256':sha256(out)};atomic_json(side,rec);del model,z;gc.collect();torch.cuda.empty_cache()
   done.append(rec);progress('RECIPE45_EVALUATION_PASS',completed=len(done),planned=90,cohort=cohort,model=model_name(r))
  del cache;gc.collect()
 atomic_json(RESULTS/'recipe45/evaluation_complete.json',{'status':'PASS','models':45,'model_cohort_records':90,'runs':done,'training_complete_sha256':sha256(RESULTS/'recipe45/training_complete.json')})
