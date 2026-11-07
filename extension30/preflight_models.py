"""Disposable complete-model checks with no clinical data or study checkpoint."""
import argparse,gc,json,platform,time
import extension_settings as S
from campaign_support import binding,bound_report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--qualify-training',action='store_true');a=p.parse_args()
 S.verify_manifest();S.setup_original()
 import torch
 import extension_models
 import numpy as np
 from information_conditions import restrict_metadata
 from train_extension import state_fingerprint
 if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
 if a.qualify_training:
  if a.device!='cuda':raise RuntimeError('Training qualification requires CUDA')
  from worker_inputs import qualify_runtime
  runtime=qualify_runtime();S.ensure_pretrained()
 else:runtime={'torch':torch.__version__,'python':platform.python_version()}
 # Import architecture definitions before paired seeding, as the preserved trainer does.
 import model_v2
 extension_models.ORIGINAL_MAKE_MODEL=getattr(model_v2,'_paa_base_model',model_v2.make_model)
 rows=[];initials={};device=torch.device(a.device)
 import sys
 sys.path.insert(0,str(S.setup_original().model_code()))
 from study_runtime import set_seed
 for arm in S.ARMS:
  set_seed(S.SEEDS[0],fast_mode=False);model=extension_models.make_model(arm,pretrained=a.qualify_training)
  if arm.startswith('late_age_'):initials[arm]=state_fingerprint(model)
  model.to(device).eval()
  if hasattr(model,'device'):model.device=device
  if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
  image=torch.randn(2,3,224,224,device=device)
  observed=np.array([[0,0,1,0,0],[1,0,0,0,1]],dtype=np.float32)
  metadata=torch.tensor(restrict_metadata(observed,S.ENCODINGS[arm]),device=device)
  logits=model(image,metadata)
  if logits.shape!=(2,6) or not torch.isfinite(logits).all():raise RuntimeError('Invalid output: '+arm)
  loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1],device=device));loss.backward()
  grads=[parameter.grad for parameter in model.parameters() if parameter.grad is not None]
  if not grads or not all(torch.isfinite(g).all() for g in grads):raise RuntimeError('Invalid active gradients: '+arm)
  peak2=torch.cuda.max_memory_allocated() if device.type=='cuda' else None
  count=sum(p.numel() for p in model.parameters())
  # Release every batch-2 graph/gradient reference BEFORE measuring batch 16.
  del image,metadata,logits,loss,grads
  model.zero_grad(set_to_none=True);gc.collect()
  if device.type=='cuda':torch.cuda.empty_cache()
  benchmark={}
  if a.qualify_training:
   model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
   scaler=torch.amp.GradScaler('cuda',enabled=True)
   xx=torch.randn(16,3,224,224,device=device)
   mm=torch.tensor(restrict_metadata(np.tile(observed,(8,1)),S.ENCODINGS[arm]),device=device)
   yy=torch.arange(16,device=device)%6
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
   # GradScaler skips overflowing steps and backs off its initial scale.
   # Qualify one actual finite optimizer step, with a bounded calibration.
   amp_backoffs=[]
   for synthetic_attempt in range(17):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda',dtype=torch.float16):
     prediction=model(xx,mm);ll=torch.nn.functional.cross_entropy(prediction,yy)
    if not torch.isfinite(ll):raise RuntimeError('Nonfinite synthetic loss: '+arm)
    scaler.scale(ll).backward();scaler.unscale_(optimizer)
    finite_gradients=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    previous_scale=scaler.get_scale()
    scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
    if finite_gradients:
     if not optimizer.state:raise RuntimeError('Synthetic optimizer did not step: '+arm)
     if not all(torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('Nonfinite updated parameters: '+arm)
     break
    if optimizer.state or not scaler.get_scale()<previous_scale:raise RuntimeError('AMP overflow did not skip/back off: '+arm)
    amp_backoffs.append({'from':previous_scale,'to':scaler.get_scale()})
   else:raise RuntimeError('Nonfinite batch16 gradients after bounded AMP calibration: '+arm)
   benchmark={'batch16_peak_bytes':torch.cuda.max_memory_allocated(),'synthetic_step_seconds':time.perf_counter()-started,'synthetic_optimizer_steps':1,'synthetic_attempts':synthetic_attempt+1,'amp_scale_backoffs':amp_backoffs,'final_amp_scale':scaler.get_scale()}
   del xx,mm,yy,prediction,ll,optimizer,scaler
  rows.append({'arm':arm,'parameters':count,'batch2_peak_bytes':peak2,**benchmark})
  del model;gc.collect()
  if device.type=='cuda':torch.cuda.empty_cache()
 if len(initials)!=2 or len(set(initials.values()))!=1:raise RuntimeError('Information conditions changed the full model initialization')
 report={**binding(),'status':'ALL_MODEL_PREFLIGHT_PASS' if a.qualify_training else 'BASIC_MODEL_CHECKS_PASS','runtime':runtime,'device':a.device,'rows':rows,'paired_initial_states_match':True,'patient_data_used':False,'study_training_started':False,'models_retained':False}
 S.atomic_json(S.RESULTS/('gpu_preflight.json' if a.qualify_training else 'basic_model_preflight.json'),report)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
