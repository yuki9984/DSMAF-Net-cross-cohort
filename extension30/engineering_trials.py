"""Four isolated one-epoch engineering trials; never reuse their weights."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,gc,json,sys,time
import extension_settings as S
from campaign_support import binding,bound_report,progress

def run_trials():
 S.verify_manifest();bound_report(S.RESULTS/'gpu_preflight.json','ALL_MODEL_PREFLIGHT_PASS')
 complete=S.RESULTS/'engineering_trials.json'
 if complete.exists():
  report=bound_report(complete,'ALL_TWO_ENGINEERING_TRIALS_PASS')
  for row in report['runs']:
   checkpoint=S.RESULTS/row['checkpoint']
   if S.sha(checkpoint)!=row['checkpoint_sha256']:raise RuntimeError('Engineering checkpoint changed')
  return report
 import numpy as np,torch
 from train_extension import load_preserved_trainer,state_fingerprint
 import extension_models
 S.ensure_pretrained();rows=[];initials={}
 attempt='attempt_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
 base=S.RESULTS/'engineering_trials'/attempt
 for arm in S.ARMS:
  progress('ENGINEERING_TRIAL_STARTED',arm=arm,epochs=1,fold=0,seed=S.SEEDS[0])
  trainer=load_preserved_trainer(arm);trainer.set_seed(S.SEEDS[0],fast_mode=False)
  bundles,stats,split_hash=trainer.make_bundles(0)
  train_loader,valid_loader,counts=trainer.make_loaders(bundles,16,0,S.SEEDS[0])
  model=extension_models.make_model(arm,pretrained=True)
  initial=state_fingerprint(model)
  if arm.startswith('late_age_'):initials[arm]=initial
  model.to('cuda:0');trainer.set_seed(S.SEEDS[0],fast_mode=False)
  device=torch.device('cuda:0');criterion=trainer.BalancedSoftmaxLoss(np.array(counts)).to(device)
  optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
  scaler=torch.amp.GradScaler('cuda',enabled=True);torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
  train_loss,train_acc=trainer.train_one_epoch(model,train_loader,criterion,optimizer,device,scaler=scaler,use_amp=True,epoch=1,total_epochs=1)
  valid=trainer.evaluate(model,valid_loader,criterion,device,bundles['valid'],6,'valid',use_amp=True,tta_horizontal_flip=False,epoch=1,total_epochs=1)
  if not np.isfinite(train_loss) or not np.isfinite(valid['bac']):raise RuntimeError('Nonfinite engineering metric')
  peak=torch.cuda.max_memory_allocated();elapsed=time.perf_counter()-started
  folder=base/arm;folder.mkdir(parents=True,exist_ok=False)
  config={'protocol':'paa_engineering_only','arm':arm,'fold':0,'seed':S.SEEDS[0],'trial':True,'epochs':1,'information_condition':S.ENCODINGS[arm],'metadata_age_train_stats':stats.to_dict(),'initial_state_sha256':initial,**binding()}
  S.atomic_json(folder/'config.json',config)
  checkpoint=folder/'engineering_only.pth';temporary=folder/'engineering_only.pth.tmp'
  torch.save({'model_state_dict':model.state_dict(),'config':config,'epoch':1},temporary);temporary.rename(checkpoint)
  model.eval();sample=next(iter(valid_loader));images,metadata=sample[0].to(device),sample[1].to(device)
  with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.float16):expected=model(images,metadata).float().cpu()
  del model,optimizer,criterion,scaler,train_loader,valid_loader;gc.collect();torch.cuda.empty_cache()
  saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
  restored=extension_models.make_model(arm,pretrained=False)
  restored.load_state_dict(saved['model_state_dict'],strict=True);restored.to(device).eval()
  if hasattr(restored,'device'):restored.device=device
  with torch.no_grad(),torch.autocast(device_type='cuda',dtype=torch.float16):actual=restored(images,metadata).float().cpu()
  delta=float((actual-expected).abs().max())
  if delta>1e-4 or not torch.equal(actual.argmax(1),expected.argmax(1)):raise RuntimeError('Engineering save/reload agreement failed: '+arm)
  row={'arm':arm,'trial':True,'epochs_completed':1,'fold':0,'seed':S.SEEDS[0],'seconds':elapsed,'peak_bytes':peak,'reload_max_logit_delta':delta,'checkpoint':checkpoint.relative_to(S.RESULTS).as_posix(),'checkpoint_sha256':S.sha(checkpoint),'initial_state_sha256':initial,'train_images':len(bundles['train'].dataframe),'valid_images':len(bundles['valid'].dataframe),'outer_or_external_used':False}
  S.atomic_json(folder/'trial_report.json',row);rows.append(row)
  del restored,saved,images,metadata,expected,actual,sample,bundles;gc.collect();torch.cuda.empty_cache()
 if len(initials)!=2 or len(set(initials.values()))!=1:raise RuntimeError('Trial initializations differ across information conditions')
 report={**binding(),'status':'ALL_TWO_ENGINEERING_TRIALS_PASS','runs':rows,'new_formal_fits_completed':0,'trial_weights_reused_for_formal_training':False,'new_outer_or_external_evaluated':False,'timing_estimates':{'hours_if_36_epochs_per_fit_excluding_overheads':sum(r['seconds'] for r in rows)*15*36/3600,'hours_at_150_epoch_cap_excluding_overheads':sum(r['seconds'] for r in rows)*15*150/3600,'scope':'Engineering timing only; never changes fixed stopping rules'}}
 S.atomic_json(complete,report);return report

def main():print(json.dumps(run_trials(),indent=2))
if __name__=='__main__':main()
