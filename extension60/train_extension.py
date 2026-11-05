"""Preserve the grouped-CV loop and adapt only model/input/output bindings."""
from pathlib import Path
import argparse,importlib.util,json,sys,hashlib
import extension_settings as S

def state_fingerprint(model):
 h=hashlib.sha256()
 for name,tensor in model.state_dict().items():
  value=tensor.detach().cpu().contiguous()
  h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
 return h.hexdigest()

def load_preserved_trainer(arm):
 if arm not in S.ARMS:raise ValueError('Unplanned model')
 S.setup_original()
 import data_v2,model_v2
 import extension_models
 from information_conditions import restrict_metadata
 if not hasattr(data_v2,'_paa_base_encoder'):data_v2._paa_base_encoder=data_v2.encode_pad_metadata
 if not hasattr(model_v2,'_paa_base_model'):model_v2._paa_base_model=model_v2.make_model
 extension_models.ORIGINAL_MAKE_MODEL=model_v2._paa_base_model
 data_v2.encode_pad_metadata=lambda frame,stats:restrict_metadata(data_v2._paa_base_encoder(frame,stats),S.ENCODINGS[arm])
 info={}
 def make_current(selected_arm,pretrained=True):
  model=extension_models.make_model(selected_arm,pretrained=pretrained)
  info['initial_state_sha256']=state_fingerprint(model)
  return model
 model_v2.make_model=make_current
 sys.modules['settings']=S
 source=S.PROJECT/'protocol_v2_supplemental_20261019/train_cv.py'
 spec=importlib.util.spec_from_file_location('preserved_paa_cv_trainer',source)
 trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
 write_original=trainer.atomic_json
 def write_current(path,value):
  if Path(path).name=='config.json':
   value.update(protocol='paa_60_base_20261029',extension_plan_sha256=S.sha(S.PLAN),extension_manifest_sha256=S.sha(S.BUNDLE/'source_manifest.json'),information_condition=S.ENCODINGS[arm],initial_state_sha256=info['initial_state_sha256'],training_entry_sha256=S.sha(Path(__file__)),method_adaptation='Common protocol and information; reference heads and visual representations may differ',trial=False)
  write_original(path,value)
 trainer.atomic_json=write_current
 return trainer

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--arm',choices=S.ARMS);p.add_argument('--fold',type=int,choices=range(5));p.add_argument('--seed',type=int,choices=S.SEEDS);p.add_argument('--validate-inputs',action='store_true')
 a=p.parse_args()
 if a.validate_inputs:print(json.dumps(S.validate_inputs(),indent=2));return
 if None in (a.arm,a.fold,a.seed):p.error('--arm, --fold and --seed are required')
 S.verify_manifest()
 from campaign_support import bound_report
 bound_report(S.RESULTS/'gpu_preflight.json','ALL_MODEL_PREFLIGHT_PASS')
 bound_report(S.RESULTS/'engineering_trials.json','ALL_FOUR_ENGINEERING_TRIALS_PASS')
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA worker required')
 S.ensure_pretrained();trainer=load_preserved_trainer(a.arm)
 sys.argv=[str(S.PROJECT/'protocol_v2_supplemental_20261019/train_cv.py'),'--arm',a.arm,'--fold',str(a.fold),'--seed',str(a.seed)]
 trainer.main()
if __name__=='__main__':main()
