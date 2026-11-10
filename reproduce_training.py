"""Run one preserved training condition with public inputs and explicit local image paths."""
from pathlib import Path
import argparse,os,subprocess,sys
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--arm',required=True,choices=['full','image_only','late_fusion','metablock_cv','rg_dermnet_adapted_cv','dsmaf_age_only','dsmaf_age_recorded','late_age_only','late_age_recorded'])
p.add_argument('--fold',type=int,choices=range(5),default=0)
p.add_argument('--seed',type=int,choices=[20261013,20261014,20261015],default=20261013)
p.add_argument('--pad-images',type=Path)
p.add_argument('--initialization-file',type=Path)
p.add_argument('--output-root',type=Path,default=R/'new_runs')
p.add_argument('--check-inputs',action='store_true')
a=p.parse_args()
if a.arm in ('full','image_only','late_fusion'):
 cmd=[sys.executable,'-B',str(R/'historical_study/train_reproduce.py'),'--study-inputs',str(R/'study_inputs'),'--experiment','cv','--arm',a.arm,'--fold',str(a.fold),'--seed',str(a.seed)]
 if a.pad_images:cmd+=['--pad-images',str(a.pad_images.resolve())]
 if a.check_inputs:cmd+=['--validate-inputs']
 else:
  if not a.initialization_file:p.error('--initialization-file is required for training')
  os.environ['DSMAF_IMAGENET_WEIGHTS']=str(a.initialization_file.resolve())
  os.environ['DSMAF_OUTPUT_ROOT']=str(a.output_root.resolve())
  os.environ['TORCH_HOME']=str(a.initialization_file.resolve().parent.parent.parent)
 subprocess.run(cmd,check=True);raise SystemExit()
folder=R/('extension30' if a.arm.startswith('late_age_') else 'extension60')
sys.path.insert(0,str(folder));import extension_settings as S
if a.pad_images:S.pad_images=lambda:a.pad_images.resolve()
if a.initialization_file:
 import shutil
 cache=a.output_root.resolve()/'cache/hub/checkpoints'/a.initialization_file.name
 cache.parent.mkdir(parents=True,exist_ok=True)
 if not cache.exists():shutil.copyfile(a.initialization_file.resolve(),cache)
 S.initialization_file=lambda:cache
S.RESULTS=a.output_root.resolve()/folder.name
S.verify_manifest()
if a.check_inputs:
 import json
 print(json.dumps(S.validate_inputs(),indent=2));raise SystemExit()
if not a.pad_images or not a.initialization_file:p.error('Training requires --pad-images and --initialization-file')
import torch
if not torch.cuda.is_available():raise RuntimeError('The preserved AMP training recipe requires CUDA.')
os.environ['TORCH_HOME']=str(a.initialization_file.resolve().parent.parent.parent)
S.ensure_pretrained()
from train_extension import load_preserved_trainer
trainer=load_preserved_trainer(a.arm)
sys.argv=['preserved_train_cv','--arm',a.arm,'--fold',str(a.fold),'--seed',str(a.seed)]
trainer.main()
