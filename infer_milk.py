"""Re-run a released frozen checkpoint on the exact published MILK input cache."""
from pathlib import Path
import argparse,json,sys
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--image-cache',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--limit',type=int)
p.add_argument('--batch-size',type=int,default=8)
a=p.parse_args()
if a.output.exists():raise FileExistsError('Use a new prediction output path.')
import numpy as np,torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
from model_api import load_frozen_checkpoint
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True);cfg=ck['config'];arm=cfg['arm'];fold=cfg['fold']
model,loaded_cfg=load_frozen_checkpoint(a.checkpoint,arm,device='cpu');del ck
D=R/'study_inputs/milk'
indices=np.load(D/'evaluation_cache_indices.npy',allow_pickle=False)
with np.load(D/'candidate_metadata_inputs.npz',allow_pickle=False) as z:
 condition='age_recorded' if arm.endswith('_age_recorded') else 'age_only' if arm.endswith('_age_only') else 'full_age_sex'
 metadata=z[f'fold_{fold}_{condition}'][indices]
cache=np.load(a.image_cache,allow_pickle=False,mmap_mode='r')
if a.limit is not None:indices=indices[:a.limit];metadata=metadata[:a.limit]
logits=[]
mean=torch.tensor([.485,.456,.406]).view(1,3,1,1);std=torch.tensor([.229,.224,.225]).view(1,3,1,1)
with torch.inference_mode():
 for start in range(0,len(indices),a.batch_size):
  rgb=np.array(cache[indices[start:start+a.batch_size]],copy=True)
  x=torch.from_numpy(rgb).permute(0,3,1,2).contiguous().float().div_(255)
  logits.append(model((x-mean)/std,torch.from_numpy(metadata[start:start+a.batch_size].copy())).numpy().astype(np.float64))
v=np.concatenate(logits);prob=np.exp(v-v.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
a.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output,logits=v,probabilities=prob)
reference=D/'predictions'/f"fold_{fold:02d}_{arm}_seed{cfg['seed']}.npz"
with np.load(reference,allow_pickle=False) as z:
 delta=float(np.max(np.abs(v-z['logits'][:len(v)])))
print(json.dumps({'status':'INFERENCE_COMPLETE','images':len(v),'arm':arm,'fold':fold,'seed':cfg['seed'],'max_absolute_logit_difference_from_archive':delta,'probabilities_ensembled':False},indent=2))
