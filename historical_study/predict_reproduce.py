"""Per-checkpoint inference using preserved preprocessing and model sources.

This entry supplies portable paths; it has not been independently validated on a
GPU. Statistical verification of the archived predictions is a separate route.
"""
from pathlib import Path
import argparse, importlib.util, json, sys
ROOT=Path(__file__).resolve().parent
LABELS=('ACK','BCC','MEL','NEV','SCC','SEK')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--study-inputs',type=Path,required=True,help='Locally authorized study-input folder; never included in the public code candidate.')
 p.add_argument('--cohort',choices=['internal','external'],required=True)
 p.add_argument('--checkpoint',type=Path)
 p.add_argument('--config',type=Path,help='Training configuration, also usable without weights for input validation.')
 p.add_argument('--images',type=Path)
 p.add_argument('--output',type=Path)
 p.add_argument('--condition',choices=['observed','unknown-sex'],default='observed')
 p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
 p.add_argument('--validate-inputs',action='store_true')
 a=p.parse_args()
 import numpy as np,pandas as pd
 from train_reproduce import verify_release
 verify_release()
 sys.path.insert(0,str(ROOT/'model_sources/protocol_v2'))
 import common,data_v2
 common.model_code=lambda:ROOT/'model_sources/compute_checks/20261014_transfer/gpu_probe/code'
 checkpoint=None
 if a.config:config=json.loads(a.config.read_text(encoding='utf-8-sig'))
 elif a.checkpoint:
  import torch
  checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False);config=checkpoint['config']
 else:p.error('Supply --config or --checkpoint.')
 stats=data_v2.AgeStats(**config['metadata_age_train_stats'])
 if a.cohort=='internal':
  if 'fold' not in config:p.error('Internal outer-test inference requires a grouped-CV configuration with a fold index.')
  frame=pd.read_csv(a.study_inputs/'internal75/splits'/f"fold_{int(config['fold']):02d}_test.csv")
  metadata=data_v2.encode_pad_metadata(frame,stats);labels=frame.diagnostic_number.to_numpy(dtype=np.int64)
  patients=frame.patient_id.astype(str).to_numpy()
 else:
  frame=pd.read_csv(a.study_inputs/'data/external_metadata_projection.csv')
  excluded=set(json.loads((a.study_inputs/'provenance.json').read_text(encoding='utf-8'))['external_excluded_img_ids'])
  frame=frame.loc[~frame.img_id.isin(excluded)].copy()
  if len(frame)!=1701 or frame.img_id.duplicated().any():raise RuntimeError('The locked external cohort must contain 1,701 unique images after the 18 exclusions.')
  # Preserve the case order used in the corresponding frozen evaluation.
  archive='matched_cohort/probabilities.npz' if 'fold' in config else 'predictions/primary_external_probabilities.npz'
  with np.load(a.study_inputs/archive,allow_pickle=False) as frozen:order=frozen['external_images' if 'fold' in config else 'img_ids'].astype(str)
  if set(frame.img_id.astype(str))!=set(order):raise RuntimeError('External case membership differs from the locked evaluation.')
  frame=frame.set_index('img_id').loc[order].reset_index()
  sys.path.insert(0,str(ROOT/'matched_cohort/inference_source'))
  from external_encoding import encode_external_metadata
  metadata=encode_external_metadata(frame,stats)
  labels=frame.diagnostic.map({n:i for i,n in enumerate(LABELS)}).to_numpy(dtype=np.int64)
  patients=frame.img_id.astype(str).to_numpy()
 if a.condition=='unknown-sex':metadata[:,2:]=np.array([0,0,1],dtype=np.float32)
 if a.images:
  missing=[str(x) for x in frame.img_id if not (a.images/str(x)).is_file()]
  if missing:raise FileNotFoundError('Missing images: '+str(missing[:3]))
 if a.validate_inputs:
  print(json.dumps({'status':'INPUTS_VERIFIED','cohort':a.cohort,'condition':a.condition,'images':len(frame),'metadata_shape':list(metadata.shape),'unknown_sex':int(metadata[:,4].sum()),'labels':np.bincount(labels,minlength=6).tolist(),'image_files_checked':a.images is not None,'neural_inference_performed':False},indent=2));return
 if not a.checkpoint or not a.images or not a.output:p.error('Inference requires --checkpoint, --images and --output.')
 if a.output.exists():raise FileExistsError('Output exists; archived predictions must not be overwritten.')
 if a.output.suffix!='.npz':p.error('--output must have the .npz extension.')
 import torch
 from torch.utils.data import DataLoader
 from model_v2 import make_model
 sys.path.insert(0,str(common.model_code()))
 from study_runtime import meta_img_dataset_test
 from study_runtime import build_transforms
 sys.path.insert(0,str(ROOT/'followup/neural_source'))
 from inference_core import direct_logits,probabilities
 if checkpoint is None:checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
 if checkpoint['config']['arm']!=config['arm'] or checkpoint['config']['metadata_age_train_stats']!=config['metadata_age_train_stats']:raise RuntimeError('Checkpoint/config mismatch.')
 device=torch.device(a.device)
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable.')
 model=make_model(config['arm'],pretrained=False);model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.to(device)
 _,transform=build_transforms(224)
 ds=meta_img_dataset_test([str(a.images/str(x)) for x in frame.img_id],metadata,labels.tolist(),transform)
 loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,pin_memory=device.type=='cuda')
 logits=direct_logits(model,loader,device,amp=device.type=='cuda')
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('xb') as stream:np.savez_compressed(stream,img_id=frame.img_id.astype(str).to_numpy(),patient_id=patients,y_true=labels,metadata=metadata,logits=logits,probabilities=probabilities(logits))
 print(json.dumps({'status':'PREDICTIONS_WRITTEN','output':str(a.output),'images':len(frame),'cohort':a.cohort,'condition':a.condition,'arm':config['arm'],'seed':config['seed'],'probability_ensemble':False},indent=2))
if __name__=='__main__':main()
