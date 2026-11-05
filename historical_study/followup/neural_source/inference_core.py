"""Equivalent frozen inference; reuse only metadata-invariant image computation."""
import numpy as np

def probabilities(logits):
 x=np.asarray(logits,dtype=np.float64);assert np.isfinite(x).all();x=x-x.max(-1,keepdims=True);p=np.exp(x);return p/p.sum(-1,keepdims=True)

def pooled(model,features):
 import torch
 import torch.nn.functional as F
 pyramid=model.afpn(list(features)) if model.use_afpn else features
 assert len(pyramid)==4
 base=pyramid[3];refined=model.da_block(base)
 if refined.shape[-2:]!=base.shape[-2:]:refined=F.interpolate(refined,size=base.shape[-2:],mode='bilinear',align_corners=False)
 final=base+model.da_res_scale*refined
 if model.use_all_afpn_outputs:return torch.cat([F.adaptive_avg_pool2d(v if i!=3 else final,1).flatten(1) for i,v in enumerate(pyramid)],1)
 return F.adaptive_avg_pool2d(final,1).flatten(1)

def direct_logits(model,loader,device,amp=True):
 import torch
 parts=[];model.eval()
 with torch.inference_mode():
  for batch in loader:
   x,m=batch[:2]
   with torch.amp.autocast(device.type,enabled=amp):z=model(x.to(device),m.to(device))
   parts.append(z.detach().cpu().numpy().astype(np.float32))
 return np.concatenate(parts)

def cached_logits(model,loader,matrices,device,amp=True):
 import torch
 assert model.use_metadata
 parts={k:[] for k in matrices};cursor=0;model.eval()
 with torch.inference_mode():
  for batch in loader:
   x=batch[0].to(device);start=cursor;cursor+=len(x)
   with torch.amp.autocast(device.type,enabled=amp):
    features=model.image_backbone(x);adapted=tuple(getattr(model,f'adapter{i+1}')(f) for i,f in enumerate(features))
    invariant=None if model.use_dsmaf else pooled(model,adapted)
    for name,metadata in matrices.items():
     q=model.meta_encoder(torch.from_numpy(metadata[start:cursor]).to(device))
     if model.use_dsmaf:
      z=pooled(model,tuple(getattr(model,f'tfpfm{i+1}')(f,q) for i,f in enumerate(adapted)))
     else:z=invariant
     if model.use_late_metadata_fusion:z=torch.cat((z,q),1)
     y=model.classifier(z);parts[name].append(y.detach().cpu().numpy().astype(np.float32))
 assert cursor==len(loader.dataset)
 return {k:np.concatenate(v) for k,v in parts.items()}

def reproduction(actual,reference,tolerance=1e-4):
 a=np.asarray(actual);b=np.asarray(reference);assert a.shape==b.shape and np.isfinite(a).all()
 error=float(abs(a-b).max());changed=int((a.argmax(-1)!=b.argmax(-1)).sum())
 assert error<=tolerance and changed==0,f'Prediction reproduction failed: max error {error}, hard changes {changed}'
 return {'max_abs_probability_error':error,'changed_hard_predictions':changed}
