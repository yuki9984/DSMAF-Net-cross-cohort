"""Pure saved-prediction statistics; model probabilities are never ensembled."""
import numpy as np
from statistics_reference import responses,summarize_effect,intervals
METRICS=[('BACC','bacc',0,100.)]+[(f'recall_{c}',f'class_{c}',0,100.) for c in range(6)]+[('NLL','mean',1,1.),('Brier','mean',2,1.)]

def validate_probabilities(y,p):
 y=np.asarray(y,dtype=np.int64);p=np.asarray(p,dtype=np.float64)
 if p.shape[-2:]!=(len(y),6) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(-1),1,atol=1e-6,rtol=0):raise ValueError('Invalid probability array')
 if set(y)!=set(range(6)):raise ValueError('All six diagnostic classes are required')
 return y,p

def fold_weights(y,folds):
 y=np.asarray(y);folds=np.asarray(folds)
 if len(y)!=len(folds) or set(folds)!=set(range(5)):raise ValueError('Five frozen outer folds are required')
 counts=np.array([[np.sum((folds==f)&(y==c)) for c in range(6)] for f in range(5)],dtype=float)
 if (counts==0).any():raise ValueError('Every fold must have all six classes')
 return counts/counts.sum(0,keepdims=True),counts.sum(1)/counts.sum()

def effective_responses(yi,pi,ye,pe,folds):
 yi,pi=validate_probabilities(yi,pi);ye,pe=validate_probabilities(ye,pe)
 if pi.ndim!=4 or pe.ndim!=5 or pi.shape[:2]!=pe.shape[:2] or pe.shape[2]!=5:raise ValueError('Expected method/seed and five external fold axes')
 cw,iw=fold_weights(yi,folds)
 ri=responses(yi,pi);re=responses(ye,pe)
 internal=np.stack((ri[...,0],ri[...,2],ri[...,3]),axis=-1)
 external=np.empty((pe.shape[0],pe.shape[1],len(ye),3),dtype=float)
 external[...,0]=(re[...,0]*cw[:,ye][None,None,:,:]).sum(2)
 external[...,1:]=(re[...,2:]*iw[None,None,:,None,None]).sum(2)
 return internal,external,cw,iw

def metric_vector(values,y):
 return np.array([summarize_effect(values[:,column],y,kind)*scale for _,kind,column,scale in METRICS])

def seed_metrics(effective,y):
 return np.stack([np.stack([metric_vector(values,y) for values in method]) for method in effective])

def calculate_statistics(arms,seeds,yi,patients_i,folds,pi,ye,patients_e,pe,pairs,repeats=2000,bootstrap_seed=2026102901):
 if len(set(arms))!=len(arms) or len(set(seeds))!=len(seeds):raise ValueError('Duplicate model or seed names')
 internal,external,cw,iw=effective_responses(yi,pi,ye,pe,folds)
 if internal.shape[:2]!=(len(arms),len(seeds)):raise ValueError('Name/axis mismatch')
 smi=seed_metrics(internal,yi);sme=seed_metrics(external,ye)
 effects_i={};effects_e={};descriptors={};seed_values={}
 for j,arm in enumerate(arms):
  for metric_index,(metric,kind,column,scale) in enumerate(METRICS):
   key=f'level|{arm}|{metric}'
   effects_i[key]=(internal[j,:,:,column].mean(0),kind,scale)
   effects_e[key]=(external[j,:,:,column].mean(0),kind,scale)
   descriptors[key]={'type':'model_level','comparison':arm,'metric':metric}
   seed_values[key]=(smi[j,:,metric_index],sme[j,:,metric_index])
 for left,right in pairs:
  li=arms.index(left);ri=arms.index(right)
  for metric_index,(metric,kind,column,scale) in enumerate(METRICS):
   key=f'contrast|{left}-minus-{right}|{metric}'
   effects_i[key]=((internal[li,:,:,column]-internal[ri,:,:,column]).mean(0),kind,scale)
   effects_e[key]=((external[li,:,:,column]-external[ri,:,:,column]).mean(0),kind,scale)
   descriptors[key]={'type':'paired_contrast','comparison':left+'-minus-'+right,'metric':metric}
   seed_values[key]=(smi[li,:,metric_index]-smi[ri,:,metric_index],sme[li,:,metric_index]-sme[ri,:,metric_index])
 ci,arrays=intervals(effects_i,effects_e,yi,patients_i,ye,patients_e,seed=bootstrap_seed,repeats=repeats)
 seed_rows=[];summary=[]
 for row in ci:
  key=row['key'];si,se=seed_values[key]
  values=si if row['target']=='internal' else se if row['target']=='external' else se-si
  if not np.isclose(row['estimate'],values.mean(),atol=2e-11,rtol=0):raise RuntimeError('Mean paired seed estimate differs from patient-effect estimate')
  description=descriptors[key]
  unit=('percentage_points' if description['type']=='paired_contrast' or row['target']=='external_minus_internal' else 'percent') if description['metric'].startswith(('BACC','recall_')) else 'mean_loss'
  summary.append({**description,'target':row['target'],'unit':unit,'estimate':row['estimate'],'ci95_low':row['ci95_low'],'ci95_high':row['ci95_high'],'seed_sd':float(np.std(values,ddof=1)),'seeds':len(seeds),'bootstrap_repeats':repeats,'interval_scope':'conditional_fixed_models_unadjusted'})
  for seed,value in zip(seeds,values):seed_rows.append({**description,'target':row['target'],'unit':unit,'seed':int(seed),'value':float(value)})
 return {'summary':summary,'by_seed':seed_rows,'arrays':arrays,'class_fold_weights':cw,'loss_fold_weights':iw}
