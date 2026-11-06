"""Lesion-stratified, paired conditional uncertainty for fixed external models."""
import numpy as np
LABELS=('ACK','BCC','MEL','NEV','SCC','SEK')
METRICS=('BACC',)+tuple('recall_'+c for c in LABELS)+('NLL','Brier')
def fold_weights(labels,folds):
 y=np.asarray(labels);f=np.asarray(folds)
 if y.shape!=f.shape or set(y)!=set(range(6)) or set(f)!=set(range(5)):raise ValueError('Invalid original outer folds')
 n=np.array([[np.sum((y==c)&(f==j)) for c in range(6)] for j in range(5)],dtype=float)
 if (n==0).any():raise ValueError('Missing class in original fold')
 return n/n.sum(0),n.sum(1)/n.sum()
def response(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=np.float64)
 if p.shape!=(len(y),6) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(-1),1,atol=1e-6,rtol=0):raise ValueError('Invalid model probabilities')
 correct=(p.argmax(-1)==y).astype(float)
 nll=-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))
 brier=((p-np.eye(6)[y])**2).sum(-1)
 return np.stack([correct,nll,brier],axis=-1)
def collapse_folds(y,probabilities,class_weights,loss_weights):
 """Each model is scored first. Probability vectors are never averaged."""
 y=np.asarray(y,dtype=int);p=np.asarray(probabilities)
 cw=np.asarray(class_weights);iw=np.asarray(loss_weights)
 if p.shape!=(5,len(y),6) or cw.shape!=(5,6) or iw.shape!=(5,):raise ValueError('Wrong fold axes')
 if (cw<0).any() or (iw<0).any() or not np.allclose(cw.sum(0),1) or not np.isclose(iw.sum(),1):raise ValueError('Invalid original fold weights')
 out=np.zeros((len(y),3))
 for f in range(5):
  r=response(y,p[f]);out[:,0]+=r[:,0]*cw[f,y];out[:,1:]+=r[:,1:]*iw[f]
 return out
def metric_vector(y,v):
 y=np.asarray(y);v=np.asarray(v)
 if v.shape!=(len(y),3) or set(y)!=set(range(6)):raise ValueError('Six classes and three per-lesion responses required')
 recalls=np.array([v[y==c,0].mean() for c in range(6)])
 return np.r_[recalls.mean()*100,recalls*100,v[:,1:].mean(0)]
def stratified_indices(y,rng):
 """One image per unique lesion, resampled within each diagnosis class."""
 return np.concatenate([rng.choice(np.flatnonzero(y==c),size=np.sum(y==c),replace=True) for c in range(6)])
def bootstrap_levels(y,effective,repeats=2000,seed=20261104):
 """effective axes: method, paired seed, lesion, response(correct/NLL/Brier).
Draws are shared by every method/seed; fitted models are held fixed.
"""
 y=np.asarray(y,dtype=int);v=np.asarray(effective,dtype=np.float64)
 if v.ndim!=4 or v.shape[2:]!=(len(y),3) or not np.isfinite(v).all() or set(y)!=set(range(6)):raise ValueError('Invalid effective responses')
 sm=np.array([[metric_vector(y,s) for s in method] for method in v])
 mean_response=v.mean(1)
 rng=np.random.default_rng(seed)
 draws=np.empty((repeats,v.shape[0],9))
 # Use shared within-class counts; batch matrix multiplication limits memory.
 for start in range(0,repeats,64):
  count=min(64,repeats-start)
  counts=np.zeros((count,len(y)),dtype=np.float64)
  for b in range(count):
   idx=stratified_indices(y,rng);counts[b]=np.bincount(idx,minlength=len(y))
  recalls=[]
  for c in range(6):
   mask=y==c
   recalls.append(counts[:,mask]@mean_response[:,mask,0].T/mask.sum())
  rec=np.stack(recalls,-1)
  draws[start:start+count,:,0]=rec.mean(-1)*100
  draws[start:start+count,:,1:7]=rec*100
  for j in range(2):draws[start:start+count,:,7+j]=counts@mean_response[:,:,1+j].T/len(y)
 return sm,draws
def summarize(arms,seeds,y,lesions,effective,pairs,repeats=2000,seed=20261104):
 if len(set(lesions))!=len(y):raise ValueError('This analysis requires one image per distinct lesion; never substitute patient labels')
 if len(arms)!=len(set(arms)) or len(seeds)!=len(set(seeds)) or np.shape(effective)[:2]!=(len(arms),len(seeds)):raise ValueError('Identity axes differ')
 sm,draws=bootstrap_levels(y,effective,repeats,seed)
 specifications=[('model_level',a,sm[i],draws[:,i]) for i,a in enumerate(arms)]
 for left,right in pairs:
  li=arms.index(left);ri=arms.index(right)
  specifications.append(('paired_contrast',left+'-minus-'+right,sm[li]-sm[ri],draws[:,li]-draws[:,ri]))
 summary=[];by_seed=[]
 for kind,name,sv,bv in specifications:
  point=sv.mean(0);lo,hi=np.quantile(bv,[.025,.975],axis=0)
  for m,metric in enumerate(METRICS):
   unit=('percentage_points' if kind=='paired_contrast' else 'percent') if m<7 else 'mean_loss'
   summary.append(dict(type=kind,comparison=name,metric=metric,unit=unit,estimate=float(point[m]),ci95_low=float(lo[m]),ci95_high=float(hi[m]),seed_sd=float(sv[:,m].std(ddof=1)),resamples=repeats,interval_scope='conditional_fitted_models_class_stratified_lesions_unadjusted',cluster_unit='lesion_not_patient'))
   for i,s in enumerate(seeds):by_seed.append(dict(type=kind,comparison=name,metric=metric,unit=unit,seed=int(s),value=float(sv[i,m])))
 return dict(summary=summary,by_seed=by_seed,level_seed_metrics=sm,level_bootstrap_draws=draws)

