"""CPU recomputation for returned predictions; never trains or loads a checkpoint."""
import numpy as np
LABELS=('ACK','BCC','MEL','NEV','SCC','SEK')
METRICS=[('bacc',0,'bacc',100),*[(f'recall_{c}',0,f'class_{i}',100) for i,c in enumerate(LABELS)],('sek_to_mel',1,'class_5',100),('nll',2,'mean',1),('brier',3,'mean',1)]
PAIRS=[('full','image_only'),('late_fusion','image_only'),('full','late_fusion')]

def softmax(logits):
 x=np.asarray(logits,dtype=np.float64)
 if not np.isfinite(x).all() or x.shape[-1]!=6:raise ValueError('Invalid six-class logits')
 e=np.exp(x-np.max(x,axis=-1,keepdims=True));return e/e.sum(axis=-1,keepdims=True)

def case_values(y,prob):
 p=np.asarray(prob,dtype=np.float64);y=np.asarray(y,dtype=int)
 if p.shape[-2:]!=(len(y),6) or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(-1),1,atol=1e-12):raise ValueError('Invalid probabilities')
 pred=np.argmax(p,axis=-1);out=np.empty(p.shape[:-1]+(4,),float)
 out[...,0]=pred==y;out[...,1]=(y==5)&(pred==2)
 correct_probability=np.take_along_axis(p,np.broadcast_to(y,p.shape[:-1])[...,None],axis=-1)[...,0]
 out[...,2]=-np.log(np.clip(correct_probability,1e-12,1))
 out[...,3]=np.sum(p*p,axis=-1)+1-2*correct_probability
 return out

def summarize(v,y,kind):
 v=np.asarray(v);y=np.asarray(y)
 if kind=='bacc':return float(sum(v[y==c].mean() for c in range(6))/6)
 if kind.startswith('class_'):return float(v[y==int(kind[6:])].mean())
 if kind!='mean':raise ValueError('Unknown outcome')
 return float(v.mean())

def metric_row(v,y,extra_flip=False):
 specs=METRICS+([('prediction_flip_percent',4,'mean',100)] if extra_flip else [])
 return {name:summarize(v[:,col],y,kind)*scale for name,col,kind,scale in specs}

def fold_weights(a):
 supports=np.array([np.bincount(a['internal_labels'][a['internal_folds']==f],minlength=6) for f in range(5)])
 if not np.all(supports>0):raise ValueError('Class missing from frozen fold')
 return supports/supports.sum(0),supports.sum(1)/supports.sum()

def combine_recipe(a,ip,ep):
 wc,wi=fold_weights(a);yi=a['internal_labels'];ye=a['external_labels']
 vi=case_values(yi,ip);ve=np.zeros(ep.shape[:2]+(len(ye),4))
 for fold in range(5):
  values=case_values(ye,ep[:,:,fold]);ve[...,:2]+=values[...,:2]*wc[fold,ye][None,None,:,None];ve[...,2:]+=values[...,2:]*wi[fold]
 return vi,ve

def patient_bootstrap(y,patients,effects,rng,repeats=2000):
 """Count-weighted matrix form, independently checked against case resampling."""
 y=np.asarray(y);_,patient_index=np.unique(np.asarray(patients).astype(str),return_inverse=True)
 n=patient_index.max()+1;keys=list(effects);data=np.column_stack([effects[k][0] for k in keys])
 if len(y)!=data.shape[0] or not np.isfinite(data).all():raise ValueError('Effects/IDs do not align')
 counts=np.zeros((n,6),int);totals=np.zeros((n,6,len(keys)))
 for c in range(6):
  mask=y==c;counts[:,c]=np.bincount(patient_index[mask],minlength=n)
  np.add.at(totals[:,c],patient_index[mask],data[mask])
 if np.any(counts.sum(0)==0):raise ValueError('Class missing from cohort')
 w=[];rejected=0
 while len(w)<repeats:
  selected=rng.integers(0,n,n);b=np.bincount(selected,minlength=n)
  if np.any(b@counts==0):rejected+=1;continue
  w.append(b)
 w=np.array(w,dtype=float);den=w@counts;num=(w@totals.reshape(n,-1)).reshape(repeats,6,len(keys))
 point=[];draw=[]
 for j,k in enumerate(keys):
  _,kind,scale=effects[k];point.append(summarize(data[:,j],y,kind)*scale)
  v=(num[:,:,j]/den).mean(1) if kind=='bacc' else num[:,int(kind[6:]),j]/den[:,int(kind[6:])] if kind.startswith('class_') else num[:,:,j].sum(1)/den.sum(1)
  draw.append(v*scale)
 return keys,np.array(point),np.column_stack(draw),rejected

def paired_intervals(internal,external,a,repeats=2000,seed=2026102302):
 if list(internal)!=list(external):raise ValueError('Unpaired effects')
 rng=np.random.default_rng(seed)
 names,pi,bi,ri=patient_bootstrap(a['internal_labels'],a['internal_patients'],internal,rng,repeats)
 _,pe,be,re=patient_bootstrap(a['external_labels'],a['external_patients'],external,rng,repeats)
 rows=[]
 for j,key in enumerate(names):
  for target,point,b in [('internal',pi[j],bi[:,j]),('external',pe[j],be[:,j]),('external_minus_internal',pe[j]-pi[j],be[:,j]-bi[:,j])]:
   lo,hi=np.quantile(b,[.025,.975]);rows.append(dict(key=key,target=target,estimate=float(point),ci95_low=float(lo),ci95_high=float(hi)))
 return rows,dict(names=names,point_internal=pi,point_external=pe,draws_internal=bi,draws_external=be,rejected_internal=ri,rejected_external=re)

def recipe_statistics(a,ip,ep):
 values={'original':combine_recipe(a,a['prob_internal'][:,:3],a['prob_external'][:,:3]),'crossentropy_natural':combine_recipe(a,ip,ep)}
 rows=[];contrasts=[];effects=[{},{}];arms=a['arms'].astype(str).tolist();seeds=a['seeds'][:3]
 for recipe,cohort_values in values.items():
  for cohort,v in zip(['internal','external'],cohort_values):
   for ai,arm in enumerate(arms):
    for si,seed in enumerate(seeds):rows.append(dict(recipe=recipe,cohort=cohort,arm=arm,seed=int(seed),**metric_row(v[ai,si],a[cohort+'_labels'])))
  for first,second in PAIRS:
   ai,bi=arms.index(first),arms.index(second);comparison=first+'-minus-'+second
   for name,col,kind,scale in METRICS:
    key=recipe+'|'+comparison+'|'+name
    for ci,v in enumerate(cohort_values):effects[ci][key]=((v[ai,:,:,col]-v[bi,:,:,col]).mean(0),kind,scale)
   for si,seed in enumerate(seeds):
    pairs={cohort:metric_row(v[ai,si]-v[bi,si],a[cohort+'_labels']) for cohort,v in zip(['internal','external'],cohort_values)}
    for target in ['internal','external','external_minus_internal']:
     metrics=pairs[target] if target in pairs else {k:pairs['external'][k]-pairs['internal'][k] for k in pairs['internal']}
     contrasts.append(dict(recipe=recipe,comparison=comparison,seed=int(seed),target=target,**metrics))
 for old in list(effects[0]):
  if old.startswith('original|'):
   new=old.replace('original|','crossentropy_natural|',1);key=old.replace('original|','recipe_difference|',1)
   for e in effects:e[key]=(e[new][0]-e[old][0],e[old][1],e[old][2])
 intervals,draws=paired_intervals(*effects,a)
 return rows,contrasts,intervals,draws
