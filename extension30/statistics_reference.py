"""Patient-cluster statistics shared across recipes and fixed-model conditions."""
import numpy as np

def responses(y,p):
 p=np.asarray(p,dtype=np.float64);y=np.asarray(y,dtype=int);pred=p.argmax(-1)
 correct=(pred==y).astype(float);sek_mel=((y==5)&(pred==2)).astype(float)
 nll=-np.log(np.clip(np.take_along_axis(p,np.broadcast_to(y,p.shape[:-1])[...,None],axis=-1)[...,0],1e-12,1))
 brier=((p-np.eye(6)[y])**2).sum(-1)
 return np.stack((correct,sek_mel,nll,brier),axis=-1)

def summarize_effect(values,y,kind):
 if kind=='bacc':return float(np.mean([values[y==c].mean() for c in range(6)]))
 if kind.startswith('class_'):return float(values[y==int(kind[6:])].mean())
 assert kind=='mean';return float(values.mean())

def cluster_draws(y,patients,effects,rng,repeats=2000):
 y=np.asarray(y,dtype=int);_,inv=np.unique(np.asarray(patients).astype(str),return_inverse=True);n=int(inv.max())+1;names=list(effects);data=np.column_stack([effects[k][0] for k in names]);assert data.shape[0]==len(y) and np.isfinite(data).all()
 counts=np.stack([np.bincount(inv[y==c],minlength=n) for c in range(6)],1);assert (counts.sum(0)>0).all()
 sums=np.zeros((n,6,len(names)))
 for c in range(6):np.add.at(sums[:,c],inv[y==c],data[y==c])
 point=np.array([summarize_effect(data[:,j],y,effects[k][1])*effects[k][2] for j,k in enumerate(names)])
 draws=[];rejected=0
 while len(draws)<repeats:
  selected=rng.integers(0,n,n);den=counts[selected].sum(0)
  if (den==0).any():rejected+=1;continue
  num=sums[selected].sum(0);ratios=num/den[:,None];row=[]
  for j,k in enumerate(names):
   kind=effects[k][1];v=ratios[:,j].mean() if kind=='bacc' else ratios[int(kind[6:]),j] if kind.startswith('class_') else num[:,j].sum()/den.sum()
   row.append(v*effects[k][2])
  draws.append(row)
 return names,point,np.stack(draws),rejected

def intervals(internal,external,yi,pi,ye,pe,seed=2026102302,repeats=2000):
 assert list(internal)==list(external);rng=np.random.default_rng(seed)
 names,pointi,bi,badi=cluster_draws(yi,pi,internal,rng,repeats);_,pointe,be,bade=cluster_draws(ye,pe,external,rng,repeats)
 rows=[]
 for j,key in enumerate(names):
  for target,point,draw in [('internal',pointi[j],bi[:,j]),('external',pointe[j],be[:,j]),('external_minus_internal',pointe[j]-pointi[j],be[:,j]-bi[:,j])]:
   lo,hi=np.quantile(draw,[.025,.975]);rows.append({'key':key,'target':target,'estimate':float(point),'ci95_low':float(lo),'ci95_high':float(hi)})
 return rows,{'names':names,'point_internal':pointi,'point_external':pointe,'draws_internal':bi,'draws_external':be,'rejected_internal':badi,'rejected_external':bade}
