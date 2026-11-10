"""Label-agnostic sex inputs; missing positions remain fixed in permutations."""
import numpy as np
REPETITIONS=100;SEED=2026102301

def permutation_indices(metadata,cohort,fold=0):
 m=np.asarray(metadata);assert m.ndim==2 and m.shape[1]==5
 sex=m[:,2:];assert np.isfinite(m).all() and np.isin(sex,[0,1]).all() and np.all(sex.sum(1)==1)
 assert cohort in ('internal','external') and fold in range(5)
 known=np.flatnonzero(m[:,4]==0);out=[]
 scope=[0,fold] if cohort=='internal' else [1,0]
 for repeat in range(REPETITIONS):
  rng=np.random.default_rng(np.random.SeedSequence([SEED,*scope,repeat]));idx=np.arange(len(m),dtype=np.int32);idx[known]=rng.permutation(known);out.append(idx)
 return np.stack(out)

def scenarios(observed,cohort,fold=0):
 m=np.asarray(observed,dtype=np.float32);indices=permutation_indices(m,cohort,fold)
 out={'observed':m.copy()};u=m.copy();u[:,2:]=[0,0,1];out['sex_unknown']=u
 for r,idx in enumerate(indices):
  z=m.copy();z[:,2:]=m[idx,2:];assert np.array_equal(z[:,4],m[:,4]);out[f'sex_recorded_perm_{r:03d}']=z
 return out,indices
