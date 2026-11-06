"""Fixed information conditions; no diagnosis or case identifier enters encoding."""
import numpy as np
CONDITIONS=('full_age_sex','age_only','age_recorded','constant_metadata')
def restrict_metadata(values,condition):
 x=np.asarray(values,dtype=np.float32)
 if x.ndim!=2 or x.shape[1]!=5 or not np.isfinite(x).all():
  raise ValueError('Expected finite metadata with shape [N,5]')
 if condition not in CONDITIONS:raise ValueError('Unknown information condition')
 if not np.isin(x[:,1:],[0,1]).all() or not np.allclose(x[:,2:].sum(1),1):
  raise ValueError('Input must use the original missing flag and three-state one-hot sex')
 out=x.copy()
 if condition=='age_only':out[:,2:]=[0,0,1]
 elif condition=='age_recorded':
  recorded=x[:,2]+x[:,3]
  out[:,2]=recorded/2;out[:,3]=recorded/2;out[:,4]=1-recorded
 elif condition=='constant_metadata':out[:]=[0,0,0,0,1]
 return out
