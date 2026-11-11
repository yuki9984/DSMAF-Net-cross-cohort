"""Build input-only metadata from original PAD training statistics; no fitting on MILK10k."""
import json,pathlib,sys,hashlib
import numpy as np,pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
sys.path.insert(0,str(PROJECT/'protocol_v2_paa_extension_20261029_portable_v2'))
import extension_settings as S
S.setup_original()
from data_v2 import AgeStats,_age_features
from information_conditions import restrict_metadata

def encode(frame,stats,condition):
 if not set(frame.sex.fillna(''))<={'male','female',''}:raise ValueError('Unmapped sex encoding')
 az,am=_age_features(frame.age,stats);sex=frame.sex.fillna('')
 x=np.column_stack([az,am,sex.eq('female'),sex.eq('male'),sex.eq('')]).astype(np.float32)
 return restrict_metadata(x,condition)
def main():
 frame=pd.read_csv(ROOT/'prepared/candidate_cases.csv');rows=[];arrays={}
 returned=PROJECT/'output/PAA60_verified_return_20261103/received/outputs/fits'
 for fold in range(5):
  configs=[json.loads(p.read_text(encoding="utf-8")) for p in (returned/f'fold_{fold:02d}').glob('*/config.json')]
  assert len(configs)==12
  stats_dict=configs[0]['metadata_age_train_stats']
  assert all(c['metadata_age_train_stats']==stats_dict for c in configs)
  _,fitted=S.frames_for_fold(fold)
  assert all(abs(getattr(fitted,k)-v)<1e-10 for k,v in stats_dict.items())
  stats=AgeStats(**stats_dict)
  conditions={k:encode(frame,stats,k) for k in ['full_age_sex','age_only','age_recorded']}
  full=conditions['full_age_sex'];b=conditions['age_only'];c=conditions['age_recorded']
  assert full.shape==(4478,5) and full[:,1].sum()==17
  assert np.array_equal(full[:,:2],b[:,:2]) and np.array_equal(full[:,:2],c[:,:2])
  assert np.all(b[:,2:]==[0,0,1]) and np.all(c[:,2:]==[.5,.5,0])
  assert np.all(full[:,4]==0) and np.all(full[:,2]+full[:,3]==1)
  for condition,array in conditions.items():arrays[f'fold_{fold}_{condition}']=array
  rows.append({'fold':fold,'training_age_stats':stats_dict,'cases':len(frame),'age_missing':17,'sex_recorded':len(frame)})
 arrays['isic_id']=frame.isic_id.to_numpy(dtype=str);arrays['lesion_id']=frame.lesion_id.to_numpy(dtype=str)
 np.savez_compressed(ROOT/'prepared/candidate_metadata_inputs.npz',**arrays)
 report={'status':'FIVE_FOLD_THREE_CONDITION_ENCODING_PASS','cases':len(frame),'metadata_columns':['age_z','age_missing','sex_female','sex_male','sex_unknown'],'diagnosis_used_as_feature':False,'external_age_statistics_fitted':False,'no_inference':True,'folds':rows}
 (ROOT/'reports/input_encoding_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
