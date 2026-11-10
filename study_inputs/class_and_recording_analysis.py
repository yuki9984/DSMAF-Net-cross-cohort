"""Post hoc saved-prediction class decomposition and full-PAD recording checks."""
from pathlib import Path
import argparse,json,zipfile,io,hashlib
import numpy as np,pandas as pd
class DirectorySource:
 def __init__(self,root):self.root=Path(root)
 def __enter__(self):return self
 def __exit__(self,*args):return False
 def read(self,name):return (self.root/name.removeprefix('OnlineResource2/')).read_bytes()
def source(path):return DirectorySource(path) if Path(path).is_dir() else zipfile.ZipFile(path)
LABELS=['ACK','BCC','MEL','NEV','SCC','SEK'];ARMS=['full','image_only','late_fusion']
def run(archive,output):
 archive=Path(archive);output=Path(output);output.mkdir(parents=True,exist_ok=True)
 with source(archive) as Z:
  raw=pd.read_csv(io.BytesIO(Z.read('OnlineResource2/data/PAD_metadata.csv')),keep_default_na=False,dtype={'img_id':str,'patient_id':str})
  A=np.load(io.BytesIO(Z.read('OnlineResource2/matched_cohort/probabilities.npz')),allow_pickle=False)
  arrays={k:A[k] for k in A.files}
  sex=raw.gender.astype(str).str.strip().str.upper();unknown=sex.eq('');assert sex.isin(['','FEMALE','MALE']).all()
  assert len(raw)==2298 and raw.img_id.is_unique and raw.patient_id.nunique()==1373
  assert int(unknown.sum())==804 and set(raw.loc[unknown,'diagnostic'])=={'ACK','NEV','SEK'}
  assert raw.loc[unknown,'biopsed'].astype(str).str.lower().eq('false').all()
  full=[]
  for c in LABELS+['All classes']:
   f=raw if c=='All classes' else raw[raw.diagnostic.eq(c)];miss=f.gender.astype(str).str.strip().eq('')
   full.append({'diagnosis':c,'images':len(f),'patients':f.patient_id.nunique(),'sex_not_recorded_images':int(miss.sum()),'missing_percent':float(miss.mean()*100),'unrecorded_sex_with_biopsy_confirmation':int((miss&f.biopsed.astype(str).str.lower().eq('true')).sum())})
  pd.DataFrame(full).to_csv(output/'full_PAD_recording_by_class.csv',index=False)
  rolechecks=[]
  for fold in range(5):
   for role in ['train','valid','test']:
    s=pd.read_csv(io.BytesIO(Z.read(f'OnlineResource2/internal75/splits/fold_{fold:02d}_{role}.csv')),dtype={'img_id':str,'patient_id':str})
    m=s.merge(raw[['img_id','patient_id','gender','diagnostic','age','biopsed']],on='img_id',validate='one_to_one',suffixes=('','_raw'))
    assert len(m)==len(s) and m.patient_id.eq(m.patient_id_raw).all()
    expect=np.column_stack((m.gender.eq('FEMALE'),m.gender.eq('MALE'))).astype(int)
    mismatch=int(np.any(m[['gender_FEMALE','gender_MALE']].to_numpy()!=expect,axis=1).sum());assert mismatch==0
    assert np.array_equal(m.diagnostic_number,m.diagnostic.map({c:i for i,c in enumerate(LABELS)}))
    agebad=int((pd.to_numeric(m.age)-pd.to_numeric(m.age_raw)).abs().gt(1e-10).sum());assert agebad==0
    rolechecks.append({'fold':fold,'role':role,'images':len(m),'patients':m.patient_id.nunique(),'sex_not_recorded_images':int(m.gender.eq('').sum()),'sex_encoding_mismatches':mismatch,'age_value_mismatches':agebad})
  pd.DataFrame(rolechecks).to_csv(output/'all_15_fold_role_encoding_checks.csv',index=False)
 yi=arrays['internal_labels'];ye=arrays['external_labels'];I=arrays['prob_internal'];E=arrays['prob_external'];folds=arrays['internal_folds']
 support=np.stack([np.bincount(yi[folds==f],minlength=6) for f in range(5)]);wc=support/support.sum(0)
 hits={'internal':{},'external':{}};seedrows=[]
 for ai,arm in enumerate(ARMS):
  hits['internal'][arm]=(I[ai].argmax(-1)==yi).astype(float)
  h=(E[ai].argmax(-1)==ye).astype(float);hits['external'][arm]=sum(wc[f,ye][None,:]*h[:,f] for f in range(5))
  for cohort,y in [('internal',yi),('external',ye)]:
   for si,seed in enumerate(arrays['seeds']):
    for ci,c in enumerate(LABELS):seedrows.append({'cohort':cohort,'arm':arm,'seed':int(seed),'diagnosis':c,'recall_percent':float(hits[cohort][arm][si,y==ci].mean()*100)})
 pd.DataFrame(seedrows).to_csv(output/'all_180_class_seed_cohort_recalls.csv',index=False)
 comparisons=[('full','late_fusion'),('full','image_only'),('late_fusion','image_only')]
 rng=np.random.default_rng(20261022);boots={};points={};rejects={}
 for cohort,y,pids in [('internal',yi,arrays['internal_patients']),('external',ye,arrays['external_patients'])]:
  patients,inv=np.unique(pids.astype(str),return_inverse=True);n=len(patients)
  d=np.stack([(hits[cohort][a]-hits[cohort][b]).mean(0) for a,b in comparisons],axis=1)
  counts=np.stack([np.bincount(inv[y==c],minlength=n) for c in range(6)],axis=1);sums=np.zeros((n,6,3))
  for c in range(6):
   for j in range(3):sums[:,c,j]=np.bincount(inv[y==c],weights=d[y==c,j],minlength=n)
  draws=[];bad=0
  while len(draws)<2000:
   selected=rng.integers(0,n,n);den=counts[selected].sum(0)
   if (den==0).any():bad+=1;continue
   draws.append(sums[selected].sum(0)/den[:,None])
  boots[cohort]=np.stack(draws);points[cohort]=np.stack([d[y==c].mean(0) for c in range(6)]);rejects[cohort]=bad
 classrows=[];baccrows=[]
 for j,(a,b) in enumerate(comparisons):
  comparison=a+'-minus-'+b
  for target,point,draw in [('internal',points['internal'][:,j],boots['internal'][:,:,j]),('external',points['external'][:,j],boots['external'][:,:,j]),('external_minus_internal',points['external'][:,j]-points['internal'][:,j],boots['external'][:,:,j]-boots['internal'][:,:,j])]:
   means=draw.mean(1);lo,hi=np.quantile(means,[.025,.975]);baccrows.append({'comparison':comparison,'target':target,'estimate':float(point.mean()*100),'ci95_low':float(lo*100),'ci95_high':float(hi*100)})
   for c,name in enumerate(LABELS):
    lo,hi=np.quantile(draw[:,c],[.025,.975]);classrows.append({'comparison':comparison,'target':target,'diagnosis':name,'estimate':float(point[c]*100),'ci95_low':float(lo*100),'ci95_high':float(hi*100),'BACC_component_pp':float(point[c]*100/6)})
 classes=pd.DataFrame(classrows);bacc=pd.DataFrame(baccrows);classes.to_csv(output/'class_contrasts_patient_cluster_intervals.csv',index=False);bacc.to_csv(output/'all_three_BACC_contrasts.csv',index=False)
 # Retain the earlier contrast estimates and confidence stream exactly.
 with source(archive) as Z:old=pd.read_csv(io.BytesIO(Z.read('OnlineResource2/matched_cohort/expected/paired_patient_cluster_intervals.csv')))
 old=old[old.metric.eq('bacc')];m=bacc.merge(old,on=['comparison','target'],suffixes=('_new','_old'));maxerr=max(float(abs(m[k+'_new']-m[k+'_old']).max()) for k in ['estimate','ci95_low','ci95_high']);assert len(m)==6 and maxerr<1e-10
 parts=classes.query("comparison == 'full-minus-image_only' and target == 'external_minus_internal'");total=float(parts.BACC_component_pp.sum());acksek=float(parts[parts.diagnosis.isin(['ACK','SEK'])].BACC_component_pp.sum())
 single=[]
 for s in range(5):single.append(float(np.mean([hits['external']['full'][s,ye==c].mean()-hits['external']['image_only'][s,ye==c].mean() for c in range(6)])*100))
 report={'status':'PASS','post_hoc':True,'full_PAD_images':2298,'full_PAD_patients':1373,'unrecorded_sex_images':804,'unrecorded_sex_percent':804/2298*100,'unrecorded_sex_diagnoses':sorted(set(raw.loc[unknown,'diagnostic'])),'unrecorded_sex_biopsy_confirmed_images':0,'fold_role_partitions_audited':15,'fold_role_rows_checked':sum(r['images'] for r in rolechecks),'fold_role_sex_encoding_mismatches':0,'fold_role_age_value_mismatches':0,'class_seed_cohort_rows':180,'class_contrast_interval_rows':54,'BACC_contrast_rows':9,'earlier_six_BACC_intervals_max_error':maxerr,'bootstrap_seed':20261022,'bootstrap_repeats':2000,'rejected_missing_class_draws':rejects,'full_minus_image_external_seed_differences_pp':single,'external_full_above_image_seed_count':sum(v>0 for v in single),'ACK_and_SEK_BACC_change_component_pp':acksek,'total_BACC_contrast_change_pp':total,'decomposition_is_causal':False,'new_neural_training_runs':0}
 (output/'class_and_full_PAD_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');np.savez_compressed(output/'paired_class_bootstrap_draws.npz',internal=boots['internal'],external=boots['external'])
 print(json.dumps(report,indent=2),flush=True)
 return report
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--archive',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();run(a.archive,a.output)
