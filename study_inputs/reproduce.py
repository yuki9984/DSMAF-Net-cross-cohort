"""CPU verification of released predictions and source-matched metadata audits."""
from pathlib import Path
import argparse,json,hashlib,csv
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
LABELS=['ACK','BCC','MEL','NEV','SCC','SEK']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=None);args=parser.parse_args()
 root=Path(__file__).resolve().parent;dest=args.output or root/'verification_output';dest.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
 for r in manifest['files']:assert sha(root/r['path'])==r['sha256'],r['path']
 prov=json.loads((root/'provenance.json').read_text(encoding='utf-8'))
 raw=pd.read_csv(root/'data/PAD_metadata.csv',keep_default_na=False)
 assert sha(root/'data/PAD_metadata.csv')==prov['official_PAD_metadata_sha256']
 train=pd.read_csv(root/'data/pad_train.csv');valid=pd.read_csv(root/'data/pad_valid.csv')
 assert not set(train.patient_id)&set(valid.patient_id)
 conflict=int((raw.groupby('patient_id').gender.nunique()>1).sum());assert conflict==0
 countrows=[];biopsyrows=[]
 for name,f in [('Training',train),('Model tuning',valid)]:
  z=f[['img_id','gender_FEMALE','gender_MALE']].merge(raw,on='img_id',validate='one_to_one')
  expected=np.stack(z.gender.map({'FEMALE':(1,0),'MALE':(0,1),'':(0,0)}));assert np.array_equal(z[['gender_FEMALE','gender_MALE']].to_numpy(),expected)
  for c in LABELS:
   q=z[z.diagnostic==c];u=int(q.gender.eq('').sum());countrows.append({'partition':name,'diagnosis':c,'images':len(q),'sex_not_recorded_images':u})
  for bp in ['false','true']:
   q=z[z.biopsed.astype(str).str.lower()==bp];biopsyrows.append({'partition':name,'biopsy_confirmed':bp,'images':len(q),'sex_not_recorded_images':int(q.gender.eq('').sum())})
 pd.DataFrame(countrows).to_csv(dest/'recording_by_class.csv',index=False);pd.DataFrame(biopsyrows).to_csv(dest/'recording_by_biopsy.csv',index=False)
 assert sum(r['sex_not_recorded_images'] for r in countrows)==631
 assert all(r['sex_not_recorded_images']==0 for r in biopsyrows if r['biopsy_confirmed']=='true')
 ext=pd.read_csv(root/'data/external_metadata_projection.csv');ext=ext[~ext.img_id.isin(prov['external_excluded_img_ids'])].copy();assert len(ext)==1701
 age=pd.to_numeric(train.age).to_numpy(float);mu=float(age.mean());sd=float(age.std(ddof=0));median=float(np.median(age))
 def features(f,external=False):
  age=pd.to_numeric(f.age,errors='coerce').to_numpy(float);missing=~np.isfinite(age);age[missing]=median
  if external:
   sx=f.gender.fillna('').astype(str).str.strip().str.upper();female=(sx=='FEMININO').to_numpy(float);male=(sx=='MASCULINO').to_numpy(float)
  else:female=f.gender_FEMALE.to_numpy(float);male=f.gender_MALE.to_numpy(float)
  return np.column_stack(((age-mu)/sd,missing,female,male,1-female-male)).astype(np.float32).astype(np.float64)
 def views(x):return {'intercept_only':np.zeros((len(x),1)),'age_only':x[:,:2],'recording_only':1-x[:,4:5],'age_recording':np.column_stack((x[:,:2],1-x[:,4])),'age_recorded_sex':x}
 models={k:LogisticRegression(C=1,solver='lbfgs',max_iter=2000,tol=1e-4,class_weight=None).fit(x,train.diagnostic_number.to_numpy()) for k,x in views(features(train)).items()}
 def metrics(y,p):
  pred=p.argmax(1)
  return {'bacc':float(np.mean([(pred[y==c]==c).mean() for c in range(6)])),'accuracy':float((pred==y).mean()),'brier':float(((p-np.eye(6)[y])**2).sum(1).mean()),'nll':float(-np.log(np.clip(p[np.arange(len(y)),y],np.finfo(float).eps,1)).mean())}
 rows=[];expected=pd.read_csv(root/'expected/demographic_reference_decomposition.csv')
 for name,f,external in [('Model tuning',valid,False),('External',ext,True)]:
  y=f.diagnostic.map({c:i for i,c in enumerate(LABELS)}).to_numpy()
  for key,x in views(features(f,external)).items():
   row={'model':key,'partition':name,**metrics(y,models[key].predict_proba(x))}
   prior=expected[(expected.model==key)&(expected.partition==name)].iloc[0]
   for metric in ['bacc','accuracy','brier','nll']:assert abs(row[metric]-prior[metric])<(1e-12 if metric in ['bacc','accuracy'] else 1e-7),(key,name,metric)
   rows.append(row)
 pd.DataFrame(rows).to_csv(dest/'demographic_references.csv',index=False)
 z=np.load(root/'predictions/primary_external_probabilities.npz',allow_pickle=False);y=z['labels'];neural=[]
 for key in [k for k in z.files if k.startswith('probs_')]:
  arm=key[6:];p=z[key];assert p.shape==(5,1701,6)
  for s in range(5):
   v=metrics(y,p[s]);assert abs(v['bacc']-z['bacc_'+arm][s])<1e-12
   neural.append({'arm':arm,'seed':20261013+s,**v})
 pd.DataFrame(neural).to_csv(dest/'neural_external_metrics.csv',index=False)
 full=z['probs_full'].argmax(2);late=z['probs_late_fusion'].argmax(2)
 delta=((full==y).astype(float)-(late==y).astype(float)).mean(0)
 interaction=((full==y).astype(float)-(z['probs_no_dsdm'].argmax(2)==y).astype(float)-(z['probs_shared_q'].argmax(2)==y).astype(float)+(z['probs_no_dsdm_shared_q'].argmax(2)==y).astype(float)).mean(0)
 rng=np.random.default_rng(20261016);groups=[np.flatnonzero(y==c) for c in range(6)];draws=[];idraws=[]
 for _ in range(2000):
  picks=[rng.choice(ids,len(ids),replace=True) for ids in groups];draws.append(np.mean([delta[q].mean() for q in picks]));idraws.append(np.mean([interaction[q].mean() for q in picks]))
 ci=np.quantile(draws,[.025,.975]);ici=np.quantile(idraws,[.025,.975]);expected_boot=json.loads((root/'expected/paired_contrasts.json').read_text(encoding="utf-8"))
 for actual,expected_value in zip([np.mean([delta[g].mean() for g in groups]),*ci,np.mean([interaction[g].mean() for g in groups]),*ici],expected_boot['values']):assert abs(actual-expected_value)<1e-12
 result={'status':'PASS','raw_development_records_checked':len(train)+len(valid),'patient_sex_conflicts':conflict,'demographic_rows_recomputed':10,'neural_seed_rows_recomputed':35,'primary_difference':float(np.mean([delta[g].mean() for g in groups])),'primary_patient_bootstrap_95ci':ci.tolist(),'interaction':float(np.mean([interaction[g].mean() for g in groups])),'interaction_patient_bootstrap_95ci':ici.tolist(),'scope':'Frozen-prediction statistics and fixed CPU demographic references; no neural retraining, images, or weights required.'}
 (dest/'verification_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
