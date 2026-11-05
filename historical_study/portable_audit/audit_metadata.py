"""Portable descriptive metadata audit. No neural training or causal attribution."""
from pathlib import Path
import argparse,json,html,platform
from audit_reporting import validate_config, enrich_report, write_report
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def measurements(y,p,classes):
 pred=p.argmax(1);counts=np.bincount(y,minlength=len(classes))
 bacc=float(np.mean([(pred[y==i]==i).mean() for i in range(len(classes))])) if (counts>0).all() else None
 return {'bacc':bacc,'accuracy':float((pred==y).mean()),
 'nll':float(-np.log(np.clip(p[np.arange(len(y)),y],np.finfo(float).eps,1)).mean()),
 'brier':float(((p-np.eye(len(classes))[y])**2).sum(1).mean()),
 'class_support_complete':bool((counts>0).all())}
def execute(config,output):
 config=Path(config).resolve();cfg=read(config);base=config.parent
 output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
 validate_config(cfg,base,output)
 classes=cfg['classes'];labels={s:i for i,s in enumerate(classes)}
 cohorts={};missing=[];standard=[];inconsistency=[]
 for spec in cfg['cohorts']:
  name=spec['name'];assert name not in cohorts
  raw=pd.read_csv(base/spec['csv'],keep_default_na=False,dtype={spec['id']:str,spec['patient']:str});ids=raw[spec['id']].astype(str)
  assert ids.is_unique and ids.str.len().gt(0).all(),name+': duplicated/missing image IDs'
  patient=raw[spec['patient']].astype(str);assert patient.str.len().gt(0).all()
  diag=raw[spec['diagnosis']].astype(str);assert diag.isin(classes).all(),name+': unmapped diagnosis'
  rawsex=raw[spec['sex']].astype(str).str.strip().str.upper()
  mapping={str(k).strip().upper():v for k,v in spec['sex_mapping'].items()}
  assert rawsex.isin(mapping).all(),name+': unmapped raw sex values; supply explicit mapping'
  sex=rawsex.map(mapping);assert sex.isin(['female','male','unknown']).all()
  age=pd.to_numeric(raw[spec['age']],errors='coerce')
  blanks=raw[spec['age']].astype(str).str.strip().eq('')
  assert not (age.isna()&~blanks).any(),name+': non-numeric age token; clean or map explicitly'
  lo,hi=cfg.get('age_range',[0,120]);assert age.dropna().between(lo,hi).all(),name+': age outside configured range'
  f=pd.DataFrame({'image_id':ids,'patient_id':patient,'diagnosis':diag,'age':age,'sex':sex,'age_missing':age.isna().astype(int),'sex_recorded':sex.ne('unknown').astype(int)})
  group=spec.get('reference_standard');f['reference_standard']=raw[group].astype(str) if group else '(not supplied)'
  cohorts[name]={'spec':spec,'raw':raw,'frame':f}
  for diagnosis,part in f.groupby('diagnosis',sort=True):
   for field,m in [('age',part.age.isna()),('sex',part.sex.eq('unknown'))]:
    missing.append({'cohort':name,'diagnosis':diagnosis,'field':field,'images':len(part),'missing_images':int(m.sum()),'missing_fraction':float(m.mean())})
  for (diagnosis,ref),part in f.groupby(['diagnosis','reference_standard'],dropna=False,sort=True):
   for field,m in [('age',part.age.isna()),('sex',part.sex.eq('unknown'))]:
    standard.append({'cohort':name,'diagnosis':diagnosis,'reference_standard':ref,'field':field,'images':len(part),'missing_images':int(m.sum()),'missing_fraction':float(m.mean())})
  conflicts=f.groupby('patient_id').sex.nunique()
  inconsistency.append({'cohort':name,'patients':f.patient_id.nunique(),'patients_with_multiple_sex_states':int((conflicts>1).sum())})
 train=cohorts[cfg['training_cohort']]['frame']
 observed=train.age.dropna()
 if len(observed):
  mu=float(observed.mean());sd=float(observed.std(ddof=0));med=float(observed.median())
  age_status='ESTIMABLE' if np.isfinite(sd) and sd>0 else 'CONSTANT_TRAINING_AGE'
 else:mu=sd=med=None;age_status='NO_OBSERVED_TRAINING_AGE'
 age_estimable=age_status=='ESTIMABLE'
 stats={'mean':mu,'std':sd,'median':med}
 encoded=[];states=[]
 for name,entry in cohorts.items():
  spec=entry['spec'];f=entry['frame'];raw=entry['raw'];cols=spec.get('encoding',{})
  if spec.get('encoded_csv'):
   z=pd.read_csv(base/spec['encoded_csv'],keep_default_na=False,dtype={spec['id']:str});assert z[spec['id']].is_unique
   assert set(z[spec['id']])==set(f.image_id),name+': encoded CSV must cover exactly the metadata IDs'
   raw=pd.DataFrame({spec['id']:f.image_id}).merge(z,on=spec['id'],how='left',validate='one_to_one')
  if cols:
   checks={}
   if 'female' in cols and 'male' in cols:
    female=pd.to_numeric(raw[cols['female']],errors='raise').to_numpy(float)
    male=pd.to_numeric(raw[cols['male']],errors='raise').to_numpy(float)
    unknown=pd.to_numeric(raw[cols['unknown']],errors='raise').to_numpy(float) if cols.get('unknown') else 1-female-male
    actual=np.column_stack((female,male,unknown))
    target=np.column_stack((f.sex.eq('female'),f.sex.eq('male'),f.sex.eq('unknown'))).astype(float)
    checks['sex']=np.any(~np.isfinite(actual)|(abs(actual-target)>1e-7),axis=1)
   if cols.get('age_z'):
    if spec.get('encoding_age_statistics') or age_estimable:
     st=spec.get('encoding_age_statistics',stats);target=((f.age.fillna(st['median'])-st['mean'])/st['std']).to_numpy(np.float32)
     actual=pd.to_numeric(raw[cols['age_z']],errors='raise').to_numpy(float)
     checks['age_z']=~np.isfinite(actual)|(abs(actual-target)>1e-6)
    else:encoded.append({'cohort':name,'field':'age_z','rows_checked':0,'mismatched_rows':0,'status':'NOT_EVALUATED_AGE_NOT_ESTIMABLE','reason':'Training age has no estimable scale and no explicit frozen encoding statistics were supplied.'})
   if cols.get('age_missing'):
    actual=pd.to_numeric(raw[cols['age_missing']],errors='raise').to_numpy(float)
    checks['age_missing']=actual!=f.age_missing.to_numpy()
   for field,bad in checks.items():encoded.append({'cohort':name,'field':field,'rows_checked':len(f),'mismatched_rows':int(bad.sum()),'status':'PASS' if not bad.any() else 'MISMATCH'})
  else:encoded.append({'cohort':name,'field':'not supplied','rows_checked':0,'mismatched_rows':0,'status':'NOT_EVALUATED'})
  for field in ['sex','age_missing','sex_recorded']:
   known=set(train[field].astype(str))
   for state,n in f[field].astype(str).value_counts().sort_index().items():
    states.append({'cohort':name,'field':field,'state':state,'images':int(n),'seen_in_training':state in known})
 refs=[]
 def features(f):
  sex=np.column_stack((f.sex.eq('female'),f.sex.eq('male'),f.sex.eq('unknown'))).astype(np.float32).astype(np.float64)
  nonage={'intercept_only':np.zeros((len(f),1)),'recording_only':1-sex[:,2:3]}
  if not age_estimable:return nonage
  x=np.column_stack(((f.age.fillna(med)-mu)/sd,f.age_missing,sex)).astype(np.float32).astype(np.float64)
  return {'intercept_only':nonage['intercept_only'],'age_only':x[:,:2],'recording_only':1-x[:,4:5],
   'age_recording':np.column_stack((x[:,:2],1-x[:,4])),'age_recorded_sex':x}
 y=train.diagnosis.map(labels).to_numpy(int)
 names=['intercept_only','age_only','recording_only','age_recording','age_recorded_sex']
 reference_status='NOT_REQUESTED';reference_coverage=[]
 available=features(train)
 if cfg.get('fixed_references',True) and set(y)==set(range(len(classes))):
  models={name:LogisticRegression(C=1,solver='lbfgs',max_iter=2000,tol=1e-4,class_weight=None).fit(x,y) for name,x in available.items()}
  for name,entry in cohorts.items():
   if name==cfg['training_cohort']:continue
   f=entry['frame'];truth=f.diagnosis.map(labels).to_numpy(int)
   for model,x in features(f).items():refs.append({'cohort':name,'reference_model':model,**measurements(truth,models[model].predict_proba(x),classes)})
  reference_status='FITTED_ON_TRAINING_COHORT_ONLY' if age_estimable else 'PARTIALLY_FITTED_NONAGE_REFERENCES'
 elif cfg.get('fixed_references',True):reference_status='NOT_ESTIMABLE_TRAINING_CLASS_SUPPORT'
 for name in names:
  fitted=reference_status in ['FITTED_ON_TRAINING_COHORT_ONLY','PARTIALLY_FITTED_NONAGE_REFERENCES'] and name in available
  reason='' if fitted else 'Optional references disabled' if reference_status=='NOT_REQUESTED' else 'Training cohort does not contain every declared class' if reference_status=='NOT_ESTIMABLE_TRAINING_CLASS_SUPPORT' else age_status
  reference_coverage.append({'reference_model':name,'status':'FITTED' if fitted else 'NOT_EVALUATED','reason':reason})
 predictions=[];prediction_arrays={}
 for item in cfg.get('prediction_inputs',[]):
  f=cohorts[item['cohort']]['frame'];p=pd.read_csv(base/item['csv'],dtype={item.get('id_column','image_id'):str})
  assert p[item.get('id_column','image_id')].astype(str).tolist()==f.image_id.tolist(),'Prediction rows must match metadata order'
  a=p[item.get('probability_columns',[f'prob_{i}' for i in range(len(classes))])].to_numpy(float)
  assert a.shape==(len(f),len(classes)) and np.isfinite(a).all() and a.min()>=0 and np.max(abs(a.sum(1)-1))<1e-6
  key=(item['cohort'],item['model_id']);prediction_arrays.setdefault(key,{})
  scenario=item['scenario'];repeat=item.get('repeat',0);assert (scenario,repeat) not in prediction_arrays[key]
  prediction_arrays[key][(scenario,repeat)]=a
 for (cohort,model),values in prediction_arrays.items():
  basep=values.get(('observed',0));assert basep is not None,'Each fixed model needs observed predictions'
  f=cohorts[cohort]['frame'];y=f.diagnosis.map(labels).to_numpy(int);baseline=measurements(y,basep,classes)
  for (scenario,repeat),p in values.items():
   row={'cohort':cohort,'model_id':model,'scenario':scenario,'repeat':repeat,**measurements(y,p,classes)}
   row['delta_bacc']=None if row['bacc'] is None else row['bacc']-baseline['bacc']
   row['prediction_flip_fraction']=float((p.argmax(1)!=basep.argmax(1)).mean());predictions.append(row)
 tables={'missing_by_diagnosis':pd.DataFrame(missing),'missing_by_reference_standard':pd.DataFrame(standard),
  'raw_to_encoding_consistency':pd.DataFrame(encoded),'states_relative_to_training':pd.DataFrame(states),
  'within_patient_consistency':pd.DataFrame(inconsistency),'fixed_reference_metrics':pd.DataFrame(refs),
  'fixed_model_perturbation_metrics':pd.DataFrame(predictions),'reference_model_coverage':pd.DataFrame(reference_coverage)}
 for name,df in tables.items():df.to_csv(output/(name+'.csv'),index=False)
 if predictions:
  keys=['cohort','model_id','scenario']
  pd.DataFrame(predictions).groupby(keys)[['bacc','delta_bacc','prediction_flip_fraction']].agg(['mean','std','count']).to_csv(output/'within_fixed_model_repetition_summary.csv')
 mismatches=sum(r['mismatched_rows'] for r in encoded)
 report={'status':'PASS' if mismatches==0 else 'ENCODING_MISMATCH_FOUND',
  'training_only_age_statistics':stats,'cohort_counts':{n:{'images':len(v['frame']),'patients':v['frame'].patient_id.nunique()} for n,v in cohorts.items()},
  'encoding_mismatched_rows':mismatches,'fixed_reference_status':reference_status,
  'new_neural_training':False,'scope':'Descriptive recording/encoding audit and fixed CPU references. Associations do not establish label leakage, causal model mechanisms or clinical benefit. Repeated perturbations are within fixed models, not independently trained replicates.',
  'python':platform.python_version(),'tables':list(tables)}
 report['training_age_reference_status']=age_status
 report['reference_model_coverage']=reference_coverage
 report=enrich_report(config,cfg,cohorts,report,tables)
 (output/'audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 write_report(report,tables,output)
 print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
 return report,tables
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path)
 a=ap.parse_args();r,_=execute(a.config,a.output or a.config.resolve().parent/'audit_output')
 if r['encoding_mismatched_rows'] or r['known_sex_conflict_patients']:raise SystemExit(2)
