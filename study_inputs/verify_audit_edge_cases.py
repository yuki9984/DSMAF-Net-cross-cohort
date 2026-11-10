from pathlib import Path
import sys,json,io,contextlib,shutil,copy
import numpy as np,pandas as pd
P=Path(__file__).resolve().parent;W=P/'verification_output';W.mkdir(exist_ok=True);D=P/'portable_audit'
sys.path.insert(0,str(D));from audit_metadata import execute
T=W/'software_validation_only';T.mkdir(exist_ok=True)
cfg0=json.loads((D/'example/audit_config.json').read_text(encoding='utf-8'))
old=pd.read_csv(D/'example/audit_output/fixed_reference_metrics.csv')
with contextlib.redirect_stdout(io.StringIO()):r,tables=execute(D/'example/audit_config.json',W/'portable_regression_example')
new=tables['fixed_reference_metrics'];keys=['cohort','reference_model'];metrics=['bacc','accuracy','nll','brier']
z=new.merge(old,on=keys,suffixes=('_new','_old'));maxdiff=max(float(abs(z[k+'_new']-z[k+'_old']).max()) for k in metrics)
assert maxdiff<1e-7,maxdiff
cov=tables['cohort_field_coverage'];assert int(cov[(cov.field=='sex')&cov.cohort.isin(['train','tuning'])].missing_images.sum())==631
assert r['encoding_mismatched_rows']==0 and not r['cohort_counts']['external']['encoding_check_available']
assert r['prediction_input_status']=='NOT_SUPPLIED'
results={'real_data_example':{'status':'PASS','logistic_rows':len(new),'maximum_absolute_metric_difference_from_prior':maxdiff,'sex_blank_development_images':631,'external_encodings':'NOT_EVALUATED','prediction_response':'NOT_SUPPLIED'},'synthetic_software_tests':[]}
basecfg={'classes':['A','B'],'training_cohort':'train','fixed_references':False,'cohorts':[]}
cols={'id':'id','patient':'patient','diagnosis':'label','age':'age','sex':'sex','sex_mapping':{'F':'female','M':'male','':'unknown'},'encoding':{'female':'f','male':'m'}}
train=pd.DataFrame({'id':['001','002','003','004'],'patient':['p1','p2','p3','p4'],'label':['A','A','B','B'],'age':[20,30,40,50],'sex':['F','M','F','M'],'f':[1,0,1,0],'m':[0,1,0,1]})
test=pd.DataFrame({'id':['005','006','007','008'],'patient':['q1','q1','q2','q3'],'label':['A','A','B','B'],'age':[21,31,41,51],'sex':['F','','F','M'],'f':[1,0,1,0],'m':[0,0,0,1]})
def run(name,change=None,expected_failure=False,check=None):
 folder=T/name;folder.mkdir(exist_ok=True);a=train.copy();b=test.copy();cfg=copy.deepcopy(basecfg)
 cfg['cohorts']=[dict(cols,name='train',csv='train.csv'),dict(cols,name='test',csv='test.csv')]
 if change:change(a,b,cfg,folder)
 a.to_csv(folder/'train.csv',index=False);b.to_csv(folder/'test.csv',index=False)
 (folder/'config.json').write_text(json.dumps(cfg),encoding='utf-8')
 try:
  with contextlib.redirect_stdout(io.StringIO()):report,t=execute(folder/'config.json',folder/'out')
 except (ValueError,AssertionError) as e:
  assert expected_failure,(name,str(e));results['synthetic_software_tests'].append({'name':name,'status':'PASS','invalid_input_rejected':True,'error':str(e)});return
 assert not expected_failure,name+' invalid input not rejected'
 if check:check(report,t,folder)
 results['synthetic_software_tests'].append({'name':name,'status':'PASS','result_status':report['status']})
def initialcheck(r,t,p):
 assert r['unseen_categorical_states']
 z=t['within_patient_consistency'].query("cohort == 'test'").iloc[0]
 assert z.patients_with_known_and_unknown_sex==1 and z.patients_with_conflicting_recorded_sex==0
 assert r['known_sex_conflict_patients']==0
run('leading_zero_ids_and_known_unknown_states',check=initialcheck)
def corrupt(a,b,c,p):b.loc[0,'f']=0
run('corrupted_encoding',corrupt,check=lambda r,t,p: (_ for _ in ()).throw(AssertionError()) if r['encoding_mismatched_rows']!=1 else None)
def conflict(a,b,c,p):b.loc[1,['sex','f','m']]=['M',0,1]
run('known_sex_conflict',conflict,check=lambda r,t,p: (_ for _ in ()).throw(AssertionError()) if r['known_sex_conflict_patients']!=1 else None)
run('unmapped_sex',lambda a,b,c,p:b.__setitem__('sex',['unexpected','','F','M']),True)
run('nonnumeric_age',lambda a,b,c,p:b.__setitem__('age',['not-an-age',31,41,51]),True)
run('duplicate_class',lambda a,b,c,p:c.__setitem__('classes',['A','A']),True)
run('partial_sex_mapping',lambda a,b,c,p:c['cohorts'][1].__setitem__('encoding',{'female':'f'}),True)
def wrongids(a,b,c,p):
 z=b[['id','f','m']].copy();z.loc[0,'id']='999';z.to_csv(p/'encoded.csv',index=False);c['cohorts'][1]['encoded_csv']='encoded.csv'
run('encoded_id_coverage_mismatch',wrongids,True)
def predinput(a,b,c,p):
 base=pd.DataFrame({'id':b.id,'prob_0':[.9,.7,.2,.1],'prob_1':[.1,.3,.8,.9]});alt=base.copy();alt.loc[0,['prob_0','prob_1']]=[.1,.9]
 base.to_csv(p/'baseline.csv',index=False);alt.to_csv(p/'alternative.csv',index=False)
 c['prediction_inputs']=[{'cohort':'test','model_id':'synthetic_fixed_model','scenario':s,'csv':v,'id_column':'id'} for s,v in [('observed','baseline.csv'),('substituted','alternative.csv')]]
def predcheck(r,t,p):
 z=t['fixed_model_perturbation_metrics'].query("scenario == 'substituted'").iloc[0]
 assert abs(z.delta_bacc+.25)<1e-12 and abs(z.prediction_flip_fraction-.25)<1e-12
 assert r['prediction_input_status']=='SUPPLIED'
run('paired_prediction_oracle',predinput,check=predcheck)

def degenerate(a,b,c,p,kind,enabled=True,encoding=False,frozen=False):
 a['age']=40 if kind=='constant' else ''
 c['fixed_references']=enabled
 if encoding:
  for f in [a,b]:
   f['age_miss']=f.age.eq('').astype(int)
   f['az']=(pd.to_numeric(f.age,errors='coerce').fillna(40)-40)/10 if frozen else 0
  for sp in c['cohorts']:
   sp['encoding']=dict(sp['encoding'],age_z='az',age_missing='age_miss')
   if frozen:sp['encoding_age_statistics']={'mean':40,'std':10,'median':40}
def degcheck(r,t,p,kind,enabled,encoding=False,frozen=False):
 assert r['training_age_reference_status']==('CONSTANT_TRAINING_AGE' if kind=='constant' else 'NO_OBSERVED_TRAINING_AGE')
 assert len(t['fixed_reference_metrics'])==(2 if enabled else 0)
 assert r['encoding_mismatched_rows']==0
 if enabled:
  assert set(t['fixed_reference_metrics'].reference_model)=={'intercept_only','recording_only'}
  assert sum(x['status']=='FITTED' for x in r['reference_model_coverage'])==2
 if encoding:
  z=t['raw_to_encoding_consistency'].query("field == 'age_z'")
  assert len(z)==2
  assert set(z.status)==({'PASS'} if frozen else {'NOT_EVALUATED_AGE_NOT_ESTIMABLE'})
 json.loads((p/'out/audit_report.json').read_text(encoding='utf-8'),parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
 assert (p/'out/audit_report.html').exists()
for kind in ['constant','missing']:
 for enabled in [True,False]:
  run(kind+'_age_refs_'+str(enabled),lambda a,b,c,p,k=kind,e=enabled:degenerate(a,b,c,p,k,e),check=lambda r,t,p,k=kind,e=enabled:degcheck(r,t,p,k,e))
 for frozen in [False,True]:
  run(kind+'_age_encoding_frozen_'+str(frozen),lambda a,b,c,p,k=kind,f=frozen:degenerate(a,b,c,p,k,True,True,f),check=lambda r,t,p,k=kind,f=frozen:degcheck(r,t,p,k,True,True,f))
def missingclass(a,b,c,p):
 a['label']='A';c['fixed_references']=True
run('training_class_absent',missingclass,check=lambda r,t,p:None if r['fixed_reference_status']=='NOT_ESTIMABLE_TRAINING_CLASS_SUPPORT' and len(t['fixed_reference_metrics'])==0 else (_ for _ in ()).throw(AssertionError()))
run('age_range_validation_preserved',lambda a,b,c,p:b.__setitem__('age',[200,31,41,51]),True)

results['scope']='Synthetic inputs validate software behavior only; not neural model or clinical study evidence.'
(W/'portable_auditor_validation.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8');print(json.dumps(results,indent=2),flush=True)
