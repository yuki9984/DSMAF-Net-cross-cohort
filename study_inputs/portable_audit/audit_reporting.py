"""Validation, provenance and presentation for the portable descriptive audit."""
from pathlib import Path
import hashlib, html, json, platform
import numpy as np
import pandas as pd
import sklearn

TOOL_VERSION = '2.1.0'
TITLES = {
 'missing_by_diagnosis':'Missing fields by diagnosis (image denominators)',
 'missing_by_reference_standard':'Missing fields by diagnosis and reference standard',
 'raw_to_encoding_consistency':'Raw fields versus supplied model encodings',
 'states_relative_to_training':'Categorical states relative to training',
 'within_patient_consistency':'Within-patient encoded sex states',
 'fixed_reference_metrics':'Fixed logistic references: held-out descriptive metrics',
 'reference_model_coverage':'Reference-model coverage and non-estimable features',
 'fixed_model_perturbation_metrics':'Optional fixed-model prediction responses'
}
def require(condition,message):
 if not condition: raise ValueError(message)
def file_hash(path):
 with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def validate_config(cfg,base,output):
 classes=cfg.get('classes',[])
 require(len(classes)>=2 and all(isinstance(x,str) and x.strip() for x in classes) and len(set(classes))==len(classes),'classes must contain unique, nonempty labels in probability-column order')
 specs=cfg.get('cohorts',[]); names=[x.get('name') for x in specs]
 require(specs and all(isinstance(x,str) and x.strip() for x in names) and len(set(names))==len(names),'cohort names must be unique and nonempty')
 require(cfg.get('training_cohort') in names,'training_cohort must name exactly one supplied cohort')
 bounds=cfg.get('age_range',[0,120]);require(len(bounds)==2 and all(np.isfinite(v) for v in bounds) and bounds[0]<bounds[1],'age_range must contain finite increasing bounds')
 for spec in specs:
  for key in ['csv','id','patient','diagnosis','age','sex','sex_mapping']: require(key in spec,f'{spec["name"]}: missing {key}')
  path=(base/spec['csv']).resolve(); require(path.is_file(),f'{spec["name"]}: metadata CSV not found')
  require(not path.is_relative_to(output), 'Output must not contain an input CSV')
  mapping=spec['sex_mapping'];require(isinstance(mapping,dict) and mapping,'sex_mapping must be explicit')
  normalized={str(k).strip().upper():v for k,v in mapping.items()}
  require(len(normalized)==len(mapping),'sex_mapping has colliding aliases after case/space normalization')
  require(set(mapping.values()) <= {'female','male','unknown'},'sex_mapping targets must be female, male or unknown')
  cols=spec.get('encoding',{})
  require(set(cols)<= {'female','male','unknown','age_z','age_missing'},'unsupported encoding key')
  require(not ({'female','male','unknown'} & set(cols)) or {'female','male'}<=set(cols),'sex encoding requires female and male columns together')
  require(all(isinstance(v,str) and v for v in cols.values()),'encoding columns must be nonempty strings')
  if spec.get('encoded_csv'):
   require(cols,'encoded_csv requires encoding column mappings')
   enc=(base/spec['encoded_csv']).resolve();require(enc.is_file(),'encoded_csv not found');require(not enc.is_relative_to(output),'Output must not contain an encoded input CSV')
  if 'encoding_age_statistics' in spec:
   st=spec['encoding_age_statistics'];require(set(st)=={'mean','std','median'} and all(np.isfinite(v) for v in st.values()) and st['std']>0,'encoding_age_statistics requires finite mean/median and positive std')
 for item in cfg.get('prediction_inputs',[]):
  require(item.get('cohort') in names,'prediction cohort is not supplied')
  require(bool(item.get('model_id')) and bool(item.get('scenario')),'prediction needs model_id and scenario')
  require(type(item.get('repeat',0)) is int and item.get('repeat',0)>=0,'prediction repeat must be a nonnegative integer')
  p=(base/item['csv']).resolve();require(p.is_file(),'prediction CSV not found');require(not p.is_relative_to(output),'Output must not contain prediction input files')

def enrich_report(config,cfg,cohorts,report,tables):
 coverage=[]; patient=[]; known_conflicts=0
 for name,entry in cohorts.items():
  f=entry['frame'];spec=entry['spec']; has_ref=bool(spec.get('reference_standard'))
  for field,mask in [('age',f.age.isna()),('sex',f.sex.eq('unknown'))]:
   coverage.append({'cohort':name,'field':field,'images':len(f),'missing_images':int(mask.sum()),'missing_percent':float(mask.mean()*100),'patients_with_any_missing_image':int(f.loc[mask,'patient_id'].nunique())})
  encoded=tables['raw_to_encoding_consistency']; checked=encoded[encoded.cohort.eq(name)]
  supplied=checked[checked.status.isin(['PASS','MISMATCH'])]
  report['cohort_counts'][name].update({'reference_standard_supplied':has_ref,'encoding_fields_checked':supplied.field.tolist(),'encoding_check_available':bool(len(supplied))})
  both=f.groupby('patient_id').sex.nunique()
  known=f[f.sex.ne('unknown')].groupby('patient_id').sex.nunique()
  conflict=int(known.gt(1).sum());known_conflicts+=conflict
  mix=int(((both>1)&both.index.isin(f.loc[f.sex.eq('unknown'),'patient_id'])).sum())
  patient.append({'cohort':name,'patients':int(f.patient_id.nunique()),'patients_with_multiple_sex_states':int((both>1).sum()),'patients_with_conflicting_recorded_sex':conflict,'patients_with_known_and_unknown_sex':mix})
 tables['within_patient_consistency']=pd.DataFrame(patient)
 tables['cohort_field_coverage']=pd.DataFrame(coverage)
 report['tables']=list(tables)
 report['tool_version']=TOOL_VERSION;report['status']='COMPLETED' if not report['encoding_mismatched_rows'] and not known_conflicts else 'COMPLETED_WITH_CONSISTENCY_FINDINGS'
 report['known_sex_conflict_patients']=known_conflicts
 report['encoding_scope']='Only explicitly mapped encoding columns are compared. A missing mapping is NOT_EVALUATED, never evidence of agreement.'
 report['state_scope']='State checks cover sex categories and binary recording/missing indicators; they do not assess continuous-age support or image-domain shift.'
 report['denominator_scope']='Missing-field fractions use images. Patients with any missing image are distinct counts, not patient missingness fractions.'
 report['uncertainty_scope']='No patient-resampling intervals or retraining uncertainty are estimated by this tool. Repetition summaries remain within fixed models.'
 report['reference_feature_definitions']={'intercept_only':['constant zero column; fitted intercept'],'age_only':['training-standardized age','age-missing indicator'],'recording_only':['sex recorded indicator'],'age_recording':['training-standardized age','age-missing indicator','sex recorded indicator'],'age_recorded_sex':['training-standardized age','age-missing indicator','female','male','unknown']}
 report['fixed_reference_specification']={'C':1,'solver':'lbfgs','max_iter':2000,'tol':1e-4,'class_weight':None,'hyperparameter_search':False,'probability_clipping':float(np.finfo(float).eps),'feature_precision':'float32 then float64 (study convention)'}
 inputs=[]
 def add(kind,path):
  p=(config.parent/path).resolve()
  inputs.append({'kind':kind,'configured_path':str(path),'bytes':p.stat().st_size,'sha256':file_hash(p)})
 add('configuration',config.name)
 for spec in cfg['cohorts']:
  add('metadata:'+spec['name'],spec['csv'])
  if spec.get('encoded_csv'):add('encoding:'+spec['name'],spec['encoded_csv'])
 for item in cfg.get('prediction_inputs',[]):add('predictions:'+item['model_id']+':'+item['scenario']+':'+str(item.get('repeat',0)),item['csv'])
 report['input_provenance']=inputs
 report['software']={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__}
 root=Path(__file__).resolve().parent
 report['tool_source_sha256']={n:file_hash(root/n) for n in ['audit_metadata.py','audit_reporting.py']}
 unseen=tables['states_relative_to_training']; unseen=unseen[~unseen.seen_in_training]
 report['unseen_categorical_states']=unseen.to_dict(orient='records')
 report['prediction_input_status']='SUPPLIED' if cfg.get('prediction_inputs') else 'NOT_SUPPLIED'
 report['warnings']=[]
 if report['training_age_reference_status']!='ESTIMABLE':report['warnings'].append('Training age is constant or entirely unobserved. Age-based references are not estimated; basic checks and available non-age references continue. No arbitrary age scale or evaluation-data age statistics are substituted.')
 if report['fixed_reference_status']=='NOT_ESTIMABLE_TRAINING_CLASS_SUPPORT':report['warnings'].append('Reference fitting is not evaluated because the training cohort lacks a declared class; descriptive checks remain available.')
 if any(not c['encoding_check_available'] for c in report['cohort_counts'].values()):report['warnings'].append('One or more cohorts have no supplied model encodings; raw-to-encoding consistency is not evaluated there.')
 if not cfg.get('prediction_inputs'):report['warnings'].append('No fixed-model prediction inputs were supplied; prediction-response analyses were not performed.')
 if len(unseen):report['warnings'].append('One or more categorical states were absent from training. This flags representation support, not a quantified performance effect.')
 if known_conflicts:report['warnings'].append('Recorded female/male values conflict within at least one patient; inspect source records before interpretation.')
 return report

def write_report(report,tables,output):
 esc=html.escape
 # Extra summary tables are written alongside the core tables after enrichment.
 for name,df in tables.items():df.to_csv(output/(name+'.csv'),index=False)
 headers=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
 '<title>Metadata recording audit</title><style>',
 'body{margin:0;background:#f5f7fa;color:#173543;font:15px/1.55 Arial,sans-serif}main{max-width:1180px;margin:0 auto;padding:40px 26px}h1,h2,h3{font-family:Georgia,serif;line-height:1.2}h1{font-size:34px;margin:8px 0 14px}h2{margin-top:32px;font-size:23px}h3{font-size:18px}p{max-width:94ch}.kicker{text-transform:uppercase;letter-spacing:.12em;font-size:11px;color:#4a6578}.cards{display:flex;gap:12px;flex-wrap:wrap}.card{flex:1;min-width:170px;background:white;border:1px solid #cfdae3;border-radius:10px;padding:16px}.card strong{display:block;font-size:24px}.muted{color:#526471}.note{background:#fff5d6;border-left:4px solid #c3963d;padding:12px 16px}.flow{display:flex;gap:8px;flex-wrap:wrap;margin:22px 0}.step{flex:1;min-width:160px;padding:12px;background:#e3f0fc;border:1px solid #adc5da;border-radius:7px}.step b{display:block}.table{overflow:auto;background:white;border:1px solid #d8e1e8;border-radius:8px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:9px 12px;text-align:left;border-bottom:1px solid #dfe6ec;white-space:nowrap}th{background:#e3f0fc}tbody tr:nth-child(even){background:#f8fafc}a{color:#17557b}summary{cursor:pointer;font-weight:bold}details{margin:14px 0}code{font-size:12px}pre{overflow:auto;background:white;padding:16px;border:1px solid #d8e1e8}footer{margin-top:30px;color:#526471;font-size:12px}@media print{body{background:white}main{padding:10px}details{break-inside:avoid}.table{overflow:visible}table{font-size:10px}th,td{padding:5px}}',
 '</style></head><body><main><div class="kicker">Portable descriptive audit / version '+esc(report['tool_version'])+'</div><h1>Check recording status before interpreting fusion</h1>',
 '<p>Keep source fields, model encodings and fixed-model responses separate. This report describes the supplied records; completion does not establish data quality, clinical utility or a causal explanation.</p>']
 parts=headers
 counts=report['cohort_counts'];total=sum(c['images'] for c in counts.values());checked=sum(r for r in tables['raw_to_encoding_consistency'].rows_checked)
 parts+=['<div class="cards"><div class="card"><span class="muted">Supplied cohort images</span><strong>'+str(total)+'</strong><small>Sum of rows, not a claim of cross-cohort uniqueness</small></div>',
 '<div class="card"><span class="muted">Encoding mismatches</span><strong>'+str(report['encoding_mismatched_rows'])+'</strong><small>Sum across checked fields; a row may contribute more than once</small></div>',
 '<div class="card"><span class="muted">Cohorts without encodings</span><strong>'+str(sum(not c['encoding_check_available'] for c in counts.values()))+'</strong><small>Not evaluated</small></div></div>',
 '<div class="flow">'+''.join('<div class="step"><b>'+str(i+1)+'. '+t+'</b>'+d+'</div>' for i,(t,d) in enumerate([('Map','Raw fields and IDs'),('Verify','Supplied encodings and patient states'),('Compare','Recording patterns and training support'),('Inspect','Fixed references and optional responses')]))+'</div>']
 if report['warnings']:parts+=['<div class="note"><b>Coverage and interpretation</b><ul>'+''.join('<li>'+esc(w)+'</li>' for w in report['warnings'])+'</ul></div>']
 def table(name,df,open_=True):
  markup=df.to_html(index=False,escape=True,na_rep='Not estimated',float_format=lambda v:f'{v:.5g}') if len(df) else '<p>Optional input was not supplied; this analysis was not performed.</p>'
  return '<details'+(' open' if open_ else '')+'><summary>'+esc(TITLES.get(name,name.replace('_',' ').title()))+'</summary><p><a href="'+esc(name)+'.csv">Download CSV</a></p><div class="table">'+markup+'</div></details>'
 parts+=['<h2>1. Counts and recording coverage</h2><p>'+esc(report['denominator_scope'])+'</p>',table('cohort_field_coverage',tables['cohort_field_coverage'])]
 parts+=['<h2>2. Source-to-encoding verification</h2><p>'+esc(report['encoding_scope'])+'</p>',table('raw_to_encoding_consistency',tables['raw_to_encoding_consistency']),table('within_patient_consistency',tables['within_patient_consistency'])]
 parts+=['<h2>3. Class associations and representation support</h2><p>Associations may reflect clinical, selection or recording processes. They do not by themselves establish label leakage.</p>',table('missing_by_diagnosis',tables['missing_by_diagnosis']),table('missing_by_reference_standard',tables['missing_by_reference_standard'],False),'<p>'+esc(report['state_scope'])+'</p>',table('states_relative_to_training',tables['states_relative_to_training'],False)]
 parts+=['<h2>4. Fixed references and optional prediction responses</h2><p>References use training-only preprocessing and fixed logistic settings. Evaluation on a tuning partition is descriptive; it is not independent internal validation.</p>',table('reference_model_coverage',tables['reference_model_coverage']),table('fixed_reference_metrics',tables['fixed_reference_metrics']),table('fixed_model_perturbation_metrics',tables['fixed_model_perturbation_metrics'],False),'<p>'+esc(report['uncertainty_scope'])+'</p>']
 parts+=['<h2>5. Provenance and scope</h2><p><a href="audit_report.json">Machine-readable report</a>. Each input and both source files have SHA-256 hashes. Dataset permissions and attribution remain applicable.</p>', '<details><summary>Software, settings and input hashes</summary><pre>'+esc(json.dumps({k:report[k] for k in ['tool_version','software','training_only_age_statistics','training_age_reference_status','reference_model_coverage','fixed_reference_specification','input_provenance','tool_source_sha256']},indent=2))+'</pre></details>', '<p>'+esc(report['scope'])+'</p><footer>Report status: '+esc(report['status'])+'. No neural training is performed. Tables and report files are local outputs.</footer></main></body></html>']
 (output/'audit_report.html').write_text('\n'.join(parts),encoding='utf-8')
