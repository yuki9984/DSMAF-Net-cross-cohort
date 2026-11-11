from pathlib import Path
import sys,json,subprocess
import pandas as pd
root=Path(__file__).resolve().parent
out=root/'verification_output/portable_audit'
subprocess.run([sys.executable,'-B',str(root/'portable_audit/audit_metadata.py'),'--config',str(root/'portable_audit/example/audit_config.json'),'--output',str(out)],check=True,cwd=root)
r=json.loads((out/'audit_report.json').read_text(encoding="utf-8"))
assert r['encoding_mismatched_rows']==0 and r['known_sex_conflict_patients']==0
assert not r['cohort_counts']['external']['encoding_check_available'] and r['prediction_input_status']=='NOT_SUPPLIED'
c=pd.read_csv(out/'cohort_field_coverage.csv')
assert int(c[c.cohort.isin(['train','tuning']) & c.field.eq('sex')].missing_images.sum())==631
actual=pd.read_csv(out/'fixed_reference_metrics.csv');expected=pd.read_csv(root/'expected/demographic_reference_decomposition.csv')
actual['partition']=actual.cohort.map({'tuning':'Model tuning','external':'External'})
z=actual.merge(expected,left_on=['partition','reference_model'],right_on=['partition','model'],suffixes=('_actual','_expected'))
assert len(z)==10
errors={k:float(abs(z[k+'_actual']-z[k+'_expected']).max()) for k in ['bacc','accuracy','nll','brier']}
assert errors['bacc']<1e-12 and errors['accuracy']<1e-12 and max(errors.values())<1e-7
report={'status':'PASS','tool_version':r['tool_version'],'fixed_reference_rows':10,'development_sex_blank_images':631,'encoding_mismatch_sum':0,'maximum_absolute_metric_errors':errors,'external_encoding_check':'NOT_EVALUATED','fixed_model_predictions':'NOT_SUPPLIED','scope':'Portable descriptive implementation, existing public example only; no new neural results or bootstrap intervals.'}
(root/'verification_output/portable_audit_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
