from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parent
for name in ['reproduce.py','verify_internal75.py','verify_portable_audit.py','verify_matched_cohort.py','verify_class_and_recording.py','verify_audit_edge_cases.py','verify_cohort_characteristics.py','verify_followup.py','verify_age_record_sensitivity.py']:
 subprocess.run([sys.executable,'-B',str(root/name)],cwd=root,check=True)
out=root/'verification_output'
report={'status':'PASS','original':json.loads((out/'verification_report.json').read_text(encoding="utf-8")),'internal75':json.loads((out/'internal75_report.json').read_text(encoding="utf-8")),'portable_audit':json.loads((out/'portable_audit_verification.json').read_text(encoding="utf-8")),'matched_cohort':json.loads((out/'matched_cohort/independent_statistics_report.json').read_text(encoding="utf-8")),'class_and_recording':json.loads((out/'class_and_recording/class_and_full_PAD_audit.json').read_text(encoding="utf-8")),'audit_software_validation':json.loads((out/'portable_auditor_validation.json').read_text(encoding="utf-8")),'scope':'Frozen-prediction statistics, records and descriptive audit; no neural retraining or receiver-side inference.'}
report['cohort_characteristics']=json.loads((out/'cohort_characteristics_report.json').read_text(encoding="utf-8"))
report['followup']=json.loads((out/'followup/verification_report.json').read_text(encoding="utf-8"))
assert report['followup']['status']=='SAVED_PREDICTIONS_AND_RECORDS_VERIFIED'
report['age_record_sensitivity']=json.loads((out/'age_record_sensitivity/verification_report.json').read_text(encoding="utf-8"))
assert report['age_record_sensitivity']['status']=='PASS'
(out/'combined_verification_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('ALL_STUDY_FOLLOWUP_AND_AGE_CHECKS_PASS',flush=True)
