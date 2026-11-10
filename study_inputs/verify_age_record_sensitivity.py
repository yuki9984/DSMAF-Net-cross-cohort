from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
from age_record_sensitivity import calculate
root=Path(__file__).resolve().parent
expected=root/'age_record_sensitivity/expected'
out=root/'verification_output/age_record_sensitivity'
plan=json.loads((root/'age_record_sensitivity/analysis_plan.json').read_text(encoding='utf-8-sig'))
ref=json.loads((expected/'report.json').read_text(encoding='utf-8-sig'))
for name,digest in ref['source_hashes'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
calculate(root,out)
actual=json.loads((out/'report.json').read_text(encoding='utf-8'))
def compare(a,b):
 if isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:compare(a[k],b[k])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):compare(x,y)
 elif isinstance(a,(int,float)) and not isinstance(a,bool):assert np.isclose(a,b,atol=1e-10,rtol=0),(a,b)
 else:assert a==b,(a,b)
compare(actual,ref)
for p in expected.glob('*.csv'):
 pd.testing.assert_frame_equal(pd.read_csv(p),pd.read_csv(out/p.name),check_exact=False,rtol=0,atol=1e-10)
with np.load(expected/'bootstrap_draws.npz',allow_pickle=False) as a,np.load(out/'bootstrap_draws.npz',allow_pickle=False) as b:
 assert a.files==b.files
 for k in a.files:
  if np.issubdtype(a[k].dtype,np.number):np.testing.assert_allclose(a[k],b[k],rtol=0,atol=1e-10)
  else:assert np.array_equal(a[k],b[k])
report={'status':'PASS','scope':'Evaluation restriction only; same 75 fixed models and five seeds. No refitting or inference.','expected_report_and_all_tables_and_draws_matched':True,'analysis':actual}
(out/'verification_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('AGE_RECORD_SENSITIVITY_VERIFIED',flush=True)
