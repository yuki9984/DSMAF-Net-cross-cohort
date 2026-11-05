"""Run retained audit/statistics code with synthetic data only, without a GPU."""
from pathlib import Path
import argparse,contextlib,csv,io,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'extension60'))
sys.path.insert(0,str(ROOT/'historical_study/portable_audit'))
from information_conditions import restrict_metadata
from analysis_core import calculate_statistics,effective_responses
from audit_metadata import execute

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();out=a.output.resolve()
 if out.exists():raise FileExistsError('Use a new output folder to preserve previous reports.')
 out.mkdir(parents=True)
 checks=[]
 x=np.array([[.3,0,1,0,0],[.3,0,0,1,0],[.3,0,0,0,1]],dtype=np.float32)
 original=x.copy();b=restrict_metadata(x,'age_only');c=restrict_metadata(x,'age_recorded')
 np.testing.assert_array_equal(b[0],b[1]);np.testing.assert_array_equal(b[1],b[2])
 np.testing.assert_array_equal(c[0],c[1]);assert not np.array_equal(c[0],c[2])
 np.testing.assert_array_equal(x,original);np.testing.assert_array_equal(c[:,:2],x[:,:2])
 checks.append('Category identity removed while recording availability and age are retained as specified.')
 try:restrict_metadata(np.array([[0,0,.5,.5,0]]),'age_only')
 except ValueError:checks.append('Invalid original sex encoding rejected.')
 else:raise AssertionError('Invalid input was accepted')
 with contextlib.redirect_stdout(io.StringIO()):
  audit,tables=execute(ROOT/'examples/synthetic/audit_config.json',out/'synthetic_audit')
 assert audit['status']=='COMPLETED' and audit['encoding_mismatched_rows']==0
 assert all(v['images']==12 for v in audit['cohort_counts'].values())
 assert all(v['patients']==12 for v in audit['cohort_counts'].values())
 coverage=tables['cohort_field_coverage']
 assert list(coverage.loc[coverage.field=='sex','missing_images'])==[4,4,4]
 checks.append('Synthetic audit counts and sex encodings match the specified synthetic inputs.')
 # Unequal frozen fold weights make metric aggregation distinguishable from ensembling.
 yi=np.tile(np.arange(6),10);folds=np.repeat([0,0,0,0,0,1,1,2,3,4],6)
 ye=np.tile(np.arange(6),4)
 pi=np.full((2,3,len(yi),6),.02)
 for i,label in enumerate(yi):pi[:,:,i,label]=.9
 pe=np.zeros((2,3,5,len(ye),6))
 for f in range(5):
  for i,label in enumerate(ye):
   pe[:,:,f,i,label]=.51 if f==0 else .01
   pe[:,:,f,i,(label+1)%6]=.49 if f==0 else .99
 ri,re,cw,iw=effective_responses(yi,pi,ye,pe,folds)
 np.testing.assert_allclose(cw[:,0],[.5,.2,.1,.1,.1]);np.testing.assert_allclose(re[...,0],.5)
 np.testing.assert_allclose(re[...,1],-.5*np.log(.51)-.5*np.log(.01),atol=1e-12)
 ensemble=(pe*iw[None,None,:,None,None]).sum(2)
 assert (ensemble.argmax(-1)!=ye).all()
 checks.append('External fold metrics use the fixed weights and are not probability ensembles.')
 result=calculate_statistics(['SYNTH_A','SYNTH_B'],[1,2,3],yi,np.array([f'SYNTH_I_PAT_{i//2}' for i in range(len(yi))]),folds,pi,ye,np.array([f'SYNTH_E_PAT_{i//2}' for i in range(len(ye))]),pe,[('SYNTH_A','SYNTH_B')],repeats=2000,bootstrap_seed=2026102901)
 contrast=[r for r in result['summary'] if r['type']=='paired_contrast']
 assert len(contrast)==27
 for r in contrast:
  assert all(r[k]==0 for k in ['estimate','ci95_low','ci95_high','seed_sd'])
 checks.append('Identical paired synthetic methods have exactly zero contrasts and conditional intervals in all metrics.')
 for name,rows in [('synthetic_summary_intervals.csv',result['summary']),('synthetic_metrics_by_seed.csv',result['by_seed'])]:
  with (out/name).open('w',encoding='utf-8',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 report={'status':'ALL_SYNTHETIC_SOFTWARE_CHECKS_PASS','scope':'Synthetic software example only. None of these estimates is a study result.','checks':checks,'audit_rows':36,'bootstrap_repeats':2000,'synthetic_contrast_rows':len(contrast),'neural_training_performed':False,'neural_inference_performed':False,'real_case_rows_read':0}
 (out/'SYNTHETIC_DEMO_REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
