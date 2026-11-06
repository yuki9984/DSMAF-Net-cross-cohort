"""Demonstrate fixed-model lesion statistics using entirely synthetic inputs."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'external_milk/scripts'))
from milk_statistics import collapse_folds,metric_vector,summarize

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();out=args.output.resolve()
 if out.exists():raise FileExistsError('Choose a new output folder.')
 y=np.repeat(np.arange(6),6);n=len(y)
 arms=['SYNTH_A','SYNTH_B'];seeds=[1,2,3]
 probabilities=np.zeros((2,3,5,n,6))
 for a in range(2):
  for s in range(3):
   for f in range(5):
    for i,label in enumerate(y):
     good=(.51+.002*(i%6)+.001*s) if f<3 else (.01+.001*(i%6)+.001*s)
     if a==1:good=.70+.01*(i%6)+.005*s
     probabilities[a,s,f,i,label]=good
     probabilities[a,s,f,i,(label+1)%6]=1-good
 cw=np.full((5,6),.2);iw=np.full(5,.2)
 effective=np.array([[collapse_folds(y,probabilities[a,s],cw,iw) for s in range(3)] for a in range(2)])
 np.testing.assert_allclose(effective[0,:,:,0],.6)
 np.testing.assert_allclose(effective[1,:,:,0],1)
 # Scoring averaged probabilities would instead misclassify every A example.
 assert (probabilities[0].mean(1).argmax(-1)!=y).all()
 for a in range(2):
  for s in range(3):
   q=probabilities[a,s]
   expected_nll=(-np.log(q[:,np.arange(n),y])).mean(0)
   expected_brier=((q-np.eye(6)[y])**2).sum(-1).mean(0)
   np.testing.assert_allclose(effective[a,s,:,1],expected_nll,atol=1e-12)
   np.testing.assert_allclose(effective[a,s,:,2],expected_brier,atol=1e-12)
 result=summarize(arms,seeds,y,[f'SYNTH_LESION_{i}' for i in range(n)],effective,
                  [('SYNTH_B','SYNTH_A')],repeats=2000,seed=20261104)
 contrast=next(r for r in result['summary'] if r['type']=='paired_contrast' and r['metric']=='BACC')
 for k in ['estimate','ci95_low','ci95_high']:np.testing.assert_allclose(contrast[k],40,atol=1e-12)
 # Independent direct-index calculation checks every bootstrap loss draw;
 # no production count matrix, aggregation or metric-vector helper is reused.
 rng=np.random.default_rng(20261104);direct=np.empty((2000,2,9))
 v=effective.mean(1)
 for b in range(2000):
  idx=np.concatenate([rng.choice(np.flatnonzero(y==c),size=6,replace=True) for c in range(6)])
  for a in range(2):
   rec=np.array([v[a,idx[y[idx]==c],0].mean() for c in range(6)])
   direct[b,a]=np.r_[rec.mean()*100,rec*100,v[a,idx,1:].mean(0)]
 err=float(np.max(np.abs(direct-result['level_bootstrap_draws'])))
 np.testing.assert_allclose(direct,result['level_bootstrap_draws'],atol=1e-10,rtol=0)
 try:summarize(arms,seeds,y,['SAME']*n,effective,[],repeats=2)
 except ValueError:pass
 else:raise AssertionError('Duplicate lesion IDs were accepted.')
 out.mkdir(parents=True)
 for name,rows in [('synthetic_summary.csv',result['summary']),('synthetic_seeds.csv',result['by_seed'])]:
  with (out/name).open('w',encoding='utf8',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 report=dict(status='SYNTHETIC_MILK_STATISTICS_PASS',scope='Synthetic demonstration, not a clinical study result.',
  synthetic_lesions=n,methods=2,seeds=3,folds=5,paired_lesion_draws=2000,
  direct_index_max_error=err,per_model_BACC_A=60,probability_ensemble_BACC_A=0,
  real_case_rows_read=0,neural_training_performed=False,neural_inference_performed=False,
  checks=['Fold metrics differ from probability ensembling.','NLL and multiclass Brier match direct formulas.',
          'All 2,000 shared lesion draws match independent direct-index calculations.',
          'Duplicate lesion identifiers are rejected.'])
 (out/'SYNTHETIC_MILK_REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
