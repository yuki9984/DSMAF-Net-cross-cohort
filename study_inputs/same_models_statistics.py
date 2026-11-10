"""Independent statistics from frozen predictions; no neural inference or fitting."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
ARMS=['full','image_only','late_fusion'];SEEDS=list(range(20261013,20261018))
def metrics(y,p):
 pred=p.argmax(1);counts=np.bincount(y,minlength=6);hits=np.bincount(y,weights=(pred==y),minlength=6)
 recalls=hits/counts;confidence=p.max(1);hit=(pred==y).astype(float)
 ece=0.
 for j in range(10):
  chosen=(confidence>=j/10)&((confidence<(j+1)/10) if j<9 else confidence<=1)
  if chosen.any():ece+=chosen.mean()*abs(confidence[chosen].mean()-hit[chosen].mean())
 return {'bacc':float(recalls.mean()),'accuracy':float(hit.mean()),'nll':float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean()),'brier':float(((p-np.eye(6)[y])**2).sum(1).mean()),'ece_10bin':float(ece),'sek_to_mel':float((pred[y==5]==2).mean()),**{f'recall_{i}':float(v) for i,v in enumerate(recalls)}}
def run(arrays,expected_folder,output):
 arrays=Path(arrays);expected_folder=Path(expected_folder);output=Path(output);output.mkdir(exist_ok=True,parents=True)
 z=np.load(arrays,allow_pickle=False);I=z['prob_internal'];E=z['prob_external'];yi=z['internal_labels'];ye=z['external_labels'];folds=z['internal_folds']
 assert I.shape==(3,5,2298,6) and E.shape==(3,5,5,1701,6)
 support=np.stack([np.bincount(yi[folds==f],minlength=6) for f in range(5)]);wc=support/support.sum(0);wn=support.sum(1)/len(yi)
 allrows=[];seedrows=[];values={'internal':{},'external':{}};maxerror=0.
 expected=pd.read_csv(expected_folder/'all_150_fold_cohort_metrics.csv')
 for ai,arm in enumerate(ARMS):
  for si,seed in enumerate(SEEDS):
   for fold in range(5):
    for cohort,y,p in [('internal',yi[folds==fold],I[ai,si,folds==fold]),('external',ye,E[ai,si,fold])]:
     m=metrics(y,p);row={'cohort':cohort,'arm':arm,'seed':seed,'fold':fold,**m};allrows.append(row)
     prior=expected.query('cohort == @cohort and arm == @arm and seed == @seed and fold == @fold');assert len(prior)==1
     for key,v in m.items():maxerror=max(maxerror,abs(v-float(prior.iloc[0][key])))
   # Average correctness/losses, never probabilities.
   hit_i=(I[ai].argmax(-1)==yi).astype(float)
   hit_e=(E[ai].argmax(-1)==ye).astype(float)
   correct_e=sum(wc[f,ye][None,:]*hit_e[:,f,:] for f in range(5))
   nll_i=-np.log(np.maximum(np.take_along_axis(I[ai],np.broadcast_to(yi,(5,len(yi)))[:,:,None],axis=2)[:,:,0],1e-12))
   nll_e=sum(wn[f]*(-np.log(np.maximum(np.take_along_axis(E[ai,:,f],np.broadcast_to(ye,(5,len(ye)))[:,:,None],axis=2)[:,:,0],1e-12))) for f in range(5))
   brier_i=((I[ai]-np.eye(6)[yi])**2).sum(-1)
   brier_e=sum(wn[f]*((E[ai,:,f]-np.eye(6)[ye])**2).sum(-1) for f in range(5))
   mel_i=(I[ai].argmax(-1)==2).astype(float)
   mel_e=sum(wc[f,ye][None,:]*(E[ai,:,f].argmax(-1)==2) for f in range(5))
   values['internal'][arm]={'bacc':hit_i,'nll':nll_i,'brier':brier_i,'sek_to_mel':mel_i}
   values['external'][arm]={'bacc':correct_e,'nll':nll_e,'brier':brier_e,'sek_to_mel':mel_e}
   for cohort,y in [('internal',yi),('external',ye)]:
    v=values[cohort][arm]
    seedrows.append({'cohort':cohort,'arm':arm,'seed':seed,'bacc':float(np.mean([v['bacc'][si,y==c].mean() for c in range(6)])),'nll':float(v['nll'][si].mean()),'brier':float(v['brier'][si].mean()),'sek_to_mel':float(v['sek_to_mel'][si,y==5].mean()),**{f'recall_{c}':float(v['bacc'][si,y==c].mean()) for c in range(6)}})
 assert maxerror<1e-10,maxerror
 rows=pd.DataFrame(allrows);seeds=pd.DataFrame(seedrows)
 rows.to_csv(output/'independent_150_fold_metrics.csv',index=False);seeds.to_csv(output/'independent_seed_metrics.csv',index=False)
 prior=pd.read_csv(expected_folder/'seed_level_matched_cohort_metrics.csv');cmp=seeds.merge(prior,on=['cohort','arm','seed'],suffixes=('_new','_old'))
 seederror=max(float(abs(cmp[k+'_new']-cmp[k+'_old']).max()) for k in ['bacc','nll','brier','sek_to_mel']+[f'recall_{c}' for c in range(6)])
 assert seederror<1e-10,seederror
 # Independent direct cluster-sum implementation; identical locked RNG stream.
 keys=[(arm,metric) for arm in ['late_fusion','image_only'] for metric in ['bacc','nll','brier','sek_to_mel']]
 rng=np.random.default_rng(20261022);boot={};point={};rejects={}
 for cohort,y,ids in [('internal',yi,z['internal_patients']),('external',ye,z['external_patients'])]:
  patient,inv=np.unique(ids.astype(str),return_inverse=True);n=len(patient)
  delta=np.stack([(values[cohort]['full'][metric]-values[cohort][arm][metric]).mean(0) for arm,metric in keys],axis=1)
  ccounts=np.stack([np.bincount(inv[y==c],minlength=n) for c in range(6)],axis=1)
  csums=np.zeros((n,6,len(keys)))
  for c in range(6):
   for j in range(len(keys)):csums[:,c,j]=np.bincount(inv[y==c],weights=delta[y==c,j],minlength=n)
  records=[];reject=0
  while len(records)<2000:
   selected=rng.integers(0,n,n);den=ccounts[selected].sum(0)
   if (den==0).any():reject+=1;continue
   num=csums[selected].sum(0);out=[]
   for j,(_,metric) in enumerate(keys):
    out.append((num[:,j]/den).mean() if metric=='bacc' else num[5,j]/den[5] if metric=='sek_to_mel' else num[:,j].sum()/den.sum())
   records.append(out)
  boot[cohort]=np.array(records);rejects[cohort]=reject
  point[cohort]=np.array([np.mean([delta[y==c,j].mean() for c in range(6)]) if metric=='bacc' else delta[y==5,j].mean() if metric=='sek_to_mel' else delta[:,j].mean() for j,(_,metric) in enumerate(keys)])
 intervals=[]
 for j,(arm,metric) in enumerate(keys):
  for target,p,b in [('internal',point['internal'][j],boot['internal'][:,j]),('external',point['external'][j],boot['external'][:,j]),('external_minus_internal',point['external'][j]-point['internal'][j],boot['external'][:,j]-boot['internal'][:,j])]:
   scale=100 if metric in ['bacc','sek_to_mel'] else 1;lo,hi=np.quantile(b,[.025,.975])
   intervals.append({'comparison':'full-minus-'+arm,'metric':metric,'target':target,'estimate':float(p*scale),'ci95_low':float(lo*scale),'ci95_high':float(hi*scale),'unit':'percentage_points' if scale==100 else 'dimensionless'})
 ci=pd.DataFrame(intervals);ci.to_csv(output/'independent_paired_intervals.csv',index=False)
 archived=pd.read_csv(expected_folder/'paired_patient_cluster_intervals.csv');m=ci.merge(archived,on=['comparison','metric','target','unit'],suffixes=('_new','_old'))
 intervalerror=max(float(abs(m[k+'_new']-m[k+'_old']).max()) for k in ['estimate','ci95_low','ci95_high']);assert len(m)==24 and intervalerror<1e-10,intervalerror
 equal=rows.groupby(['cohort','arm','seed']).bacc.mean().rename('equal_fold_bacc').reset_index();m=seeds.merge(equal,on=['cohort','arm','seed']);m['difference_percentage_points']=(m.bacc-m.equal_fold_bacc)*100
 m.to_csv(output/'equal_fold_sensitivity_recomputed.csv',index=False)
 summary=seeds.groupby(['cohort','arm'])[['bacc','nll','brier','sek_to_mel']].agg(['mean','std']);summary.to_csv(output/'matched_summary.csv')
 report={'status':'PASS','fold_cohort_metrics_rows':150,'fold_metric_values_recomputed':len(allrows)*12,'seed_cohort_rows':30,'intervals_recomputed':24,'maximum_fold_metric_error':maxerror,'maximum_seed_metric_error':seederror,'maximum_interval_error':intervalerror,'bootstrap_seed':20261022,'bootstrap_repeats':2000,'rejected_missing_class_draws':rejects,'maximum_equal_vs_class_support_weighted_external_bacc_difference_pp':float(m[m.cohort.eq('external')].difference_percentage_points.abs().max()),'probability_ensemble':False,'all_seeds_folds_retained':True,'new_neural_training':False,'scope':'Independent recomputation from saved probabilities, with direct patient-cluster sums; fixed model collection and descriptive post hoc inference.'}
 (output/'independent_statistics_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
 return report
if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--arrays',required=True,type=Path);ap.add_argument('--expected',required=True,type=Path);ap.add_argument('--output',required=True,type=Path);a=ap.parse_args();run(a.arrays,a.expected,a.output)
