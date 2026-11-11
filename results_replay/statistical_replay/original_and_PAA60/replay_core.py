"""Run the PAA60 and original prespecified comparisons from saved, coded numerical inputs."""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
from analysis_core import calculate_statistics
ROOT=Path(__file__).resolve().parent
def check(a,b):
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-10)
def tab(path): 
    with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def verify_table(rows,ref,keys,numeric):
    def index(records):return {tuple(str(r[k]) for k in keys):r for r in records}
    left,right=index(rows),index(ref)
    assert len(left)==len(rows)==len(right)==len(ref) and set(left)==set(right)
    values=[]
    for key,row in left.items():
        for k in row:
            v=right[key][k]
            if k in numeric:values.append(abs(float(row[k])-float(v)))
            else:assert str(row[k])==str(v),(key,k)
    check(values,0)
    return max(values,default=0)
def main():
    manifest=json.loads((ROOT/'file_hashes.json').read_text('utf8'))
    for name,expected in manifest.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Input mismatch: '+name)
    plan=json.loads((ROOT/'plan.json').read_text(encoding="utf-8"))
    with np.load(ROOT/'matched.npz',allow_pickle=False) as z:
        a={k:z[k] for k in z.files}
    result=calculate_statistics(plan['methods'],plan['seeds'],a['yi'],a['gi'],a['folds'],a['pi'],a['ye'],a['ge'],a['pe'],plan['pairs'],2000,plan['bootstrap_seed'])
    cols=['type','comparison','metric','target','unit']
    one=verify_table(result['summary'],tab(ROOT/'summary_intervals.csv'),cols,{'estimate','ci95_low','ci95_high','seed_sd'})
    two=verify_table(result['by_seed'],tab(ROOT/'metrics_and_contrasts_by_seed.csv'),cols+['seed'],{'value'})
    errors=[]
    with np.load(ROOT/'bootstrap_reference.npz',allow_pickle=False) as ref:
        assert list(ref['names'])==list(result['arrays']['names'])
        for k in ('point_internal','point_external','draws_internal','draws_external','class_fold_weights','loss_fold_weights'):
            v=result['arrays'][k] if k in result['arrays'] else result[k]
            check(v,ref[k]);errors.append(float(np.max(np.abs(v-ref[k]))))
    expected=json.loads((ROOT/'primary_reference.json').read_text(encoding="utf-8"))
    with np.load(ROOT/'primary.npz',allow_pickle=False) as z:
        y=z['labels'];g=[np.flatnonzero(y==i) for i in range(6)]
        correct={k[6:]:(z[k].argmax(axis=-1)==y) for k in z.files if k.startswith('probs_')}
        for arm,v in correct.items():check([np.mean([row[k].mean() for k in g]) for row in v],expected['bacc'][arm])
    dif=(correct['full'].astype(float)-correct['late_fusion']).mean(0)
    interaction=(correct['full'].astype(float)-correct['no_dsdm']-correct['shared_q']+correct['no_dsdm_shared_q']).mean(0)
    rng=np.random.default_rng(20261016);draws=[]
    for _ in range(2000):
        ix=[rng.choice(k,len(k),replace=True) for k in g]
        draws.append([np.mean([v[k].mean() for k in ix]) for v in (dif,interaction)])
    values=[]
    for i,v in enumerate((dif,interaction)):
        values.extend([np.mean([v[k].mean() for k in g]),*np.quantile(np.asarray(draws)[:,i],[.025,.975])])
    check(values,expected['contrasts'])
    print(json.dumps({'status':'PASS','primary_models':35,'primary_resamples':2000,'matched_models':105,'matched_summary_rows':len(result['summary']),'matched_seed_rows':len(result['by_seed']),'matched_resamples':2000,'max_table_error':max(one,two),'max_array_error':max(errors),'max_primary_error':float(np.max(np.abs(np.array(values)-expected['contrasts']))),'scope':'Original primary and matched extension from frozen predictions'},indent=2))
if __name__=='__main__':main()

