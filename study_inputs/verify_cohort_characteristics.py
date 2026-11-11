from pathlib import Path
import csv,math,statistics,json,hashlib
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parent
LABELS={'ACK':0,'BCC':1,'MEL':2,'NEV':3,'SCC':4,'SEK':5}
def read_rows(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def raw_value(row):
 age=float(row['age']);assert math.isfinite(age),'Non-finite raw age'
 sex=row['gender'].strip().upper();assert sex in ['','FEMALE','MALE'],'Invalid raw sex'
 return row['patient_id'],age,sex

def raw_patient_summary(rows):
 patients=defaultdict(list)
 for r in sorted(rows,key=lambda x:(x['patient_id'],x['img_id'])):patients[r['patient_id']].append(raw_value(r)[1:])
 assert all(len({s for a,s in values})==1 for values in patients.values()),'Conflicting within-patient sex'
 values=[v[0] for v in patients.values()];ages=[a for a,s in values];n=len(values)
 varying={p:v for p,v in patients.items() if len({a for a,s in v})>1}
 return {'images':len(rows),'patients':n,'age_mean':statistics.mean(ages),'age_sd':statistics.stdev(ages),'female':sum(s=='FEMALE' for a,s in values),'male':sum(s=='MALE' for a,s in values),'unrecorded':sum(s=='' for a,s in values),'age_variant_patients':len(varying),'age_variant_images':sum(len(v) for v in varying.values())}

def split_summary(rows):
 groups=defaultdict(list)
 for r in rows:
  f,m=int(r['gender_FEMALE']),int(r['gender_MALE']);assert (f,m) in [(0,0),(1,0),(0,1)]
  groups[r['patient_id']].append(r)
 values=[];variant=0;var_images=0
 for pid,rr in groups.items():
  assert len({(int(r['gender_FEMALE']),int(r['gender_MALE'])) for r in rr})==1
  first=min(rr,key=lambda r:r['img_id'])
  values.append((float(first['age']),int(first['gender_FEMALE']),int(first['gender_MALE'])))
  if len({float(r['age']) for r in rr})>1:variant+=1;var_images+=len(rr)
 values=np.asarray(values,dtype=float)
 return {'images':len(rows),'patients':len(values),'age_mean':float(np.mean(values[:,0])),'age_sd':float(np.std(values[:,0],ddof=1)),'female':int(np.sum(values[:,1])),'male':int(np.sum(values[:,2])),'unrecorded':int(np.sum(values[:,1]+values[:,2]==0)),'age_variant_patients':variant,'age_variant_images':var_images}

def calculate(root=ROOT):
 raw=read_rows(root/'data/PAD_metadata.csv');index={r['img_id']:r for r in raw}
 assert len(raw)==len(index)==2298
 full=raw_patient_summary(raw);assert full['patients']==1373 and full['age_variant_patients']==90 and full['age_variant_images']==316
 age_sets=defaultdict(set)
 for row in raw:age_sets[row['patient_id']].add(float(row['age']))
 ranges=[max(v)-min(v) for v in age_sets.values() if len(v)>1]
 assert len(ranges)==90 and min(ranges)==1 and max(ranges)==30
 range_report={'minimum':min(ranges),'maximum':max(ranges),'histogram':{str(int(v)):ranges.count(v) for v in sorted(set(ranges))}}
 assert sum(r['gender'].strip()=='' for r in raw)==804
 summaries=[dict(partition='Full PAD',fold=None,role='all',**full)]
 paths=[root/'data/PAD_metadata.csv'];checked=0;outer_ids=[];outer_patients=[]
 for fold in range(5):
  groups={role:read_rows(root/f'internal75/splits/fold_{fold:02d}_{role}.csv') for role in ['train','valid','test']}
  ids={role:{r['img_id'] for r in rows} for role,rows in groups.items()}
  pids={role:{r['patient_id'] for r in rows} for role,rows in groups.items()}
  hashes={role:{r['image_sha256'] for r in rows} for role,rows in groups.items()}
  for role in groups:assert len(ids[role])==len(groups[role])
  assert set.union(*ids.values())==set(index)
  for a,b in [('train','valid'),('train','test'),('valid','test')]:
   assert not ids[a]&ids[b] and not pids[a]&pids[b] and not hashes[a]&hashes[b],'Role overlap'
  for role,rows in groups.items():
   paths.append(root/f'internal75/splits/fold_{fold:02d}_{role}.csv')
   selected=[]
   for r in rows:
    source=index[r['img_id']];pid,age,sex=raw_value(source)
    assert r['patient_id']==pid and float(r['age'])==age
    assert int(r['diagnostic_number'])==LABELS[source['diagnostic']]
    assert (int(r['gender_FEMALE']),int(r['gender_MALE']))==(int(sex=='FEMALE'),int(sex=='MALE'))
    selected.append(source);checked+=1
   one=raw_patient_summary(selected);two=split_summary(rows)
   for key in one:assert abs(one[key]-two[key])<1e-10,(fold,role,key,one[key],two[key])
   assert one['female']+one['male']+one['unrecorded']==one['patients']
   label={'train':'Training','valid':'Inner selection','test':'Outer test'}[role]
   summaries.append(dict(partition=f'Fold {fold}: {label}',fold=fold,role=role,**one))
  if fold==0:
   oracle=split_summary([r for rows in groups.values() for r in rows])
   for key in full:assert abs(full[key]-oracle[key])<1e-10,key
   frozen_sets=defaultdict(set)
   for role_rows in groups.values():
    for row in role_rows:frozen_sets[row['patient_id']].add(float(row['age']))
   assert dict(frozen_sets)==dict(age_sets)
  outer_ids+=list(ids['test']);outer_patients+=list(pids['test'])
 original=[];original_groups={}
 for role,label in [('train','Original train'),('valid','Original tuning')]:
  path=root/f'data/pad_{role}.csv';paths.append(path);rr=read_rows(path);original_groups[role]=rr
  selected=[index[r['img_id']] for r in rr]
  for frozen,source in zip(rr,selected):
   pid,age,sex=raw_value(source)
   assert frozen['patient_id']==pid and float(frozen['age'])==age
   assert int(frozen['diagnostic_number'])==LABELS[source['diagnostic']]
   assert (int(frozen['gender_FEMALE']),int(frozen['gender_MALE']))==(int(sex=='FEMALE'),int(sex=='MALE'))
  one=raw_patient_summary(selected);two=split_summary(rr)
  for key in one:assert abs(one[key]-two[key])<1e-10,(role,key)
  original.append(dict(partition=label,fold=None,role=role,**one))
 assert not {r['img_id'] for r in original_groups['train']}&{r['img_id'] for r in original_groups['valid']}
 assert not {r['patient_id'] for r in original_groups['train']}&{r['patient_id'] for r in original_groups['valid']}
 assert sum(r['images'] for r in original)==1822 and sum(r['patients'] for r in original)==1097
 assert len(outer_ids)==len(set(outer_ids))==2298
 assert len(outer_patients)==len(set(outer_patients))==1373
 assert checked==11490 and len(summaries)==16
 return {'status':'PASS','rows':summaries,'original_development_partitions':original,'original_rows_cross_checked':1822,'within_patient_age_range_years':range_report,'age_and_sex_unit':'Distinct patients within partition','age_sd_ddof':1,'patient_age_summary_rule':'First image after lexicographic patient_id,img_id sorting; descriptive only','per_image_model_inputs_unchanged':True,'rows_cross_checked':checked,'new_model_training':False,'new_neural_inference':False,'source_hashes':{p.relative_to(root).as_posix():sha(p) for p in paths},'scope':'Descriptive re-expression of frozen source records; five outer role views are not independent cohorts.'}

if __name__=='__main__':
 result=calculate()
 expected=ROOT/'cohort_characteristics/expected.json'
 if expected.exists():
  old=json.loads(expected.read_text(encoding="utf-8"))
  for a,b in zip(result['rows'],old['rows']):
   assert a.keys()==b.keys()
   for k,v in a.items():
    assert abs(v-b[k])<1e-10 if isinstance(v,float) else v==b[k],(a['partition'],k)
 out=ROOT/'verification_output';out.mkdir(exist_ok=True)
 (out/'cohort_characteristics_report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print('COHORT_CHARACTERISTICS_18_ROWS_RAW_AND_FROZEN_CHECKS_PASS',flush=True)
