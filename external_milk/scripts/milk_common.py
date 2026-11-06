"""MILK10k paths, input identities and immutable frozen-fit inventory."""
from pathlib import Path
import csv,hashlib,json,re,sys,os
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
BUNDLE=PROJECT/'protocol_v2_paa_extension_20261029_portable_v2'
PAA30=PROJECT/'revision_work/paa_focused_revision_20261103/handoff/DSMAF-PAA30-LateFusion'
SEEDS=(20261013,20261014,20261015)
ARMS=('image_only','late_fusion','full','metablock_cv','rg_dermnet_adapted_cv','dsmaf_age_only','dsmaf_age_recorded','late_age_only','late_age_recorded')
CONDITIONS={a:('age_recorded' if a.endswith('_age_recorded') else 'age_only' if a.endswith('_age_only') else 'full_age_sex') for a in ARMS}
LABELS=('ACK','BCC','MEL','NEV','SCC','SEK')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def json_hash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');q.replace(p)
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rs):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def setup():
 sys.dont_write_bytecode=True
 for p in [PROJECT/'revision_work/paa_focused_revision_20261103/check_dependencies',ROOT/'dependencies',BUNDLE]:
  sys.path.insert(0,str(p))
 os.environ.setdefault('HF_HOME',str(ROOT/'tmp/hf'))
 os.environ.setdefault('TORCH_HOME',str(ROOT/'tmp/torch'))
 import extension_settings as S
 S.setup_original()
 return S
def matrix():return [(f,a,s) for a in ARMS for s in SEEDS for f in range(5)]
def task_id(f,a,s):return f'fold_{f:02d}_{a}_seed{s}'
def validate_campaign(manifest,tag):
 arms=ARMS[3:7] if tag=='PAA60' else ARMS[7:]
 count=len(arms)*15
 if manifest.get('status')!=f'ALL_{count}_FITS_FROZEN':raise ValueError('Entire campaign must be frozen: '+tag)
 actual=[(r['fold'],r['arm'],r['seed']) for r in manifest['runs']]
 if len(actual)!=count or set(actual)!={(f,a,s) for f in range(5) for a in arms for s in SEEDS}:raise ValueError('Wrong campaign fold/seed grid')
 plan=BUNDLE/'analysis_plan.json' if tag=='PAA60' else PAA30/'analysis_plan.json'
 if manifest['plan_sha256']!=sha(plan):raise ValueError('Freeze manifest plan differs from delivered training plan')
 if tag=='PAA60':
  trusted=PROJECT/'output/PAA60_verified_return_20261103/received/outputs/training_complete.json'
  if json_hash(manifest)!=json_hash(read(trusted)):raise ValueError('PAA60 differs from previously returned frozen manifest')
 return manifest['runs']
def inventory():
 records=[];missing=[]
 legacy=read(ROOT/'prepared/checkpoint_requirements.json')['legacy_matched_checkpoints']
 cfg={(r['fold'],r['arm'],r['seed']):r['config'] for r in read(BUNDLE/'materials/control_configs.json')}
 for r in legacy:
  m=re.search(r'fold_(\d+)/(.+)_seed(\d+)/best_valid.pth',r['path']);f,a,s=int(m[1]),m[2],int(m[3])
  p=Path(r['absolute_path'])
  if not p.is_file():missing.append(str(p))
  records.append(dict(fold=f,arm=a,seed=s,path=str(p),checkpoint_sha256=r['sha256'],config=cfg[(f,a,s)],campaign='legacy'))
 for tag in ['PAA60','PAA30']:
  base=ROOT/'weights'/tag
  manifest=base/'outputs/training_complete.json'
  if not manifest.exists():
   missing.append(str(manifest));continue
  for r in validate_campaign(read(manifest),tag):
   folder=base/'outputs/fits'/f"fold_{r['fold']:02d}"/f"{r['arm']}_seed{r['seed']}"
   p=folder/'best_valid.pth';c=folder/'config.json'
   if not p.is_file() or not c.is_file():missing.append(str(folder));continue
   if sha(c)!=r['config_sha256']:raise ValueError('Changed config '+str(c))
   conf=read(c)
   if (conf['fold'],conf['arm'],conf['seed'])!=(r['fold'],r['arm'],r['seed']) or conf.get('trial') is not False:raise ValueError('Configuration identity/trial mismatch')
   if conf['information_condition']!=CONDITIONS[r['arm']]:raise ValueError('Training information condition mismatch')
   records.append(dict(fold=r['fold'],arm=r['arm'],seed=r['seed'],path=str(p),checkpoint_sha256=r['checkpoint_sha256'],config=conf,campaign=tag))
 if not missing and (len(records)!=135 or {(r['fold'],r['arm'],r['seed']) for r in records}!=set(matrix())):raise ValueError('Wrong 135-model inventory')
 return sorted(records,key=lambda r:(ARMS.index(r['arm']),r['seed'],r['fold'])),missing
def check_cohort():
 import numpy as np
 lock=read(ROOT/'prepared/cohort_lock.json')
 if lock['status']!='COHORT_FIXED_BEFORE_OUTCOMES':raise ValueError('Cohort is not fixed')
 for path,digest in lock['sha256'].items():
  if sha(ROOT/path)!=digest:raise ValueError('Changed locked input: '+path)
 cases=rows(ROOT/'prepared/evaluation_cases.csv')
 idx=np.load(ROOT/'prepared/evaluation_cache_indices.npy',allow_pickle=False)
 ids=np.array([r['isic_id'] for r in cases])
 if len(cases)!=lock['images'] or len(set(ids))!=len(cases):raise ValueError('Wrong case list')
 if not np.array_equal(np.load(ROOT/'prepared/candidate_rgb224_ids.npy',allow_pickle=False)[idx],ids):raise ValueError('Image cache order mismatch')
 return cases,idx,lock

