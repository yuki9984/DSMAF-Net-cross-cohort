"""Synthetic software validation ONLY; no study models, training or clinical results."""
from pathlib import Path
import os,sys,json,hashlib,ast,importlib.util,types,contextlib,io
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent;OUT=Path(os.environ.get('DSMAF_SOFTWARE_TEST_OUTPUT',ROOT/'software_validation_only'));OUT.mkdir(parents=True,exist_ok=True)
(OUT/'README_SYNTHETIC_ONLY.txt').write_text('All generated values in this directory are synthetic software fixtures, not study results. No real model was trained or inferred.\n',encoding='utf-8')
from interventions import scenarios,permutation_indices
from inference_core import probabilities,reproduction
from stat_core import responses,summarize_effect,cluster_draws,intervals
checks=[]
def passed(name,**extra):checks.append({'name':name,'status':'PASS',**extra})

m=np.array([[.1,0,1,0,0],[.2,0,0,1,0],[.3,0,0,0,1],[.4,0,1,0,0],[.5,0,0,1,0],[.6,0,0,1,0],[.7,0,0,0,1]],np.float32)
s,idx=scenarios(m,'internal',0);assert len(s)==102 and np.array_equal(s['observed'],m)
assert np.array_equal(s['sex_unknown'][:,:2],m[:,:2]) and np.all(s['sex_unknown'][:,4]==1)
for r in range(100):
 z=s[f'sex_recorded_perm_{r:03d}'];assert np.array_equal(z[:,:2],m[:,:2]) and np.array_equal(z[:,4],m[:,4]);assert np.array_equal(np.sort(idx[r]),np.arange(len(m)));assert np.array_equal(z[:,2:].sum(0),m[:,2:].sum(0))
assert np.array_equal(idx,permutation_indices(m,'internal',0));assert not np.array_equal(idx,permutation_indices(m,'internal',1))
assert np.array_equal(permutation_indices(m,'external',0),permutation_indices(m,'external',4))
passed('100_permutations_preserve_missing_positions_age_and_category_counts')
u=m.copy();u[:,2:]=[0,0,1];u_indices=permutation_indices(u,'internal',0);assert np.array_equal(u_indices,np.tile(np.arange(len(m)),(100,1)));passed('all_unknown_permutation_identity')
try:permutation_indices(np.ones((3,5)), 'internal',0);raise RuntimeError('invalid one-hot not rejected')
except AssertionError:passed('invalid_one_hot_rejected')
p=np.ones((12,6))/6;y=np.tile(np.arange(6),2);v=responses(y,p);assert np.allclose(v[:,2],np.log(6)) and np.allclose(v[:,3],5/6);assert np.isfinite(probabilities(np.array([[100000,-100000,0,0,0,0]]))).all();passed('probability_NLL_Brier_oracles')
try:reproduction(np.eye(6),np.roll(np.eye(6),1,axis=1));raise RuntimeError('changed predictions not rejected')
except AssertionError:passed('prediction_reproduction_fails_closed')
# Direct independent patient resampling oracle: patients may carry multiple classes/images.
y=np.tile(np.arange(6),4);patients=np.repeat(np.arange(12).astype(str),2);x=np.linspace(-.6,.7,len(y));effects={'effect':(x,'bacc',100),'same':(x,'bacc',100),'mean':(x,'mean',1)}
rng=np.random.default_rng(27);names,point,draw,bad=cluster_draws(y,patients,effects,rng,100)
unique=np.unique(patients);rng=np.random.default_rng(27);naive=[]
while len(naive)<100:
 chosen=rng.integers(0,len(unique),len(unique));indices=np.concatenate([np.flatnonzero(patients==unique[k]) for k in chosen]);yy=y[indices]
 if len(np.unique(yy))<6:continue
 xx=x[indices];naive.append([np.mean([xx[yy==c].mean() for c in range(6)])*100,np.mean([xx[yy==c].mean() for c in range(6)])*100,xx.mean()])
assert np.max(abs(draw-np.asarray(naive)))<1e-10 and np.array_equal(draw[:,0],draw[:,1]);passed('patient_cluster_draws_match_independent_image_reconstruction',draws=100)
# AST comparison pins the calls responsible for augmentation/training/selection.
old=ast.parse((ROOT/'original_trainer_reference.txt').read_text(encoding='utf-8-sig'));new=ast.parse((ROOT/'train_recipe.py').read_text(encoding='utf-8-sig'))
def calls(tree,name):return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id==name) or isinstance(n.func,ast.Attribute) and n.func.attr==name)]
for name in ['train_one_epoch','evaluate','AdamW','CosineAnnealingLR','build_transforms','make_model','set_seed','DataLoader']:assert calls(old,name)==calls(new,name),name
source=(ROOT/'train_recipe.py').read_text(encoding='utf-8-sig');assert 'WeightedRandomSampler' not in source and 'BalancedSoftmaxLoss' not in source;assert 'RandomSampler(train_ds, replacement=False, generator=sampling_rng)' in source;assert 'CrossEntropyLoss(weight=None, label_smoothing=0.0)' in source
for p in ROOT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
passed('training_calls_unchanged_only_loss_sampling_namespace_and_timing_changed')
# Optional local torch CPU inference test; never substitutes for the mandatory laptop GPU gate.
try:
 import torch
 from torch import nn
 from torch.utils.data import TensorDataset,DataLoader
 from inference_core import cached_logits,direct_logits,pooled
 class Backbone(nn.Module):
  def forward(self,x):return x,x*.7,x*.5,x*.3
 class Gate(nn.Module):
  def forward(self,x,q):return x*(1+q[:,0,None,None,None])
 class Toy(nn.Module):
  def __init__(self,deep):
   super().__init__();self.use_metadata=True;self.use_dsmaf=deep;self.use_afpn=False;self.use_all_afpn_outputs=True;self.use_late_metadata_fusion=not deep;self.image_backbone=Backbone();self.meta_encoder=nn.Linear(5,2);self.da_block=nn.Identity();self.da_res_scale=.1
   for i in range(1,5):setattr(self,f'adapter{i}',nn.Identity());setattr(self,f'tfpfm{i}',Gate())
   self.classifier=nn.Linear(12+(0 if deep else 2),6)
  def forward(self,x,meta):
   q=self.meta_encoder(meta);features=self.image_backbone(x)
   if self.use_dsmaf:features=tuple(f*(1+q[:,0,None,None,None]) for f in features)
   z=pooled(self,features)
   return self.classifier(z if self.use_dsmaf else torch.cat((z,q),1))
 from torch.utils.data import RandomSampler
 sample=list(RandomSampler(range(37),replacement=False,generator=torch.Generator().manual_seed(7)));assert sorted(sample)==list(range(37)) and sample==list(RandomSampler(range(37),replacement=False,generator=torch.Generator().manual_seed(7)))
 passed('natural_sampler_exactly_once_per_case_and_seed_reproducible')
 torch.manual_seed(4);x=torch.randn(len(m),3,4,4);ld=DataLoader(TensorDataset(x,torch.from_numpy(m)),batch_size=3)
 for deep in [False,True]:
  model=Toy(deep).eval();subset={k:s[k] for k in ['observed','sex_unknown','sex_recorded_perm_000']};got=cached_logits(model,ld,subset,torch.device('cpu'),amp=False)
  for name,z in subset.items():
   direct=direct_logits(model,DataLoader(TensorDataset(x,torch.from_numpy(z)),batch_size=3),torch.device('cpu'),amp=False);assert np.max(abs(got[name]-direct))<1e-6
 passed('cached_vs_direct_toy_CPU_both_metadata_paths',torch=str(torch.__version__))
except ImportError as e:checks.append({'name':'cached_vs_direct_toy_CPU_both_metadata_paths','status':'NOT_AVAILABLE','reason':str(e)})
# End-to-end synthetic oracle, kept outside production outputs.
fi=[];fy=[]
for f in range(5):
 for c in range(6):
  for k in range(f+1):fi.append(f);fy.append(c)
folds=np.asarray(fi);yi=np.asarray(fy);ye=np.tile(np.arange(6),3);ni=len(yi);ne=len(ye)
logi=np.eye(6)[yi]*12;loge=np.eye(6)[ye]*12;ip=np.broadcast_to(probabilities(logi),(3,5,ni,6)).copy();ep=np.broadcast_to(probabilities(loge),(3,5,5,ne,6)).copy()
a={'internal_labels':yi,'external_labels':ye,'internal_folds':folds,'internal_images':np.array([f'i{k:03d}' for k in range(ni)]),'external_images':np.array([f'e{k:03d}' for k in range(ne)]),'internal_patients':np.array([f'p{k//2:03d}' for k in range(ni)]),'external_patients':np.array([f'q{k:03d}' for k in range(ne)]),'prob_internal':ip,'prob_external':ep}
fixture=OUT/'synthetic_pipeline_only';fixture.mkdir(exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def savez(p,**kw):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,**kw)
def modelname(r):return f"fold_{r['fold']:02d}_{r['arm']}_seed{r['seed']}"
settings=types.ModuleType('followup_settings');items={'RESULTS':fixture,'ARMS':('full','image_only','late_fusion'),'SEEDS':(20261013,20261014,20261015),'ALL_SEEDS':tuple(range(20261013,20261018)),'LABELS':('ACK','BCC','MEL','NEV','SCC','SEK'),'original_arrays':lambda:a,'atomic_npz':savez,'atomic_json':put,'read':read,'sha256':digest,'model_name':modelname}
for k,v in items.items():setattr(settings,k,v)
previous=sys.modules.get('followup_settings');sys.modules['followup_settings']=settings
try:
 spec=importlib.util.spec_from_file_location('synthetic_analysis_only',ROOT/'analyze_followup.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
finally:
 if previous is None:del sys.modules['followup_settings']
 else:sys.modules['followup_settings']=previous
original_intervals=mod.intervals
mod.intervals=lambda *args,**kwargs:original_intervals(*args,**{**kwargs,'repeats':100})
mod.load_new_arrays=lambda unused:(ip[:,:3].copy(),ep[:,:3].copy())
with contextlib.redirect_stdout(io.StringIO()):mod.analyze_recipe()
r=read(fixture/'analysis/recipe45/complete.json');df=pd.read_csv(fixture/'analysis/recipe45/all_recipe_contrasts_and_interactions.csv');z=df[df.key.str.startswith('recipe_difference|')]
assert r['seed_metrics_rows']==36 and r['contrast_interval_rows']==270 and len(z)==90
assert np.max(abs(z[['estimate','ci95_low','ci95_high']].to_numpy()))<1e-10
passed('full_recipe_pipeline_zero_interaction_oracle',seed_metric_rows=36,interval_rows=270)
# A deliberately incorrect probability-ensemble shortcut must disagree with fixed-fold correctness.
q=ep.copy();q[:]=.001;q[...,0]=.994
# Alternate fold decisions while keeping very different confidence magnitudes.
for f in range(5):
 for j,c in enumerate(ye):
  q[:,:,f,j]=.001;q[:,:,f,j,c if f<3 else (c+1)%6]=.994
vi,ve=mod.recipe_values(a,ip[:,:3],q[:,:3]);wc,wi=mod.weights(a)
expected=sum(wc[f,ye]*(q[0,0,f].argmax(1)==ye) for f in range(5));assert np.max(abs(ve[0,0,:,0]-expected))<1e-12
ensemble=q[0,0].mean(0).argmax(1)==ye;assert not np.array_equal(expected,ensemble.astype(float));passed('fixed_fold_correctness_not_probability_ensemble')
records=[]
for cohort in ['internal','external']:
 for f in range(5):
  ix=np.flatnonzero(folds==f) if cohort=='internal' else np.arange(ne);y=yi[ix] if cohort=='internal' else ye;ids=a['internal_images'][ix] if cohort=='internal' else a['external_images'];patients=a['internal_patients'][ix] if cohort=='internal' else a['external_patients'];base=logi[ix] if cohort=='internal' else loge
  for arm in settings.ARMS:
   for seed in settings.ALL_SEEDS:
    rec={'cohort':cohort,'fold':f,'arm':arm,'seed':seed};p=fixture/'interventions'/cohort/(modelname(rec)+'.npz');kw={'img_ids':ids,'patient_ids':patients,'labels':y,'logits_observed':base}
    if arm!='image_only':
     unknown=np.roll(base,1,axis=1) if arm=='full' else base.copy();perms=np.repeat(base[None],100,axis=0)
     if arm=='full':perms[:2]=np.roll(base,1,axis=1)
     kw.update(logits_sex_unknown=unknown,logits_sex_recorded_perm=perms)
    savez(p,**kw);records.append({**rec,'artifact_sha256':digest(p)})
put(fixture/'interventions/complete.json',{'status':'PASS','runs':records})
with contextlib.redirect_stdout(io.StringIO()):mod.analyze_interventions()
r=read(fixture/'analysis/interventions/complete.json');df=pd.read_csv(fixture/'analysis/interventions/all_intervention_patient_cluster_intervals.csv');summary=pd.read_csv(fixture/'analysis/interventions/permutation_seed_SD_permutation_SD_and_MCSE.csv')
assert r['seed_delta_rows']==60 and r['patient_interval_rows']==198 and r['repetition_rows']==3000
for condition,value in [('sex_unknown',-100),('sex_recorded_permutation',-2)]:
 z=df[(df.key==condition+'|full|bacc') & df.target.isin(['internal','external'])];assert len(z)==2 and np.max(abs(z[['estimate','ci95_low','ci95_high']].to_numpy()-value))<1e-9
z=df[(df.key=='sex_recorded_permutation|full|bacc') & df.target.eq('external_minus_internal')];assert np.max(abs(z[['estimate','ci95_low','ci95_high']].to_numpy()))<1e-9
z=summary[summary.arm.eq('full')];assert np.max(abs(z.trained_seed_SD_pp))<1e-9 and (z.permutation_SD_of_five_seed_means_pp>0).all();assert np.max(abs(z.MCSE_pp-z.permutation_SD_of_five_seed_means_pp/10))<1e-12
passed('full_intervention_pipeline_nonzero_delta_and_variability_oracles',seed_rows=60,interval_rows=198,repetition_rows=3000)
report={'status':'PASS','scope':'Synthetic software tests only; no study model inference/training or clinical result. CPU toy test does not replace mandatory original-laptop GPU qualification.','checks':checks,'checks_passed':sum(r['status']=='PASS' for r in checks),'GPU_runtime_or_real_checkpoint_equivalence_tested':False,'actual_new_study_training_fits':0}
put(OUT/'software_validation_report.json',report);print(json.dumps(report,indent=2),flush=True)
