"""Cache deterministic original evaluation crops, without a model or clinical predictions."""
import ast,concurrent.futures,hashlib,io,json,pathlib,sys,time,zipfile
import numpy as np,pandas as pd
from PIL import Image
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(PROJECT/'revision_work/paa_focused_revision_20261103/check_dependencies'))
import torch,torchvision
from torchvision import transforms

def main():
 source=PROJECT/'protocol_v2_paa_extension_20261029_portable_v2/runtime/compute_checks/20261014_transfer/gpu_probe/code/train_hfpa.py'
 tree=ast.parse(source.read_text(encoding='utf-8-sig'));fn=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_transforms'];assert len(fn)==1
 ns={'transforms':transforms};exec(compile(ast.Module(body=fn,type_ignores=[]),str(source),'exec'),ns)
 _,original=ns['build_transforms'](224);spatial=transforms.Compose(original.transforms[:2]);tail=transforms.Compose(original.transforms[2:])
 cases=pd.read_csv(ROOT/'prepared/candidate_cases.csv');ids=cases.isic_id.tolist();n=len(ids)
 target=ROOT/'prepared/candidate_rgb224.npy';partial=target.with_suffix('.npy.part')
 cache=np.lib.format.open_memmap(partial,mode='w+',dtype=np.uint8,shape=(n,224,224,3))
 z=zipfile.ZipFile(io.BytesIO((ROOT/'raw/milk10k.zip').read_bytes()));entries={pathlib.PurePosixPath(x).stem:x for x in z.namelist() if pathlib.PurePosixPath(x).suffix.lower() in ['.jpg','.jpeg','.png']}
 sampled=set(np.linspace(0,n-1,24,dtype=int).tolist());checks=[]
 def crop(pair):
  i,image_id=pair;data=z.read(entries[image_id])
  with Image.open(io.BytesIO(data)) as im:
   rgb=im.convert('RGB');arr=np.asarray(spatial(rgb)).copy()
   if i in sampled:
    assert torch.equal(original(rgb),tail(arr)),'Caching changed original preprocessing'
  return i,arr
 started=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for done,(i,arr) in enumerate(pool.map(crop,enumerate(ids)),1):
   cache[i]=arr
   if done%1000==0:print('Cached original crops',done,'/',n,flush=True)
 cache.flush();del cache;z.close();partial.replace(target)
 np.save(ROOT/'prepared/candidate_rgb224_ids.npy',np.asarray(ids,dtype=str),allow_pickle=False)
 h=hashlib.sha256()
 with target.open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 report={'status':'ORIGINAL_EVAL_CROPS_CACHED','shape':[n,224,224,3],'dtype':'uint8','size_bytes':target.stat().st_size,'sha256':h.hexdigest(),'case_selection_sha256':hashlib.sha256((ROOT/'prepared/candidate_cases.csv').read_bytes()).hexdigest(),'original_transform_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'torch':torch.__version__,'torchvision':torchvision.__version__,'exact_cached_vs_original_transform_checks':len(sampled),'seconds':time.monotonic()-started,'candidate_superset_final_exclusions_pending':True,'model_loaded':False,'model_inference':False}
 (ROOT/'reports/image_cache_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
