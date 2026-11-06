"""Verify archive and screen clinical images against PAD and DermatoNet; no inference."""
import concurrent.futures,collections,hashlib,json,pathlib,zipfile,io
import numpy as np,pandas as pd
from PIL import Image
from scipy.fft import dctn
ROOT=pathlib.Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def signature(item):
 cohort,path,*payload=item
 data=payload[0] if payload else path.read_bytes()
 with Image.open(io.BytesIO(data)) as im:
  im.load();size=im.size;mode=im.mode;rgb=im.convert('RGB');pix=hashlib.sha256(str(size).encode()+rgb.tobytes()).hexdigest()
  gray=rgb.convert('L');a=np.asarray(gray.resize((32,32),Image.Resampling.LANCZOS),dtype=float)
  small=dctn(a,type=2,norm='ortho')[:8,:8];bits=small>np.median(small.flat[1:]);bits[0,0]=False
  ph=int.from_bytes(np.packbits(bits.reshape(-1)).tobytes(),'big')
  b=np.asarray(gray.resize((9,8),Image.Resampling.LANCZOS));dh=int.from_bytes(np.packbits((b[:,1:]>b[:,:-1]).reshape(-1)).tobytes(),'big')
 return {'cohort':cohort,'path':str(path),'image_id':path.stem,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'pixel_sha256':pix,'width':size[0],'height':size[1],'mode':mode,'phash':f'{ph:016x}','dhash':f'{dh:016x}'}
def main():
 archive=ROOT/'raw/milk10k.zip';receipt=json.loads(archive.with_suffix('.zip.download.json').read_text());assert sha(archive)==receipt['sha256']
 all_cases=pd.read_csv(ROOT/'prepared/all_clinical_selection.csv');cases=pd.read_csv(ROOT/'prepared/candidate_cases.csv')
 candidate_data={}
 with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as z:
  assert z.testzip() is None,'ZIP CRC failed'
  mapping={}
  for info in z.infolist():
   if pathlib.PurePosixPath(info.filename).suffix.lower() in ['.jpg','.jpeg','.png']:
    stem=pathlib.PurePosixPath(info.filename).stem
    assert stem not in mapping
    mapping[stem]=info
  assert len(mapping)==10480
  for image_id in cases.isic_id:
   assert image_id in mapping
   candidate_data[image_id]=z.read(mapping[image_id])
  archive_entries={image_id:mapping[image_id].filename for image_id in all_cases.isic_id}
 (ROOT/'prepared/clinical_archive_entries.json').write_text(json.dumps(archive_entries,indent=2),encoding='utf-8')
 print('Complete archive CRC passed; clinical data read directly from verified ZIP',flush=True)
 refs=[('PAD',PROJECT/'teacher_materials/20261014/original/DSMAF/PAD-UFES-20/images'),('DermatoNet',PROJECT/'external_data/DermatoNet_v2/images')]
 items=[('MILK10k',pathlib.Path(i+'.jpg'),candidate_data[i]) for i in cases.isic_id]
 for cohort,ref in refs:
  assert ref.is_dir(),ref
  items.extend((cohort,p) for p in ref.rglob('*') if p.suffix.lower() in ['.jpg','.jpeg','.png'])
 rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  for row in pool.map(signature,items):
   rows.append(row)
   if len(rows)%1000==0:print('fingerprints',len(rows),'/',len(items),flush=True)
 df=pd.DataFrame(rows);df.to_csv(ROOT/'reports/image_fingerprints.csv',index=False)
 exact=[]
 for key in ['sha256','pixel_sha256']:
  for h,g in df.groupby(key):
   milk=g[g.cohort.eq('MILK10k')]
   if len(milk) and len(g)>1:exact.append({'method':key,'digest':h,'members':g[['cohort','image_id']].to_dict('records')})
 milk=df[df.cohort.eq('MILK10k')].reset_index(drop=True);ref=df[~df.cohort.eq('MILK10k')].reset_index(drop=True)
 near=[]
 for target,within in [(ref,False),(milk,True)]:
  hashes=np.array([int(v,16) for v in target.phash],dtype=np.uint64)
  for i,row in milk.iterrows():
   dist=np.bitwise_count(np.bitwise_xor(hashes,np.uint64(int(row.phash,16))))
   idx=np.flatnonzero(dist<=4)
   if within:idx=idx[idx>i]
   for j in idx:
    other=target.iloc[j]
    if row.sha256==other.sha256 or row.pixel_sha256==other.pixel_sha256:continue
    near.append({'milk_image_id':row.image_id,'other_cohort':other.cohort,'other_image_id':other.image_id,'phash_distance':int(dist[j]),'dhash_distance':(int(row.dhash,16)^int(other.dhash,16)).bit_count(),'review':'PENDING_NOT_AUTOMATIC_EXCLUSION'})
 columns=['milk_image_id','other_cohort','other_image_id','phash_distance','dhash_distance','review']
 pd.DataFrame(near,columns=columns).to_csv(ROOT/'reports/near_duplicate_candidates.csv',index=False)
 (ROOT/'reports/exact_duplicate_groups.json').write_text(json.dumps(exact,indent=2),encoding='utf-8')
 report={'status':'IMAGE_SCREENING_REQUIRES_REVIEW' if near or exact else 'NO_IMAGE_DUPLICATES_DETECTED_BY_SPECIFIED_SCREEN','zip_crc_pass':True,'clinical_entries_indexed':len(all_cases),'image_storage':'verified original ZIP; no full extraction required','candidate_images_decoded':len(milk),'reference_images':ref.cohort.value_counts().to_dict(),'exact_duplicate_groups':len(exact),'near_duplicate_pairs':len(near),'dimensions_min':{'width':int(milk.width.min()),'height':int(milk.height.min())},'dimensions_max':{'width':int(milk.width.max()),'height':int(milk.height.max())},'image_fingerprints_sha256':sha(ROOT/'reports/image_fingerprints.csv'),'does_not_establish_patient_independence':True,'no_model_inference':True}
 (ROOT/'reports/image_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
