"""Audit public metadata, retaining raw values and explicit six-class selection."""
import pathlib,json,hashlib,datetime
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parents[1]
LABELS=['ACK','BCC','MEL','NEV','SCC','SEK']
MAP={'Solar or actinic keratosis':'ACK','Basal cell carcinoma':'BCC','Melanoma Invasive':'MEL','Melanoma in situ':'MEL','Nevus':'NEV','Squamous cell carcinoma, Invasive':'SCC','Seborrheic keratosis':'SEK'}
EXPECTED={'ACK':'AKIEC','BCC':'BCC','MEL':'MEL','NEV':'NV','SCC':'SCCKA','SEK':'BKL'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 frames={};sources={}
 for name,count in [('milk10k.csv',10480),('training_supp.csv',10480),('training_input.csv',10480),('training_gt.csv',5240)]:
  p=ROOT/'raw'/name;r=json.loads(p.with_suffix('.csv.download.json').read_text())
  assert p.stat().st_size==r['size_bytes'] and sha(p)==r['sha256']
  d=pd.read_csv(p,keep_default_na=False);assert len(d)==count,(name,len(d))
  key='lesion_id' if name=='training_gt.csv' else 'isic_id'
  assert d[key].ne('').all() and d[key].is_unique
  frames[name]=d;sources[name]={'rows':count,'sha256':r['sha256'],'url':r['url'],'etag_md5_verified':r['etag_md5_verified']}
 m=frames['milk10k.csv'];supp=frames['training_supp.csv'];inp=frames['training_input.csv'];gt=frames['training_gt.csv']
 for d in [supp,inp]:assert set(d.isic_id)==set(m.isic_id)
 assert set(m.lesion_id)==set(gt.lesion_id)
 assert set(m.image_type)=={'clinical: close-up','dermoscopic'}
 groups=m.groupby('lesion_id')
 assert groups.size().eq(2).all() and groups.image_type.nunique().eq(2).all()
 for c in ['age_approx','sex','diagnosis_1','diagnosis_2','diagnosis_3','diagnosis_4','diagnosis_confirm_type']:assert groups[c].nunique(dropna=False).eq(1).all(),c
 merged=m.merge(inp,on='isic_id',validate='one_to_one',suffixes=('','_input'))
 for c in ['lesion_id','image_type','age_approx','sex','copyright_license']:assert merged[c].astype(str).equals(merged[c+'_input'].astype(str)),c
 ms=m.merge(supp,on='isic_id',validate='one_to_one',suffixes=('','_supp'))
 assert ms.diagnosis_confirm_type.equals(ms.diagnosis_confirm_type_supp)
 most_specific=ms.diagnosis_4.where(ms.diagnosis_4.ne(''),ms.diagnosis_3).where(lambda s:s.ne(''),ms.diagnosis_2)
 assert most_specific.equals(ms.diagnosis_full),'granular diagnosis sources disagree'
 assert gt.drop(columns='lesion_id').sum(axis=1).eq(1).all()
 clinical=ms.loc[ms.image_type.eq('clinical: close-up')].copy();assert len(clinical)==5240 and clinical.lesion_id.is_unique
 clinical['mapped_class']=clinical.diagnosis_3.map(MAP).fillna('')
 clinical['included_candidate']=clinical.mapped_class.ne('')
 clinical['exclusion_reason']=''
 clinical.loc[~clinical.included_candidate,'exclusion_reason']='Outside strict six-class definition: '+clinical.loc[~clinical.included_candidate,'diagnosis_full']
 clinical['label']=clinical.mapped_class.map({k:i for i,k in enumerate(LABELS)}).fillna(-1).astype(int)
 clinical['img_id']=clinical.isic_id+'.jpg';clinical['age']=pd.to_numeric(clinical.age_approx,errors='coerce');clinical['sex_recorded']=clinical.sex.ne('')
 clinical['cluster_id']=clinical.lesion_id;clinical['cluster_unit']='lesion_not_patient'
 selected=clinical[clinical.included_candidate].copy().sort_values('isic_id')
 assert set(selected.mapped_class)==set(LABELS)
 g=gt.set_index('lesion_id')
 for r in selected.itertuples():assert g.loc[r.lesion_id,EXPECTED[r.mapped_class]]==1
 report_rows=[]
 for label in LABELS:
  d=selected[selected.mapped_class.eq(label)]
  report_rows.append({'class':label,'lesions':len(d),'clinical_images':len(d),'age_missing':int(d.age.isna().sum()),'sex_missing':int(d.sex.eq('').sum()),'female':int(d.sex.eq('female').sum()),'male':int(d.sex.eq('male').sum()),'histopathology':int(d.diagnosis_confirm_type.eq('histopathology').sum())})
 clinical.sort_values('isic_id').to_csv(ROOT/'prepared/all_clinical_selection.csv',index=False)
 selected.to_csv(ROOT/'prepared/candidate_cases.csv',index=False)
 clinical[~clinical.included_candidate].sort_values('isic_id').to_csv(ROOT/'prepared/excluded_diagnoses.csv',index=False)
 pd.DataFrame(report_rows).to_csv(ROOT/'reports/class_support.csv',index=False)
 clinical.groupby(['diagnosis_3','diagnosis_full','mapped_class'],dropna=False).size().rename('clinical_images').reset_index().to_csv(ROOT/'reports/diagnosis_mapping_counts.csv',index=False)
 report={'status':'METADATA_COMPLETE_SIX_CLASS_CANDIDATE_FEASIBLE_NOT_YET_FINAL_COHORT','audit_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':sources,'total_images':len(m),'total_lesions':5240,'clinical_images':len(clinical),'selected_candidate_lesions':len(selected),'excluded_lesions':len(clinical)-len(selected),'selected_age_missing':int(selected.age.isna().sum()),'selected_sex_missing':int(selected.sex.eq('').sum()),'source_sex_missing':int(clinical.sex.eq('').sum()),'patient_id_available':False,'candidate_cases_sha256':sha(ROOT/'prepared/candidate_cases.csv'),'class_support':report_rows,'strict_mapping':MAP,'unresolved':['Image integrity and duplicate screening','Patient overlap cannot be excluded from metadata; no patient linkage supplied','No explicit prior ISIC identifier field found in these four distributed CSVs despite descriptive mention','Frozen checkpoint availability and CPU versus CUDA numerical qualification'],'no_model_predictions_viewed':True}
 (ROOT/'reports/metadata_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in report.items() if k not in ['sources','strict_mapping']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
