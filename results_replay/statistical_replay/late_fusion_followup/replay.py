"""Recompute the late-fusion follow-up from coded saved probabilities."""
from pathlib import Path
import hashlib,json,csv
import numpy as np
from analysis_core import calculate_statistics
R=Path(__file__).resolve().parent
def rows(name):
 with (R/name).open(encoding="utf8",newline="") as f:return list(csv.DictReader(f))
def verify(actual,reference,keys,numeric):
 def idx(v):return {tuple(str(r[k]) for k in keys):r for r in v}
 a,b=idx(actual),idx(reference)
 assert len(a)==len(actual)==len(b)==len(reference) and set(a)==set(b)
 error=0.
 for key,row in a.items():
  for k,v in row.items():
   if k in numeric:error=max(error,abs(float(v)-float(b[key][k])))
   else:assert str(v)==str(b[key][k]),(key,k)
 assert error<1e-10
 return error
def main():
 for n,h in json.loads((R/"file_hashes.json").read_text(encoding="utf-8")).items():
  assert Path(n).name==n and hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
 p=json.loads((R/"plan.json").read_text(encoding="utf-8"))
 with np.load(R/"matched.npz",allow_pickle=False) as z:a={k:z[k] for k in z.files}
 result=calculate_statistics(p["methods"],p["seeds"],a["yi"],a["gi"],a["folds"],a["pi"],a["ye"],a["ge"],a["pe"],p["pairs"],p["bootstrap_repeats"],p["bootstrap_seed"])
 keys=["type","comparison","metric","target","unit"]
 err1=verify(result["summary"],rows("summary_intervals.csv"),keys,{"estimate","ci95_low","ci95_high","seed_sd"})
 err2=verify(result["by_seed"],rows("metrics_and_contrasts_by_seed.csv"),keys+["seed"],{"value"})
 errors=[]
 with np.load(R/"bootstrap_reference.npz",allow_pickle=False) as z:
  assert z["names"].tolist()==list(result["arrays"]["names"])
  for k in ["point_internal","point_external","draws_internal","draws_external","class_fold_weights","loss_fold_weights"]:
   value=result["arrays"][k] if k in result["arrays"] else result[k]
   np.testing.assert_allclose(value,z[k],atol=1e-10,rtol=0)
   errors.append(float(np.max(np.abs(value-z[k]))))
 print(json.dumps(dict(status="PASS",models=75,new_models=30,summary_rows=len(result["summary"]),seed_rows=len(result["by_seed"]),paired_patient_draws=2000,max_table_error=max(err1,err2),max_array_error=max(errors),scope="Saved-prediction statistics; no training or inference"),indent=2))
if __name__=="__main__":main()
