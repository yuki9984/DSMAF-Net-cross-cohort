from pathlib import Path
import json,re
import numpy as np
from same_models_statistics import run
root=Path(__file__).resolve().parent
matched=root/'matched_cohort'
z=np.load(matched/'probabilities.npz',allow_pickle=False);old=np.load(root/'internal75/pooled_predictions.npz',allow_pickle=False)
for newkey,oldkey in [('internal_labels','labels'),('internal_patients','patient_ids'),('internal_images','img_ids'),('internal_folds','outer_folds')]:assert np.array_equal(z[newkey],old[oldkey])
for i,arm in enumerate(['full','image_only','late_fusion']):assert np.array_equal(z['prob_internal'][i],old['probs_'+arm])
inventory=json.loads((root/'internal75/checkpoint_inventory.json').read_text(encoding="utf-8"))['files'];weightmap={}
for r in inventory:
 m=re.search(r'fold_(\d+)/(.*)_seed(\d+)/best_valid.pth$',r['path']);assert m
 weightmap[(int(m[1]),m[2],int(m[3]))]=r['sha256']
binding=json.loads((matched/'prediction_bindings.json').read_text(encoding="utf-8"))
for stage in ['reproduction_complete','external_inference_complete']:
 records=binding[stage]['records'];assert len(records)==75
 assert set((r['fold'],r['arm'],r['seed']) for r in records)==set(weightmap)
 for r in records:assert r['checkpoint_sha256']==weightmap[(r['fold'],r['arm'],r['seed'])]
assert all(r['changed_hard_predictions']==0 and r['max_abs_probability_error']<=1e-4 for r in binding['reproduction_complete']['records'])
report=run(matched/'probabilities.npz',matched/'expected',root/'verification_output/matched_cohort')
assert report['status']=='PASS'
print('SAME_FIXED_MODELS_AND_ALL_24_CONTRAST_INTERVALS_PASS',flush=True)
