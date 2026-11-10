from pathlib import Path
import json
import pandas as pd,numpy as np
from class_and_recording_analysis import run
root=Path(__file__).resolve().parent;out=root/'verification_output/class_and_recording'
r=run(root,out)
for name in ['full_PAD_recording_by_class.csv','all_15_fold_role_encoding_checks.csv','all_180_class_seed_cohort_recalls.csv','class_contrasts_patient_cluster_intervals.csv','all_three_BACC_contrasts.csv']:
 a=pd.read_csv(out/name);b=pd.read_csv(root/'class_and_recording/expected'/name)
 pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=1e-10,atol=1e-10)
assert r['status']=='PASS' and r['external_full_above_image_seed_count']==1
assert r['unrecorded_sex_images']==804 and r['fold_role_partitions_audited']==15
assert r['earlier_six_BACC_intervals_max_error']<1e-10
print('CLASS_CONTRASTS_DIRECT_FUSION_AND_FULL_PAD_RECORDING_PASS',flush=True)
