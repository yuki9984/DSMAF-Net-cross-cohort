# Ordinary late-fusion information follow-up

This snapshot records the two new conditions, age only (B) and age plus sex-recording status (C), under the original five folds and three paired seeds. Full-information late fusion (D) is reused. Thirty new fits are followed by 60 internal and DermatoNet evaluations. B/C keep the graph and five input dimensions unchanged.

analysis_plan.json is the unchanged pre-training local plan. Its readiness flags describe the time it was written, not the current completion state. The follow-up was chosen after the earlier DSMAF-Net results were known; it is exploratory.

information_conditions.py defines the encodings; train_extension.py, evaluate_extension.py and run_campaign.py record the worker procedures. analysis_core.py and statistics_reference.py implement the paired statistics from saved probabilities.

The public snapshot is not a standalone neural-training bundle. The original worker runtime, authorized materials, source/input manifests, splits and worker_locations.json must be supplied separately with matching hashes. Obtain the third-party components only as permitted by their owners, as documented in the root THIRD_PARTY.md. Clinical inputs, per-case predictions, trained weights and private worker manifests are deliberately absent here. Do not disable integrity checks or substitute another run to obtain a successful receipt.

The public synthetic example can be run without these neural dependencies. Confidential statistical review materials cover the saved-output calculations, not a repeat of neural training.
