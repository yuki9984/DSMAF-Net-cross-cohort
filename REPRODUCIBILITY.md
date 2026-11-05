# Reproduction and data access

## Data sources
* PAD-UFES-20 clinical photographs and metadata: https://doi.org/10.17632/zr7vgbcyr2.1
* DermatoNet-TelessaudeRS v2 clinical photographs and metadata: https://doi.org/10.17632/g5bj76p2z3.2
* MILK10k clinical and dermoscopic image release: https://api.isic-archive.com/doi/milk10k/

Obtain clinical data directly from the providers under the terms of the respective releases. The public example uses only generated rows. Exact neural reruns require the original images, fixed patient-grouped partitions, ordered external metadata, inner-training age statistics and the documented model dependencies. Exact recomputation of the paper's clinical intervals also requires authorized saved predictions. The arrangements for reviewer access to those research-specific inputs are pending.

## Training protocol
The historical worker used Python 3.11.9, PyTorch 2.6.0 with CUDA 12.4, torchvision 0.21.0 and an RTX 4060. The matched extension used five patient-grouped folds, three paired seeds, batch size 16 and ResNet50 ImageNet V2 initialization. The training protocol and checkpoint selection rule are in extension60/analysis_plan.json. Paths in that file identify the source protocol at the time of planning.

With an authorized local input directory, one can first check a fold:

    python -B historical_study/train_reproduce.py --experiment cv --fold 0 --study-inputs /path/to/study-inputs --validate-inputs

For the grouped analysis, the directory must provide internal75/splits/fold_00_train.csv, fold_00_valid.csv and fold_00_test.csv and the corresponding files for the other folds. The primary analysis uses neural_setup/original_split/. Input columns are img_id, patient_id, age, gender_FEMALE, gender_MALE and diagnostic_number.

For the full neural execution, first supply the external components described in THIRD_PARTY.md and the authorized study inputs. Use the preserved training entry point with --pad-images and the initialization specified by the plan. Per-checkpoint prediction is implemented in historical_study/predict_reproduce.py. For the 60-fit extension, train_extension.py and evaluate_extension.py show the studied training and evaluation procedures; the worker's private source/input manifests and machine configuration are separate from this release. The local checkpoints, splits, worker return archives, individual predictions and histories are excluded from the public tree.
