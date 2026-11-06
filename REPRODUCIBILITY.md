# Reproduction and data access

## Data sources
* PAD-UFES-20 clinical photographs and metadata: https://doi.org/10.17632/zr7vgbcyr2.1
* DermatoNet-TelessaudeRS v2 clinical photographs and metadata: https://doi.org/10.17632/g5bj76p2z3.2
* MILK10k clinical and dermoscopic image release: https://api.isic-archive.com/doi/milk10k/

Obtain clinical data directly from the providers under the terms of their respective releases. The public examples contain only synthetic rows. Exact neural reruns require the original images, fixed patient-grouped partitions, ordered external metadata, inner-training age parameters and the documented model dependencies. Exact clinical statistical recomputation requires authorized saved outputs.

Two confidential review packages have distinct scopes. The coded core covers the original primary and matched seven-condition extension, the ordinary late-fusion follow-up, and the nine-condition MILK analysis. It contains coded grouping information, reference labels and saved outputs; the MILK module uses per-model correctness and loss responses rather than full probability vectors. This core excludes original research identifiers, demographic fields, photographs and weights. A separate historical extended archive retains the source-release research identifiers, demographic fields, fixed partitions and saved outputs needed to check the earlier record audits, input interventions, recipe sensitivity and restricted-age analysis. It excludes photographs and weights. Neither archive is part of this public repository.

The three core replay modules passed CPU checks and encrypted-package round-trip verification. These are saved-output statistical checks, not new training or inference. A confidential access channel is being arranged with the journal; approval and reviewer delivery must not be assumed. The corresponding author can coordinate access through the handling editor. Readers can run the two public synthetic examples immediately.

## Training protocol
The historical worker used Python 3.11.9, PyTorch 2.6.0 with CUDA 12.4, torchvision 0.21.0 and an RTX 4060. The matched extension used five patient-grouped folds, three paired seeds, batch size 16 and ResNet50 ImageNet V2 initialization. The training protocol and checkpoint selection rule are in extension60/analysis_plan.json. Paths in that file identify the source protocol at the time of planning.

With an authorized local input directory, one can first check a fold:

    python -B historical_study/train_reproduce.py --experiment cv --fold 0 --study-inputs /path/to/study-inputs --validate-inputs

For the grouped analysis, the directory must provide internal75/splits/fold_00_train.csv, fold_00_valid.csv and fold_00_test.csv and the corresponding files for the other folds. The primary analysis uses neural_setup/original_split/. Input columns are img_id, patient_id, age, gender_FEMALE, gender_MALE and diagnostic_number.

For the full neural execution, first supply the external components described in THIRD_PARTY.md and the authorized study inputs. Use the preserved training entry point with --pad-images and the initialization specified by the plan. Per-checkpoint prediction is implemented in historical_study/predict_reproduce.py. For the 60-fit extension, train_extension.py and evaluate_extension.py show the studied training and evaluation procedures; the worker's private source/input manifests and machine configuration are separate from this release. The local checkpoints, splits, worker return archives, individual predictions and histories are excluded from the public tree.


## Follow-up and additional external evaluation
The ordinary late-fusion B/C follow-up used the same five folds and three paired seeds as the corresponding full-information control. Its pre-training plan and unchanged numerical sources are in extension30/. All 30 selected checkpoints were frozen before their new PAD and DermatoNet evaluations. The follow-up was motivated by earlier DSMAF-Net findings; it does not turn reused DermatoNet outcomes into an independent confirmatory test.

external_milk/ contains the unchanged metadata audit, image-screening, fixed-model inference and statistics source. Its protocol retained 4,474 clinical images with distinct lesion identifiers, and a 4,315-lesion histopathology sensitivity subset. Patient linkage is unavailable, and the bootstrap unit is the lesion. All sex values are recorded. These constraints matter when interpreting the transfer of recording-status-trained models.

MILK inference used CPU float32 and batch eight, with fixed model-specific PAD age parameters and the original deterministic image transform. Original PAD class-support weights aggregate external class recalls; image-support weights aggregate probability losses. Model probabilities are not ensembled. One hundred and five frozen-model predictions were staged while the final 30 fits were still running, without calculating or reviewing their performance. The full analysis required all 135 verified predictions. Independent recomputation checked both complete 2,000-draw bootstrap arrays and confirmed that the 105 staged predictions were unchanged.

The preserved entry points require the original authorized directory structure and hash-locked inputs. The public source snapshot and synthetic examples are not a turnkey reconstruction of the historical neural environment. Outstanding third-party provenance and permission issues remain documented in THIRD_PARTY.md; a passed statistics check does not resolve those issues.
