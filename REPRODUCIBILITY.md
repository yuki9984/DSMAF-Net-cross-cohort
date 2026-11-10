# Reproduction guide

## Public access and attribution

The exact processed study inputs, fixed partitions, labels and saved individual outputs are now directly available in this repository. No author approval or private delivery is needed to obtain them. The weights and MILK image cache are individual download assets in the [reproduction release](https://github.com/yuki9984/DSMAF-Net-cross-cohort/releases/tag/reproduction-2026-11-10). Original image releases: [PAD-UFES-20 v1](https://doi.org/10.17632/zr7vgbcyr2.1), [DermatoNet v2](https://doi.org/10.17632/g5bj76p2z3.2), [MILK10k](https://api.isic-archive.com/doi/milk10k/). Read DATA_LICENSES.md and retain source attribution and source-specific non-commercial restrictions.

## 1. Statistical reproduction without a GPU

Use Python 3.13, install `requirements_verification.txt`, then run:

```bash
python -B results_replay/statistical_replay/replay_all.py --output verification_results
python -B results_replay/image_review_sensitivity/replay.py
python -B study_inputs/reproduce_all.py
```

Use a new output folder. The first two scripts verify the reported numerical results and every stored paired bootstrap draw. The last command covers the earlier record audits, patient-grouped comparison, interventions, training-recipe sensitivity and restricted-age analysis. Data arrays use NumPy's non-pickle numerical format. Code and data are exposed as individual repository files.

## 2. Neural environment and dependencies

The historical training worker used Python 3.11.9, PyTorch 2.6.0/CUDA 12.4, torchvision 0.21.0 and RTX 4060. For CPU checkpoint inference, use `requirements_neural.txt`. For the original GPU training recipe, create a Python 3.11 environment and install `torch==2.6.0` and `torchvision==0.21.0` from the CUDA 12.4 PyTorch wheel index, followed by `requirements_training.txt`. The CPU qualification environment has different PyTorch versions and must not be represented as the original training environment. Then review the upstream licenses linked in DEPENDENCY_ACCESS.md. After explicitly accepting those licenses, obtain the exact upstream files:

```bash
python -B fetch_dependencies.py --accept-emcad-research-license --accept-rg-noncommercial-license
python -B check_external_dependencies.py
```

The source hashes are enforced. The EMCAD source is fetched directly from its owner and is never republished here. Source code and numerical weights are separate artifacts. Source licenses continue to govern use of the executable model.

## 3. Validate the actual training inputs

```bash
python -B reproduce_training.py --arm full --fold 0 --seed 20261013 --check-inputs
python -B reproduce_training.py --arm dsmaf_age_only --check-inputs
python -B reproduce_training.py --arm late_age_recorded --check-inputs
```

The tables include source research IDs, labels and the exact PAD age/sex fields. Validation checks patient separation and all training information encodings. No demographic normalization is fitted to an external cohort.

## 4. Train a selected model

Download the original PAD images and torchvision ResNet50 ImageNet V2 initialization (`resnet50-11ad3fa6.pth`; full SHA256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`). Supply actual local paths:

```bash
python -B reproduce_training.py --arm dsmaf_age_only --fold 0 --seed 20261013 --pad-images /data/PAD/images --initialization-file /data/imagenet/resnet50-11ad3fa6.pth --output-root new_runs
```

Repeat the same command for the nine conditions listed in `study_inputs/model_inventory.json`, all five folds and seeds 20261013, 20261014 and 20261015. The preserved loops retain batch 16, 150-epoch maximum, patience 15, original augmentation/sampling, optimizer, schedule and inner-validation checkpoint selection. The public wrapper supplies paths and input manifests without requiring the historical worker's private directory tree. CUDA is required for the original AMP training recipe; CPU compatibility does not certify bit-identical CUDA retraining.

## 5. Infer using a released frozen model

For an original control checkpoint on the locked DermatoNet cohort:

```bash
python -B historical_study/predict_reproduce.py --study-inputs study_inputs --cohort external --checkpoint /data/weights/fold_00_full_seed20261013.pth --images /data/DermatoNet/images --device cpu --output new_predictions/dermatonet.npz
```

For any of the nine conditions on MILK, download its weight file and `candidate_rgb224.npy` from the reproduction release:

```bash
python -B infer_milk.py --checkpoint /data/weights/fold_00_dsmaf_age_only_seed20261013.pth --image-cache /data/milk/candidate_rgb224.npy --output new_predictions/milk.npz
```

The image cache contains 4,478 candidate images; the published indices select the 4,474 evaluation images. Per-image attribution for all cache entries is in `study_inputs/milk/image_cache_attribution.csv`. The script uses the exact stored order, training-derived metadata encodings, deterministic normalization and batch eight CPU float32 inference. It reports the maximum logit difference from the saved output. `--limit 8` permits a small engineering check. Fold outcomes are weighted using PAD class/image support; probabilities are not averaged into an ensemble.

## Limits of the reproduced analysis

Resampling conditions on the fitted models. PAD and DermatoNet use patient clusters; MILK uses lesion clusters because patient linkage is unavailable. MILK has complete sex recording. These materials expose the actual study data and execution route; they do not make the reused external cohort a new confirmatory dataset or establish clinical validity.

## Earlier experiments
The original seven-condition, five-seed study uses `historical_study/train_reproduce.py --experiment primary --arm full --seed 20261013 --study-inputs study_inputs --pad-images /data/PAD/images --download-initialization`. The same entry accepts its six listed comparator arms. The historical grouped comparison uses `--experiment cv` with seeds 20261013 through 20261017. The alternative-recipe comparison uses `--experiment alternative` and only the three seeds 20261013 through 20261015. Their original partitions/plans and individual outputs are retained in study_inputs/.

## Published input projections
The historical training CSVs carried unrelated clinical attributes, although the model used only image ID, patient grouping ID, label, age and two sex indicators. Public tables retain these actual inputs (plus image hashes for grouped tables). PUBLIC_INPUT_PROJECTION.json records the original and published file hashes. All 15 grouped-role tables were checked for identical input values and row order before publication. The original pre-training fold manifest is preserved. The loader verifies the explicit projection mapping instead of silently accepting a changed file. No trained feature value, partition or prediction was changed.
