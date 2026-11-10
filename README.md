# Age-and-Sex Fusion Across Cohorts

**Release status:** Complete reproduction release. All 135 numerical checkpoints and the exact MILK input cache are available in the linked release; every server SHA256 matches the local artifact manifest. See ASSET_VERIFICATION.json.

**Archived snapshot:** The same frozen code, processed study inputs, 135 core numerical checkpoints and exact MILK cache are preserved at [10.5281/zenodo.24110942](https://doi.org/10.5281/zenodo.24110942). The archive retains the component-specific source terms and references the original frozen source commit.

Code, study inputs and individual model outputs for **Age-and-Sex Fusion Across Cohorts: Separating Recording Status from Recorded Values in Skin Lesion Classification**.

The study separates sex-recording status from recorded female/male categories within DSMAF-Net and ordinary late fusion. Nine conditions use five patient-grouped PAD folds and three paired seeds; the same 135 frozen models are evaluated on DermatoNet and MILK10k.

## Recompute the reported statistics

Use Python 3.13 and the pinned numerical dependencies:

```bash
python -m pip install -r requirements_verification.txt
python -B results_replay/statistical_replay/replay_all.py --output verification_results
python -B results_replay/image_review_sensitivity/replay.py
```

The first command checks the original primary comparison, the matched-method extension, the late-fusion information follow-up and both complete/histopathology-only MILK analyses, including all 2,000 paired resamples. The second checks the conservative author-review image-exclusion sensitivity. Earlier audits and sensitivity analyses can be rerun with `python -B study_inputs/reproduce_all.py`.

## Obtain the inputs and rerun models

* **Exact partitions, age/sex inputs, labels, configurations and individual outputs:** directly available under [study_inputs](study_inputs/).
* **All 135 full MILK probability/logit files:** [study_inputs/milk/predictions](study_inputs/milk/predictions/).
* **Frozen numerical weights and exact MILK RGB input cache:** the [reproduction release](https://github.com/yuki9984/DSMAF-Net-cross-cohort/releases/tag/reproduction-2026-11-10), with checksums and individual files rather than an opaque project archive.
* **Original images:** obtain PAD-UFES-20, DermatoNet and MILK10k from the providers listed in [data licensing and attribution](DATA_LICENSES.md). Exact research-release case IDs are included, so the study cohort does not need to be inferred from descriptions.

[REPRODUCIBILITY.md](REPRODUCIBILITY.md) gives commands for input validation, training, per-checkpoint inference and result verification. [DEPENDENCY_ACCESS.md](DEPENDENCY_ACCESS.md) identifies the upstream dependencies and their terms. The EMCAD owner prohibits redistribution; readers obtain that pinned dependency directly after reviewing its research license.

## Source layout

* `historical_study/`: original training/inference routines and release-compatible model assembly.
* `extension60/`: matched MetaBlock, adapted RG-DermNet and the two DSMAF information conditions.
* `extension30/`: the two ordinary late-fusion information conditions.
* `external_milk/`: original external-cohort preparation and analysis sources.
* `study_inputs/`: exact tables, partitions, model configurations, historical audits and saved predictions.
* `results_replay/`: numerical inputs and scripts for direct statistical reproduction.

All original measurements and fitted-model predictions are retained. Public checkpoint containers omit machine-specific paths and unnecessary worker state; their tensor values are checked against the original checkpoints. Publishing materials does not constitute a new independent training replication. Research records retain their original source terms; see DATA_LICENSES.md. No personal correspondence, credentials, environment directories or unlicensed historical modules are included.
