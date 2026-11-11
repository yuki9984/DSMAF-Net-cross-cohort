# Reproduction version 2026.11.11

- Corrected `/blob/` and `/tree/` separators when constructing raw upstream download URLs. Four pinned source files were fetched in a cache-free directory and their normalized hashes matched.
- Added explicit UTF-8 encoding to 24 text reads in 12 public entry/verification scripts. Numerical arrays and training configuration values are unchanged.
- Added a post hoc equal-fold aggregation analysis for all nine conditions on DermatoNet and MILK10k, with 2,000 paired conditional resamples. All 12 external design/seed recording-status BACC contrasts retain negative direction.
- Original numerical weights and MILK image-cache bytes remain at the immutable `reproduction-2026-11-10` release; their hashes and source terms are unchanged. Corrected source is frozen under `reproduction-2026-11-11`.
- The corrected Zenodo version is 2026.11.11, DOI 10.5281/zenodo.24124181. The preceding DOI remains the preserved original snapshot.

These checks distinguish source execution, saved-output statistics and representative frozen-model inference from full independent retraining.
