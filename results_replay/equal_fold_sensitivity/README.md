# Equal-fold sensitivity

Post hoc analysis specified after the original outcomes. All nine conditions, five fixed folds and three paired seeds are retained. Separately scored fold-model outcomes are averaged with equal weights; probabilities are never ensembled. The original fixed-PAD-support-weighted point estimates are checked before this sensitivity is computed. DermatoNet uses patient resampling; MILK10k uses class-stratified lesion resampling. Every method, seed, metric and weighting shares 2,000 within-cohort draws (seed 2026111101). Intervals condition on fitted models and are unadjusted.

Run `python -B results_replay/equal_fold_sensitivity/replay.py --output verification_results_equal` from the repository root. The expected summaries and all resample arrays are verified. No training or neural inference is repeated.
