# Age-and-Sex Fusion Across Cohorts
Study code for *Age-and-Sex Fusion Across Cohorts: Separating Recording Status from Recorded Values in Skin Lesion Classification*.

The training conditions distinguish two inputs that are often combined: whether a sex value was recorded and which category was recorded. The example below runs the study routines for input encoding, recording audits, weighted fold comparisons and paired patient-cluster intervals on synthetic inputs.

## Run the example
Use Python 3.13 in a clean environment, from the repository root:

    python -m pip install -r requirements_verification.txt
    python -B run_synthetic_demo.py --output demo_output

The example checks the encodings, audits 36 synthetic records and computes 2,000 paired resamples. A successful run writes the check results to demo_output/SYNTHETIC_DEMO_REPORT.json. Choose a new output folder for each execution.

The separate MILK statistics example also runs without a GPU or real cases:

    python -B run_synthetic_milk.py --output demo_output_milk

It distinguishes scoring frozen fold models from averaging their probabilities, verifies NLL and multiclass Brier against direct formulas, and checks all 2,000 paired lesion resamples against an independent direct-index calculation. Its report is synthetic and provides no evidence of clinical performance.

## Study implementation
* historical_study/ contains the model assembly and the original training, prediction and audit entry points.
* extension60/ contains the four matched training conditions, the original analysis plan, the reference adapters and the paired statistical analysis.
* extension30/ preserves the ordinary late-fusion information follow-up source and pre-training plan.
* external_milk/ preserves metadata-audit, fixed-model inference and lesion-bootstrap source for MILK10k.
* examples/synthetic/ contains the synthetic records used in the demonstration.

For dataset links, required local inputs and third-party dependency provenance, see [Reproduction and data access](REPRODUCIBILITY.md) and [Third-party notices](THIRD_PARTY.md). The runnable example demonstrates software behavior. Reproducing the full neural experiments also requires the identified local inputs and historical dependencies. The 30-fit ordinary late-fusion follow-up and all 135 fixed-model MILK evaluations passed independent statistical verification. The original 105 staged MILK prediction arrays were unchanged. Both the eligible and histopathology-only MILK analyses were independently recomputed with all 2,000 paired lesion draws. These checks verify saved-output calculations and provenance; they do not repeat neural training or establish clinical validity. See each directory README for its scope and required private inputs.
