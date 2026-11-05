# Age-and-Sex Fusion Across Cohorts
Study code for *Age-and-Sex Fusion Across Cohorts: Separating Recording Status from Recorded Values in Skin Lesion Classification*.

The training conditions distinguish two inputs that are often combined: whether a sex value was recorded and which category was recorded. The example below runs the published code paths for input encoding, recording audits, weighted fold comparisons and paired patient-cluster intervals on synthetic inputs.

## Run the example
Use Python 3.13 in a clean environment, from the repository root:

    python -m pip install -r requirements_verification.txt
    python -B run_synthetic_demo.py --output demo_output

The example checks the encodings, audits 36 generated records and computes 2,000 paired resamples. A successful run writes the check results to demo_output/SYNTHETIC_DEMO_REPORT.json. Choose a new output folder for each execution.

## Study implementation
* historical_study/ contains the model assembly and the original training, prediction and audit entry points.
* extension60/ contains the four matched training conditions, the original analysis plan, the reference adapters and the paired statistical analysis.
* examples/synthetic/ contains the generated records for the demonstration.

For dataset links, required local inputs and third-party dependency provenance, see [Reproduction and data access](REPRODUCIBILITY.md) and [Third-party notices](THIRD_PARTY.md). The runnable example demonstrates software behavior. Reproducing the full neural experiments also requires the identified local inputs and historical dependencies. This release covers the historical experiments and the 60-fit extension. The pending ordinary-late-fusion extension and MILK evaluation will be documented separately after joint verification.
