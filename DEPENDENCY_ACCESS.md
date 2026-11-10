# Neural dependency access
This candidate separates distributable compatibility code from externally obtained
research dependencies. No legacy Agent Attention helper, BFAM/B2CNet, HLNet network,
inverse wavelet module, or embedded AFPN/MSCB fallback is included.

## Included compatibility code
- AFPN: Apache-2.0 upstream, commit 0930b563b43830c68825241689b80d8e430f2576.
  Remove the MMDetection registry and unused fifth pooling output. Four outputs,
  parameter names and initialization are retained.
- Forward wavelet: Wave-ViT (YehLi/ImageNetModel), Apache-2.0,
  commit 0b4302e0b10ff12013b03b799dda50393efa69a9.
  Use float32 filter buffers, match filter device/dtype to the input and make the
  backward input contiguous. Only DWT_Function and DWT_2D are retained.
- Dual attention: MIT-licensed DANet attention, commit
  b0a0fa008148f102321cdaa7ca38fb0ddd0239d1, with an explicitly adapted head.
  The release head preserves the historical full-width depthwise Q/K/V,
  reduction by 16, dropout 0.05, output activations and named parameter layout.
  This is a compatibility implementation, not a new method or a claim that
  the original anonymous local file has acquired a license.
Full license texts are in LICENSES/. The private historical copies remain
unchanged and are not part of this distribution.

## Obtain EMCAD directly
Review the [UT Austin Research License](https://github.com/SLDGroup/EMCAD/blob/26c9c31f731f749b62c5fe83f44376dac75f3aa8/LICENSE).
Obtain [lib/decoders.py](https://github.com/SLDGroup/EMCAD/blob/26c9c31f731f749b62c5fe83f44376dac75f3aa8/lib/decoders.py)
at that commit from its owner. Set DSMAF_EMCAD_DECODER to your local file path,
or place it at external_dependencies/emcad/decoders.py. This repository does
not redistribute the decoder or a copied fallback. The loader checks the
newline-normalized source hash and fails if it differs. The upstream license
continues to govern its use; linking it does not grant additional rights.

## Obtain RG-DermNet directly
Review [license.md](https://github.com/life-ufes/rg-dermnet/blob/940c1cd3598b5a887aaec118371164dc256d1cb3/license.md)
(CC BY-NC 4.0). Obtain gatedResidualBlock.py, metablock.py and
multimodalIntraInterModal.py from src/scripts/benchmark/models/ at that commit.
Place them in external_dependencies/rg_dermnet/ or set DSMAF_RG_DIR to the
containing directory. No case data, predictions or checkpoints are downloaded
by these instructions.

Before importing the models, check the four external files without executing them:

    python -B check_external_dependencies.py

The command returns a nonzero exit status for missing or altered files. It does
not download dependencies or accept upstream terms on behalf of the user. The
model loaders independently enforce the same hashes.

## Build or load a frozen model
Install requirements_neural.txt in a dedicated environment. The qualification
environment uses CPU float32; this is not a claim of CUDA AMP parity.

    from model_api import build_model, load_frozen_checkpoint
    model = build_model("full")
    model, config = load_frozen_checkpoint("YOUR_LOCAL_CHECKPOINT.pth", "full")

The API defaults to CPU and rejects a condition that differs from the saved
checkpoint configuration. To use another device, pass device="cuda" explicitly.
For the age-only and recording-status conditions, apply the corresponding
restriction in extension60/information_conditions.py before the forward call;
the architecture alone does not remove information from a supplied input.

Use the saved training age statistics and information condition from the
checkpoint configuration. Do not re-fit normalization on an external cohort.
Qualification results and their precise scope are in COMPATIBILITY_REPORT.json.

Historical training plans describe the executed experiments, not a promise
that newly trained random initializations will reproduce every old bit.
The study partitions, configurations and case inputs are now public under study_inputs/. See REPRODUCIBILITY.md.
