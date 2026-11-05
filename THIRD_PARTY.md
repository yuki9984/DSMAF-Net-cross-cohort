# Component provenance
The study's MetaBlock adaptation follows Pacheco et al., https://doi.org/10.1109/JBHI.2021.3062002. Its MIT notice is in extension60/references/METABLOCK_LICENSE.

The RG-DermNet adapter uses the author's [original implementation](https://github.com/life-ufes/rg-dermnet/tree/940c1cd3598b5a887aaec118371164dc256d1cb3). The study uses the one-hot metadata branch on the common ResNet50 backbone. For a locally authorized run, obtain gatedResidualBlock.py, metablock.py and multimodalIntraInterModal.py from src/scripts/benchmark/models/ at that commit and place them in extension60/references/. Refer to the upstream license.md for the CC BY-NC 4.0 terms.

The historical DSMAF assembly imports Haar decomposition, attention, multi-scale convolution and feature-pyramid components through module_wrappers.py. That wrapper embeds third-party class implementations too. The legacy component copies and the wrapper are withheld pending file-specific permission. A list of required study-snapshot filenames and checksums is in NEURAL_DEPENDENCIES.json. An authorized local copy must match the recorded snapshot for exact reruns. The archived workers used the historical originals; different versions require verification. The synthetic example does not import these modules.

Torch, torchvision, timm, einops and PyWavelets are external dependencies under their respective licenses.
