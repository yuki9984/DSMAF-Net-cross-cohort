"""Seven locked model arms built from the qualified release-compatible architecture."""
from __future__ import annotations

import sys
import types

import torch
from torch import nn

from common import ARMS, model_code
from data_v2 import META_COLUMNS

sys.path.insert(0, str(model_code()))
from models.hfpa_net import HFPA_Net, HLFDBWrapper  # noqa: E402
from models import module_wrappers as wrappers  # noqa: E402


class CapacityResidualMLP(nn.Module):
    """Fixed-size parameter control on the 1280-dimensional late-fusion vector."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1280, 3832), nn.GELU(), nn.Linear(3832, 1280)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _reject_dwt_fallback(self: HLFDBWrapper, _tensor: torch.Tensor) -> None:
    raise RuntimeError("DWT fallback would change the locked architecture")


def strict_architecture_checks(model: HFPA_Net, arm: str) -> None:
    needed = ["licensed_afpn.py", "licensed_attention.py"]
    if model.use_dsmaf:
        needed.append("decoders-mscam.py")
    if model.use_dsdm:
        needed.append("licensed_wavelet.py")
    failures = {name: wrappers._LOAD_ERRORS[name] for name in needed
                if wrappers._LOAD_ERRORS.get(name)}
    if failures or wrappers.DA_Block is None:
        raise RuntimeError(f"Required architecture sources unavailable: {failures}")
    if model.afpn is None or not getattr(model.afpn, "_use_impl", False):
        raise RuntimeError("AFPN fallback is prohibited")
    if model.use_dsdm:
        if wrappers.DWT_2D is None:
            raise RuntimeError("Wavelet DWT unavailable")
        for module in model.modules():
            if isinstance(module, HLFDBWrapper):
                if not module.use_dwt:
                    raise RuntimeError("DWT disabled")
                module._fallback = types.MethodType(_reject_dwt_fallback, module)
    if arm in ("shared_q", "no_dsdm_shared_q"):
        for stage in range(1, 5):
            block = getattr(model, f"tfpfm{stage}")
            if block.q_high is not block.q_low:
                raise RuntimeError("Outer query projection is not shared")
            if block.high_attn.q_proj is not block.low_attn.q_proj:
                raise RuntimeError("Inner query projection is not shared")


def make_model(arm: str, *, pretrained: bool = True) -> HFPA_Net:
    if arm not in ARMS:
        raise ValueError(f"Unknown model arm: {arm}")
    dsmaf = arm in ("full", "no_dsdm", "shared_q", "no_dsdm_shared_q")
    metadata = arm != "image_only"
    dsdm = arm in ("full", "shared_q")
    model = HFPA_Net(
        num_classes=6, meta_in_dim=len(META_COLUMNS), fusion_dim=256,
        num_heads=4, dropout=0.1, pretrained_backbone=pretrained,
        backbone_name="resnet50", use_all_afpn_outputs=True,
        use_sade=False, use_metadata=metadata, use_mgca=dsmaf,
        use_dsdm=dsdm, use_dsmaf=dsmaf, use_afpn=True,
        use_cluster_aux_head=False, verbose=False,
    )
    if arm in ("shared_q", "no_dsdm_shared_q"):
        for stage in range(1, 5):
            block = getattr(model, f"tfpfm{stage}")
            block.q_low = block.q_high
            block.low_attn.q_proj = block.high_attn.q_proj
    if arm == "late_fusion_cap":
        classifier = model.classifier
        if classifier.in_features != 1280:
            raise RuntimeError("Capacity comparator expects 1280 fusion features")
        model.classifier = nn.Sequential(CapacityResidualMLP(), classifier)
    strict_architecture_checks(model, arm)
    return model


def count_trainable(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def check_capacity_match(full: nn.Module, control: nn.Module) -> dict[str, float | int]:
    full_count = count_trainable(full)
    control_count = count_trainable(control)
    relative = abs(control_count - full_count) / full_count
    if relative > 0.01:
        raise RuntimeError(f"Capacity comparator differs by {relative:.2%}")
    return {"full": full_count, "late_fusion_cap": control_count,
            "absolute_difference": abs(control_count - full_count),
            "relative_difference": relative}
