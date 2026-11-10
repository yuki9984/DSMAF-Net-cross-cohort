"""Portable frozen-model construction; original preprocessing remains mandatory."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
def build_model(condition,device="cpu"):
    import torch
    sys.path.insert(0,str(ROOT/"extension60"))
    import extension_settings as settings
    settings.setup_original()
    from model_v2 import make_model
    from extension_models import make_model as extension
    if condition in ("metablock_cv","rg_dermnet_adapted_cv","dsmaf_age_only","dsmaf_age_recorded"):
        model=extension(condition,pretrained=False)
    elif condition in ("late_age_only","late_age_recorded"):
        model=make_model("late_fusion",pretrained=False)
    else:
        model=make_model(condition,pretrained=False)
    target=torch.device(device)
    if hasattr(model,"device"):model.device=target
    return model.to(target)
def load_frozen_checkpoint(path,condition,device="cpu"):
    import torch
    # Public release checkpoints contain tensors and primitive configuration values.
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    config=checkpoint.get("config",{})
    if config.get("arm")!=condition:
        raise ValueError("Requested condition differs from the checkpoint configuration")
    model=build_model(condition,device=device)
    model.load_state_dict(checkpoint["model_state_dict"],strict=True)
    return model.eval(),config
