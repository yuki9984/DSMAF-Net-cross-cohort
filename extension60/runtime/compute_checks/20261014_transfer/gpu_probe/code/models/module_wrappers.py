"""Minimal dependency bindings for the locked study models. No legacy fallbacks."""
from pathlib import Path
import hashlib, importlib.util, os
from torch import nn
from .licensed_afpn import AFPN as _AFPN
from .licensed_wavelet import DWT_2D
from .licensed_attention import DA_Block
_LOAD_ERRORS={}
HLFDB_LOAD_ERROR=None
def _load_decoder():
    root=next(p for p in Path(__file__).resolve().parents if (p/"NEURAL_DEPENDENCIES.json").exists())
    path=Path(os.environ.get("DSMAF_EMCAD_DECODER",root/"external_dependencies/emcad/decoders.py"))
    if not path.is_file():
        raise FileNotFoundError("Obtain EMCAD directly under its research license; see DEPENDENCY_ACCESS.md. Set DSMAF_EMCAD_DECODER.")
    text=path.read_text(encoding="utf-8-sig")
    if hashlib.sha256(text.encode()).hexdigest() != "4b36981117376ea005aed8844f4ea9c846f432286695b85c1530797a20157a18":
        raise RuntimeError("EMCAD decoder differs from the qualified version")
    spec=importlib.util.spec_from_file_location("dsmaf_external_emcad",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
_decoder=None
def MSCBLayer(in_channels,out_channels,n=1,stride=1,kernel_sizes=[1,3,5],expansion_factor=2):
    global _decoder
    if _decoder is None:
        try: _decoder=_load_decoder()
        except Exception as error:
            _LOAD_ERRORS["decoders-mscam.py"]=repr(error)
            raise RuntimeError("Required EMCAD unavailable; no fallback is allowed") from error
    return _decoder.MSCBLayer(in_channels,out_channels,n=n,stride=stride,kernel_sizes=kernel_sizes,expansion_factor=expansion_factor)
class AFPN(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.impl=_AFPN(in_channels=in_channels,out_channels=out_channels)
        self._use_impl=True
    def forward(self,x):
        return self.impl(x)
def _removed(*args,**kwargs):
    raise RuntimeError("This legacy branch was not used by the locked study and is excluded from the release.")
B2C_BFAM=CBM=diff_moudel=_removed
