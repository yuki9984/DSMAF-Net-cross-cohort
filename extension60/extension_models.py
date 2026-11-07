"""Adapt reference fusion mechanisms without changing their executable class ASTs."""
from pathlib import Path
import ast,importlib.util,sys,os,hashlib
from extension_settings import BUNDLE,setup_original
ORIGINAL_MAKE_MODEL=None

def load_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
 return module

def class_from_reference(path,name,namespace):
 tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
 definitions=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name]
 if len(definitions)!=1:raise RuntimeError('Ambiguous author class: '+name)
 # Execute the original class body with explicit dependencies; optional unused
 # model imports are not executed. The selected fusion branch is unchanged.
 body=ast.Module(body=definitions,type_ignores=[])
 exec(compile(body,str(path),'exec'),namespace)
 return namespace[name]

def rg_model(pretrained=True):
 import torch
 from torch import nn
 from torchvision.models import resnet50,ResNet50_Weights
 ref=Path(os.environ.get('DSMAF_RG_DIR',str(BUNDLE.parent/'external_dependencies/rg_dermnet')))
 for name,expected in {'gatedResidualBlock.py': '2e5ee97690a5e9bcb9d25fbdbdafe27a1d3fb4d704572108ee0d06831465ba1f', 'metablock.py': 'd2c40b2d4c2845a80c0baf6853f10c56e5c2ac9d21ac4baefb877a3a7b01f18d', 'multimodalIntraInterModal.py': 'bc61d427aedad2075f61489a6336bf4f4b1a08798fca3ed9f4a3f58492a78e51'}.items():
  path=ref/name
  if not path.is_file():raise FileNotFoundError("Obtain the pinned RG-DermNet sources; see DEPENDENCY_ACCESS.md")
  if hashlib.sha256(path.read_text(encoding="utf-8-sig").encode()).hexdigest()!=expected:raise RuntimeError("RG-DermNet source differs from qualified version: "+name)
 gate=load_module('paa_rg_gates',ref/'gatedResidualBlock.py')
 meta=load_module('paa_rg_metablock',ref/'metablock.py')
 class CommonBackboneLoader:
  @staticmethod
  def loadModelImageEncoder(cnn_model_name,backbone_train_mode):
   if cnn_model_name!='resnet-50':raise ValueError('Only the fixed common backbone is allowed')
   model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
   model.fc=nn.Identity()
   for parameter in model.parameters():parameter.requires_grad_(True)
   return model,2048
 namespace={'torch':torch,'nn':nn,'loadModels':CommonBackboneLoader,'MetaBlock':meta.MetaBlock,'GatedAlteredResidualBlock':gate.GatedAlteredResidualBlock,'StackedGatedResidualBlock':gate.StackedGatedResidualBlock}
 cls=class_from_reference(ref/'multimodalIntraInterModal.py','MultimodalModel',namespace)
 return cls(num_classes=6,num_heads=8,device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),cnn_model_name='resnet-50',text_model_name='one-hot-encoder',vocab_size=5,common_dim=512,text_encoder_dim_output=512,unfreeze_weights='all_unfrozen_weights',attention_mecanism='att-intramodal+residual+cross-attention-metadados')

def make_model(arm,pretrained=True):
 global ORIGINAL_MAKE_MODEL
 setup_original()
 if arm in ('dsmaf_age_only','dsmaf_age_recorded'):
  if ORIGINAL_MAKE_MODEL is None:
   from model_v2 import make_model as original
   ORIGINAL_MAKE_MODEL=original
  return ORIGINAL_MAKE_MODEL('full',pretrained=pretrained)
 if arm=='metablock_cv':
  module=load_module('paa_metablock_protocol',BUNDLE/'references/metablock_protocol_adapter.py')
  return module.MetaBlockResNet50(pretrained=pretrained)
 if arm=='rg_dermnet_adapted_cv':return rg_model(pretrained=pretrained)
 raise ValueError('Unknown extension model arm')
