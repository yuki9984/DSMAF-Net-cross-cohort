"""Two information conditions of the unchanged ordinary late-fusion network."""
from extension_settings import setup_original
ORIGINAL_MAKE_MODEL=None

def make_model(arm,pretrained=True):
 global ORIGINAL_MAKE_MODEL
 setup_original()
 if arm not in ('late_age_only','late_age_recorded'):
  raise ValueError('Unknown late-fusion information condition: '+arm)
 if ORIGINAL_MAKE_MODEL is None:
  from model_v2 import make_model as original
  ORIGINAL_MAKE_MODEL=original
 return ORIGINAL_MAKE_MODEL('late_fusion',pretrained=pretrained)
