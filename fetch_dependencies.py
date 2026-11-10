"""Fetch pinned upstream files only after the requested license acknowledgment."""
from pathlib import Path
import argparse,hashlib,json,urllib.request
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--accept-emcad-research-license',action='store_true')
p.add_argument('--accept-rg-noncommercial-license',action='store_true')
a=p.parse_args()
items=json.loads((R/'NEURAL_DEPENDENCIES.json').read_text())['external']
if not a.accept_emcad_research_license or not a.accept_rg_noncommercial_license:
 p.error('Read DEPENDENCY_ACCESS.md and the linked upstream licenses, then explicitly acknowledge both with the corresponding flags.')
for x in items:
 if x['component']=='EMCAD MSCBLayer':
  files=[('decoders.py',x['url'].replace('github.com/','raw.githubusercontent.com/').replace('/blob/',''),x['normalized_source_sha256'],R/'external_dependencies/emcad')]
 else:
  base=x['url'].replace('github.com/','raw.githubusercontent.com/').replace('/tree/','')
  files=[(n,base+'/'+n,h,R/'external_dependencies/rg_dermnet') for n,h in x['normalized_source_sha256'].items()]
 for name,url,expected,folder in files:
  with urllib.request.urlopen(url,timeout=90) as f:raw=f.read()
  text=raw.decode('utf-8-sig').replace('\r\n','\n')
  if hashlib.sha256(text.encode()).hexdigest()!=expected:raise RuntimeError('Pinned source hash mismatch: '+name)
  folder.mkdir(parents=True,exist_ok=True);(folder/name).write_text(text,encoding='utf8',newline='\n')
  print('VERIFIED',name)
print('Dependencies obtained from original owners. Their licenses continue to apply.')
