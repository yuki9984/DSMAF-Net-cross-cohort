"""Check locally obtained, pinned external sources without importing them."""
from pathlib import Path
import hashlib,json,os

ROOT=Path(__file__).resolve().parent

def check():
    manifest=json.loads((ROOT/'NEURAL_DEPENDENCIES.json').read_text(encoding='utf-8'))
    checks=[]
    for entry in manifest['external']:
        if entry['component']=='EMCAD MSCBLayer':
            files=[(Path(os.environ.get('DSMAF_EMCAD_DECODER',ROOT/'external_dependencies/emcad/decoders.py')),entry['normalized_source_sha256'])]
        elif entry['component']=='RG-DermNet':
            folder=Path(os.environ.get('DSMAF_RG_DIR',ROOT/'external_dependencies/rg_dermnet'))
            files=[(folder/name,digest) for name,digest in entry['normalized_source_sha256'].items()]
        else:
            raise ValueError('Unknown external dependency: '+entry['component'])
        for path,expected in files:
            actual=hashlib.sha256(path.read_text(encoding='utf-8-sig').encode()).hexdigest() if path.is_file() else None
            checks.append({'component':entry['component'],'file':path.name,'status':'PASS' if actual==expected else 'MISSING' if actual is None else 'HASH_MISMATCH'})
    return {'status':'PASS' if all(x['status']=='PASS' for x in checks) else 'DEPENDENCIES_REQUIRED','checks':checks}

if __name__=='__main__':
    result=check()
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['status']=='PASS' else 1)
