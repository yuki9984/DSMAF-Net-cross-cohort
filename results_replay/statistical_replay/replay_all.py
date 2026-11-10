"""Replay all public statistical modules; no neural training or inference."""
from pathlib import Path
import argparse,subprocess,sys,os,json,hashlib
R=Path(__file__).resolve().parent
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();out=args.output.resolve()
 if out.exists():raise FileExistsError('Use a new verification-output folder.')
 manifest=json.loads((R/'PACKAGE_MANIFEST.json').read_text('utf8'))
 for row in manifest['files']:
  p=(R/row['path']).resolve()
  assert p.is_relative_to(R.resolve()) and p.is_file()
  assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
 out.mkdir(parents=True);reports={}
 for name,script in [('original_and_PAA60','replay_core.py'),('late_fusion_followup','replay.py'),('MILK10k','replay.py')]:
  run=subprocess.run([sys.executable,'-X','utf8','-B',str(R/name/script)],cwd=R/name,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'},capture_output=True,text=True,encoding='utf8')
  (out/(name+'_stdout.txt')).write_text(run.stdout,encoding='utf8')
  (out/(name+'_stderr.txt')).write_text(run.stderr,encoding='utf8')
  if run.returncode:raise RuntimeError(name+' failed; its log is preserved.')
  report=json.loads(run.stdout)
  assert report['status']=='PASS'
  reports[name]=report
 result=dict(status='ALL_THREE_SAVED_OUTPUT_REPLAYS_PASS',modules=reports,
             scope='Saved-output statistical verification; no source-data audit, neural inference or training.',
             public_upload_allowed=True)
 (out/'REVIEW_REPORT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
