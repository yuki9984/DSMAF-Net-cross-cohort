from pathlib import Path
import subprocess,sys
r=Path(__file__).resolve().parent
subprocess.run([sys.executable,'-B',str(r/'followup/verify_followup.py')],cwd=r,check=True)
