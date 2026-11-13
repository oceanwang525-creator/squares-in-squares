import subprocess,sys,pathlib
r=pathlib.Path(__file__).resolve().parents[4]
subprocess.run([sys.executable,str(r/'poc/vacancy_diamonds/poc/extract.py')],cwd=r,check=True)
subprocess.run([sys.executable,str(r/'poc/vacancy_diamonds/poc/slp.py')],cwd=r/'poc/vacancy_diamonds',check=True)
