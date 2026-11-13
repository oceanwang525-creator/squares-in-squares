import pathlib,subprocess,sys
r=pathlib.Path(__file__).resolve().parents[4]
for name in ['poc1_sheared_pockets.py','poc2_off45_sturmian_band.py','poc3_skyline_dp.py']:
 subprocess.run([sys.executable,str(r/'poc/hydrogen_brainstorm/poc'/name)],cwd=r,check=True)
