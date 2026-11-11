import sys,pickle,json,subprocess
sys.path.insert(0,'/Users/lucaszhang/squares-in-squares/poc/baseline/poc/src')
import pandas as pd
from squares.core import read_submission,write_submission
R='/Users/lucaszhang/squares-in-squares/';F=R+'poc/fresh_public_sources/'
m=read_submission(R+'poc/pell_corrugation/data/border-scan/merged.csv')
w=pickle.load(open(F+'data/wins.pkl','rb'))
for n,v in w.items(): m[n]=v['rows']
write_submission(F+'out/fresh_merged.csv',m,validate=False)
pd.read_csv(F+'out/fresh_merged.csv').to_csv(F+'out/fresh_merged_pandas.csv',index=False)
print('built',len(w),flush=True)
for nm in ['fresh_merged','fresh_merged_pandas']:
    subprocess.run(['/Users/lucaszhang/miniforge3/bin/squares','score',F+f'out/{nm}.csv','--report',F+f'out/{nm}_report.json'],check=False)
