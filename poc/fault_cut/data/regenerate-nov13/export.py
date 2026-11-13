import pathlib,sys,json
r=pathlib.Path(__file__).resolve().parents[4]
sys.path[:0]=[str(r/'poc/fault_cut/poc'),str(r/'poc/baseline/poc/src')]
import fault_cut as m
from squares.core import read_submission,write_submission
groups=read_submission(r/'regeneration-proof/incumbent.csv')
results=[]
for source_n in [27,51,102,635]:
 result=m.run_cut_eval({source_n:groups[source_n]},1)
 if result['candidate_packing']:write_submission(pathlib.Path(__file__).with_name('n'+str(source_n-1)+'.csv'),{source_n-1:result['candidate_packing']},validate=False)
 results.append(result)
 print(source_n,result['certified'],result['final_L'],flush=True)
pathlib.Path(__file__).with_name('results.json').write_text(json.dumps(results,indent=2,default=lambda x:x.item()))
