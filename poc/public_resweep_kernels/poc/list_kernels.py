"""List all competition kernels (anonymous Kaggle endpoint), paginated; data only."""
import json,sys,time,urllib.request
H={'content-type':'application/json','User-Agent':'curl/8.4.0'}
def post(obj):
    for i in range(6):
        try:
            r=urllib.request.Request('https://www.kaggle.com/api/i/kernels.KernelsService/ListKernels',json.dumps(obj).encode(),H)
            return json.load(urllib.request.urlopen(r,timeout=60))
        except Exception as e:
            err=e;time.sleep(2**i)
    raise err
allk={};log=[]
for sort in ['DATE_RUN','DATE_CREATED','HOTNESS','VOTE_COUNT','SCORE_ASCENDING','SCORE_DESCENDING','COMMENT_COUNT','VIEW_COUNT']:
  for q in ['']:
    pg=1;pages=0
    while True:
        lr={"pageSize":100,"sortBy":sort,"competitionId":166661,"page":pg}
        try: d=post({"kernelFilterCriteria":{"search":q,"listRequest":lr}})
        except Exception as e: log.append((sort,q,repr(e)));break
        pages+=1
        for k in d.get('kernels',[]): allk[k['scriptUrl']]=k
        pg+=1
        if not d.get('kernels'):break
    log.append((sort,q,pages,len(allk)))
    print(log[-1],flush=True)
json.dump(allk,open(sys.argv[1],'w'))
