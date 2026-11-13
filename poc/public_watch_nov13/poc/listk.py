import json,urllib.request
def post(url,obj):
    r=urllib.request.Request(url,json.dumps(obj).encode(),{'content-type':'application/json','User-Agent':'curl/8.4.0'})
    return json.load(urllib.request.urlopen(r,timeout=60))
ks={}
for q in ['','squares in squares','polish','merge','contact','certified','portfolio','lb','band','atlas','22.1','packing','optimizer','ensemble','best']:
    for sort in ['DATE_RUN','HOTNESS','SCORE_ASCENDING','VOTE_COUNT']:
        for page in [None]:
            try:
                d=post('https://www.kaggle.com/api/i/kernels.KernelsService/ListKernels',{"kernelFilterCriteria":{"search":q,"listRequest":{"pageSize":100,"sortBy":sort,"competitionId":166661}}})
            except Exception as e: print('fail',q,sort,e);continue
            for k in d.get('kernels',[]):
                ks[k['scriptUrl']]=dict(url=k['scriptUrl'],title=k['title'],run=k.get('lastRunTime'),score=k.get('bestPublicScore'),out=k.get('hasDataOutputFiles'))
L=sorted(ks.values(),key=lambda k:(k['score'] or 99))
json.dump(L,open('kernels.json','w'),indent=1)
print(len(L))
for k in L[:40]: print(k['score'],k['run'],k['out'],k['url'])
