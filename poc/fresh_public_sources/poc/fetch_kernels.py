"""Download output CSVs of public Squares in Squares kernels (anonymous Kaggle endpoints). Data only."""
import json,sys,os,urllib.request,urllib.parse,time
opener=urllib.request.build_opener();opener.addheaders=[('User-Agent','curl/8.4.0'),('Accept','*/*')];urllib.request.install_opener(opener)
OUT=sys.argv[1]; N=int(sys.argv[2])
def post(url,obj):
    r=urllib.request.Request(url,json.dumps(obj).encode(),{'content-type':'application/json','User-Agent':'curl/8.4.0'})
    return json.load(urllib.request.urlopen(r,timeout=60))
ks=[];seen=set()
for q in ['','squares in squares','polish','fq','merge','contact','quench','certified','portfolio','lb 22']:
    for sort in ['DATE_RUN','HOTNESS']:
        try:
            d=post('https://www.kaggle.com/api/i/kernels.KernelsService/ListKernels',{"kernelFilterCriteria":{"search":q,"listRequest":{"pageSize":50,"sortBy":sort,"competitionId":166661}}})
        except Exception as e: print('list fail',q,e);continue
        for k in d['kernels']:
            if k['scriptUrl'] not in seen and k.get('hasDataOutputFiles'):
                seen.add(k['scriptUrl']);ks.append(k)
ks.sort(key=lambda k:k['lastRunTime'],reverse=True)
json.dump([dict(url=k['scriptUrl'],title=k['title'],run=k['lastRunTime'],score=k.get('bestPublicScore')) for k in ks],open(OUT+'/kernel_index.json','w'),indent=1)
print(len(ks),'kernels with outputs')
for k in ks[:N]:
    u,s=k['scriptUrl'].split('/')[2:4]
    d=f'{OUT}/{u}__{s}';
    if os.path.exists(d+'/.done'):continue
    os.makedirs(d,exist_ok=True)
    try:
        files=[];tok=None
        while True:
            url=f'https://www.kaggle.com/api/v1/kernels/output?userName={u}&kernelSlug={s}'+(f'&pageToken={urllib.parse.quote(tok)}' if tok else '')
            o=json.load(urllib.request.urlopen(url,timeout=60))
            files+=o.get('files',[]);tok=o.get('nextPageToken')
            if not tok:break
        for f in files:
            nm=f['fileName']
            if nm.lower().endswith('.csv') and '/' not in nm:
                try: urllib.request.urlretrieve(f['url'],f'{d}/{nm}')
                except Exception as e: print('dl fail',u,s,nm,e)
        print(u,s,[f['fileName'] for f in files if f['fileName'].endswith('.csv')],flush=True)
        open(d+'/.done','w').write('1')
    except Exception as e: print('fail',u,s,e,flush=True)
