import json,os,urllib.request
D='/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/data/sigma/'
os.makedirs(D,exist_ok=True)
def post(url,obj):
    r=urllib.request.Request(url,json.dumps(obj).encode(),{'content-type':'application/json','User-Agent':'curl/8.4.0'})
    return json.load(urllib.request.urlopen(r,timeout=60))
v=json.load(open(D+'../sig_versions.json'))['items']
for i in v:
    if i['run']['status']!='COMPLETE':continue
    vn=i['version']['versionNumber'];rid=i['run']['id']
    d=f'{D}v{vn}/'
    if os.path.exists(d+'.done'):continue
    os.makedirs(d,exist_ok=True)
    items=[];tok=None
    while True:
        o=post('https://www.kaggle.com/api/i/kernels.KernelsService/ListKernelSessionOutput',{"kernelSessionId":rid,"pageSize":500,**({"pageToken":tok} if tok else {})})
        items+=o.get('items',[]);tok=o.get('nextPageToken')
        if not tok:break
    names=[x['fullPath'] for x in items]
    print(vn,len(items),[n for n in names if not n.startswith('runtime_deps') and not n.endswith('.pyc')],flush=True)
    for x in items:
        p=x['fullPath']
        if p.startswith('runtime_deps') or p.endswith(('.pyc','.png','.py')):continue
        fn=d+p.replace('/','__')
        try:urllib.request.urlretrieve(x['downloadUrl'],fn)
        except Exception as e:print('fail',p,e)
    open(d+'.done','w').write('1')
