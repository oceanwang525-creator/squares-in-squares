"""Fetch every output file (csv/zip/gz, nested paths included) for every listed kernel. Data only, never executed."""
import json,sys,os,time,shutil,urllib.request,urllib.parse
from concurrent.futures import ThreadPoolExecutor
R='/Users/lucaszhang/squares-in-squares/poc/'
OLD=R+'fresh_public_sources/data/kernels/'
D=R+'public_resweep_kernels/data/kernels/'
UA={'User-Agent':'curl/8.4.0','Accept':'*/*'}
EXT=('.csv','.zip','.gz','.csv.gz','.tsv','.txt','.json','.parquet')
def get(url,retries=6,stream_to=None):
    err=None
    for i in range(retries):
        try:
            r=urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=120)
            if stream_to:
                with open(stream_to+'.part','wb') as f: shutil.copyfileobj(r,f,1<<20)
                os.replace(stream_to+'.part',stream_to);return None
            return r.read()
        except urllib.error.HTTPError as e:
            err=f'HTTP {e.code}'
            if e.code in(401,403,404):break
            time.sleep(2**i)
        except Exception as e:
            err=repr(e);time.sleep(2**i)
    raise RuntimeError(err)
def one(k):
    u,s=k['scriptUrl'].split('/')[2:4]
    d=f'{D}{u}__{s}';os.makedirs(d,exist_ok=True)
    rec=dict(kernel=f'{u}/{s}',files=[],errors=[])
    if os.path.exists(d+'/.done'):
        return json.load(open(d+'/.rec.json'))
    try:
        files=[];tok=None
        while True:
            url=f'https://www.kaggle.com/api/v1/kernels/output?userName={u}&kernelSlug={s}'+(f'&pageToken={urllib.parse.quote(tok)}' if tok else '')
            o=json.loads(get(url));files+=o.get('files',[]);tok=o.get('nextPageToken')
            if not tok:break
    except Exception as e:
        rec['errors'].append(('list',str(e)));json.dump(rec,open(d+'/.rec.json','w'));return rec
    for f in files:
        nm=f['fileName'];rec['files'].append(nm)
        if not nm.lower().endswith(EXT) or nm.startswith('__pycache__'):continue
        dst=d+'/'+nm.replace('/','__NEST__')
        if os.path.exists(dst):continue
        old=OLD+f'{u}__{s}/{nm}'
        if '/' not in nm and os.path.exists(old):
            shutil.copy(old,dst);continue
        try:get(f['url'],stream_to=dst)
        except Exception as e:rec['errors'].append((nm,str(e)))
    json.dump(rec,open(d+'/.rec.json','w'))
    if not rec['errors']:open(d+'/.done','w').write('1')
    print(rec['kernel'],len(rec['files']),rec['errors'],flush=True)
    return rec
ks=[k for k in json.load(open(sys.argv[1])).values()]
ks.sort(key=lambda k:k['lastRunTime'],reverse=True)
with ThreadPoolExecutor(4) as ex: recs=list(ex.map(one,ks))
json.dump(recs,open(R+'public_resweep_kernels/docs/fetch_log.json','w'),indent=1)
print('done',len(recs),sum(1 for r in recs if r['errors']))
