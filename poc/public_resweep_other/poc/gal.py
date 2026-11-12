import json,os,urllib.request,concurrent.futures as cf
D='/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/data/gallery_n/'
os.makedirs(D,exist_ok=True)
def get(n):
    for t in range(4):
        try:
            r=urllib.request.Request(f'https://squares-in-squares-gallery.vercel.app/data/{n}.json',headers={'User-Agent':'curl/8.4.0'})
            d=json.load(urllib.request.urlopen(r,timeout=60))
            json.dump(dict(n=n,L=d.get('L'),s=d['s'],keys=list(d)),open(D+f'g{n}.json','w'));return n,list(d)
        except Exception as e:err=str(e)
    return n,err
with cf.ThreadPoolExecutor(3) as ex:
    res=list(ex.map(get,range(1,1001)))
bad=[r for r in res if isinstance(r[1],str)]
print(len(bad),bad[:5],res[5])
