"""Extract every zip (nested too) into its own new dir under data/extract; csv/txt/json members only."""
import zipfile,os,sys,glob,json
B='/Users/lucaszhang/squares-in-squares/poc/public_resweep_kernels/data/'
done=set();log={}
def ex(z,dst,depth=0):
    os.makedirs(dst,exist_ok=False)
    try:
        zf=zipfile.ZipFile(z)
    except Exception as e:
        log[z]=repr(e);return
    for i,info in enumerate(zf.infolist()):
        nm=info.filename
        if info.is_dir():continue
        low=nm.lower()
        if low.endswith(('.csv','.csv.gz','.tsv','.txt','.json')) and info.file_size<3e9:
            safe=f'{i}__'+nm.replace('/','__NEST__').replace('..','_')
            with zf.open(info) as s,open(dst+'/'+safe,'wb') as o:
                while (b:=s.read(1<<20)):o.write(b)
        elif low.endswith('.zip') and depth<2 and info.file_size<2e9:
            p=dst+f'/{i}__inner.zip'
            with zf.open(info) as s,open(p,'wb') as o:
                while (b:=s.read(1<<20)):o.write(b)
            ex(p,dst+f'/{i}__inner_x',depth+1)
    log[z]='ok'
for z in sorted(glob.glob(B+'kernels/**/*.zip',recursive=True)):
    ex(z,B+'extract/'+os.path.relpath(z,B+'kernels').replace('/','__'))
json.dump(log,open(B+'../docs/zip_log.json','w'),indent=1)
print(len(log),sum(v!='ok' for v in log.values()))
