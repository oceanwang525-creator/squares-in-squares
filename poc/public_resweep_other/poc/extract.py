import zipfile,glob,os,sys
D='/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/data/'
for z in sorted(glob.glob(D+'zips/*.zip')):
    name=os.path.basename(z)[:-4];dst=D+'src/'+name
    if os.path.exists(dst):continue
    os.makedirs(dst)
    try:
        with zipfile.ZipFile(z) as zf:zf.extractall(dst)
    except Exception as e:print('FAIL',name,e)
