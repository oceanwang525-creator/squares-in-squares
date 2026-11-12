import json,re,urllib.request
def post(url,obj):
    r=urllib.request.Request(url,json.dumps(obj).encode(),{'content-type':'application/json','User-Agent':'curl/8.4.0'})
    return json.load(urllib.request.urlopen(r,timeout=60))
D='/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/data/'
old=json.load(open(D+'ds_list.json'));seen={x['datasetUrl']:x for x in old}
qs=['think select reclose','certified snapshot','fault dp','fault-line','squares champion','squares blend','square deal','supporting files','phantom progress','packing 1000 squares','sis certified','contact lp','l-band','stretched packings','squares v4','private packings','best packings','sigmaborov','alexchilton','squares in squares data','squares in squares solution','squares submission','side lengths squares packing','tilted squares','n=1000 squares','squares-in-squares-2','erich friedman packing','packing center','squares in squares weights','squares in squares input']
for q in qs:
  for sort in ['DATASET_SORT_BY_UPDATED','DATASET_SORT_BY_RELEVANCE']:
   for page in (1,2):
    try: d=post('https://www.kaggle.com/api/i/datasets.DatasetService/SearchDatasets',{"search":q,"pageSize":50,"page":page,"sortBy":sort})
    except Exception as e: print(q,e);break
    for x in d['datasetList']['items'] if isinstance(d['datasetList'],dict) else d['datasetList']: seen.setdefault(x['datasetUrl'],x)
    if not d.get('hasMore'):break
json.dump(list(seen.values()),open(D+'ds_list.json','w'),indent=1)
print(len(seen))
sel=[]
for u,x in seen.items():
    slug=u.strip('/').split('/',1)[1];t=x['datasource'].get('title','')
    if x['dateUpdated']<'2026-09-01':continue
    if re.search(r'square|packing|sis-|-sis|layout|quench|think|reclose|supporting-files-v25|phantom|fault|blend|snapshot|champion',(slug+' '+t).lower()) and not re.search(r'tic-tac|qm9|machine-learning|groundwater|sentinel|home-growth|vqcc|eeg|sea|toronto|poland|job',slug): sel.append(slug)
sel=[s for s in sel if 'qcc-frozen' not in s]
open(D+'ds_slugs.txt','w').write('\n'.join(sorted(set(sel)))+'\n');print(len(set(sel)));print('\n'.join(sorted(set(sel))))
