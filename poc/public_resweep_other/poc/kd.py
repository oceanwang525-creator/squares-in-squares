import json,sys,urllib.request
def post(url,obj):
    r=urllib.request.Request(url,json.dumps(obj).encode(),{'content-type':'application/json','User-Agent':'curl/8.4.0'})
    return json.load(urllib.request.urlopen(r,timeout=60))
seen={}
qs=['squares in squares','squares-in-squares','packing squares','square packing','squares packing','santa squares','certified packings','squares in square','packings','unit squares','sis packing','squares bank','layout atlas','quenched','contact polish','kaggle squares','squares gallery','submission squares in squares']
for q in qs:
  for sort in ['DATASET_SORT_BY_UPDATED','DATASET_SORT_BY_RELEVANCE','DATASET_SORT_BY_HOT','DATASET_SORT_BY_PUBLISHED']:
   for page in (1,2,3):
    try:
        d=post('https://www.kaggle.com/api/i/datasets.DatasetService/SearchDatasets',{"search":q,"pageSize":50,"page":page,"sortBy":sort})
    except Exception as e:
        print(q,sort,'ERR',str(e)[:100]);break
    for x in d.get('datasetList',{}).get('items',[]) if isinstance(d.get('datasetList'),dict) else d.get('datasetList',[]):
        seen[x.get('datasetSlug',x.get('slug'))+'|'+str(x.get('ownerUser',x.get('ownerSlug')))]=x
    if not d.get('hasMore'):break
print(len(seen))
json.dump(list(seen.values()),open('/Users/lucaszhang/squares-in-squares/poc/public_resweep_other/data/ds_list.json','w'),indent=1)
