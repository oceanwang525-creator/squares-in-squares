import json,urllib.request,urllib.parse
seen={}
for q in ['squares in squares','squares-in-squares','square packing','packings']:
  for p in [1,2]:
    try: d=json.load(urllib.request.urlopen(urllib.request.Request(f'https://www.kaggle.com/api/v1/datasets/list?search={urllib.parse.quote(q)}&sortBy=updated&page={p}',headers={'User-Agent':'curl/8.4.0'}),timeout=60))
    except Exception as e: continue
    for x in d: seen[x['ref']]=[x.get('lastUpdated'),x.get('totalBytes'),x.get('title')]
json.dump(seen,open('datasets.json','w')); print(len(seen),'datasets')
