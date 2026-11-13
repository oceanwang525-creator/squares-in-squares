#!/usr/bin/env python3
"""Export only reviewed regeneration folders, never modifying the incumbent."""
import argparse, csv, io, json, pathlib, subprocess, sys
ROOT=pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'poc/auto_union/poc'))
import normalize as norm
u=norm.u

def row_groups(data):
    """Read explicit full coordinate records from reviewed search JSON evidence."""
    found=[]
    def walk(x, expected=None):
        if isinstance(x,dict):
            expected=x.get('n',expected)
            if isinstance(x.get('rows'),list) and x['rows']:
                rows=x['rows']
                if all(isinstance(r,dict) and {'x','y','angle'}<=r.keys() for r in rows):
                    n=int(expected) if expected is not None else 0
                    if 1<=n<=1000 and len(rows)==n:
                        found.append({n:[dict(square_id=str(i),x=str(r['x']),y=str(r['y']),angle=str(r['angle'])) for i,r in enumerate(rows)]})
            for key,value in x.items():walk(value,int(key) if str(key).isdigit() else expected)
        elif isinstance(x,list):
            for value in x:walk(value,expected)
    walk(data)
    return found

def export(folder,cache,sides,source):
    f=ROOT/'poc'/folder
    entries=[];audit=[];tasks=[]
    paths=list((f/'data/regenerate-nov13').rglob('*'))
    # Legacy programs have documented output paths outside their new evidence dir.
    paths+=list((f/'data').glob('*.npy'))+list((f/'data').glob('*.json'))
    paths+=list((f/'docs').glob('*.npy'))+list(f.glob('data_*.json'))
    seen=set()
    for path in sorted(paths):
        if not path.is_file() or path in seen or path.suffix not in ('.csv','.npy','.npz','.json'):continue
        seen.add(path)
        if path.name in ('execution.json','proof.json'):continue
        rec={'path':str(path.relative_to(ROOT)),'sha256':u.digest(path.read_bytes())}
        groups=[]
        try:
            a,b=norm.candidate(path);groups.append((a,b))
        except Exception as e:
            rec['decode_note']=str(e)
            if path.suffix=='.json':
                try:
                    for raw in row_groups(json.loads(path.read_text())):
                        text=u.csv_text(raw)
                        groups.append((u.parse_text(text),u.parse_text(u.pd.read_csv(io.StringIO(text)).to_csv(index=False))))
                except Exception as e:rec['row_decode_note']=str(e)
        for a,b in groups:
            for n,rows in a.items():
                task=(n,rows,b.get(n,[]));entries.append((task,rec['path']));tasks.append(task)
        rec['groups']=sum(len(a) for a,b in groups);audit.append(rec)
    u.certify_many(tasks,1,cache)
    best={};values={};sources={}
    for task,path in entries:
        n=task[0];result=cache[u.key(task)]
        if n in sides and u.qualifies(result,sides[n]):
            pair=tuple(map(u.Decimal,result))
            if n not in values or all(a<b for a,b in zip(pair,values[n])):
                best[n]=task[1];values[n]=pair;sources[n]=path
    data=u.final_text(best).encode()
    a=u.parse_text(data.decode());b=u.parse_text(u.pd.read_csv(io.StringIO(data.decode())).to_csv(index=False))
    exported=[(n,rows,b[n]) for n,rows in a.items()]
    u.certify_many(exported,1,cache)
    excluded=[t[0] for t in exported if not u.qualifies(cache[u.key(t)],sides[t[0]])]
    for n in excluded:best.pop(n);sources.pop(n)
    data=u.final_text(best).encode()
    final=f/'final.csv';before=final.read_bytes() if final.exists() else None
    data,kept=norm.preserve_final_groups(before,data,sides,1,cache)
    a=u.parse_text(data.decode());b=u.parse_text(u.pd.read_csv(io.StringIO(data.decode())).to_csv(index=False))
    finaltasks=[(n,rows,b[n]) for n,rows in a.items()]
    u.certify_many(finaltasks,1,cache)
    assert all(u.qualifies(cache[u.key(t)],sides[t[0]]) for t in finaltasks)
    u.atomic(final,data)
    evidence=f/'data/regenerate-nov13';evidence.mkdir(parents=True,exist_ok=True)
    proof=dict(folder=folder,source_main=source,incumbent_sha256=u.digest(u.BASE.read_bytes()),
        canonical_unchanged=True,pandas=u.pd.__version__,literal_and_actual_pandas=True,
        candidate_files=audit,candidate_groups=len(tasks),winning_ns=sorted(a),preserved_ns=kept,
        conversion_excluded=excluded,final_sha256=u.digest(data),sources=sources,
        candidate_results=[{'n':t[0],'key':u.key(t),'official':cache[u.key(t)]} for t in tasks],
        final_results=[{'n':t[0],'official':cache[u.key(t)],'incumbent':[str(v) for v in sides[t[0]]]} for t in finaltasks])
    u.atomic(evidence/'proof.json',json.dumps(proof,indent=2).encode())
    print('EXPORTED',folder,sorted(a),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='+');args=p.parse_args()
    initial=u.BASE.read_bytes();assert u.pd.__version__=='3.1.0'
    cachepath=ROOT/'regeneration-proof/cache.json'
    cache=json.loads(cachepath.read_text()) if cachepath.exists() else {}
    _,sides=u.baseline_checks(u.BASE,1,cache)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for f in args.folders:export(f,cache,sides,source)
    assert u.BASE.read_bytes()==initial
    u.atomic(cachepath,json.dumps(cache).encode())
if __name__=='__main__':main()
