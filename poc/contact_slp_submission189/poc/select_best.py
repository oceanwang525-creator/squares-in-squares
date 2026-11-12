import csv
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import monotonic

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'baseline' / 'poc' / 'src'))
from squares.core import evaluate as evaluate_single_puzzle

FIELDS = ['id','n','square_id','x','y','angle']
TOLERANCE = 1e-8

def load(path):
    grouped=[[] for _ in range(1001)]
    with open(path, encoding='utf-8-sig', newline='') as source:
        reader=csv.reader(source)
        if next(reader)!=FIELDS:
            raise ValueError(f'{path}: unexpected headers')
        for i,(key,n,sid,x,y,a) in enumerate(reader):
            N=int(n)
            if not 1<=N<=1000 or int(sid)!=len(grouped[N]) or int(key)!=i:
                raise ValueError(f'{path}: invalid IDs at row {i}')
            grouped[N].append((x,y,a))
    if any(len(grouped[n])!=n for n in range(1,1001)):
        raise ValueError(f'{path}: missing coordinates')
    return grouped

def width(geometry):
    l=b=math.inf
    r=t=-math.inf
    for x0,y0,a0 in geometry:
        x,y,a=float(x0),float(y0),float(a0)
        h=(abs(math.cos(a))+abs(math.sin(a)))/2
        l,r,b,t=min(l,x-h),max(r,x+h),min(b,y-h),max(t,y+h)
    return max(r-l,t-b)

def cert(n, geometry):
    rows=[dict(square_id=sid,x=x,y=y,angle=a) for sid,(x,y,a) in enumerate(geometry)]
    return evaluate_single_puzzle(n,rows)

def certify(job):
    n, baseline, approx, proposed=job
    _, baseline_score, _, baseline_status=cert(n,baseline)
    best=baseline
    chosen=None
    best_score=baseline_score
    best_side=None
    rejects=[]
    for estimate,name,geometry in proposed:
        if baseline_status=='ok' and estimate/math.sqrt(n)-1>=best_score-TOLERANCE:
            break
        _, score, side, status=cert(n,geometry)
        if status!='ok':
            rejects.append((name,status))
            continue
        if score<best_score-TOLERANCE or (best_side is None and baseline_status!='ok' and score<=best_score+TOLERANCE):
            best,chosen,best_score,best_side=geometry,name,score,side
    return n,best,chosen,best_score,best_side,baseline_score,baseline_status,rejects

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Keep officially certified lower-score whole puzzles')
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('candidates', nargs='+', type=Path)
    args = parser.parse_args()
    baseline = args.baseline
    started=monotonic()
    base=load(baseline)
    limits=[0.0]+[width(base[n]) for n in range(1,1001)]
    estimate_scores={baseline.name:math.fsum(limits[n]/math.sqrt(n)-1 for n in range(1,1001))}
    shortlists=[[] for _ in range(1001)]
    seen=[set() for _ in range(1001)]
    for path in args.candidates:
        if not path.exists():
            continue
        data=load(path)
        lengths=[0.0]+[width(data[n]) for n in range(1,1001)]
        estimate_scores[path.name]=math.fsum(lengths[n]/math.sqrt(n)-1 for n in range(1,1001))
        for n in range(1,1001):
            if lengths[n] < limits[n] - TOLERANCE*math.sqrt(n):
                geom=tuple(data[n])
                if geom not in seen[n]:
                    shortlists[n].append((lengths[n],path.name,geom))
                    seen[n].add(geom)
    print('Source estimates',json.dumps(estimate_scores),flush=True)
    jobs=[(n,base[n],limits[n],sorted(shortlists[n],key=lambda x:x[0])) for n in range(1,1001) if shortlists[n]]
    print(len(jobs),'groups to certify, shortlist ready in',round(monotonic()-started,1),'s',flush=True)
    adopted={}
    rejects={}
    base_scores={n:limits[n]/math.sqrt(n)-1 for n in range(1,1001)}
    selected_scores=base_scores.copy()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(certify,jobs),1):
            n, rows, name, score, side, baseline_score, baseline_status, failures=result
            base_scores[n]=baseline_score
            selected_scores[n]=score
            if name is not None:
                base[n]=rows
                adopted[n]=dict(source=name,score=score,side=side,gain=baseline_score-score)
            if failures or baseline_status!='ok':
                rejects[n]=dict(baseline_status=baseline_status,candidates=failures)
            if i%25==0:
                print(i,'checked,',len(adopted),'adopted',flush=True)
    count_by_source={}
    for row in adopted.values():
        count_by_source[row['source']]=count_by_source.get(row['source'],0)+1
    summary=dict(approx_source_scores=estimate_scores,baseline_score=math.fsum(base_scores.values()),score=math.fsum(selected_scores.values()),
                 gain=math.fsum(row['gain'] for row in adopted.values()),changed=len(adopted),count_by_source=count_by_source,failed=rejects)
    print(json.dumps(summary),flush=True)
    with args.output.open('w',encoding='utf-8',newline='') as out:
        writer=csv.writer(out,lineterminator='\n')
        writer.writerow(FIELDS)
        key=0
        for n in range(1,1001):
            for sid,(x,y,a) in enumerate(base[n]):
                writer.writerow((key,n,sid,x,y,a))
                key+=1
    args.output.with_suffix('.json').write_text(json.dumps(dict(summary=summary,groups=adopted),indent=2)+'\n')

if __name__=='__main__':
    main()


