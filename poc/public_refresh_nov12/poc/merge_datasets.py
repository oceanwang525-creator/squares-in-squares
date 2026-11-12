"""Same as merge_public, but for public Kaggle datasets (zip of CSVs). Data only."""
import json, os, sys, tempfile, urllib.request, zipfile
from multiprocessing import Pool
import merge_public as mp

SKIP_WORDS = ('qm9', 'tic-tac', 'groundwater', 'sentinel', 'home-growth', 'polygonatum', 'daily-machine',
              'vqcc', 'medal-line', 'big-universe', 'phantom-progress')


def main():
    best = mp.Best(*sys.argv[1:4])
    ds = json.load(open(sys.argv[4]))
    done_p = f'{mp.STATE}/done_ds.json'
    done = set(json.load(open(done_p))) if os.path.exists(done_p) else set()
    refs = sorted(ds, key=lambda r: ds[r][0] or '', reverse=True)
    with Pool(8) as pool:
        for ref in refs:
            upd, size, title = ds[ref]
            if ref in done or any(w in ref for w in SKIP_WORDS) or (size or 0) > 400e6:
                continue
            mp.log(f'dataset {ref} ({(size or 0)/1e6:.1f} MB, {upd})')
            fd, tmp = tempfile.mkstemp(suffix='.zip', dir='.'); os.close(fd)
            try:
                req = urllib.request.Request(f'https://www.kaggle.com/api/v1/datasets/download/{ref}', headers=mp.UA)
                with urllib.request.urlopen(req, timeout=900) as r, open(tmp, 'wb') as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b: break
                        f.write(b)
                with zipfile.ZipFile(tmp) as z:
                    for info in z.infolist():
                        if not info.filename.lower().endswith('.csv') or info.file_size > 150e6:
                            continue
                        text = z.read(info).decode('utf-8', 'replace')
                        w = mp.scan_text(best, text, f'dataset:{ref}:{info.filename}', pool)
                        if w:
                            mp.log(f'  {info.filename}: {w} wins; total gain {best.gain():.6f}')
            except Exception as e:
                mp.log('  fail', ref, e)
            finally:
                os.remove(tmp)
            done.add(ref); json.dump(sorted(done), open(done_p, 'w'))
    mp.log('DS DONE total gain', best.gain(), 'n improved', len(best.rows))


if __name__ == '__main__':
    main()
