"""Scan CSVs in public GitHub repos with the same certified per-n rule. Data only, never executed.
usage: merge_github.py merged.csv lit.json pandas.json owner/repo [owner/repo ...]
"""
import gzip, io, json, os, sys, urllib.request, zipfile
from multiprocessing import Pool
import merge_public as mp


def get(url, timeout=300):
    return urllib.request.urlopen(urllib.request.Request(url, headers=mp.UA), timeout=timeout).read()


def main():
    best = mp.Best(*sys.argv[1:4])
    done_p = f'{mp.STATE}/done_gh.json'
    done = set(json.load(open(done_p))) if os.path.exists(done_p) else set()
    with Pool(8) as pool:
        for repo in sys.argv[4:]:
            if repo in done:
                continue
            try:
                info = json.loads(get(f'https://api.github.com/repos/{repo}'))
                br = info['default_branch']
                tree = json.loads(get(f'https://api.github.com/repos/{repo}/git/trees/{br}?recursive=1'))
            except Exception as e:
                mp.log('gh list fail', repo, e); continue
            files = [t for t in tree.get('tree', []) if t['type'] == 'blob' and t.get('size', 0) < 150e6
                     and t['path'].lower().endswith(('.csv', '.csv.gz', '.zip'))]
            mp.log(f'repo {repo}: {len(files)} csv/zip files (truncated={tree.get("truncated")})')
            for t in files:
                url = f'https://raw.githubusercontent.com/{repo}/{br}/{urllib.parse.quote(t["path"])}'
                try:
                    data = get(url, 600)
                except Exception as e:
                    mp.log('  dl fail', t['path'], e); continue
                texts = []
                p = t['path'].lower()
                try:
                    if p.endswith('.gz'):
                        texts.append((t['path'], gzip.decompress(data).decode('utf-8', 'replace')))
                    elif p.endswith('.zip'):
                        with zipfile.ZipFile(io.BytesIO(data)) as z:
                            for i in z.infolist():
                                if i.filename.lower().endswith('.csv') and i.file_size < 150e6:
                                    texts.append((t['path'] + ':' + i.filename, z.read(i).decode('utf-8', 'replace')))
                    else:
                        if data[:200].startswith(b'version https://git-lfs'):
                            continue
                        texts.append((t['path'], data.decode('utf-8', 'replace')))
                except Exception as e:
                    mp.log('  parse fail', t['path'], e); continue
                del data
                for name, text in texts:
                    w = mp.scan_text(best, text, f'github:{repo}:{name}', pool)
                    if w:
                        mp.log(f'  {name}: {w} wins; total gain {best.gain():.6f}')
            done.add(repo); json.dump(sorted(done), open(done_p, 'w'))
    mp.log('GH DONE total gain', best.gain(), 'n improved', len(best.rows))


if __name__ == '__main__':
    import urllib.parse
    main()
