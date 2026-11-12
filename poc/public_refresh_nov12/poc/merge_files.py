"""Scan local full-submission CSVs with the same certified per-n rule as merge_public.
usage: merge_files.py merged.csv lit.json pandas.json file1.csv [file2.csv ...]
"""
import os, sys
from multiprocessing import Pool
import merge_public as mp


def main():
    best = mp.Best(*sys.argv[1:4])
    with Pool(8) as pool:
        for path in sys.argv[4:]:
            text = open(path, encoding='utf-8', errors='replace').read()
            w = mp.scan_text(best, text, f'local:{os.path.basename(path)}', pool)
            mp.log(f'local {os.path.basename(path)}: {w} wins; total gain {best.gain():.6f}')
    mp.log('FILES DONE total gain', best.gain(), 'n improved', len(best.rows))


if __name__ == '__main__':
    main()
