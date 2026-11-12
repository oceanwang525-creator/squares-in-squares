#!/usr/bin/env python3
"""Clone or refresh the GitHub packing banks surfaced by repository search.

These are independent researchers' banks that the November 10 re-sweep never
saw. Data only: repositories are cloned and parsed, never executed.
"""
from __future__ import annotations

import argparse
import os
import subprocess

REPOS = [
    "JoostdeWinter/square-packing",
    "TheSnakeFang/squarepack-certs",
    "DigitalEuan/squares_in_squares",
    "cubres/kaggle-squares-in-squares",
    "raojohng/squares-in-squares",
    "anopheles/packingproblem",
    "TobiasHFS/squares-in-squares",
    "omerfisler/kaggle-unit-squares-packing",
    "rushikesh648/Squares-in-square-",
    "mohsensbu-jpg/square_packing_records_karimi",
    "chromoany/r68-square-packing",
    "jlevy/squares",
]

# The kenaznia-doga s<n>-upper-bound family is small and versioned; each repo
# ships the raw witness for one published bound.
KENAZNIA = [
    "kenaznia-doga/s156-upper-bound",
    "kenaznia-doga/s172-upper-bound",
    "kenaznia-doga/s179-upper-bound",
    "kenaznia-doga/s179-upper-bound-13880960371",
    "kenaznia-doga/s180-upper-bound",
    "kenaznia-doga/s199-upper-bound",
    "kenaznia-doga/s204-upper-bound",
    "kenaznia-doga/s232-upper-bound",
    "kenaznia-doga/s237-upper-bound",
    "kenaznia-doga/s268-upper-bound",
    "kenaznia-doga/s268-upper-bound-16872029849",
    "kenaznia-doga/s269-upper-bound",
    "kenaznia-doga/s269-upper-bound-1688973007",
    "kenaznia-doga/s269-upper-bound-16899469076",
    "kenaznia-doga/s270-upper-bound",
    "kenaznia-doga/s299-upper-bound",
    "kenaznia-doga/s299-upper-bound-17801004",
    "kenaznia-doga/s303-upper-bound",
    "kenaznia-doga/s307-upper-bound",
]


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-kenaznia", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    repos = REPOS + (KENAZNIA if args.include_kenaznia else [])
    for url in repos:
        name = url.rsplit("/", 2)[-2] + "_" + url.rsplit("/", 1)[-1]
        target = os.path.join(args.out, name)
        if os.path.isdir(os.path.join(target, ".git")):
            run("git", "-C", target, "fetch", "--depth", "1", "origin")
            run("git", "-C", target, "reset", "--hard", "FETCH_HEAD")
            print("updated", target, flush=True)
        else:
            run("git", "clone", "--depth", "1", "--quiet",
                "https://github.com/" + url, target)
            print("cloned", target, flush=True)


if __name__ == "__main__":
    main()
