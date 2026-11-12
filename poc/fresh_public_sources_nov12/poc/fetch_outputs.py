#!/usr/bin/env python3
"""Download public Kaggle kernel outputs anonymously. Data only; no submission.

The kernel list is the November 12 2026 snapshot used by this POC. Output files
are written under ``--out/<owner>__<slug>/``. Nested output paths are kept as
relative names so anchor CSVs can be scanned too.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request

USER_AGENT = "curl/8.4.0"

KERNELS = [
    "jonathan1506/50-top-2-6-22-1793-fault-dp-fq-stretch",
    "prvsiyan/a-square-deal-research-to-verified-packings",
    "malyshevdanil/squares-in-squares-living-atlas-of-tight-packings",
    "jazivxt/boxed-in-sub",
    "alexchilton/fault-line-dp-basin-hopping-lb-22-2620",
    "lucifer19/think-select-reclose",
    "nilanjansaha123/squares-gpu-seam-dp-soft-rigid-highs",
    "kmmmvs/squares-public-merge-daily",
    "busyaprime/lb-22-1945500-sacred-geometry-of-1000-squares",
    "mzishran/squares-in-squares-validated-merge-local-search",
]

# Keep the download bounded: full submissions plus optional anchor files.
DEFAULT_NAMES = {
    "submission.csv",
    "submission_merged.csv",
    "submission_ours.csv",
    "embedded_baseline.csv",
}


def opener() -> urllib.request.OpenerDirector:
    op = urllib.request.build_opener()
    op.addheaders = [("User-Agent", USER_AGENT), ("Accept", "*/*")]
    return op


def list_files(op, owner: str, slug: str) -> list[dict]:
    files: list[dict] = []
    token = None
    while True:
        url = (
            "https://www.kaggle.com/api/v1/kernels/output"
            f"?userName={urllib.parse.quote(owner)}"
            f"&kernelSlug={urllib.parse.quote(slug)}"
        )
        if token:
            url += "&pageToken=" + urllib.parse.quote(token)
        payload = json.load(urllib.request.urlopen(url, timeout=60))
        files.extend(payload.get("files", []))
        token = payload.get("nextPageToken")
        if not token:
            return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--anchors", action="store_true",
                        help="also fetch lucifer19 v*_anchor.csv files")
    args = parser.parse_args()

    op = opener()
    os.makedirs(args.out, exist_ok=True)
    index = {}
    for path in KERNELS:
        owner, slug = path.split("/", 1)
        target = os.path.join(args.out, f"{owner}__{slug}")
        os.makedirs(target, exist_ok=True)
        try:
            files = list_files(op, owner, slug)
        except Exception as error:  # noqa: BLE001 - one unavailable kernel is not fatal
            print("list failed", path, error)
            continue
        keep = []
        for entry in files:
            name = entry.get("fileName", "")
            base = os.path.basename(name)
            if base in DEFAULT_NAMES or (
                args.anchors and base.endswith("_anchor.csv")
            ):
                keep.append(entry)
        index[path] = [entry["fileName"] for entry in keep]
        for entry in keep:
            name = entry["fileName"]
            # Flatten nested output paths; names are unique within a kernel.
            dest = os.path.join(target, name.replace("/", "__"))
            if os.path.exists(dest):
                continue
            try:
                urllib.request.urlretrieve(entry["url"], dest)
                print("got", path, name, os.path.getsize(dest))
            except Exception as error:  # noqa: BLE001
                print("download failed", path, name, error)
    with open(os.path.join(args.out, "kernel_outputs.json"), "w",
              encoding="utf-8") as handle:
        json.dump(index, handle, indent=1)


if __name__ == "__main__":
    main()
