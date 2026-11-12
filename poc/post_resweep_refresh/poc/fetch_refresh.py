#!/usr/bin/env python3
"""Download current CSV outputs of kernels missed by or newer than the resweep.

Anonymous Kaggle output API only; data is downloaded, never submitted.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request

USER_AGENT = "curl/8.4.0"

KERNELS = [
    "kmmmvs/squares-public-merge-daily",
    "prvsiyan/a-square-deal-research-to-verified-packings",
    "malyshevdanil/squares-in-squares-living-atlas-of-tight-packings",
    "alexchilton/squares-101-the-starter-kit-10-600-downloads",
    "alexchilton/fault-line-dp-basin-hopping-lb-22-2620",
    "penguin069/squares-boxed-public-certified-union",
    "keshigami/squares-in-squares-validate-first-tilt-one-block",
    "saamhm/lb-22-257046-auto-optimizer-packing-unit-squares",
    "amirhezekiahespanto/debut",
]


def list_files(owner: str, slug: str) -> list[dict]:
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
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
        payload = json.load(urllib.request.urlopen(request, timeout=60))
        files.extend(payload.get("files", []))
        token = payload.get("nextPageToken")
        if not token:
            return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for path in KERNELS:
        owner, slug = path.split("/", 1)
        target = os.path.join(args.out, f"{owner}__{slug}")
        os.makedirs(target, exist_ok=True)
        try:
            files = list_files(owner, slug)
        except Exception as error:  # noqa: BLE001 - one kernel is not fatal
            print("list failed", path, error)
            continue
        for entry in files:
            name = entry.get("fileName", "")
            if not name.endswith(".csv") or name.startswith("."):
                continue
            dest = os.path.join(target, name.replace("/", "__"))
            if os.path.exists(dest):
                continue
            try:
                request = urllib.request.Request(
                    entry["url"],
                    headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
                with urllib.request.urlopen(request, timeout=300) as source, \
                        open(dest, "wb") as sink:
                    sink.write(source.read())
                print("got", path, name, os.path.getsize(dest))
            except Exception as error:  # noqa: BLE001
                print("download failed", path, name, error)


if __name__ == "__main__":
    main()
