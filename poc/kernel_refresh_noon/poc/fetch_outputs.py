#!/usr/bin/env python3
"""Download current CSV outputs of the newest public kernels.

Anonymous Kaggle output API only; data is downloaded, never submitted. The
kernel list is re-queried across several sort orders because the public list
endpoint caps at 100 results per sort.
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request

USER_AGENT = "curl/8.4.0"


def request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})


def list_kernels() -> list[dict]:
    seen: dict[str, dict] = {}
    for page in (1, 2, 3):
        for sort in ("DATE_RUN", "DATE_CREATED", "HOTNESS", "SCORE", "VOTES"):
            body = json.dumps({
                "kernelFilterCriteria": {
                    "search": "",
                    "listRequest": {
                        "pageSize": 100, "page": page, "sortBy": sort,
                        "competitionId": 166661,
                    },
                },
            }).encode()
            req = urllib.request.Request(
                "https://www.kaggle.com/api/i/kernels.KernelsService/ListKernels",
                body, {"content-type": "application/json",
                       "User-Agent": USER_AGENT})
            try:
                payload = json.load(urllib.request.urlopen(req, timeout=60))
            except Exception:  # noqa: BLE001 - one sort failing is not fatal
                continue
            for kernel in payload.get("kernels", []):
                seen.setdefault(kernel["scriptUrl"], kernel)
    return sorted(seen.values(), key=lambda k: k.get("lastRunTime", ""),
                  reverse=True)


def list_files(owner: str, slug: str) -> list[dict]:
    files: list[dict] = []
    token = None
    while True:
        url = ("https://www.kaggle.com/api/v1/kernels/output"
               f"?userName={urllib.parse.quote(owner)}"
               f"&kernelSlug={urllib.parse.quote(slug)}")
        if token:
            url += "&pageToken=" + urllib.parse.quote(token)
        payload = json.load(urllib.request.urlopen(request(url), timeout=60))
        files.extend(payload.get("files", []))
        token = payload.get("nextPageToken")
        if not token:
            return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--kernels", default="",
                        help="comma-separated owner/slug; default: newest 12")
    args = parser.parse_args()

    kernels = list_kernels()
    print("kernels", len(kernels))
    paths = [p for p in args.kernels.split(",") if p.strip()]
    if not paths:
        paths = [k["scriptUrl"].replace("/code/", "")
                 for k in kernels[:12] if k.get("hasDataOutputFiles")]

    os.makedirs(args.out, exist_ok=True)
    for path in paths:
        owner, slug = path.split("/", 1)
        target = os.path.join(args.out, f"{owner}__{slug}")
        os.makedirs(target, exist_ok=True)
        try:
            files = list_files(owner, slug)
        except Exception as error:  # noqa: BLE001
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
                with urllib.request.urlopen(request(entry["url"]),
                                            timeout=300) as source, \
                        open(dest, "wb") as sink:
                    sink.write(source.read())
                print("got", path, name, os.path.getsize(dest))
            except Exception as error:  # noqa: BLE001
                print("download failed", path, name, error)


if __name__ == "__main__":
    main()
