"""Certify final.csv from its own bytes in literal and pandas transports."""
import csv
import json
import math
import sys
from pathlib import Path

import mpmath as mp
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "poc" / "gadget_squeeze" / "poc"))

import common  # noqa: E402


def groups_from_csv_text(text):
    groups = {}
    for row in csv.DictReader(text.splitlines()):
        n, sid = row["id"].split("_")
        with mp.workdps(80):
            angle = mp.mpf(row["deg"].lstrip("s")) * mp.pi / 180
        groups.setdefault(int(n), []).append(dict(
            n=str(n), square_id=sid,
            x=row["x"].lstrip("s"), y=row["y"].lstrip("s"),
            angle=mp.nstr(angle, 80),
        ))
    return groups


def main(path):
    text = Path(path).read_text()
    incumbent = str(ROOT / "merged-22.15074331648.csv")
    if not Path(incumbent).exists():
        incumbent = "/tmp/incumbent.csv"
    ours = common.ours_L(incumbent, cache=str(Path(__file__).parents[1] / "data" / "oursL.json"))
    report = {"incumbent": incumbent, "groups": {}}
    total = 0.0
    for n, rows in groups_from_csv_text(text).items():
        rows.sort(key=lambda r: int(r["square_id"]))
        _, literal, _, literal_why = common.certify(n, rows)
        frame = pd.read_csv(__import__("io").StringIO(text))
        pandas_text = frame.to_csv(index=False)
        _, transported, _, pandas_why = common.certify(
            n, groups_from_csv_text(pandas_text)[n])
        entry = dict(
            count=len(rows), complete=len(rows) == n,
            literal=(literal is not None), pandas=(transported is not None),
            literal_L=str(literal), pandas_L=str(transported),
            literal_why=str(literal_why), pandas_why=str(pandas_why),
            incumbent_L=ours.get(n),
        )
        if literal is not None and transported is not None and n in ours:
            d_lit = float(literal) - ours[n]
            d_pd = float(transported) - ours[n]
            entry.update(dL_literal=d_lit, dL_pandas=d_pd,
                         strict_win=min(d_lit, d_pd) <= -1e-9)
            total += (d_lit + d_pd) / (2 * math.sqrt(n))
        report["groups"][str(n)] = entry
        print(f"n={n}: complete={entry['complete']} literal={literal} "
              f"pandas={transported} dL={entry.get('dL_literal')} "
              f"{entry.get('dL_pandas')} strict={entry.get('strict_win')}")
    report["score_gain"] = total
    out = Path(__file__).parents[1] / "docs" / "final_verify.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"total score gain: {total:.12e}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         Path(__file__).parents[1] / "final.csv")
