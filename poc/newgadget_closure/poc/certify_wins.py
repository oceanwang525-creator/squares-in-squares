"""Build and certify the DP prediction against the requested merge."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
sys_dir = ROOT / "poc" / "gadget_squeeze" / "poc"
import sys
sys.path.insert(0, str(sys_dir))

import common  # noqa: E402
import compose  # noqa: E402
import squeeze  # noqa: E402


def main():
    incumbent = "/tmp/incumbent.csv"
    pred_path = HERE / "data" / "pred_current.json"
    lib_path = HERE / "data" / "lib_current.json"
    pred = json.loads(pred_path.read_text())
    lib = compose.load_lib(str(lib_path))
    ours = common.ours_L(incumbent, cache=str(HERE / "data" / "oursL.json"))
    wins = {}
    for ns, prediction in pred.items():
        n = int(ns)
        k = prediction["K"]
        f = prediction["f"]
        parts = [tuple(part) for part in prediction["parts"]]
        points = compose.realise(lib, k, f, parts, n)
        description = " + ".join(
            f"{lib[i]['w']}x{lib[i]['h']}+{lib[i]['g']}"
            f"{'T' if transpose else ''}[n={lib[i].get('src')},f={lib[i]['f']:.12g}]"
            for i, transpose in parts)
        if points is None or len(points) != n:
            print(f"n={n}: build failed")
            continue
        violation = squeeze.violation(points, k + f, k + f)
        x0, x1, y0, y1 = squeeze.bbox(points)
        if violation > 1e-9 or max(x1 - x0, y1 - y0) > k + f + 1e-9:
            print(f"n={n}: float gate failed ({violation:.3e})")
            continue
        rows = common.rows_from_P(n, points, eps=2e-12, s_prefix=False)
        _, literal, transported, why = common.certify(n, rows)
        if literal is None or transported is None:
            print(f"n={n}: certification failed ({why})")
            continue
        d_literal = float(literal) - ours[n]
        d_pandas = float(transported) - ours[n]
        print(f"n={n}: literal={literal} pandas={transported} "
              f"dL={d_literal:.12e}/{d_pandas:.12e} {description}")
        if min(d_literal, d_pandas) <= -1e-9:
            wins[n] = dict(
                L_literal=str(literal), L_pandas=str(transported),
                dL_literal=d_literal, dL_pandas=d_pandas, K=k, f=f,
                rows=[{key: str(row[key]) for key in
                       ("square_id", "x", "y", "angle")} for row in rows],
                description=description,
            )
    (HERE / "data" / "wins.json").write_text(
        json.dumps({str(k): v for k, v in wins.items()}, indent=1) + "\n")
    groups = {n: [{"square_id": row["square_id"], "x": row["x"], "y": row["y"],
                   "angle": row["angle"]} for row in win["rows"]]
              for n, win in wins.items()}
    common.write_final(str(HERE / "final.csv"), groups)
    print(f"certified wins: {sorted(wins)}; wrote {HERE / 'final.csv'}")


if __name__ == "__main__":
    main()
