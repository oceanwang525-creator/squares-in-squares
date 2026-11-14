"""Overlay PR #72 final geometries onto the official upstream donor CSV.

The result is a donor source only. Official win/loss comparisons always use the
unmodified upstream merge artifact.
"""
import argparse
import csv
import io

import mpmath as mp


PR_GROUPS = {
    "h128": ({634, 695, 861, 926}, {926}),
    "h131": ({634, 695, 861}, {634, 695, 861}),
    "h132": ({442, 485, 530, 577, 588}, {442, 485, 530, 577, 588}),
}


def read_final(path):
    groups = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "x", "y", "deg"]:
            raise ValueError(f"{path}: expected final header id,x,y,deg")
        for row in reader:
            n_text, square_text = row["id"].split("_", 1)
            n = int(n_text)
            groups.setdefault(n, {})[int(square_text)] = row
    for n, rows in groups.items():
        if set(rows) != set(range(n)):
            raise ValueError(f"{path}: n={n} is incomplete ({len(rows)}/{n})")
    return groups


def value(text):
    return text[1:] if text.startswith("s") else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("h128")
    ap.add_argument("h131")
    ap.add_argument("h132")
    ap.add_argument("output")
    args = ap.parse_args()

    finals = {
        "h128": read_final(args.h128),
        "h131": read_final(args.h131),
        "h132": read_final(args.h132),
    }
    selected = {}
    for label, (expected, chosen) in PR_GROUPS.items():
        actual = set(finals[label])
        if actual != expected:
            raise ValueError(f"{label}: groups {sorted(actual)} != {sorted(expected)}")
        for n in chosen:
            selected[n] = (label, finals[label][n])

    if set(selected) != {442, 485, 530, 577, 588, 634, 695, 861, 926}:
        raise ValueError(f"unexpected PR #72 group set: {sorted(selected)}")

    with open(args.baseline, newline="", encoding="utf-8") as f:
        baseline_rows = list(csv.DictReader(f))
    baseline = {}
    for row in baseline_rows:
        baseline.setdefault(int(row["n"]), []).append(row)
    if set(baseline) != set(range(1, 1001)):
        raise ValueError("baseline must contain exactly n=1..1000")

    out = io.StringIO(newline="")
    writer = csv.DictWriter(
        out, fieldnames=["id", "n", "square_id", "x", "y", "angle"]
    )
    writer.writeheader()
    row_id = 0
    with mp.workdps(100):
        for n in range(1, 1001):
            if n in selected:
                rows = selected[n][1]
                for square_id in range(n):
                    row = rows[square_id]
                    angle = mp.mpf(value(row["deg"])) * mp.pi / 180
                    writer.writerow(
                        {
                            "id": row_id,
                            "n": n,
                            "square_id": square_id,
                            "x": value(row["x"]),
                            "y": value(row["y"]),
                            "angle": mp.nstr(angle, 75),
                        }
                    )
                    row_id += 1
            else:
                for row in sorted(baseline[n], key=lambda r: int(r["square_id"])):
                    writer.writerow(
                        {
                            "id": row_id,
                            "n": n,
                            "square_id": row["square_id"],
                            "x": row["x"],
                            "y": row["y"],
                            "angle": row["angle"],
                        }
                    )
                    row_id += 1

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        f.write(out.getvalue())
    print(
        f"wrote {args.output}: {row_id} rows; overlaid "
        + ", ".join(f"{n}<-{selected[n][0]}" for n in sorted(selected))
    )


if __name__ == "__main__":
    main()
