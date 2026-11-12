# Measured results

## Portfolio

| check | score | accepted | missing | rejected |
|---|---:|---:|---:|---:|
| literal decimal strings | **22.173995906838663** | 1000 | 0 | 0 |
| actual pandas `read_csv` / `to_csv` transport | **22.173995906838663** | 1000 | 0 | 0 |

Previous incumbent: `22.17529852480167`.
Certified reduction: `0.001302617963011434`.
Strict per-`n` wins: **114**.

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `e5ef17f711ea2e674eb03b006dfa6b3c018fd3ac5e053cf7166896b0715cda92`.

Exact command and output:

```text
$ . .venv/bin/activate && python /tmp/opencode/score_both.py \
    /tmp/opencode/merged_loop10.csv
rows 500500 n 1000
LITERAL {"score": 22.173995906838663, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
literal seconds 123.66382241249084
PANDAS {"score": 22.173995906838663, "max_n": 1000, "accepted": 1000, "missing": 0, "rejected": 0}
pandas seconds 115.2159378528595
```

The transport run used actual pandas **3.1.0**. Both runs use the unchanged
official v20 metric.

## Largest wins

| n | score gain | old L | new certified L | source |
|---:|---:|---:|---:|---|
| 642 | `3.4316e-4` | `25.826248643` | `25.817553669` | `jazivxt/boxed-in-sub` |
| 504 | `1.6570e-4` | `22.915135075` | `22.911415009` | `jazivxt/boxed-in-sub` |
| 862 | `1.2158e-4` | `29.830859962` | `29.827290504` | `jonathan1506/50-top-2-6-22-1783` |
| 922 | `1.1755e-4` | `30.830859962` | `30.827290504` | `jonathan1506/50-top-2-6-22-1783` |
| 507 | `7.0067e-5` | `22.961862385` | `22.960284706` | `jonathan1506/50-top-2-6-22-1783` |

Full precision for all 114 is in `docs/wins.json`; the complete per-`n` reports
are `docs/final-literal-official.json` and `docs/final-pandas-official.json`.

## Method

1. Fetch current CSV outputs of the four re-run kernels via the anonymous
   output API.
2. Prefilter each `n` by floating bounding box below the incumbent.
3. Certify the literal strings and the actual pandas round trip; require both
   strictly below the incumbent.
4. Replace only those groups, then re-certify all 1000 groups from scratch.

No dilation was needed for any of the 114 wins; every accepted row certifies
identically in both transports.
