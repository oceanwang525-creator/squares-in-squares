# Harvest watch (Nov 12 2026)

One command for a full harvest cycle: list the competition kernels, fetch CSV
outputs of every kernel that ran since the saved timestamp, poll the live
square-packing feed for sides below the incumbent, and report whether the team
pushed a new tracked file to diff per `n`.

## Measured result

The watcher itself found no win in the first cycle after Loop 10:

- kernels newer than `2026-11-12T09:33:56`: 1 (`saamhm`), 0 candidates;
- live feed (`at` 2026-11-12T01:38-08:00): 0 candidates above a `1e-12`
  threshold;
- `origin/main`: no new team commit.

## Repository and dataset discovery

GitHub search and Kaggle dataset search around the same time also surfaced
these previously unscanned sources; all were cloned/downloaded and certified
against the `22.173995906838663` incumbent:

| source | result |
|---|---|
| `BuddenPing/square-packing-certificates` (n=105, 146, 153, 207, 261, 272, 302) | 0 candidates |
| `lollipoll/couzo-five-exact-certificates` | 0 candidates |
| `SidG2k1/square-packing-refinements` (14 exact refinements, ~`1e-13`) | 0 candidates |
| `ebdeleeuw/square-packing-n70` | 0 candidates |
| 9 Kaggle-competition repos (`ialirezaesrafili`, `denis-42ds`, `Arnab-Chakraborti`, `1zuki`, `tiagocabo`, `fab64443`, `EvanOman`, `Pdbz199`, ...) | 0 (the only hits were `spheres.json` false positives) |
| `samanyu1808/sis-final-packings` dataset | 0 candidates |
| `saamhm` refreshed submission | 0 candidates |

The exact-refinement repos improve *published* sides by `~1e-13`, but the
tracked portfolio already matches or beats every one of those references, so
none is a strict win.

## Reproduce

```sh
. .venv/bin/activate
python poc/harvest_watch/poc/cycle.py \
  --state /tmp/sis_harvest/state.json \
  --out /tmp/sis_harvest/kout \
  --report poc/pell_corrugation/data/border-scan/merged-literal-official.json
```

Then gate and merge the staged CSVs:

```sh
python poc/kernel_refresh_morning/poc/scan_and_merge.py \
  --inputs /tmp/sis_harvest/kout \
  --incumbent poc/pell_corrugation/data/border-scan/merged.csv \
  --report poc/pell_corrugation/data/border-scan/merged-literal-official.json \
  --candidates /tmp/harvest_cands.json \
  --output /tmp/merged_harvest.csv --wins /tmp/harvest_wins.json
```

Data only; no third-party code is executed and no Kaggle submission is made.
Public authors retain credit.
