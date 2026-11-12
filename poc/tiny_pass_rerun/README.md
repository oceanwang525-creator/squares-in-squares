# Tiny pass re-run on the current portfolio

The team's `poc/public_refresh_nov12/poc/tiny_pass.py` runs two cheap certified
moves:

1. **Deletion:** take the packing for `n+k` (k=1..5) and delete `k` squares,
   keeping all rows; it is valid for `n` at the same side.
2. **Border wrap:** take the packing for `n0`, add `t` rows/columns of upright
   squares in the east/north strips (side grows to `L0 + t + eps`), then delete
   down to the target `n`.

Their original run used it on the November 12 intermediate file and added
`2.49e-10`. The tracked file has changed at more than 200 `n` since, so the
pass was re-run against the current `22.173995906838663` incumbent.

## Measured result

**8** strict wins, all certified in both literal and pandas transport:

| n | source | old L | new certified L | dL |
|---:|---|---:|---:|---:|
| 904 | `wrap-845-t1` | `30.563136023802043` | `30.563136023765352` | `3.669e-11` |
| 442 | `delete-from-443` | `21.535533905932896` | `21.535533905932831` | `6.455e-14` |
| 732 | `wrap-679-t1` | `27.535533905932977` | `27.535533905932904` | `7.306e-14` |
| 678 | `delete-from-679` | `26.535533905932888` | `26.535533905932853` | `3.500e-14` |
| 577 | `delete-from-578` | `24.535533905932872` | `24.535533905932851` | `2.185e-14` |
| 964 | `wrap-679-t5` | `31.535533905933007` | `31.535533905932983` | `2.305e-14` |
| 627 | `delete-from-628` | `25.535533905932880` | `25.535533905932877` | `3.250e-15` |
| 492 | `delete-from-493` | `22.707106781186654` | `22.707106781186654` | `1.925e-17` |

The merged portfolio scores **22.173995906837433** on both transports, with
**1000/1000 accepted, 0 missing, 0 rejected**. Certified reduction:
`1.2292409126146337e-12`.

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `308fc61effc296bebefa57617d095b5b81055e425541fbb25d9b55296c25d302`.

## Negative checks

- Wrap candidates at n=442, 530, 577, 678, 786, 843, 902, 963 all returned
  `no gain`: the wrapped side is `1e-13` above the incumbent, so no other wrap
  is usable.
- Deletion over `n+1..n+5` wins only at the five flat pairs
  (442/443, 577/578, 627/628, 678/679, 492/493), and only by
  `1e-17`..`1e-13`.

## Reproduce

```sh
. .venv/bin/activate
cd poc/public_refresh_nov12/poc
python tiny_pass.py \
  /mnt/e/squares-in-squares/poc/pell_corrugation/data/border-scan/merged.csv \
  /mnt/e/squares-in-squares/poc/pell_corrugation/data/border-scan/merged-literal-official.json \
  /mnt/e/squares-in-squares/poc/pell_corrugation/data/border-scan/merged-pandas-official.json \
  /tmp/tiny_wins.json
```

Then merge the eight rows and re-certify the full file. The pass is dust-level
by construction; it is re-run because it is nearly free and every strict win is
legitimate. `docs/wins.json` has the merge gate's output.
