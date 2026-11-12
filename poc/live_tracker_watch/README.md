# Live tracker watch (Nov 12 2026)

The `s269-upper-bound-1688973007` repo points at a **live** public tracker:
`https://dogaozden.com/math/live/square-packing/`. Its JSON feed
(`data/feed.json`) publishes the lab's current per-`n` results, a 5-decimal
drawing for 292 puzzles, and a record history. The same account
(`kenaznia-doga`) publishes one exact-rational witness repository per new upper
bound, named `s<n>-upper-bound[<run id>]`.

This POC polls the feed, flags any `n` whose side is below the tracked
incumbent, and clones/certifies the matching witness repos.

## Measured result

The feed is valid and live (`at` 2026-11-12T00:55-08:00). Scanning every
numeric side in `results`, `drawings`, `events`, `record_history`, `targets`
and `lower_bounds` against the `22.17589232224942` incumbent leaves exactly one
candidate, at `n=269`:

| n | old L | new certified L | gain |
|---:|---:|---:|---:|
| 269 | `16.89946907585504652716309549503674454576565242778760` | `16.88973007359563669924200100534631805369196464263972` | `5.9379744775e-4` |

Source: `kenaznia-doga/s269-upper-bound-1688973007`, exact-rational witness
`n-269-decimal-pose.yaml` (the JSON `n269-witness.json` gives the same rows).
It is `0.009739` below the previous tracked row and about `5.5e-9` below the
Kaggle per-puzzle leaderboard entry for n=269 (`16.889730079125`, elegans).

The portfolio scores **22.17529852480167** on literal decimal strings and
**22.175298524801672** after actual pandas 3.1.0 transport, with **1000/1000
accepted, 0 missing, 0 rejected**. Previous incumbent:
`22.17589232224942`. Certified reduction: `5.937974477501639e-4`.

Merged file: `poc/pell_corrugation/data/border-scan/merged.csv`
SHA256: `9195b04515268df0cd53143dd1442fdd63e7f32c0d3a170100665525f353926b`.

## Negative checks

- The other 291 drawings are 5-decimal poses, so their apparent `~4e-6` gaps
  are rounding; the feed's own declared `side` for them is at or above the
  incumbent. Only n=269 survives.
- The two earlier `s269` repos (`s269-upper-bound`,
  `s269-upper-bound-16899469076`) are the previously imported
  `16.900994579951325` and `16.899469075855046` values, both superseded here.
- `s179-upper-bound-13880960371` publishes `13.880960371558225`, which is
  `3e-12` **worse** than our tracked `13.880960371555588`.
- `s237-upper-bound` publishes the same `15.893805580572574` we already track.

## Reproduce

```sh
. .venv/bin/activate
python poc/live_tracker_watch/poc/poll_feed.py \
  --report poc/pell_corrugation/data/border-scan/merged-literal-official.json \
  --out /tmp/live_candidates.json
python poc/live_tracker_watch/poc/fetch_bounds_repos.py \
  --out /tmp/sis_bounds --inventory /tmp/sis_bounds_inventory.json
python poc/external_bank_expansion/poc/scan_and_merge.py \
  --repos /tmp/sis_bounds \
  --incumbent poc/pell_corrugation/data/border-scan/merged.csv \
  --report poc/pell_corrugation/data/border-scan/merged-literal-official.json \
  --candidates /tmp/bounds_cands.json \
  --output /tmp/merged_bounds.csv --wins /tmp/bounds_wins.json
```

Data only; no third-party code is executed. Public authors retain credit. No
Kaggle submission was made.
