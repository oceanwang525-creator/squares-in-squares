# Public dataset harvest (Nov 12 2026)

The harvest watcher only scans Kaggle **kernels** and the live feed. Kaggle's
**dataset** index carried many fresh full submissions, several updated the same
day, so this pass fetched and scanned every packing dataset.

## Datasets fetched (anonymous `kagglehub`)

- `cshara/squares-in-squares-current-best`
- `prvsiyan/squares-verified-best-of-1105`
- `jonathan1506/joegun-squares-packings`
- `samanyu1808/sis-final-packings`
- `busyaprime/squares-in-squares-fq-quenched-packings`
- `mdmuqtadirfuad/sis-certified-portfolio`
- `avazaku/squares-contact-lp-cusp-bounds-465-998`
- `az05192000gmailcom/squares-in-squares-l-band-packings`
- `az05192000gmailcom/squares-in-squares-stretched-packings`
- `lostdecimal27/squares-in-squares-bank` (source of the leader curve)
- `pablodiegoo/unit-square-packing-certified-bounds`
- `ruoxuanzhuang/squares-in-squares-two-fault-dp-packings`
- `alexchilton/squares-in-squares-starter-kit`
- `kmmmvs/squares-in-squares-private-packings` (1,838 files, 737 MB)
- `prvsiyan/squares-auto-updater-live`
- `octaviograu/squares-certified-blend-inputs` (60 public submissions incl.
  Aqib Raza's top-LB blend)
- `dariushafshar/squares-in-squares-certified-packings`
- `lightningv08/squares-in-squares-layout-atlas`
- `kmmmvs/squares-in-squares-best-packings`

## Result: negative

`scan_refresh.py` over the submission-shaped CSVs and `scan_new.py` over the
non-CSV witness formats returned **0 candidate n** in every case. The bundled
`submission.csv` in `lostdecimal27` reaches the same public frontier as the
incumbent but not the private per-puzzle bests: e.g. its n=908 is
`30.638533`, n=971 `31.678780`, n=972 `31.692388`, versus our
`30.634071`, `31.664318`, `31.683990`.

No CSV was written and `merged.csv` was unchanged by this pass.
