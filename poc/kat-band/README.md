# Kat band / FTMC

Thick-band geometry, FTMC moves, ruin/rebuild, Gensane and portable pilot bundles. Python and flat bundle sources remain together under `poc/`; strict C++ post-run certification is under `cuda/`. PyTorch is required for relaxation tests.

Run from the repository root after installing the shared package.

```sh
python poc/kat-band/poc/seed_cli.py --help
```

[Band research notes](docs/README.md). Bundle creation uses `python poc/kat-band/poc/ftmc_bundle.py SOURCE_DIR bundle.tar.gz`; supply the run-specific `baseline.json` in SOURCE_DIR. No baseline is fabricated.
