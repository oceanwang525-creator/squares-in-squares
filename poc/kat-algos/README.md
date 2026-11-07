# Kat algorithms

Standalone fp64 compression, basin hopping and portfolio polishing. Separate from the main loop; benchmarks compare certified geometry under equal wall budgets.

Run from the repository root after installing the shared package.

```sh
make -C poc/kat-algos/cuda all ARCH=sm_80
python poc/kat-algos/poc/algos_bench.py --help
```

[Algorithm and benchmark notes](docs/ALGOS.md). Benchmarks also need the main loop packing and certifier executables.
