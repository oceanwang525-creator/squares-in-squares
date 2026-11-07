# Main CUDA loop

The original packing and seeded kernels, shared C++ certifier/family enumerator and resumable search loop. Host Python is in `poc/`; compiled executables are built beside their sources in `cuda/`.

Run from the repository root after installing the shared package.

```sh
make -C poc/cuda-loop/cuda certifier families_cli
python poc/cuda-loop/poc/endless.py --help
```

[Host/kernel guide](docs/README.md), [kernel brief](docs/KERNEL_BRIEF.md), [historical reports](docs/results/).
