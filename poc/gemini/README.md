# Gemini experimental kernels

Experimental kernels and driver, deliberately separate from the established CUDA loop. They share `cuda-loop/cuda/families.hpp` and the main C++ certifier. No quality improvement is claimed.

Run from the repository root after installing the shared package.

```sh
make -C poc/gemini/cuda all ARCH=sm_80
python poc/gemini/poc/gemini_full_run.py --help
```

Use a separate output directory, never another solver's resumable state.
