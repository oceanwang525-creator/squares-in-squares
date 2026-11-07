NVCC ?= nvcc
CXX ?= c++
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d .)
ARCH ?= $(if $(DETECTED_CC),sm_$(DETECTED_CC),sm_89)
NVCCFLAGS ?= -O3 -std=c++17
CXXFLAGS ?= -O3 -std=c++17 -pthread
PROGRAMS = gemini_packing gemini_seeded
.PHONY: all clean
all: $(PROGRAMS)
gemini_packing gemini_seeded: %: %.cu ../../cuda-loop/cuda/families.hpp
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
clean:
	rm -f $(PROGRAMS)
