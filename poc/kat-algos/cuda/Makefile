NVCC ?= nvcc
CXX ?= c++
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d .)
ARCH ?= $(if $(DETECTED_CC),sm_$(DETECTED_CC),sm_89)
NVCCFLAGS ?= -O3 -std=c++17
CXXFLAGS ?= -O3 -std=c++17 -pthread
PROGRAMS = algos algos_test
.PHONY: all check clean
all: $(PROGRAMS)
algos: algos.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
algos_test: tests/algos_test.cu algos.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
check: all
	./algos_test
clean:
	rm -f $(PROGRAMS)
