## Makefile — CUDA GEMM Kernel Study (v2, 11 kernels)
CUDA_HOME ?= $(shell dirname $(shell which nvcc))/..
NVCC      := nvcc
CXX       := g++

NVCC_FLAGS := -O3 -std=c++14 \
              --ptxas-options=-v \
              -lineinfo \
              -Xcompiler "-Wall -Wextra" \
              -I../kernels

PASCAL_FLAGS := -arch=sm_61
TURING_FLAGS := -arch=sm_75

LDFLAGS := -lcublas -lcuda

# All kernel sources (kernels 0–7 for Pascal; 0–8 for Turing)
COMMON_SRCS := \
    ../kernels/00_naive.cu \
    ../kernels/01_coalesced.cu \
    ../kernels/02_smem_tiling.cu \
    ../kernels/03_1d_blocktile.cu \
    ../kernels/04_2d_blocktile.cu \
    ../kernels/05_vectorized.cu \
    ../kernels/06_warptiling.cu \
    ../kernels/07_double_buffering.cu

TC_SRCS     := ../kernels/08_tensor_core_wmma.cu

BENCH_SRC := bench_runner.cu

PASCAL_BIN := bench_runner_sm61
TURING_BIN := bench_runner_sm75

.PHONY: all pascal turing clean ptxas_info check_env

all: pascal turing

pascal: $(PASCAL_BIN)

turing: $(TURING_BIN)

# Pascal: no TC kernel
$(PASCAL_BIN): $(BENCH_SRC) $(COMMON_SRCS)
	@echo ""
	@echo "=== Building Pascal (sm_61) — kernels 0..7 + cuBLAS FP32 ==="
	$(NVCC) $(NVCC_FLAGS) $(PASCAL_FLAGS) $(LDFLAGS) \
		-o $@ \
		$(BENCH_SRC) $(COMMON_SRCS) \
		2>&1 | tee ptxas_sm61.log
	@echo "Binary: $@"
	@echo "ptxas log: ptxas_sm61.log"
	@$(MAKE) --no-print-directory _extract_ptxas LOG=ptxas_sm61.log ARCH=sm_61

# Turing: includes TC kernel + cublasGemmEx FP16 path
$(TURING_BIN): $(BENCH_SRC) $(COMMON_SRCS) $(TC_SRCS)
	@echo ""
	@echo "=== Building Turing (sm_75) — kernels 0..9 + cuBLAS FP32/FP16 ==="
	$(NVCC) $(NVCC_FLAGS) $(TURING_FLAGS) $(LDFLAGS) \
		-o $@ \
		$(BENCH_SRC) $(COMMON_SRCS) $(TC_SRCS) \
		2>&1 | tee ptxas_sm75.log
	@echo "Binary: $@"
	@echo "ptxas log: ptxas_sm75.log"
	@$(MAKE) --no-print-directory _extract_ptxas LOG=ptxas_sm75.log ARCH=sm_75

# Extract and display ptxas register/smem summary
_extract_ptxas:
	@echo ""
	@echo "--- ptxas resource usage ($(ARCH)) ---"
	@grep "ptxas info.*Function\|ptxas info.*Used" $(LOG) 2>/dev/null | \
	 awk '/Function/{k=$$NF} /Used/{printf "  %-45s %s\n", k, $$0}' || true
	@echo "-----------------------------------------------"

ptxas_info:
	@$(MAKE) --no-print-directory _extract_ptxas LOG=ptxas_sm61.log ARCH=sm_61 2>/dev/null || true
	@$(MAKE) --no-print-directory _extract_ptxas LOG=ptxas_sm75.log ARCH=sm_75 2>/dev/null || true

# Quick sanity check: run each kernel at tiny size (128³)
check_env:
	@echo "=== Sanity check: 128×128×128 on available GPU ==="
	@GPU_CAP=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.'); \
	 if [ "$$GPU_CAP" -ge "75" ] && [ -f $(TURING_BIN) ]; then \
	   BIN=$(TURING_BIN); \
	 elif [ -f $(PASCAL_BIN) ]; then \
	   BIN=$(PASCAL_BIN); \
	 else echo "No binary found. Run 'make' first."; exit 1; fi; \
	 for kid in 0 1 2 3 4 5 6 7 10; do \
	   echo -n "  Kernel $$kid: "; \
	   $$BIN $$kid 128 128 128 2 5 1 0 2>/dev/null | cut -d, -f2,6,13 || echo "FAILED"; \
	 done

clean:
	rm -f $(PASCAL_BIN) $(TURING_BIN) *.log *.o
