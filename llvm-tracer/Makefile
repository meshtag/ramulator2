LLVM_CONFIG ?= /usr/local/bin/llvm-config

CXX      := clang++
CC       := clang
CXXFLAGS := $(shell $(LLVM_CONFIG) --cxxflags) -fPIC
LDFLAGS  := $(shell $(LLVM_CONFIG) --ldflags)

BUILD_DIR := build

# ── Targets ────────────────────────────────────────────────────────────

.PHONY: all clean

all: $(BUILD_DIR)/MemTracePass.dylib $(BUILD_DIR)/ComputeTracePass.dylib \
     $(BUILD_DIR)/libtrace_runtime.dylib $(BUILD_DIR)/libim_runtime.dylib \
     $(BUILD_DIR)/libsimdram_runtime.dylib

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Pass plugin: memory instrumentation (loaded by opt --load-pass-plugin)
$(BUILD_DIR)/MemTracePass.dylib: src/MemTracePass.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -shared -undefined dynamic_lookup -o $@ $<

# Pass plugin: arithmetic instrumentation for SIMDRAM
$(BUILD_DIR)/ComputeTracePass.dylib: src/ComputeTracePass.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -shared -undefined dynamic_lookup -o $@ $<

# Runtime library (linked into instrumented program)
$(BUILD_DIR)/libtrace_runtime.dylib: runtime/trace_runtime.c | $(BUILD_DIR)
	$(CC) -O2 -fPIC -shared -o $@ $<

# IM execution-model runtime (bank-ID / program-ID primitives)
$(BUILD_DIR)/libim_runtime.dylib: runtime/im_runtime.c | $(BUILD_DIR)
	$(CC) -O2 -fPIC -shared -o $@ $<

# SIMDRAM runtime (bit-serial memory + MAJ gate computation traces).
# Includes im_runtime.c (for __pim_get_program_id*) and im_addr_dedup.c
# (for the operand-load coalescing dedup). The actual kernel-link step
# in pass_ablation.py / run_triton_im_ramulator2_e2e.py recompiles these
# alongside the instrumented IR; this rule produces a standalone dylib
# that's complete enough to load on its own for sanity testing.
$(BUILD_DIR)/libsimdram_runtime.dylib: runtime/simdram_runtime.c runtime/simdram_runtime.h \
                                       runtime/im_addr_dedup.c runtime/im_addr_dedup.h \
                                       runtime/im_runtime.c runtime/im_runtime.h | $(BUILD_DIR)
	$(CC) -O2 -fPIC -shared -o $@ \
	    runtime/simdram_runtime.c runtime/im_addr_dedup.c runtime/im_runtime.c

clean:
	rm -rf $(BUILD_DIR)
