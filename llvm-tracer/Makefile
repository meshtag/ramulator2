LLVM_CONFIG ?= /usr/local/bin/llvm-config

CXX      := clang++
CC       := clang
CXXFLAGS := $(shell $(LLVM_CONFIG) --cxxflags) -fPIC
LDFLAGS  := $(shell $(LLVM_CONFIG) --ldflags)

BUILD_DIR := build

# ── Targets ────────────────────────────────────────────────────────────

.PHONY: all clean

all: $(BUILD_DIR)/MemTracePass.dylib $(BUILD_DIR)/ComputeTracePass.dylib \
     $(BUILD_DIR)/libim_runtime.dylib

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Pass plugin: memory instrumentation (loaded by opt --load-pass-plugin)
$(BUILD_DIR)/MemTracePass.dylib: src/MemTracePass.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -shared -undefined dynamic_lookup -o $@ $<

# Pass plugin: arithmetic instrumentation for SIMDRAM
$(BUILD_DIR)/ComputeTracePass.dylib: src/ComputeTracePass.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -shared -undefined dynamic_lookup -o $@ $<

# IM execution-model runtime (bank-ID / program-ID primitives)
$(BUILD_DIR)/libim_runtime.dylib: runtime/im_runtime.c | $(BUILD_DIR)
	$(CC) -O2 -fPIC -shared -o $@ $<

clean:
	rm -rf $(BUILD_DIR)
