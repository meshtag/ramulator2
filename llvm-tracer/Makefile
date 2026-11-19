LLVM_CONFIG ?= /usr/local/bin/llvm-config

CXX      := clang++
CXXFLAGS := $(shell $(LLVM_CONFIG) --cxxflags) -fPIC

BUILD_DIR := build

# ── Targets ────────────────────────────────────────────────────────────

.PHONY: all clean

all: $(BUILD_DIR)/MemTracePass.dylib $(BUILD_DIR)/ComputeTracePass.dylib

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Pass plugin: memory instrumentation (loaded by opt --load-pass-plugin)
$(BUILD_DIR)/MemTracePass.dylib: src/MemTracePass.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -shared -undefined dynamic_lookup -o $@ $<

# Pass plugin: arithmetic instrumentation for SIMDRAM
$(BUILD_DIR)/ComputeTracePass.dylib: src/ComputeTracePass.cpp runtime/simdram_runtime.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -shared -undefined dynamic_lookup -o $@ $<

clean:
	rm -rf $(BUILD_DIR)
