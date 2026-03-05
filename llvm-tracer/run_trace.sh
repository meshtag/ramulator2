#!/usr/bin/env bash
#
# run_trace.sh — LLVM IR → Ramulator2 memory trace pipeline
#
# Usage:
#   ./run_trace.sh <input.ll or input.c>  [ramulator_config.yaml]
#
# If a .c file is given, it is first compiled to LLVM IR.
# The script optimizes the IR (to eliminate stack spills), instruments it
# to capture memory accesses, executes it through a cache-filtered runtime,
# and optionally runs Ramulator2 on the resulting DRAM-only trace.
#
# Environment variables:
#   OPT_LEVEL       LLVM optimization level: "O1", "O2", "O3", or "none"
#                   (default: "O2")
#   CACHE_ENABLE    "1" (default) to filter through cache model, "0" for raw trace
#   CACHE_SIZE_KB   Cache size in KB    (default: 1024 = 1 MB)
#   CACHE_ASSOC     Associativity       (default: 16)
#   CACHE_LINE_BYTES Line size in bytes (default: 64)
#   RAMULATOR2      Path to ramulator2 binary (skip simulation if unset)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

PASS_LIB="$BUILD_DIR/MemTracePass.dylib"
RUNTIME_LIB="$BUILD_DIR/libtrace_runtime.dylib"

OPT="${OPT:-/usr/local/bin/opt}"
CLANG="${CLANG:-clang}"
RAMULATOR2="${RAMULATOR2:-}"
OPT_LEVEL="${OPT_LEVEL:-O2}"

TRACE_OUTPUT="${TRACE_OUTPUT:-ramulator_trace.txt}"
CONFIG="${2:-$SCRIPT_DIR/ramulator_config.yaml}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.ll | input.c> [ramulator_config.yaml]"
    exit 1
fi

INPUT="$1"
EXT="${INPUT##*.}"

# ── Step 0: Compile .c to .ll if needed ──────────────────────────────
if [ "$EXT" = "c" ] || [ "$EXT" = "cpp" ]; then
    LL_FILE="${INPUT%.*}.ll"
    echo "[1/5] Compiling $INPUT → $LL_FILE"
    $CLANG -S -emit-llvm -O0 -Xclang -disable-O0-optnone -o "$LL_FILE" "$INPUT"
    INPUT="$LL_FILE"
else
    echo "[1/5] Using provided IR: $INPUT"
fi

# ── Step 1: Optimize IR ──────────────────────────────────────────────
OPTIMIZED="${INPUT%.ll}.opt.ll"
if [ "$OPT_LEVEL" = "none" ]; then
    echo "[2/5] Skipping IR optimization (OPT_LEVEL=none)"
    cp "$INPUT" "$OPTIMIZED"
else
    echo "[2/5] Optimizing IR (-${OPT_LEVEL})"
    $OPT -${OPT_LEVEL} -S "$INPUT" -o "$OPTIMIZED"
fi

# ── Step 2: Instrument the optimized IR ──────────────────────────────
INSTRUMENTED="${INPUT%.ll}.instrumented.ll"
echo "[3/5] Instrumenting → $INSTRUMENTED"
$OPT --load-pass-plugin="$PASS_LIB" --passes="mem-trace" \
     -S "$OPTIMIZED" -o "$INSTRUMENTED"

# ── Step 3: Execute and produce trace ────────────────────────────────
echo "[4/5] Executing instrumented IR (trace → $TRACE_OUTPUT)"
export RAMULATOR_TRACE_OUT="$TRACE_OUTPUT"
$CLANG -O0 "$INSTRUMENTED" "$RUNTIME_LIB" -o "${INSTRUMENTED%.ll}.exe"
"./${INSTRUMENTED%.ll}.exe"

TRACE_LINES=$(wc -l < "$TRACE_OUTPUT" | tr -d ' ')
echo "     Trace generated: $TRACE_LINES DRAM requests"

# ── Step 4: Run Ramulator2 (if binary is available) ──────────────────
if [ -n "$RAMULATOR2" ] && [ -x "$RAMULATOR2" ]; then
    echo "[5/5] Running Ramulator2 with config $CONFIG"
    $RAMULATOR2 --config_file "$CONFIG"
else
    echo "[5/5] Skipped Ramulator2 (set RAMULATOR2=/path/to/ramulator2 to run)"
    echo "     To run manually:"
    echo "       ramulator2 --config_file $CONFIG"
fi

echo ""
echo "Done. Trace file: $TRACE_OUTPUT"
