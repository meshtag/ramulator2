#!/usr/bin/env bash
#
# run_pim_trace.sh - C source -> LLVM IR -> PIM trace -> Ramulator2-PIM pipeline
#
# Usage:
#   ./run_pim_trace.sh <input.c>  [ramulator_pim_config.yaml]
#
# This script:
#   1. Compiles a C source (with PIM annotations) to LLVM IR
#   2. Optimizes the IR
#   3. Instruments loads/stores via MemTracePass
#   4. Links with pim_runtime and executes to produce a PIM trace
#   5. Runs Ramulator2 with HBM3_PIM model on the trace
#
# Environment variables:
#   OPT_LEVEL       "O1", "O2", "O3", or "none" (default: "O2")
#   PIM_TRACE_OUT   Output PIM trace file (default: pim_trace.txt)
#   RAMULATOR2      Path to ramulator2 binary

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

PASS_LIB="$BUILD_DIR/MemTracePass.dylib"
PIM_RUNTIME="$SCRIPT_DIR/runtime/pim_runtime.c"

OPT="${OPT:-/usr/local/bin/opt}"
CLANG="${CLANG:-clang}"
RAMULATOR2="${RAMULATOR2:-$SCRIPT_DIR/../build/ramulator2}"
OPT_LEVEL="${OPT_LEVEL:-O2}"

PIM_TRACE_OUT="${PIM_TRACE_OUT:-pim_trace.txt}"
CONFIG="${2:-$SCRIPT_DIR/config/hbmpim_config.yaml}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.c> [hbmpim_config.yaml]"
    exit 1
fi

INPUT="$1"

# ── Step 1: Compile C to LLVM IR ─────────────────────────────────────
LL_FILE="${INPUT%.*}.ll"
echo "[1/6] Compiling $INPUT -> $LL_FILE"
$CLANG -S -emit-llvm -O0 -Xclang -disable-O0-optnone \
       -I"$SCRIPT_DIR/runtime" -o "$LL_FILE" "$INPUT"

# ── Step 2: Optimize IR ──────────────────────────────────────────────
OPTIMIZED="${INPUT%.*}.opt.ll"
if [ "$OPT_LEVEL" = "none" ]; then
    echo "[2/6] Skipping optimization"
    cp "$LL_FILE" "$OPTIMIZED"
else
    echo "[2/6] Optimizing IR (-${OPT_LEVEL})"
    $OPT -${OPT_LEVEL} -S "$LL_FILE" -o "$OPTIMIZED"
fi

# ── Step 3: Instrument loads/stores ──────────────────────────────────
INSTRUMENTED="${INPUT%.*}.instrumented.ll"
echo "[3/6] Instrumenting -> $INSTRUMENTED"
$OPT --load-pass-plugin="$PASS_LIB" --passes="mem-trace" \
     -S "$OPTIMIZED" -o "$INSTRUMENTED"

# ── Step 4: Compile and link with PIM runtime ────────────────────────
EXE="${INPUT%.*}.pim.exe"
echo "[4/6] Compiling instrumented IR + PIM runtime -> $EXE"
$CLANG -O0 "$INSTRUMENTED" "$PIM_RUNTIME" -lm -o "$EXE"

# ── Step 5: Execute to generate PIM trace ────────────────────────────
echo "[5/6] Executing (PIM trace -> $PIM_TRACE_OUT)"
export PIM_TRACE_OUT
"$EXE"

TRACE_LINES=$(wc -l < "$PIM_TRACE_OUT" | tr -d ' ')
echo "     PIM trace generated: $TRACE_LINES operations"

# ── Step 6: Run Ramulator2-PIM ───────────────────────────────────────
if [ -x "$RAMULATOR2" ]; then
    # Create a config with the correct trace path
    TMP_CONFIG="$SCRIPT_DIR/.pim_config_tmp.yaml"
    sed "s|path:.*|path: $PIM_TRACE_OUT|" "$CONFIG" > "$TMP_CONFIG"

    echo "[6/6] Running Ramulator2-PIM with HBM3_PIM"
    "$RAMULATOR2" --config_file "$TMP_CONFIG"
    rm -f "$TMP_CONFIG"
else
    echo "[6/6] Skipped Ramulator2 (binary not found at $RAMULATOR2)"
    echo "     Set RAMULATOR2=/path/to/ramulator2 or build ramulator2 first"
fi

echo ""
echo "Done. PIM trace: $PIM_TRACE_OUT"
