#!/usr/bin/env bash
#
# test_all_pim.sh - Build everything and run all PIM test programs
#
# This script:
#   1. Builds the MemTracePass LLVM plugin (make)
#   2. Builds Ramulator2 (cmake)
#   3. Runs each test through the pipeline:
#      C source -> LLVM IR -> optimize -> instrument -> link w/ pim_runtime -> execute -> Ramulator2
#
# Usage:
#   ./test_all_pim.sh              # build + run all tests
#   ./test_all_pim.sh axpy_pim     # build + run a single test by name (without .c)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
TEST_DIR="$SCRIPT_DIR/test"
WORK_DIR="$SCRIPT_DIR/.test_work"
RAMULATOR2_BUILD_DIR="$SCRIPT_DIR/../build"

PASS_LIB="$BUILD_DIR/MemTracePass.dylib"
PIM_RUNTIME="$SCRIPT_DIR/runtime/pim_runtime.c"
CONFIG="$SCRIPT_DIR/config/hbmpim_config.yaml"

OPT="${OPT:-/usr/local/bin/opt}"
CLANG="${CLANG:-clang}"
RAMULATOR2="${RAMULATOR2:-$RAMULATOR2_BUILD_DIR/ramulator2}"
OPT_LEVEL="${OPT_LEVEL:-O2}"

TESTS=("axpy_pim" "attention_pim")

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BOLD='\033[1m'
NC='\033[0m'

passed=0
failed=0
failed_names=()

# ── Pre-flight checks ────────────────────────────────────────────────
if ! command -v "$CLANG" &>/dev/null; then
    echo -e "${RED}ERROR: clang not found${NC}"
    exit 1
fi

# ── Build MemTracePass (LLVM pass plugin) ─────────────────────────────
echo -e "${BOLD}[build] MemTracePass${NC}"
if ! make -C "$SCRIPT_DIR" -j 2>&1 | tail -3; then
    echo -e "${RED}ERROR: MemTracePass build failed${NC}"
    exit 1
fi

# ── Build Ramulator2 ──────────────────────────────────────────────────
echo -e "${BOLD}[build] Ramulator2${NC}"
mkdir -p "$RAMULATOR2_BUILD_DIR"
if ! (cd "$RAMULATOR2_BUILD_DIR" && cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1 && \
      cmake --build . -j 2>&1 | tail -3); then
    echo -e "${RED}ERROR: Ramulator2 build failed${NC}"
    exit 1
fi
echo ""

if [ ! -x "$RAMULATOR2" ]; then
    echo -e "${YELLOW}WARNING: ramulator2 binary not found at $RAMULATOR2 after build${NC}"
    echo "         Trace generation will run, but simulation will be skipped."
    echo ""
fi

# Filter to a single test if specified
if [ $# -ge 1 ]; then
    TESTS=("$1")
fi

mkdir -p "$WORK_DIR"

echo -e "${BOLD}=======================================${NC}"
echo -e "${BOLD}  PIM Test Suite${NC}"
echo -e "${BOLD}=======================================${NC}"
echo ""

for test_name in "${TESTS[@]}"; do
    SRC="$TEST_DIR/${test_name}.c"
    if [ ! -f "$SRC" ]; then
        echo -e "${RED}SKIP${NC} $test_name — source not found: $SRC"
        failed=$((failed + 1))
        failed_names+=("$test_name")
        continue
    fi

    echo -e "${BOLD}── $test_name ──${NC}"

    WD="$WORK_DIR/$test_name"
    mkdir -p "$WD"

    LL="$WD/${test_name}.ll"
    OPT_LL="$WD/${test_name}.opt.ll"
    INST_LL="$WD/${test_name}.instrumented.ll"
    EXE="$WD/${test_name}.pim.exe"
    TRACE="$WD/pim_trace.txt"
    TMP_CONFIG="$WD/config.yaml"
    STATS="$WD/ramulator_stats.txt"
    ok=true

    # Step 1: Compile to LLVM IR
    echo -n "  [1/6] Compile to IR ... "
    if $CLANG -S -emit-llvm -O0 -Xclang -disable-O0-optnone \
              -I"$SCRIPT_DIR/runtime" -o "$LL" "$SRC" 2>"$WD/compile.log"; then
        echo "ok"
    else
        echo -e "${RED}FAIL${NC}"
        cat "$WD/compile.log"
        ok=false
    fi

    # Step 2: Optimize
    if $ok; then
        echo -n "  [2/6] Optimize (-$OPT_LEVEL) ... "
        if [ "$OPT_LEVEL" = "none" ]; then
            cp "$LL" "$OPT_LL"
            echo "skipped"
        elif $OPT -$OPT_LEVEL -S "$LL" -o "$OPT_LL" 2>"$WD/opt.log"; then
            echo "ok"
        else
            echo -e "${RED}FAIL${NC}"
            cat "$WD/opt.log"
            ok=false
        fi
    fi

    # Step 3: Instrument
    if $ok; then
        echo -n "  [3/6] Instrument ... "
        if $OPT --load-pass-plugin="$PASS_LIB" --passes="mem-trace" \
                -S "$OPT_LL" -o "$INST_LL" 2>"$WD/instrument.log"; then
            echo "ok"
        else
            echo -e "${RED}FAIL${NC}"
            cat "$WD/instrument.log"
            ok=false
        fi
    fi

    # Step 4: Link with PIM runtime
    if $ok; then
        echo -n "  [4/6] Link + compile ... "
        if $CLANG -O0 "$INST_LL" "$PIM_RUNTIME" -lm -o "$EXE" 2>"$WD/link.log"; then
            echo "ok"
        else
            echo -e "${RED}FAIL${NC}"
            cat "$WD/link.log"
            ok=false
        fi
    fi

    # Step 5: Execute to generate trace
    if $ok; then
        echo -n "  [5/6] Generate trace ... "
        if (cd "$WD" && "$EXE") 2>"$WD/exec.log"; then
            lines=$(wc -l < "$TRACE" | tr -d ' ')
            echo "ok ($lines trace lines)"
            if [ "$lines" -eq 0 ]; then
                echo -e "  ${RED}ERROR: trace is empty${NC}"
                ok=false
            fi
        else
            echo -e "${RED}FAIL${NC}"
            cat "$WD/exec.log"
            ok=false
        fi
    fi

    # Step 6: Ramulator2 simulation
    if $ok && [ -x "$RAMULATOR2" ]; then
        echo -n "  [6/6] Ramulator2 simulate ... "
        sed "s|path:.*|path: $TRACE|" "$CONFIG" > "$TMP_CONFIG"
        if "$RAMULATOR2" --config_file "$TMP_CONFIG" 2>"$WD/ramulator.log" 1>"$STATS"; then
            cycles=$(grep -o 'memory_system_cycles.*' "$STATS" 2>/dev/null | head -1 || echo "")
            reads=$(grep -o 'total_num_read_requests.*' "$STATS" 2>/dev/null | head -1 || echo "")
            bkreads=$(grep -o 'total_num_bkread_requests.*' "$STATS" 2>/dev/null | head -1 || echo "")
            bkwrites=$(grep -o 'total_num_bkwrite_requests.*' "$STATS" 2>/dev/null | head -1 || echo "")
            echo "ok"
            echo "       $cycles"
            echo "       $reads"
            echo "       $bkreads"
            echo "       $bkwrites"
        else
            echo -e "${RED}FAIL${NC}"
            cat "$WD/ramulator.log"
            ok=false
        fi
    elif $ok; then
        echo "  [6/6] Ramulator2 simulate ... skipped (binary not found)"
    fi

    # Result
    if $ok; then
        echo -e "  ${GREEN}PASSED${NC}"
        passed=$((passed + 1))
    else
        echo -e "  ${RED}FAILED${NC}"
        failed=$((failed + 1))
        failed_names+=("$test_name")
    fi
    echo ""
done

# ── Summary ──────────────────────────────────────────────────────────
echo -e "${BOLD}=======================================${NC}"
echo -e "${BOLD}  Results: ${GREEN}$passed passed${NC}, ${RED}$failed failed${NC}"
if [ ${#failed_names[@]} -gt 0 ]; then
    echo -e "  Failed: ${RED}${failed_names[*]}${NC}"
fi
echo -e "${BOLD}=======================================${NC}"

exit $failed
