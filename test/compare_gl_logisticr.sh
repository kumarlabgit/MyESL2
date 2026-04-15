#!/bin/bash
#
# Bit-for-bit comparison of MyESL2's gl_logisticr port against the MyESL
# reference binary. Runs both on the same tiny dataset at the same lambda1
# value and diffs the resulting XML outputs.
#
# Requires:
#   - /claude/MyESL/bin/gl_logisticr (statically linked reference build)
#   - $MYESL2_ROOT/build/bin/gl_logisticr_compare (built with
#     -DBUILD_SOLVER_TESTS=ON)
#
# Usage:
#   ./test/compare_gl_logisticr.sh [lambda1=0.5]
#
set -e

LAMBDA1="${1:-0.5}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/gl_logisticr"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MYESL_BIN="/claude/MyESL/bin/gl_logisticr"
MYESL2_BIN="$ROOT_DIR/build/bin/gl_logisticr_compare"

if [ ! -x "$MYESL_BIN" ]; then
    echo "ERROR: MyESL reference binary not found or not executable: $MYESL_BIN"
    exit 2
fi
if [ ! -x "$MYESL2_BIN" ]; then
    echo "ERROR: MyESL2 comparison binary not found: $MYESL2_BIN"
    echo "Build with: cmake -B build -DBUILD_SOLVER_TESTS=ON && cmake --build build"
    exit 2
fi

cd "$DATA_DIR"

# Clean stale outputs
rm -f myesl_out.xml myesl2_out.xml

# Run MyESL reference
"$MYESL_BIN" \
    -f features.csv \
    -n groups.csv \
    -r response.csv \
    -w myesl_out \
    --lambda1 "$LAMBDA1" \
    --lambda2 0.0 > /dev/null
# MyESL writes to myesl_out.xml (the .xml extension is appended by the binary)

# Run MyESL2 port
"$MYESL2_BIN" features.csv groups.csv response.csv myesl2_out.xml "$LAMBDA1" > /dev/null

# Byte-identical diff
if diff -q myesl_out.xml myesl2_out.xml > /dev/null; then
    echo "PASS: gl_logisticr outputs are byte-identical (lambda1=$LAMBDA1)"
    exit 0
else
    echo "FAIL: gl_logisticr outputs differ at lambda1=$LAMBDA1"
    echo
    diff myesl_out.xml myesl2_out.xml | head -40
    exit 1
fi
