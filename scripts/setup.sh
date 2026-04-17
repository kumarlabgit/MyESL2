#!/usr/bin/env bash
# setup.sh — Download the latest MyESL2 release binary for this platform
#            and place it (plus its runtime libraries) into ./bin/.
#
# Usage:   bash scripts/setup.sh [--tag vX.Y.Z] [--repo owner/repo]
# Example: bash scripts/setup.sh                       # latest release
#          bash scripts/setup.sh --tag v0.1.2          # specific tag
#
# Supports:  Linux x86_64, macOS arm64 (Apple Silicon).
# Windows:   use scripts/setup.ps1 from a PowerShell prompt instead.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
BIN_DIR="$REPO_ROOT/bin"

# ── Args ──────────────────────────────────────────────────────────
REPO="kumarlabgit/MyESL2"
TAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)  TAG="$2"; shift 2 ;;
        --repo) REPO="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ── Platform detection ────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)
        [[ "$ARCH" == "x86_64" ]] \
            || { echo "ERROR: Linux binaries are x86_64 only (got $ARCH)." >&2; exit 1; }
        BINARY_ASSET="myesl2-linux-x64"
        LIB_ASSETS=("libopenblas.so" "libgfortran.so.5")
        LOCAL_BIN_NAME="myesl2"
        ;;
    Darwin)
        [[ "$ARCH" == "arm64" ]] \
            || { echo "ERROR: macOS binaries are arm64 (Apple Silicon) only (got $ARCH)." >&2; exit 1; }
        BINARY_ASSET="myesl2-macos-arm64"
        LIB_ASSETS=("libopenblas.dylib" "libgfortran.5.dylib" "libquadmath.0.dylib")
        LOCAL_BIN_NAME="myesl2"
        ;;
    *)
        echo "ERROR: Unsupported OS '$OS'. Use scripts/setup.ps1 on Windows." >&2
        exit 1
        ;;
esac

echo "=== MyESL2 binary setup ==="
echo "  Repo:     $REPO"
echo "  Platform: $OS $ARCH"
echo "  Target:   $BIN_DIR/"
echo ""

# ── Resolve release tag ──────────────────────────────────────────
if [[ -z "$TAG" ]]; then
    API_URL="https://api.github.com/repos/${REPO}/releases/latest"
    echo "Resolving latest release tag..."
    TAG=$(curl -fsSL "$API_URL" \
        | grep -oE '"tag_name":[[:space:]]*"[^"]+"' \
        | head -1 \
        | sed -E 's/.*"([^"]+)"$/\1/')
    [[ -n "$TAG" ]] \
        || { echo "ERROR: could not resolve latest release tag from $API_URL" >&2; exit 1; }
fi
echo "  Tag:      $TAG"
echo ""

BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"

# ── Download binary + libs into ./bin/ ───────────────────────────
mkdir -p "$BIN_DIR"

download() {
    local asset="$1" dest="$2" required="${3:-yes}"
    local url="$BASE_URL/$asset"
    echo "  ↓ $asset  →  bin/$(basename "$dest")"
    if ! curl -fsSL -o "$dest" "$url"; then
        rm -f "$dest"
        if [[ "$required" == "yes" ]]; then
            echo "ERROR: download failed: $url" >&2
            exit 1
        else
            echo "    (not found in release — skipping)" >&2
        fi
    fi
}

download "$BINARY_ASSET" "$BIN_DIR/$LOCAL_BIN_NAME"
chmod +x "$BIN_DIR/$LOCAL_BIN_NAME"

# First lib (libopenblas) is required; Fortran runtime libs are optional
# (older releases may not include them).
download "${LIB_ASSETS[0]}" "$BIN_DIR/${LIB_ASSETS[0]}" yes
for lib in "${LIB_ASSETS[@]:1}"; do
    download "$lib" "$BIN_DIR/$lib" optional
done

# ── Copy data_defs.ini next to the binary ────────────────────────
# The binary looks up data_defs.ini in its own directory first; placing
# a copy here lets users run `./bin/myesl2 ...` from any working dir.
DATA_DEFS_SRC="$REPO_ROOT/data_defs.ini"
if [[ -f "$DATA_DEFS_SRC" ]]; then
    cp -f "$DATA_DEFS_SRC" "$BIN_DIR/data_defs.ini"
    echo "  = data_defs.ini  →  bin/data_defs.ini"
else
    echo "WARNING: $DATA_DEFS_SRC not found; myesl2 will fall back to cwd lookup." >&2
fi

echo ""
echo "Done. Installed to $BIN_DIR/:"
ls -la "$BIN_DIR"
echo ""
echo "Try it:"
echo "  ./bin/$LOCAL_BIN_NAME"
