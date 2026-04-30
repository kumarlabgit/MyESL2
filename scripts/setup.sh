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
        # The standard linux-x64 artifact is built on ubuntu-latest, which
        # currently ships glibc 2.35. Hosts with older glibc (e.g. RHEL 7's
        # 2.17) need the portable artifact built inside manylinux2014.
        STANDARD_GLIBC_MIN="2.35"
        HOST_GLIBC=$(ldd --version 2>/dev/null | head -1 \
            | grep -oE '[0-9]+\.[0-9]+' | tail -1)
        if [[ -z "$HOST_GLIBC" ]]; then
            echo "WARNING: could not detect host glibc version; assuming portable build is needed." >&2
            USE_PORTABLE=1
        elif [[ "$(printf '%s\n%s\n' "$STANDARD_GLIBC_MIN" "$HOST_GLIBC" | sort -V | head -1)" != "$STANDARD_GLIBC_MIN" ]]; then
            echo "  Host glibc $HOST_GLIBC < $STANDARD_GLIBC_MIN; using portable artifact."
            USE_PORTABLE=1
        else
            USE_PORTABLE=0
        fi
        if [[ "$USE_PORTABLE" == "1" ]]; then
            ARCHIVE_ASSET="myesl2-linux-x64-portable.tar.gz"
        else
            ARCHIVE_ASSET="myesl2-linux-x64.tar.gz"
        fi
        LOCAL_BIN_NAME="myesl2"
        ;;
    Darwin)
        [[ "$ARCH" == "arm64" ]] \
            || { echo "ERROR: macOS binaries are arm64 (Apple Silicon) only (got $ARCH)." >&2; exit 1; }
        ARCHIVE_ASSET="myesl2-macos-arm64.tar.gz"
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

# ── Download archive and extract into ./bin/ ─────────────────────
mkdir -p "$BIN_DIR"

ARCHIVE_URL="$BASE_URL/$ARCHIVE_ASSET"
TMP_ARCHIVE=$(mktemp -t myesl2.XXXXXX.tar.gz)
trap 'rm -f "$TMP_ARCHIVE"' EXIT

echo "  ↓ $ARCHIVE_ASSET"
if ! curl -fsSL -o "$TMP_ARCHIVE" "$ARCHIVE_URL"; then
    echo "ERROR: download failed: $ARCHIVE_URL" >&2
    exit 1
fi

echo "  ⇒ extracting into bin/"
tar -xzf "$TMP_ARCHIVE" -C "$BIN_DIR"
chmod +x "$BIN_DIR/$LOCAL_BIN_NAME"

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
