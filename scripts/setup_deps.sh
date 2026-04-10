#!/usr/bin/env bash
# setup_deps.sh — Download and install pinned dependency versions.
# Run from the project root: bash scripts/setup_deps.sh
#
# This script downloads the same Armadillo, OpenBLAS, and LAPACK versions
# used by CI, ensuring local builds match CI exactly.
#
# On Windows: downloads prebuilt binaries + Armadillo headers into include/
# On Linux/macOS: uses vcpkg to build OpenBLAS/LAPACK from source, downloads
#                 Armadillo headers into include/

set -euo pipefail
cd "$(dirname "$0")/.."

# Load pinned versions
source deps.env

echo "=== MyESL2 dependency setup ==="
echo "  Armadillo: $ARMADILLO_VERSION"
echo "  OpenBLAS:  $OPENBLAS_VERSION"
echo "  LAPACK:    $LAPACK_VERSION"
echo ""

# ── Armadillo headers (all platforms) ─────────────────────────────
if [[ -f include/armadillo_bits/arma_version.hpp ]]; then
    CURRENT=$(grep ARMA_VERSION_MAJOR include/armadillo_bits/arma_version.hpp | head -1 | grep -oE '[0-9]+')
    CURRENT+="."
    CURRENT+=$(grep ARMA_VERSION_MINOR include/armadillo_bits/arma_version.hpp | head -1 | grep -oE '[0-9]+')
    CURRENT+="."
    CURRENT+=$(grep ARMA_VERSION_PATCH include/armadillo_bits/arma_version.hpp | head -1 | grep -oE '[0-9]+')
    if [[ "$CURRENT" == "$ARMADILLO_VERSION" ]]; then
        echo "Armadillo $ARMADILLO_VERSION already installed, skipping."
    else
        echo "Armadillo version mismatch (have $CURRENT, need $ARMADILLO_VERSION), updating..."
        rm -rf include/armadillo include/armadillo_bits
    fi
fi

if [[ ! -f include/armadillo ]]; then
    echo "Downloading Armadillo $ARMADILLO_VERSION..."
    curl -L -o /tmp/armadillo.tar.xz \
        "https://sourceforge.net/projects/arma/files/armadillo-${ARMADILLO_VERSION}.tar.xz/download"
    tar xf /tmp/armadillo.tar.xz -C /tmp
    cp /tmp/armadillo-${ARMADILLO_VERSION}/include/armadillo include/armadillo
    cp -r /tmp/armadillo-${ARMADILLO_VERSION}/include/armadillo_bits include/armadillo_bits
    rm -rf /tmp/armadillo.tar.xz /tmp/armadillo-${ARMADILLO_VERSION}
    echo "Armadillo $ARMADILLO_VERSION installed."
fi

# ── OpenBLAS + LAPACK ─────────────────────────────────────────────
OS=$(uname -s)

if [[ "$OS" == "MINGW"* || "$OS" == "MSYS"* || "$OS" == "CYGWIN"* || "$OS" == *"NT"* ]]; then
    echo ""
    echo "Windows detected."
    echo "OpenBLAS/LAPACK: use vcpkg or place prebuilt .lib/.dll files in include/"
    echo "  vcpkg install openblas lapack --triplet x64-windows"
    echo "  Then copy the libs:"
    echo "    cp \$VCPKG_INSTALLED/lib/openblas.lib include/libopenblas.lib"
    echo "    cp \$VCPKG_INSTALLED/bin/openblas.dll include/libopenblas.dll"
    echo "    cp \$VCPKG_INSTALLED/lib/lapack.lib   include/liblapack.lib"
    echo "    cp \$VCPKG_INSTALLED/bin/lapack.dll   include/liblapack.dll"

    if [[ -f include/libopenblas.dll && -f include/liblapack.dll ]]; then
        echo "  (libs already present in include/)"
    else
        echo "  WARNING: libs not found in include/ — build will fail until they are placed there."
    fi
else
    echo ""
    echo "Unix detected ($OS)."
    if command -v vcpkg &>/dev/null; then
        ARCH=$(uname -m)
        if [[ "$OS" == "Darwin" ]]; then
            [[ "$ARCH" == "arm64" ]] && TRIPLET="arm64-osx" || TRIPLET="x64-osx"
        else
            TRIPLET="x64-linux"
        fi
        echo "Installing OpenBLAS $OPENBLAS_VERSION + LAPACK $LAPACK_VERSION via vcpkg (triplet: $TRIPLET)..."
        vcpkg install openblas lapack --triplet "$TRIPLET"
        INSTALLED="$(vcpkg env --triplet "$TRIPLET" printenv VCPKG_INSTALLED_DIR 2>/dev/null || echo "$VCPKG_INSTALLATION_ROOT/installed")/$TRIPLET"
        echo ""
        echo "When building, pass these to CMake:"
        echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \\"
        echo "    -DBLA_VENDOR=OpenBLAS \\"
        echo "    -DCMAKE_LIBRARY_PATH=\"$INSTALLED/lib\""
        echo ""
        echo "When running, set:"
        echo "  export LD_LIBRARY_PATH=\"$INSTALLED/lib:\$LD_LIBRARY_PATH\""
    else
        echo "vcpkg not found. Install OpenBLAS $OPENBLAS_VERSION and LAPACK $LAPACK_VERSION via your package manager:"
        echo "  Ubuntu:  sudo apt install libopenblas-dev liblapack-dev"
        echo "  macOS:   brew install openblas lapack"
        echo ""
        echo "NOTE: Package manager versions may differ from the pinned version ($OPENBLAS_VERSION)."
        echo "      For exact version matching, install vcpkg and re-run this script."
    fi
fi

echo ""
echo "Done."
