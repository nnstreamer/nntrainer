#!/bin/bash

# Build script for CausalLM Android application
# This script builds libcausallm_core.so and nntrainer_causallm executable
set -e

# Parse options
USE_BUILD_CACHE=0
USE_HTP=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --cache)
            USE_BUILD_CACHE=1
            shift
            ;;
        --profile)
            USE_PROFILE=1
            shift
            ;;
        --htp)
            USE_HTP=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cache] [--htp] [--profile]"
            echo "  --cache  Reuse existing nntrainer builddir if available"
            echo "  --htp    Build with Hexagon HTP acceleration enabled"
            echo "           (-Denable-htp=true). Requires HEXAGON_SDK_ROOT."
            echo "  --profile  Build nntrainer with -Denable-profile=true and the"
            echo "           app with -DPROFILE, so nntrainer_causallm prints"
            echo "           per-layer-type time totals after the run (doc 45 Gate 1)."
            exit 1
            ;;
    esac
done


# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN} $1 ${NC}"
    echo -e "${CYAN}========================================${NC}"
}

log_step() {
    echo -e "\n${YELLOW}[Step $1]${NC} $2"
    echo -e "${YELLOW}----------------------------------------${NC}"
}

# Function to check and fix artifact location
check_artifact() {
    local filename=$1
    local libs_path="libs/arm64-v8a/$filename"
    local obj_path="obj/local/arm64-v8a/$filename"

    if [ -f "$libs_path" ]; then
        size=$(ls -lh "$libs_path" | awk '{print $5}')
        echo -e "  ${GREEN}[OK]${NC} $filename ($size)"
        return 0
    elif [ -f "$obj_path" ]; then
        echo -e "  ${YELLOW}[WARN]${NC} $filename found in obj but not in libs. Copying..."
        mkdir -p "libs/arm64-v8a"
        cp "$obj_path" "$libs_path"
        if [ -x "$obj_path" ]; then
            chmod +x "$libs_path"
        fi
        size=$(ls -lh "$libs_path" | awk '{print $5}')
        echo -e "  ${GREEN}[OK]${NC} $filename ($size) (Copied from obj)"
        return 0
    else
        echo -e "  ${RED}[ERROR]${NC} $filename not found!"
        log_info "  Checked paths:"
        log_info "    - $libs_path"
        log_info "    - $obj_path"
        return 1
    fi
}

# Determine ANDROID_NDK: honor an existing value, else fall back to a
# candidate list, else error out by name -- mirrors the repo-root
# build_android.sh's own NDK-detection block (that script previously
# force-overwrote ANDROID_NDK to one hardcoded path here, unconditionally,
# before this same check ran -- which meant this check could never fail
# even when the hardcoded path didn't exist on the machine running it).
if [ -z "$ANDROID_NDK" ]; then
    for candidate in \
        "$HOME/Desktop/workspace/android-ndk-r26d" \
        /opt/android-ndk-r26d; do
        if [ -d "$candidate" ]; then
            export ANDROID_NDK="$candidate"
            break
        fi
    done
fi

if [ -z "$ANDROID_NDK" ] || [ ! -d "$ANDROID_NDK" ]; then
    log_error "ANDROID_NDK is not set (or does not exist). Please set it to your Android NDK path."
    log_info "Example: export ANDROID_NDK=/path/to/android-ndk-r26d"
    exit 1
fi
export PATH="$PATH:$ANDROID_NDK"

# Set NNTRAINER_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export NNTRAINER_ROOT

log_header "Build CausalLM Android Application"
log_info "NNTRAINER_ROOT: $NNTRAINER_ROOT"
log_info "Build cache: $([ "$USE_BUILD_CACHE" -eq 1 ] && echo 'enabled' || echo 'disabled (default)')"
log_info "ANDROID_NDK: $ANDROID_NDK"
log_info "Working directory: $(pwd)"

# Step 1: Build nntrainer for Android if not already built
log_step "1/4" "Build nntrainer for Android"

PACKAGE_ANDROID_ARGS=()
: "${USE_PROFILE:=0}"
if [ "$USE_PROFILE" -eq 1 ]; then
    log_info "Profiling enabled: -Denable-profile=true, app built with -DPROFILE"
    PACKAGE_ANDROID_ARGS+=("-Denable-profile=true")
fi
if [ "$USE_HTP" -eq 1 ]; then
    : "${HEXAGON_SDK_ROOT:?set HEXAGON_SDK_ROOT to a Hexagon SDK checkout (--htp requires it)}"
    log_info "HTP acceleration enabled: HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT"
    # Generates generated/nntr_hvx_stub.c, which meson.build errors out by
    # name for if missing -- see that script's own header comment for why
    # this is a manual step rather than a meson custom_target.
    bash "$NNTRAINER_ROOT/nntrainer/tensor/htp_backend/generate_stub.sh"

    # The DSP skel is NOT built here and never was; test/htp/build.sh builds
    # it and the caller pushes it. That is easy to forget, and forgetting it
    # produces AEE_EBADPARM at the first call whose IDL changed rather than
    # anything that names the skel -- three measurement cycles so far. Say so
    # while the build is running, when there is still time to act on it.
    log_info "reminder: this does NOT rebuild libnntr_hvx_skel.so."
    log_info "  if the IDL changed:  HEXKL_ROOT=... ./test/htp/build.sh"
    log_info "  then push test/htp/build/libnntr_hvx_skel.so to the device"
    # hexkl-lib-subdir already defaults to armv8_android26 (meson_options.txt),
    # matching the device recipe in docs/htp_attention/40_moe_ffn_htp_task.md
    # section6.2.
    #
    # HEXKL_ROOT wins where it is set, because the beta2 drop this project
    # runs on unpacks to its own directory rather than into the SDK. meson
    # derives the SDK root as two levels up from the addon, which is right
    # only for the bundled layout, so pass it explicitly either way -- the
    # derivation on a standalone drop lands on the parent of the checkout and
    # stops at a missing incs/ that names neither variable.
    HEXKL_ADDON="${HEXKL_ROOT:-$HEXAGON_SDK_ROOT/addons/hexkl_addon}"
    if [ ! -d "$HEXKL_ADDON" ]; then
        log_error "HexKL addon not found: $HEXKL_ADDON"
        log_error "  set HEXKL_ROOT to the hexkl_addon directory"
        exit 1
    fi
    # Two lib layouts. The SDK-bundled addon is lib/<arch>/libsdkl.so; the
    # standalone beta2 drop puts a version between them,
    # lib/<sdk_ver>/<arch>/libsdkl.so, which is what test/htp/build.sh already
    # walks for libhexkl_micro.a. meson's default assumes the first, and
    # getting it wrong stops at "C shared or static library 'sdkl' not found"
    # without naming a path. Look for the file instead of assuming.
    HEXKL_ARCH="${HEXKL_ARCH:-armv8_android26}"
    if [ -f "$HEXKL_ADDON/lib/$HEXKL_ARCH/libsdkl.so" ]; then
        HEXKL_LIB_SUBDIR="$HEXKL_ARCH"
    else
        ver="${HEXKL_SDK_VER:-$(ls -1 "$HEXKL_ADDON/lib" 2>/dev/null | sort -V | tail -1)}"
        HEXKL_LIB_SUBDIR="$ver/$HEXKL_ARCH"
    fi
    if [ ! -f "$HEXKL_ADDON/lib/$HEXKL_LIB_SUBDIR/libsdkl.so" ]; then
        log_error "No libsdkl.so under $HEXKL_ADDON/lib for $HEXKL_ARCH."
        log_error "  looked at: lib/$HEXKL_ARCH and lib/<version>/$HEXKL_ARCH"
        log_error "  available: $(ls -1 "$HEXKL_ADDON/lib" 2>/dev/null | tr '\n' ' ')"
        log_error "  set HEXKL_SDK_VER, or HEXKL_ARCH for a different device"
        exit 1
    fi
    log_info "HexKL: $HEXKL_ADDON/lib/$HEXKL_LIB_SUBDIR/libsdkl.so"
    PACKAGE_ANDROID_ARGS+=("-Denable-htp=true" \
        "-Dhexkl-sdk-root=$HEXKL_ADDON" \
        "-Dhexkl-lib-subdir=$HEXKL_LIB_SUBDIR" \
        "-Dhexagon-sdk-root=$HEXAGON_SDK_ROOT")
fi

if [ "$USE_BUILD_CACHE" -eq 1 ] && [ -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    log_info "Build cache enabled: reusing existing nntrainer builddir (skipping)"
else
    log_info "Building nntrainer for Android..."
    cd "$NNTRAINER_ROOT"
    if [ -d "$NNTRAINER_ROOT/builddir" ]; then
        log_info "Removing existing builddir..."
        rm -rf builddir
    fi
    ./tools/package_android.sh "${PACKAGE_ANDROID_ARGS[@]}"
fi

# Check if build was successful
if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    log_error "nntrainer build failed. Please check the build logs."
    exit 1
fi
log_success "nntrainer ready"

# Step 2: Build tokenizer library if not present
log_step "2/4" "Build Tokenizer Library"

cd "$SCRIPT_DIR"
if [ ! -f "lib/libtokenizers_android_c.a" ]; then
    log_warning "libtokenizers_android_c.a not found in lib directory."
    log_info "Attempting to build tokenizer library..."
    if [ -f "build_tokenizer_android.sh" ]; then
        ./build_tokenizer_android.sh
    else
        log_error "tokenizer library not found and build script is missing."
        log_info "Please build or download the tokenizer library for Android arm64-v8a"
        log_info "and place it in: $SCRIPT_DIR/lib/libtokenizers_android_c.a"
        exit 1
    fi
else
    log_info "Tokenizer library already built (skipping)"
fi
log_success "Tokenizer library ready"

# Step 3: Prepare json.hpp if not present
log_step "3/4" "Prepare json.hpp"

if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
    log_info "json.hpp not found. Downloading..."
    # prepare_encoder.sh expects target directory as first argument and version as second
    # It copies json.hpp to ../Applications/CausalLM/ if version is 0.2
    "$NNTRAINER_ROOT/jni/prepare_encoder.sh" "$NNTRAINER_ROOT/builddir" "0.2"
    
    if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
        log_error "Failed to download json.hpp"
        exit 1
    fi
else
    log_info "json.hpp already exists (skipping)"
fi
log_success "json.hpp ready"

# Step 4: Build CausalLM (libcausallm_core.so and nntrainer_causallm)
log_step "4/4" "Build CausalLM Core (library + executable)"

cd "$SCRIPT_DIR/jni"

# Clean previous builds
rm -rf libs obj

log_info "Building with ndk-build (builds causallm_core, nntrainer_causallm, nntr_quantize, nntr_safetensors_info)..."
# We explicitly set paths to ensure outputs are predictable
if ndk-build NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs NDK_OUT=./obj APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk CAUSALLM_PROFILE=$USE_PROFILE CAUSALLM_HTP=$USE_HTP causallm_core nntrainer_causallm  nntr_quantize nntr_safetensors_info -j $(nproc); then
    log_success "Build completed successfully"
else
    log_error "Build failed"
    exit 1
fi

# Verify outputs
log_info "Build artifacts:"

check_artifact "libcausallm_core.so" || exit 1
check_artifact "nntrainer_causallm" || exit 1
check_artifact "nntr_quantize" || exit 1
check_artifact "nntr_safetensors_info" || exit 1

# Summary
log_header "Build Summary"
log_success "Build completed successfully!"
log_info "Output files are in: $SCRIPT_DIR/jni/libs/arm64-v8a/"
log_info "Executables:"
log_info "  - nntrainer_causallm (main application), nntr_quantize, nntr_safetensors_info"
log_info "Libraries:"
log_info "  - libcausallm_core.so (CausalLM Core library)"
log_info "  - libnntrainer.so (nntrainer library)"
log_info "  - libccapi-nntrainer.so (nntrainer C/C API)"
log_info "  - libc++_shared.so (C++ runtime)"
log_info "To build API library, run:"
log_info "  ./build_api_lib.sh"
log_info "To install and run:"
log_info "  ./install_android.sh"
