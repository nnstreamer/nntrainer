#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
##
# @file    env.sh
# @brief   Environment for native HTP builds on the Ubuntu workstation
# @author  dlwlzzero <dlwlzzero@gmail.com>
#
# Contract: docs/plans/0001-htp-moe-decode-agent-system.md §4.1.
# Source this before test/htp/build.sh, build_android.sh --htp or the
# host build:
#
#     source tools/htp/env.sh
#
# Overrides: HEXAGON_SDK_ROOT (default 6.4.0.1 under /local/mnt/workspace),
# HEXKL_ROOT (the HexKL package root that holds lib/<sdk version>/...),
# HEXKL_SDK_VER (the lib/ subdirectory to use), ANDROID_NDK, NNTR_MODEL_DIR,
# HEX_ARCH (v79, the S25 Ultra).

_nntr_sdk="${HEXAGON_SDK_ROOT:-/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1}"
if [ ! -f "$_nntr_sdk/setup_sdk_env.source" ]; then
  echo "env.sh: no SDK at $_nntr_sdk" >&2
  return 1 2>/dev/null || exit 1
fi
# setup_sdk_env.source returns at once when HEXAGON_SDK_ROOT is already set.
unset HEXAGON_SDK_ROOT
# shellcheck disable=SC1091
source "$_nntr_sdk/setup_sdk_env.source" >/dev/null
unset _nntr_sdk

export HEXKL_ROOT="${HEXKL_ROOT:-$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon}"
export HEXKL_SDK_VER="${HEXKL_SDK_VER:-$(basename "$HEXAGON_SDK_ROOT")}"
export ANDROID_NDK="${ANDROID_NDK:-$HOME/android-ndk-r30}"
export ANDROID_NDK_HOME="$ANDROID_NDK"
export NNTR_MODEL_DIR="${NNTR_MODEL_DIR:-/local/mnt/workspace/models/lfm2.5-8b-a1b}"
export HEX_ARCH="${HEX_ARCH:-v79}"
for _f in "$HEXKL_ROOT/lib/$HEXKL_SDK_VER/hexagon_toolv19_$HEX_ARCH/libhexkl_micro.a" \
          "$HEXKL_ROOT/lib/$HEXKL_SDK_VER/armv8_android26/libsdkl.so" \
          "$ANDROID_NDK/ndk-build"; do
  [ -e "$_f" ] || echo "env.sh: missing $_f" >&2
done
unset _f
# Tools/bin (hexagon-clang), the NDK, ~/.cargo/bin (cargo for the tokenizer
# library of the Android app) and ~/.local/bin (clang-format-14)
for _d in "$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin" "$ANDROID_NDK" "$HOME/.cargo/bin" "$HOME/.local/bin"; do
  case ":$PATH:" in *":$_d:"*) ;; *) export PATH="$_d:$PATH" ;; esac
done
unset _d

echo "htp env: SDK $HEXAGON_SDK_ROOT ($DEFAULT_TOOLS_VARIANT), HexKL $HEXKL_ROOT/lib/$HEXKL_SDK_VER, NDK $ANDROID_NDK, model $NNTR_MODEL_DIR, arch $HEX_ARCH"
