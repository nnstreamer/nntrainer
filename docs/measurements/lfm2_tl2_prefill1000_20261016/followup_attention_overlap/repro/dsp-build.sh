#!/bin/bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$here/../../.." && pwd)"
export ANDROID_NDK=/home/jungone-lee/workspace/nntrainer/android-ndk-r30
export PATH="$ANDROID_NDK:$PATH"
export HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1
export DEFAULT_HEXAGON_TOOLS_ROOT="$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04"
export HEXKL_ROOT=/home/jungone-lee/workspace/nntrainer/Qualcomm/hexkl-1.0-beta.2/hexkl_addon
export HEXKL_SDK_VER=6.4.0.2 HEX_ARCH=v81
cd "$R"
git submodule update --init --depth 1
mkdir -p builddir
bash test/htp/build.sh
