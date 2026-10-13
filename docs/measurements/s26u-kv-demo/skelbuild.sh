#!/bin/bash
# skel only; extra -D flags as $1 (e.g. "-DHEXKL_MOE_NO_DIRECT")
cd ~/workspace/nntrainer
export HEXKL_ROOT=$HOME/workspace/hxkl-beta2/hexkl_addon
SDK=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1
(unset HEXAGON_SDK_ROOT; cd $SDK; source ./setup_sdk_env.source >/dev/null 2>&1; cd ~/workspace/nntrainer/test/htp && HEX_ARCH=v81 HEX_EXTRA_CFLAGS="$1" ./build.sh) 2>&1 | grep -E "error|ARCH OK|UNDEFINED" | head -5
md5sum test/htp/build/libnntr_hvx_skel.so | cut -c1-12
