#!/bin/bash
# app only (libcausallm_core.so, nntrainer_causallm); build.sh does skel + libnntrainer + app
cd ~/workspace/nntrainer
export ANDROID_NDK=$HOME/workspace/android-ndk-r26d PATH=$HOME/workspace/android-ndk-r26d:$HOME/.local/bin:$PATH
export HEXKL_ROOT=$HOME/workspace/hxkl-beta2/hexkl_addon HEXAGON_SDK_ROOT=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1
(cd Applications/CausalLM && ./build_android.sh --htp --cache) > ~/workspace/s26u-work/app.out 2>&1 || { grep -E "error" -A 3 ~/workspace/s26u-work/app.out | head -30; echo "APP FAILED"; exit 1; }
md5sum Applications/CausalLM/jni/libs/arm64-v8a/{nntrainer_causallm,libcausallm_core.so} | cut -c1-12,34-
