#!/usr/bin/env bash

set -e

TARGET=$1
[ -z $1 ] && TARGET=$(pwd)
echo $TARGET


if [ ! -d $TARGET ]; then
    if [[ $1 == -D* ]]; then
	TARGET=$(pwd)
	echo $TARGET
    else
	echo $TARGET is not a directory. please put project root of nntrainer
	exit 1
    fi
fi

pushd $TARGET

filtered_args=()

for arg in "$@"; do
    if [[ $arg == -D* ]]; then
	filtered_args+=("$arg")
    fi
done

if [ ! -d builddir ]; then
    #default value of openblas num threads is 1 for android
    #enable-tflite-interpreter=false is just temporally until ci system is stabel
    #enable-opencl=true will compile OpenCL related changes or remove this option to exclude OpenCL compilations.
  meson builddir -Dplatform=android -Dopenblas-num-threads=1 -Denable-tflite-interpreter=false -Denable-tflite-backbone=false -Denable-fp16=false -Domp-num-threads=1 -Dhgemm-experimental-kernel=false -Denable-onnx-interpreter=true ${filtered_args[@]}
  
  # Prepare protobuf for Android build when ONNX interpreter is enabled
  echo "Preparing protobuf for Android build"
  # Use our prepare_protobuf.sh script to build protobuf for Android
  ANDROID_NDK_PATH=${ANDROID_NDK:-/opt/android-ndk}
  if [ ! -d "$ANDROID_NDK_PATH" ]; then
    ANDROID_NDK_PATH=${ANDROID_NDK_HOME:-/usr/local/android-ndk}
  fi
  if [ ! -d "$ANDROID_NDK_PATH" ]; then
    echo "Warning: Android NDK not found. Please set ANDROID_NDK or ANDROID_NDK_HOME environment variable."
    echo "Using existing protobuf libraries from subprojects (may cause compatibility issues)"
    # Use existing protobuf from subprojects
    if [ ! -d "builddir/protobuf-25.2" ]; then
      echo "Copying protobuf from subprojects to build directory"
      cp -r subprojects/protobuf-25.2 builddir/
    fi
    # Copy protobuf libraries and headers to jni directory for Android NDK
    echo "Copying protobuf libraries and headers to jni directory"
    mkdir -p builddir/jni/protobuf-25.2/lib
    # Check if we have built libraries in the build directory
    if [ -f "build/subprojects/protobuf-25.2/lib/arm64-v8a/libprotobuf.a" ] && [ -f "build/subprojects/protobuf-25.2/lib/arm64-v8a/libprotobuf-lite.a" ]; then
      echo "Using protobuf libraries from build directory"
      cp build/subprojects/protobuf-25.2/lib/arm64-v8a/libprotobuf.a builddir/jni/protobuf-25.2/lib/
      cp build/subprojects/protobuf-25.2/lib/arm64-v8a/libprotobuf-lite.a builddir/jni/protobuf-25.2/lib/
    else
      echo "Using existing protobuf libraries from subprojects"
      # Copy the libraries from the subproject (this is a temporary solution)
      # In a real scenario, these should be built for Android
      find subprojects/protobuf-25.2 -name "libprotobuf.a" -exec cp {} builddir/jni/protobuf-25.2/lib/ \;
      find subprojects/protobuf-25.2 -name "libprotobuf-lite.a" -exec cp {} builddir/jni/protobuf-25.2/lib/ \;
    fi
    # Copy protobuf headers
    cp -r subprojects/protobuf-25.2/src builddir/jni/protobuf-25.2/
    # Copy abseil headers for protobuf
    mkdir -p builddir/jni/protobuf-25.2/third_party
    # Try different paths for abseil-cpp
    if [ -d "subprojects/abseil-cpp-20250814.1" ]; then
      cp -r subprojects/abseil-cpp-20250814.1 builddir/jni/protobuf-25.2/third_party/abseil-cpp
    elif [ -d "../subprojects/abseil-cpp-20250814.1" ]; then
      cp -r ../subprojects/abseil-cpp-20250814.1 builddir/jni/protobuf-25.2/third_party/abseil-cpp
    elif [ -d "../../subprojects/abseil-cpp-20250814.1" ]; then
      cp -r ../../subprojects/abseil-cpp-20250814.1 builddir/jni/protobuf-25.2/third_party/abseil-cpp
    else
      echo "Warning: abseil-cpp-20250814.1 not found, skipping copy"
    fi
  else
    echo "Using Android NDK at $ANDROID_NDK_PATH"
    # Use our prepare_protobuf.sh script to build protobuf for Android
    ./jni/prepare_protobuf.sh 25.2 builddir "$ANDROID_NDK_PATH"
    # Copy protobuf libraries
    mkdir -p builddir/jni/protobuf-25.2/lib
    if [ -d "builddir/protobuf-25.2/lib" ]; then
      cp -r builddir/protobuf-25.2/lib/* builddir/jni/protobuf-25.2/lib/
    fi
    # Copy protobuf headers
    mkdir -p builddir/jni/protobuf-25.2
    if [ -d "builddir/protobuf-25.2/src" ]; then
      cp -r builddir/protobuf-25.2/src builddir/jni/protobuf-25.2/
    elif [ -d "subprojects/protobuf-25.2/src" ]; then
      cp -r subprojects/protobuf-25.2/src builddir/jni/protobuf-25.2/
    fi
    # Copy abseil headers for protobuf
    mkdir -p builddir/jni/protobuf-25.2/third_party
    # Try different paths for abseil-cpp
    if [ -d "subprojects/abseil-cpp-20250814.1" ]; then
      cp -r subprojects/abseil-cpp-20250814.1 builddir/jni/protobuf-25.2/third_party/abseil-cpp
    elif [ -d "../subprojects/abseil-cpp-20250814.1" ]; then
      cp -r ../subprojects/abseil-cpp-20250814.1 builddir/jni/protobuf-25.2/third_party/abseil-cpp
    elif [ -d "../../subprojects/abseil-cpp-20250814.1" ]; then
      cp -r ../../subprojects/abseil-cpp-20250814.1 builddir/jni/protobuf-25.2/third_party/abseil-cpp
    else
      echo "Warning: abseil-cpp-20250814.1 not found, skipping copy"
    fi
  fi
else
  echo "warning: $TARGET/builddir has already been taken, this script tries to reconfigure and try building"
  pushd builddir
    #default value of openblas num threads is 1 for android
    #enable-tflite-interpreter=false is just temporally until ci system is stabel  
    #enable-opencl=true will compile OpenCL related changes or remove this option to exclude OpenCL compilations.
    meson configure -Dplatform=android -Dopenblas-num-threads=1 -Denable-tflite-interpreter=false -Denable-tflite-backbone=false -Denable-fp16=false -Domp-num-threads=1 -Dhgemm-experimental-kernel=false -Denable-onnx-interpreter=true ${filtered_args[@]}
    meson --wipe
    
  # Prepare protobuf for Android build when ONNX interpreter is enabled
  echo "Preparing protobuf for Android build"
  # Use our prepare_protobuf.sh script to build protobuf for Android
  ANDROID_NDK_PATH=${ANDROID_NDK:-/opt/android-ndk}
  if [ ! -d "$ANDROID_NDK_PATH" ]; then
    ANDROID_NDK_PATH=${ANDROID_NDK_HOME:-/usr/local/android-ndk}
  fi
  if [ ! -d "$ANDROID_NDK_PATH" ]; then
    echo "Warning: Android NDK not found. Please set ANDROID_NDK or ANDROID_NDK_HOME environment variable."
    echo "Using existing protobuf libraries from subprojects (may cause compatibility issues)"
    # Use existing protobuf from subprojects
    if [ ! -d "../builddir/protobuf-25.2" ]; then
      echo "Copying protobuf from subprojects to build directory"
      cp -r ../subprojects/protobuf-25.2 ../builddir/
    fi
    # Copy protobuf libraries and headers to jni directory for Android NDK
    echo "Copying protobuf libraries and headers to jni directory"
    mkdir -p ../builddir/jni/protobuf-25.2/lib
    # Check if libraries exist in the build directory
    if [ -f "../../build/subprojects/protobuf-25.2/lib/arm64-v8a/libprotobuf.a" ] && [ -f "../../build/subprojects/protobuf-25.2/lib/arm64-v8a/libprotobuf-lite.a" ]; then
      echo "Using protobuf libraries from build directory"
      cp ../../build/subprojects/protobuf-25.2/lib/arm64-v8a/libprotobuf.a ../builddir/jni/protobuf-25.2/lib/
      cp ../../build/subprojects/protobuf-25.2/lib/arm64-v8a/libprotobuf-lite.a ../builddir/jni/protobuf-25.2/lib/
    else
      echo "Using existing protobuf libraries from subprojects"
      # Copy the libraries from the subproject (this is a temporary solution)
      # In a real scenario, these should be built for Android
      find ../subprojects/protobuf-25.2 -name "libprotobuf.a" -exec cp {} ../builddir/jni/protobuf-25.2/lib/ \;
      find ../subprojects/protobuf-25.2 -name "libprotobuf-lite.a" -exec cp {} ../builddir/jni/protobuf-25.2/lib/ \;
    fi
    # Copy protobuf headers
    cp -r ../builddir/protobuf-25.2/src ../builddir/jni/protobuf-25.2/
    # Copy abseil headers for protobuf
    mkdir -p ../builddir/jni/protobuf-25.2/third_party
    # Try different paths for abseil-cpp
    if [ -d "../subprojects/abseil-cpp-20250814.1" ]; then
      cp -r ../subprojects/abseil-cpp-20250814.1 ../builddir/jni/protobuf-25.2/third_party/abseil-cpp
    elif [ -d "../../subprojects/abseil-cpp-20250814.1" ]; then
      cp -r ../../subprojects/abseil-cpp-20250814.1 ../builddir/jni/protobuf-25.2/third_party/abseil-cpp
    elif [ -d "../../../subprojects/abseil-cpp-20250814.1" ]; then
      cp -r ../../../subprojects/abseil-cpp-20250814.1 ../builddir/jni/protobuf-25.2/third_party/abseil-cpp
    else
      echo "Warning: abseil-cpp-20250814.1 not found, skipping copy"
    fi
  else
    echo "Using Android NDK at $ANDROID_NDK_PATH"
    # Use our prepare_protobuf.sh script to build protobuf for Android
    ${TARGET}/jni/prepare_protobuf.sh 25.2 .. "$ANDROID_NDK_PATH"
    # Copy protobuf libraries
    mkdir -p ../builddir/jni/protobuf-25.2/lib
    if [ -d "../builddir/protobuf-25.2/lib" ]; then
      cp -r ../builddir/protobuf-25.2/lib/* ../builddir/jni/protobuf-25.2/lib/
    fi
    # Copy protobuf headers
    mkdir -p ../builddir/jni/protobuf-25.2
    if [ -d "../builddir/protobuf-25.2/src" ]; then
      cp -r ../builddir/protobuf-25.2/src ../builddir/jni/protobuf-25.2/
    elif [ -d "../../subprojects/protobuf-25.2/src" ]; then
      cp -r ../../subprojects/protobuf-25.2/src ../builddir/jni/protobuf-25.2/
    fi
    # Copy abseil headers for protobuf
    mkdir -p ../builddir/jni/protobuf-25.2/third_party
    # Try different paths for abseil-cpp
    if [ -d "../subprojects/abseil-cpp-20250814.1" ]; then
      cp -r ../subprojects/abseil-cpp-20250814.1 ../builddir/jni/protobuf-25.2/third_party/abseil-cpp
    elif [ -d "../../subprojects/abseil-cpp-20250814.1" ]; then
      cp -r ../../subprojects/abseil-cpp-20250814.1 ../builddir/jni/protobuf-25.2/third_party/abseil-cpp
    elif [ -d "../../../subprojects/abseil-cpp-20250814.1" ]; then
      cp -r ../../../subprojects/abseil-cpp-20250814.1 ../builddir/jni/protobuf-25.2/third_party/abseil-cpp
    else
      echo "Warning: abseil-cpp-20250814.1 not found, skipping copy"
    fi
  fi
  popd
fi

pushd builddir
ninja install

tar -czvf $TARGET/nntrainer_for_android.tar.gz --directory=android_build_result .

popd
popd
