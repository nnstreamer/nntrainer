#!/usr/bin/env bash
VERSION=$1
TARGET=$2
NDK_PATH=$3

set -e
echo "PREPARING PROTOBUF ${VERSION} at ${TARGET}"
echo "prepare_protobuf.sh called with VERSION=${VERSION}, TARGET=${TARGET}, NDK_PATH=${NDK_PATH}"

if [ ! -d ${TARGET} ]; then
  mkdir -p ${TARGET}
fi

pushd ${TARGET}

# Store the original directory
ORIG_DIR=$(pwd)

#Get protobuf
if [ ! -d "protobuf-${VERSION}" ]; then
    if [ ! -f "protobuf-${VERSION}.tar.xz" ]; then
      echo "[PROTOBUF] Download protobuf-${VERSION}"
      URL="https://github.com/nnstreamer/nnstreamer-android-resource/raw/master/external/protobuf-${VERSION}.tar.xz"
      if ! wget -q ${URL} ; then
        echo "[PROTOBUF] Failed to download from GitHub, trying to use local subproject"
        # Try to copy from subprojects directory
        SUBPROJECTS_DIR="${ORIG_DIR}/../../subprojects/protobuf-${VERSION}"
        echo "[PROTOBUF] Checking subprojects directory: ${SUBPROJECTS_DIR}"
        if [ -d "${SUBPROJECTS_DIR}" ]; then
          echo "[PROTOBUF] Copying from subprojects directory"
          cp -r "${SUBPROJECTS_DIR}" .
        else
          echo "[PROTOBUF] Subproject directory not found at ${SUBPROJECTS_DIR}"
          # Try alternative paths
          ALT_SUBPROJECTS_DIR1="${ORIG_DIR}/../subprojects/protobuf-${VERSION}"
          ALT_SUBPROJECTS_DIR2="${ORIG_DIR}/subprojects/protobuf-${VERSION}"
          echo "[PROTOBUF] Trying alternative path 1: ${ALT_SUBPROJECTS_DIR1}"
          if [ -d "${ALT_SUBPROJECTS_DIR1}" ]; then
            echo "[PROTOBUF] Copying from alternative path 1"
            cp -r "${ALT_SUBPROJECTS_DIR1}" .
          else
            echo "[PROTOBUF] Alternative path 1 not found"
            echo "[PROTOBUF] Trying alternative path 2: ${ALT_SUBPROJECTS_DIR2}"
            if [ -d "${ALT_SUBPROJECTS_DIR2}" ]; then
              echo "[PROTOBUF] Copying from alternative path 2"
              cp -r "${ALT_SUBPROJECTS_DIR2}" .
            else
              echo "[PROTOBUF] Alternative path 2 not found"
              echo "[PROTOBUF] There was an error while downloading protobuf and subproject not found, check if you have specified right version"
              exit $?
            fi
          fi
        fi
      else
        echo "[PROTOBUF] Finish downloading protobuf-${VERSION}"
        echo "[PROTOBUF] untar protobuf-${VERSION}"
        mkdir -p protobuf-${VERSION}
        tar -xf protobuf-${VERSION}.tar.xz -C protobuf-${VERSION}
        rm "protobuf-${VERSION}.tar.xz"
      fi
    fi
else
  echo "[PROTOBUF] folder already exist, exiting without downloading"
  # Even if the folder exists, let's check if it has the CMakeLists.txt file
  if [ ! -f "protobuf-${VERSION}/CMakeLists.txt" ]; then
    echo "[PROTOBUF] CMakeLists.txt not found in existing folder, trying to copy from subprojects"
    SUBPROJECTS_DIR="${ORIG_DIR}/../../subprojects/protobuf-${VERSION}"
    echo "[PROTOBUF] Checking subprojects directory: ${SUBPROJECTS_DIR}"
    if [ -d "${SUBPROJECTS_DIR}" ]; then
      echo "[PROTOBUF] Copying from subprojects directory"
      rm -rf "protobuf-${VERSION}"
      cp -r "${SUBPROJECTS_DIR}" .
    else
      echo "[PROTOBUF] Subproject directory not found at ${SUBPROJECTS_DIR}"
      # Try alternative paths
      ALT_SUBPROJECTS_DIR1="${ORIG_DIR}/../subprojects/protobuf-${VERSION}"
      ALT_SUBPROJECTS_DIR2="${ORIG_DIR}/subprojects/protobuf-${VERSION}"
      echo "[PROTOBUF] Trying alternative path 1: ${ALT_SUBPROJECTS_DIR1}"
      if [ -d "${ALT_SUBPROJECTS_DIR1}" ]; then
        echo "[PROTOBUF] Copying from alternative path 1"
        rm -rf "protobuf-${VERSION}"
        cp -r "${ALT_SUBPROJECTS_DIR1}" .
      else
        echo "[PROTOBUF] Alternative path 1 not found"
        echo "[PROTOBUF] Trying alternative path 2: ${ALT_SUBPROJECTS_DIR2}"
        if [ -d "${ALT_SUBPROJECTS_DIR2}" ]; then
          echo "[PROTOBUF] Copying from alternative path 2"
          rm -rf "protobuf-${VERSION}"
          cp -r "${ALT_SUBPROJECTS_DIR2}" .
        else
          echo "[PROTOBUF] Alternative path 2 not found"
          echo "[PROTOBUF] Subproject directory not found, cannot proceed"
          exit 1
        fi
      fi
    fi
  fi
fi

# Initialize submodules if .gitmodules file exists and .git directory exists
if [ -f "protobuf-${VERSION}/.gitmodules" ] && [ -d "protobuf-${VERSION}/.git" ]; then
  echo "[PROTOBUF] Initializing submodules"
  pushd "protobuf-${VERSION}"
  git submodule update --init --recursive
  popd
elif [ -f "protobuf-${VERSION}/.gitmodules" ]; then
  echo "[PROTOBUF] .gitmodules file found but no .git directory, skipping submodule initialization"
fi

# Copy abseil-cpp from subprojects to protobuf third_party directory if it doesn't exist
if [ ! -d "protobuf-${VERSION}/third_party/abseil-cpp" ] || [ -z "$(ls -A "protobuf-${VERSION}/third_party/abseil-cpp")" ]; then
  echo "[PROTOBUF] Copying abseil-cpp from subprojects"
  # Try to find abseil-cpp in subprojects directory
  ABSL_DIR=""
  if [ -d "${ORIG_DIR}/../../subprojects/abseil-cpp-20250814.1" ]; then
    ABSL_DIR="${ORIG_DIR}/../../subprojects/abseil-cpp-20250814.1"
  elif [ -d "${ORIG_DIR}/../subprojects/abseil-cpp-20250814.1" ]; then
    ABSL_DIR="${ORIG_DIR}/../subprojects/abseil-cpp-20250814.1"
  elif [ -d "${ORIG_DIR}/subprojects/abseil-cpp-20250814.1" ]; then
    ABSL_DIR="${ORIG_DIR}/subprojects/abseil-cpp-20250814.1"
  fi
  
  if [ -n "${ABSL_DIR}" ]; then
    echo "[PROTOBUF] Found abseil-cpp at ${ABSL_DIR}"
    mkdir -p "protobuf-${VERSION}/third_party/abseil-cpp"
    # Copy the contents of abseil-cpp directory, not the directory itself
    cp -r "${ABSL_DIR}/"* "protobuf-${VERSION}/third_party/abseil-cpp/"
  else
    echo "[PROTOBUF] abseil-cpp not found in subprojects directory"
  fi
fi

# Store the protobuf directory path
PROTOBUF_DIR="$(pwd)/protobuf-${VERSION}"

# Build protobuf for Android if NDK path is provided
if [ ! -z "$NDK_PATH" ] && [ -d "$NDK_PATH" ]; then
  echo "[PROTOBUF] Building protobuf for Android"
  
  # Check if already built
  if [ ! -f "protobuf-${VERSION}/lib/arm64-v8a/libprotobuf.a" ]; then
    # Create build directories
    mkdir -p protobuf-${VERSION}/build/arm64-v8a
    
    # Build for arm64-v8a
    echo "[PROTOBUF] Building for arm64-v8a"
    pushd protobuf-${VERSION}/build/arm64-v8a
    echo "[PROTOBUF] Running cmake with PROTOBUF_DIR=${PROTOBUF_DIR}"
    echo "[PROTOBUF] Running cmake with NDK_PATH=${NDK_PATH}"
    cmake ${PROTOBUF_DIR} \
      -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-21 \
      -DCMAKE_BUILD_TYPE=Release \
      -Dprotobuf_BUILD_TESTS=OFF \
      -Dprotobuf_BUILD_EXAMPLES=OFF \
      -DCMAKE_INSTALL_PREFIX=../install/arm64-v8a \
      -Dprotobuf_BUILD_PROTOC_BINARIES=OFF
    echo "[PROTOBUF] Running make"
    make -j$(nproc)
    echo "[PROTOBUF] Copying libraries"
    mkdir -p ../../lib/arm64-v8a
    cp libprotobuf.a ../../lib/arm64-v8a/
    cp libprotobuf-lite.a ../../lib/arm64-v8a/
    popd
    
    # Create libabsl.a from individual abseil-cpp libraries
    echo "[PROTOBUF] Creating libabsl.a from individual abseil-cpp libraries"
    if [ -d "protobuf-${VERSION}/build/arm64-v8a/third_party/abseil-cpp/absl" ]; then
      echo "[PROTOBUF] Found abseil-cpp build directory"
      pushd protobuf-${VERSION}/build/arm64-v8a/third_party/abseil-cpp/absl
      # Find all .a files and combine them into libabsl.a
      echo "[PROTOBUF] Copying individual abseil-cpp libraries"
      mkdir -p ../../../../../lib/arm64-v8a
      find . -name "*.a" -exec cp {} ../../../../../lib/arm64-v8a/ \;
      popd
    else
      echo "[PROTOBUF] abseil-cpp build directory not found"
    fi
    
    # Create a list of all abseil-cpp libraries with correct paths for Android.mk
    if [ -d "protobuf-${VERSION}/build/arm64-v8a/third_party/abseil-cpp/absl" ]; then
      echo "[PROTOBUF] Creating list of abseil-cpp libraries"
      find protobuf-${VERSION}/build/arm64-v8a/third_party/abseil-cpp/absl -name "*.a" | sed 's|protobuf-25.2/build/arm64-v8a/third_party/abseil-cpp/absl|../protobuf-25.2/build/arm64-v8a/third_party/abseil-cpp/absl|g' > protobuf-${VERSION}/lib/arm64-v8a/abseil_libs.txt
    else
      echo "[PROTOBUF] abseil-cpp build directory not found"
    fi
  else
    echo "[PROTOBUF] Android libraries already built"
  fi
fi

popd
