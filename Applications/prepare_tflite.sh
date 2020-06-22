#!/usr/bin/env bash
VERSION=$1
TARGET=$2

echo "PREPARING TENSORFLOW ${VERSION} at ${TARGET}"

pushd ${TARGET}

#Get tensorflow
if [ ! -d "tensorflow-${VERSION}" ]; then
    if [ ! -f "tensorflow-lite-${VERSION}.tar.xz" ]; then
      echo "[TENSORFLOW-LITE] Download tensorflow-${VERSION}"
      URL="https://github.com/nnstreamer/nnstreamer-android-resource/raw/master/external/tensorflow-lite-${VERSION}.tar.xz"
      if ! wget ${URL} ; then
        echo "[TENSORFLOW-LITE] There was an error while downloading tflite, check if you have specified right version"
        exit $?
      fi
      echo "[TENSORFLOW-LITE] Finish downloading tensorflow-${VERSION}"
      echo "[TENSORFLOW-LITE] untar tensorflow-${VERSION}"
    fi
    mkdir -p tensorflow-${VERSION}
    tar -xf tensorflow-lite-${VERSION}.tar.xz -C tensorflow-${VERSION}
    rm "tensorflow-lite-${VERSION}.tar.xz"
else
  echo "[TENSORFLOW-LITE] folder already exist, exiting without downloading"
fi

popd
