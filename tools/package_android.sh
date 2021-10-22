#!/usr/bin/env bash

set -e

TARGET=$1
[ -z $1 ] && TARGET=$(pwd)
echo $TARGET

if [ ! -d $TARGET ]; then
  echo $TARGET is not a directory. please put project root of nntrainer
  exit 1
fi

pushd $TARGET

if [ !d builddir ]; then
  meson builddir -Dplatform=android
else
  echo "warning: $TARGET/builddir has already been taken, this script tries to reconfigure and try building"
  pushd builddir
    meson configure -Dplatform=android
    meson --wipe
  popd
fi

pushd builddir
ninja install

tar -czvf $TARGET/nntrainer_for_android.tar.gz --directory=android_build_result .

popd
popd

