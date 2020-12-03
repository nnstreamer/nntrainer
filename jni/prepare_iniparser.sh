#!/usr/bin/env bash
TARGET=$1

if [ ! -d ${TARGET} ]; then
  mkdir -p ${TARGET}
fi

pushd ${TARGET}

# Get Iniparser
if [ ! -d "iniparer" ]; then
    echo "PREPARING ini parser at ${TARGET}"
    git clone https://github.com/ndevilla/iniparser.git iniparser
fi

popd
