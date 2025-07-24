#!/bin/bash

set -e
TARGET_PATH=$1

if [ $# -eq 0 ]; then
  echo "No argument provided, Target path = $(pwd)"
  TARGET_PATH=$(pwd)
fi

if [ ! -d "$TARGET_PATH" ]; then
  echo "given arugment is not directory, aborting!"
  exit 1
fi

dataset_path=$TARGET_PATH/cifar-100-binary
zip_path=$dataset_path.tar.gz
wget -O $zip_path https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
tar -zxvf $zip_path
