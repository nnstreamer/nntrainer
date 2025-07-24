#!/usr/bin/env bash

# This is to create the nntrainer documents (https://nntrainer.github.io/).
# Our documentation uses hotdoc, you should usually refer to here: http://hotdoc.github.io/ .
# Run this script on the root path of the NNTrainer.

echo "Generate NNTrainer documents"

v=$( grep -w version: meson.build | perl -pe 'if(($v)=/([0-9]+([.][0-9]+)+)/){print"$v\n";exit}$_=""' )
deps_file_path="$(pwd)/docs/NNTrainer.deps"

echo "NNTrainer version: $v"
echo "Dependencies file path: $deps_file_path"

hotdoc run -i index.md -o docs/NNTrainer-doc --sitemap=docs/hotdoc/sitemap.txt --deps-file-dest=$deps_file_path \
           --html-extra-theme=docs/hotdoc/theme/extra --project-name=NNTrainer --project-version=$v
