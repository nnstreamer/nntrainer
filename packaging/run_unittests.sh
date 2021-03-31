#!/usr/bin/env bash
##
## @file run_unittests.sh
## @author Jijoong Moon <jijoong.moon@samsung.com>
## @date 03 April 2020
## @brief Run unit test for NNTrainer
##

ret=0
pushd build

run_entry() {
  entry=$1
  ${entry} --gtest_output="xml:${entry##*/}.xml"
  return $?
}

if [ -f "$1" ]; then
    echo $1
    run_entry $1
    exit $?
elif [ -d "$1" ]; then
    testlist=$(find $1 -type f -executable -name "unittest_*")
    for t in ${testlist}; do
	echo "running: ${t} @$(pwd)"
	run_entry ${t}
	ret=$?
	if [ $ret -ne 0 ]; then
	    break
	fi
    done
fi

popd
exit $ret
