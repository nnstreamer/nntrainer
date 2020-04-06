#!/usr/bin/env bash
##
## @file run_unittests.sh
## @author Jijoong Moon <jijoong.moon@samsung.com>
## @date 03 April 2020
## @brief Run unit test for NNTrainer
##

ret=0
pushd build

if [ -f "$1" ]; then
    echo $1
    {$1}
    return $?
elif [ -d "$1" ]; then
    testlist=(`find "$1" -type f -executable -name "unittest_*"`)
    for t in ${testlist}; do
	${t}
	ret=$?
	if [ $ret -ne 0 ]; then
	    break
	fi
    done
fi

popd
