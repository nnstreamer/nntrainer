#!/bin/bash


ret_vals=(0 0 0 0 0)

for ((c=1; c<=10000; c++))
do
  /data/nntrainer2/build/Applications/toy_classification/toy_classification >> /dev/null > /dev/null

  ret=$?
  ((ret_vals[(($ret))]++))

  # if (( $ret == 0 )) ; then
  #   echo "success"
  # elif (( $ret == 1 )) ; then
  #   echo "construct fail"
  # elif (( $ret == 2 )) ; then
  #   echo "compile fail"
  # elif (( $ret == 3 )) ; then
  #   echo "run fail"
  # else
  #   echo "unknown fail"
  # fi
done

echo "success ${ret_vals[0]}"
echo "construct fail ${ret_vals[1]}"
echo "compile fail ${ret_vals[2]}"
echo "run fail ${ret_vals[3]}"
echo "unknown ${ret_vals[4]}"
