#!/bin/bash

#echo "Started vowpal wabbit at $(date)"

vw_path=$1/vowpalwabbit/vw
model=$2
test_file=$3
args=$4
dev_file=$5

#echo "VW: $vw_path"
#echo "model: $model"
#echo "test file: $test_file"
#echo "dev file: $dev_file"
#echo "ARGS: $args"
#if [ -z "$dev_file" ]; then
#    echo "Dev file is empty"
#fi
#exit 1;

#echo "Use model from $model\n"

if [ $dev_file = "_" ]; then
    data=$test_file
else
    data="$test_file.test_dev"
    if [ ! -f "$data" ]; then
        #echo "Combine dev and test"
        cat $dev_file $test_file > $data
    fi
fi

#echo "\nCommand: $vw_path -i $model -t $data -p /dev/stdout --loss_function=logistic --quiet $args\n"
$vw_path -i $model -t $data -p /dev/stdout --loss_function=logistic --quiet $args

#echo "Finished sh script ($(date))"