#!/usr/bin/env bash

# Source: https://github.com/JohnLangford/vowpal_wabbit.git

working_dir=$1

mkdir -p $working_dir/tools

cd $working_dir/tools/
git clone https://github.com/JohnLangford/vowpal_wabbit.git

cd vowpal_wabbit

make
make test

echo 'Finished installation of vowpal wabbit'