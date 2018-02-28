#!/usr/bin/env bash

# Source: https://github.com/clab/fast_align

working_dir=$1

cd $working_dir/tools/
git clone https://github.com/clab/fast_align.git

cd fast_align

mkdir -p $working_dir/tools/fast_align/build
cd build
cmake ..
make

echo 'Finished installation of fast align'