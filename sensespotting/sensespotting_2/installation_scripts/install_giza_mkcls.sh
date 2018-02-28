#!/usr/bin/env bash

working_dir=$1

cd $working_dir/tools
git clone https://github.com/moses-smt/giza-pp
cd giza-pp
mv GIZA++-v2 $working_dir/tools
mv mkcls-v2 $working_dir/tools

cd $working_dir/tools

rm -rf giza-pp

cd $working_dir/tools/GIZA++-v2
make

cd $working_dir/tools/mkcls-v2
make