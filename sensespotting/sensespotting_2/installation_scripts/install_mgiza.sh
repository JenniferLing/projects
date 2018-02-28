#!/usr/bin/env bash

# Source: http://www.statmt.org/moses/?n=Development.GetStarted

working_dir=$1

cd $working_dir/tools/
git clone https://github.com/moses-smt/mgiza.git
cd mgiza/mgizapp
cmake .
make
make install

echo 'Finished mgiza installation!'