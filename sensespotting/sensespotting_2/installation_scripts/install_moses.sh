#!/usr/bin/env bash

# Source: http://www.statmt.org/moses/?n=Development.GetStarted

working_dir=$1

cd $working_dir/tools/
git clone https://github.com/moses-smt/mosesdecoder.git
cd mosesdecoder

# Run the following to install a recent version of Boost (the default version on your system might be too old), as well as cmph (for CompactPT), irstlm (language model from FBK, required to pass the regression tests), and xmlrpc-c (for moses server). By default, these will be installed in ./opt in your working directory
make -f contrib/Makefiles/install-dependencies.gmake 

# To compile moses, run 
./compile.sh

echo 'Finished Moses installation!'