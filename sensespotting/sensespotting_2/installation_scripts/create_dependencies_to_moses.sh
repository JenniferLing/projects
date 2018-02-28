#!/usr/bin/env bash

# Source: http://www.cfilt.iitb.ac.in/Moses-Tutorial.pdf

working_dir=$1

cd $working_dir/tools/mosesdecoder/scripts
mkdir -p exports
cd exports

# GIZA++ files
cp $working_dir/tools/GIZA++-v2/GIZA++ $working_dir/tools/GIZA++-v2/snt2cooc.out ./

# Mkcls files
cp $working_dir/tools/mkcls-v2/mkcls ./

# mgiza files
mkdir -p mgizapp
cp $working_dir/tools/mgiza/mgizapp/bin/* ./mgizapp
cp $working_dir/tools/mgiza/mgizapp/scripts/merge_alignment.py ./