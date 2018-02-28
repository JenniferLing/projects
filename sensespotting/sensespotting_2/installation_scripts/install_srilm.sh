#!/usr/bin/env bash

# Source: http://www.speech.sri.com/projects/srilm/download.html

working_dir=$1
srilm_path=$2

mkdir -p $working_dir/tools/srilm
cd $working_dir/tools/srilm

if [ ! -f "$srilm_path" ]; then
    echo "Download source file from http://www.speech.sri.com/projects/srilm/download.html and save it in the same folder as this script!"
    exit 1
fi

cp $srilm_path $working_dir/tools/srilm/srilm.tar.gz

# unpack files
tar xfvz $working_dir/tools/srilm/srilm.tar.gz

rm -f $working_dir/tools/srilm/srilm.tar.gz

echo "Now, follow INSTALL file (or follow Steps 1-6)"
echo "Step 1: Go to srilm folder (command: cd $working_dir/tools/srilm)"
echo "Step 2: Add to Makefile: SRILM = $working_dir/tools/srilm/"
echo "Step 3: execute command: make World"
echo "Step 4: execute command: export PATH=$working_dir/tools/srilm/bin:$working_dir/tools/srilm/bin/i686-m64:\$PATH"
echo "Step 5: execute command: make test"
echo "Step 6: execute command: make cleanest"
# - add path to Makefile
# - make World
# - export PATH=$working_dir/tools/srilm/bin:$working_dir/tools/srilm/bin/i686-m64:$PATH
# - make test
# - make cleanest