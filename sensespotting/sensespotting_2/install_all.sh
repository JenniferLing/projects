#!/usr/bin/env bash
#
#echo "Current directory: $(dirname ${BASH_SOURCE[0]} && pwd)"
#
#current_dir="$(cd "$( dirname "$0" )" && pwd )"
#echo "Current directory: $current_dir"
#
### Define paths:
### path to the installation scripts
script_path=/mounts/Users/student/lingj/sensespotting/installation_scripts
#
### path to the working directory
working_dir=/mounts/Users/student/lingj/sensespotting
#
### path to the directory where the tools are installed 
### (directory will be created if it does not exist)
tool_path=/mounts/Users/student/lingj/sensespotting/tools
#
mkdir -p $tool_path
#
if [ ! -d $tool_path/srilm ]; then
    echo "Install SRILM"
    sh $script_path/install_srilm.sh $working_dir;
fi
#
if [ ! -d $tool_path/tree_tagger ]; then
    sh $script_path/install_tree_tagger.sh $working_dir;
fi
# 
if [ ! -d $tool_path/fast_align ]; then
    sh $script_path/install_fast_align.sh $working_dir;
fi
# 
if [ ! -d $tool_path/GIZA++-v2 ]; then
    sh $script_path/install_giza_mkcls.sh $working_dir;
fi
# 
if [ ! -d $tool_path/mgiza ]; then
    sh $script_path/install_mgiza.sh $working_dir;
fi
# 
if [ ! -d $tool_path/mosesdecoder ]; then
    sh $script_path/install_moses.sh $working_dir;
fi
# 
if [ ! -d $tool_path/mosesdecoder/scripts/exports ]; then
    sh $script_path/create_dependencies_to_moses.sh $working_dir;
fi
#
if [ ! -d $tool_path/vowpal_wabbit ]; then
    sh $script_path/install_vowpal_wabbit.sh $working_dir;
fi 
#
echo "Finished installation!"