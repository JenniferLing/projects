#!/usr/bin/env bash

working_dir=$1

mkdir -p $working_dir/tools/tree_tagger
cd $working_dir/tools/tree_tagger

# Linux version (Source: http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/)
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-linux-3.2.1.tar.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tagger-scripts.tar.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/install-tagger.sh
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/french-par-linux-3.2-utf8.bin.gz
wget http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/english-par-linux-3.2-utf8.bin.gz

sh install-tagger.sh

