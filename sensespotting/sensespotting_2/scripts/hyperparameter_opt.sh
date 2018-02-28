#!/bin/bash
# Optimize l1 and l2 regularization!

vw_path=$1/vowpalwabbit/vw
vw_hypersearch=$1/utl/vw-hypersearch

# "/big/l/lingj/token_context-token_ngram_prob-type_context-type_ngram_prob-type_rel_freq-type_topic.feat"
training_file=$2
dev_file=$3
param=$4
largeReg=$5
stepReg=$6

if [ ! -f "$training_file.shuf" ]
then
    cat $training_file | shuf > $training_file.shuf
else
    echo "$training_file.shuf exists"
fi

if [ ! -f "$dev_file.shuf" ]
then
    cat $dev_file | shuf > $dev_file.shuf
else
    echo "$dev_file.shuf exists"
fi

#largeReg="0.00028355923"
#echo $largeReg
#stepReg="0.00005671184"
#echo $stepReg

if [[ "$param" == "l1" ]]; then
  $vw_hypersearch -t $dev_file.shuf -L 0. $largeReg $vw_path --l1 % $training_file.shuf
elif [["$param" == "l2" ]]; then
  $vw_hypersearch -t $dev_file.shuf -L $stepReg $largeReg $vw_path --l2 % $training_file.shuf
else
  echo "No step defined for args $param"   
fi

#$vw_path -d $training_file.shuf.train --eval auroc --logistic --passes 2 --orsearch --l1 0. $largeReg +$stepReg --l2 $stepReg $largeReg +$stepReg --exact_adaptive_norm --power_t 0.5
#/home/l/lingj/sensespotting/tools/vowpal_wabbit/vowpalwabbit/vw --help
#$(vw_path) -d $(training_file).shuf.train --dev $(training_file).shuf.dev --eval auroc --logistic --passes 1 --orsearch --l1 0. $largeReg +$stepReg --l2 $stepReg $largeReg +$stepReg --exact_adaptive_norm --power_t 0.5

echo "Finished"