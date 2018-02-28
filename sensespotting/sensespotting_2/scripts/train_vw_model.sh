#!/bin/bash

vw_path=$1/vowpalwabbit/vw
working_dir=$2
training_file=$3
dev_file=$4
model_path=$5
patience=$6
cache_name=$7
shift
shift
shift
shift
shift
shift
shift
args="$@"

#echo "VW: $vw_path \n"
#echo "dir: $working_dir \n"
#echo "train: $training_file \n"
#echo "model: $model_path \n"
#echo "early_stop: $patience \n"
#echo "ARGS: $args \n"
#echo "\n"
#exit 1

cache_path=$working_dir/$cache_name.cache
#echo "$cache_path \n"
#exit 1

if [ $dev_file = "_" ]; then
    data="$training_file.shuf"
    if [ ! -f "$data" ]; then
        echo "Shuffle $data"
        cat $training_file | shuf > $data
    #else
    #    echo "$training_file.shuf exists"
    fi
else
    data="$training_file.train_dev.shuf"
    if [ ! -f "$data" ]; then
        echo "Combine dev and train and shuffle $data.shuf"
        cat $training_file $dev_file | shuf > $data
    #else
    #    echo "$training_file.shuf exists"
    fi
fi

if [ -f "$cache_path" ]
then
    rm -f $cache_path
fi

echo "\nCommand: $vw_path -d $data --loss_function=logistic -f $model_path -p $model_path.pred --cache_file cache_path --save_per_pass --early_terminate $patience $args\n"

# train classifier:
$vw_path -d $data --loss_function=logistic -f $model_path -p $model_path.pred --cache_file cache_path --save_per_pass --early_terminate $patience $args 
#$vw_path -d $training_file.shuf --loss_function=logistic -f $model_path --oaa $(cat $training_file | awk -F "|" '{print $1}' | sort | uniq | wc -l) --probabilities --passes 20 -l 1 --cache_file $cache_path --compressed --early_terminate 5


#$(vw_path) -d $(training_file).shuf --dev $(dev_file).shuf --eval auroc --logistic --passes $bestPass --noearlystop --readable $(model_path) --args $(bestConfig) -p $(model_path).predictions

#echo "Model saved at $model_path\n"
#echo "Finished sh script ($(date))"