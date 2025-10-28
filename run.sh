#!/bin/sh

# config=$"./configs/vitb_384.yaml"
# gpus=$1
# output=$"output/"

config=$1
gpus=$2
output=$3


#参数的验证部分
if [ -z $config ]
then
    echo "No config file found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi


if [ -z $output ]
then
    echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi


#额外参数的处理
shift 3
opts=${@}

python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --resume \
 OUTPUT_DIR $output \
 $opts



