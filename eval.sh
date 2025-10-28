#!/bin/sh
# sh eval.sh configs/vitb_384_test.yaml 1 output/ MODEL.WEIGHTS  ./output/model_0004999.pth

config=$1
gpus=$2
output=$3

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

shift 3
opts=${@}

#Pascal Loveda
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
#  TEST.SLIDING_WINDOW "True" \
#  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
#  MODEL.WEIGHTS $output/model_final.pth \
 $opts



                                  