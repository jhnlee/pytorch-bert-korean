#!/usr/bin/env bash

pretrained_folder='./model_saved_pretrain/lr0.0001,batch256,total255600,warmup0.005,len128,skt'

lines=`find $pretrained_folder -name '*model.bin'`

for line in $lines
do
    echo use $line
    python train_classification.py\
        --pretrained_type=skt\
        --pretrained_model_path=$line
done
