#!/usr/bin/env bash

#model params
PRETRAINED_TYPE="etri"

TRAIN_BATCH_SIZE=100
EVAL_BATCH_SIZE=100
LEARNING_RATE=1e-5

#train params
BATCH_SIZE=1024
EVAL_BATCH_SIZE=512
LEARNING_RATE=3e-5
EPOCHS=3
EVAL_STEP=1
LOGGING_STEP=1000
GRAD_CLIP_NORM=1.0

#other parameters
NUM_WORKERS=8
DEVICE="cuda"
FP16=1
FP16_OPT_LEVEL="O1"
#SEED=0

#run traininer
for ATTENTION_METHOD in "concat" "general"
do
	for SEED in 0 1 2 3 4
	do
		#path params
		SAVE_PATH="model_saved/layer=${NUM_HIDDEN_LAYER}.embed=${EMBEDDING_HIDDEN_DIM}.hidden=${GRU_HIDDEN_DIM}.batch=${BATCH_SIZE}.epoch=${EPOCHS}.attention=${ATTENTION_METHOD}"

		TMP_PATH="${SAVE_PATH}/seed${SEED}"

		python train.py\
			--embedding_hidden_dim=${EMBEDDING_HIDDEN_DIM}\
			--num_hidden_layer=${NUM_HIDDEN_LAYER}\
			--gru_hidden_dim=${GRU_HIDDEN_DIM}\
			--dropout_p=${DROPOUT_P}\
			--attention_method=${ATTENTION_METHOD}\
			--batch_size=${BATCH_SIZE}\
			--eval_batch_size=${EVAL_BATCH_SIZE}\
			--learning_rate=${LEARNING_RATE}\
			--logging_step=${LOGGING_STEP}\
			--epochs=${EPOCHS}\
			--eval_step=${EVAL_STEP}\
			--grad_clip_norm=${GRAD_CLIP_NORM}\
			--device=${DEVICE}\
			--fp16=${FP16}\
			--fp16_opt_level=${FP16_OPT_LEVEL}\
			--seed=${SEED}\
			--save_path=${TMP_PATH}

		python inference.py\
			--bestmodel_path=${TMP_PATH}
	done
done