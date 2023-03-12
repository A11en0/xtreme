#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


REPO=$PWD
MODEL=${1:-xlm-roberta-base}
#MODEL=${1:-bert-base-multilingual-cased}
GPU=${2:-1}
DATA_DIR=${3:-"$REPO/download/"}
OUT_DIR=${4:-"$REPO/outputs/"}
MODEL_TYPE=${5:-xlmr-mh}   # Modify this to control xlm-roberta-base or its rewrite multi-head version. [xlmr, xlmr-mh]
WEIGHT_TYPE=${6:-uniform}  # uniform / less_forgetting
TRAIN_LANGS=${7:-"en,de,fr,pa,pl,pt,qu,ro"}
PREDICT_HEAD=${8:-mean}

export CUDA_VISIBLE_DEVICES=$GPU
TASK='panx'
LANGS="ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu,qu,pl,uk,az,lt,pa,gu,ro"

NUM_EPOCHS=10
MAX_LENGTH=128
LR=2e-5

LC=""

if [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-roberta-large" ]; then
  BATCH_SIZE=2
  GRAD_ACC=16
else
  BATCH_SIZE=8
  GRAD_ACC=4
fi

#DATA_DIR=$DATA_DIR/${TASK}/${TASK}_${MODEL_TYPE}_processed_maxlen${MAX_LENGTH}/
DATA_DIR=$DATA_DIR/${TASK}/${TASK}_xlmr_processed_maxlen${MAX_LENGTH}/
OUTPUT_DIR="$OUT_DIR/$TASK/${MODEL}_${MODEL_TYPE}_TL${TRAIN_LANGS}_PH${PREDICT_HEAD}_LR${LR}-epoch${NUM_EPOCHS}-MaxLen${MAX_LENGTH}-${WEIGHT_TYPE}/"
echo $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
python $REPO/third_party/my_run_tag.py \
  --data_dir $DATA_DIR \
  --model_type $MODEL_TYPE \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --gradient_accumulation_steps $GRAD_ACC \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --per_gpu_eval_batch_size 32 \
  --save_steps 1000 \
  --seed 1 \
  --learning_rate $LR \
  --do_train \
  --do_eval \
  --do_predict \
  --predict_langs $LANGS \
  --train_langs $TRAIN_LANGS \
  --log_file $OUTPUT_DIR/train.log \
  --eval_all_checkpoints \
  --eval_patience -1 \
  --overwrite_output_dir \
  --save_only_best_checkpoint $LC \
  --weight_type $WEIGHT_TYPE \
  --predict_head $PREDICT_HEAD