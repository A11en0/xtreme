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
DATA_DIR=${2:-"$REPO/download/"}

TASK='panx'
MAXL=128
LANGS="ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu,qu,pl,uk,az,lt,pa,gu,ro"

LC=""
if [ $MODEL == "bert-base-multilingual-cased" ]; then
  MODEL_TYPE="bert"
elif [ $MODEL == "xlm-mlm-100-1280" ] || [ $MODEL == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ $MODEL == "xlm-roberta-large" ] || [ $MODEL == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
fi

echo "Download panx NER dataset"
if [ -f $DIR/AmazonPhotos.zip ]; then
    base_dir=$DATA_DIR/panx_dataset/
    unzip -qq -j $DATA_DIR/AmazonPhotos.zip -d $base_dir
    cd $base_dir
    langs=(ar he vi id jv ms tl eu ml ta te af nl en de el bn hi mr ur fa fr it pt es bg ru ja ka ko th sw yo my zh kk tr et fi hu qu pl uk az lt pa gu ro)
    for lg in ${langs[@]}; do
        tar xzf $base_dir/${lg}.tar.gz
        for f in dev test train; do mv $base_dir/$f $base_dir/${lg}-${f}; done
    done
    echo "Unzip finished"
    cd ..
    python $REPO/utils_preprocess.py \
        --data_dir $base_dir \
        --output_dir $DATA_DIR/$TASK \
        --task panx
    rm -rf $base_dir
    echo "Successfully downloaded data at $DATA_DIR/panx" >> $DATA_DIR/download.log
else
    echo "Please download the AmazonPhotos.zip file on Amazon Cloud Drive manually and save it to $DIR/AmazonPhotos.zip"
    echo "https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN"
fi

SAVE_DIR="$DATA_DIR/$TASK/${TASK}_${MODEL_TYPE}_processed_maxlen${MAXL}"
echo $SAVE_DIR
mkdir -p $SAVE_DIR
python3 $REPO/utils_preprocess.py \
  --data_dir $DATA_DIR/$TASK/ \
  --task panx_tokenize \
  --model_name_or_path $MODEL \
  --model_type $MODEL_TYPE \
  --max_len $MAXL \
  --output_dir $SAVE_DIR \
  --languages $LANGS $LC >> $SAVE_DIR/preprocess.log
if [ ! -f $SAVE_DIR/labels.txt ]; then
  cat $SAVE_DIR/*/*.${MODEL} | cut -f 2 | grep -v "^$" | sort | uniq > $SAVE_DIR/labels.txt
fi



