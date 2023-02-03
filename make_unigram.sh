#!/bin/bash

mkdir -p ../unigram-corpus
mkdir -p ../../spm-model

# sentencepiece model の学習
#spm_train --input=../raw-corpus/train.ja --model_prefix=../../spm-model/spm.ja --vocab_size=32000 --character_coverage=0.9995 --model_type=unigram
spm_train --input=../raw-corpus/train.en --model_prefix=../../spm-model/spm.en --vocab_size=32000 --character_coverage=1.0 --model_type=unigram --input_sentence_size=14500000 --shuffle_input_sentence=true

for lang in en
do
    for type in train valid test
    do
    spm_encode --model=../../spm-model/spm.$lang.model --output_format=piece < ../raw-corpus/$type.$lang > ../unigram-corpus/$type.$lang
    done
done