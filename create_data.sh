#!/bin/bash

cd ..

# JParaCrawl-v3.0 のダウンロード、解凍
wget https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/bitext/en-ja.tar.gz
tar -zxvf en-ja.tar.gz
rm -r en-ja.tar.gz

cd scripts

# パラレルコーパス作成
python make_corpus.py ../en-ja/en-ja.bicleaner05.txt ../raw-corpus/train.en ../raw-corpus/train.ja

cd ..

# 検証・評価用データ (WMT2020) 入手
mkdir raw-corpus

sacrebleu -t wmt20/dev -l en-ja --echo src > raw-corpus/valid.en
sacrebleu -t wmt20/dev -l en-ja --echo ref > raw-corpus/valid.ja
sacrebleu -t wmt20 -l en-ja --echo src > raw-corpus/test.en
sacrebleu -t wmt20 -l en-ja --echo ref > raw-corpus/test.ja

cd scripts

# サブワード分割
mkdir -p ../unigram-corpus
mkdir -p ../../spm-model

# sentencepiece model の学習
spm_train --input=../raw-corpus/train.ja --model_prefix=../../spm-model/spm.ja --vocab_size=32000 --character_coverage=0.9995 --model_type=unigram
spm_train --input=../raw-corpus/train.en --model_prefix=../../spm-model/spm.en --vocab_size=32000 --character_coverage=1.0 --model_type=unigram --input_sentence_size=14500000 --shuffle_input_sentence=true

for lang in en ja
do
    for type in train valid test
    do
    spm_encode --model=../../spm-model/spm.$lang.model --output_format=piece < ../raw-corpus/$type.$lang > ../unigram-corpus/$type.$lang
    done
done

python remove_250.py