# PBL-translation-teamB
2022年学部共通PBL(探求型)翻訳B班

# ディレクトリ構造図

home/  
  └sugihara/  
       ├Downloads/  
       |    └Anaconda3-2021.05-Linux-x86_64.sh  
       ├utils/  
       |  ├apex/  
       |  ├fairseq/  
       |  ├sentencepiece/  
       |  └vcpkg/  
       └workspace/  
            ├pbl/  
	    | ├data/  
	    | |  ├raw-corpus/  
	    | |  |    ├train.en(ja)  
	    | |  |    ├valid.en(ja)  
	    | |  |    └test.en(ja)  
	    | |  ├scripts/  
	    | |  |   ├make_corpus.py  
	    | |  |   ├make_unigram.py  
	    | |  |   └remove_250.py  
	    | |  └unigram-corpus/  
	    | |          ├train.en(ja)  
	    | |          ├valid.en(ja)  
	    | |          ├test.en(ja)  
	    | |          └clean250.en(ja)  
	    | ├scripts/  
	    | |   ├preprocess/  
	    | |   |     └preprocess.sh  
	    | |   ├train/  
	    | |   |  ├train_base.sh  
	    | |   |  ├train_big.sh  
	    | |   |  └train_efficient.sh  
	    | |   └generate/  
	    | |   　   └generate.sh  
	    | └spm-model/  
	    |      ├spm.en.model(vocab)  
	    |      └spm.ja.model(vocab)  
	    └PBL_APPLICATION  
	            └application/  
		           └app.py  
       
# 仮想環境構築
- Anaconda のインストール
~~~
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
~~~

- 仮想環境の作成、起動
~~~
conda create -n PBL_t python=3.8 -y
conda activate PBL_t
~~~

# ライブラリのインストール
- fairseq
~~~
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..
~~~

- apex
~~~
git clone https://github.com/NVIDIA/apex
cd apex
CUDA_HOME=/usr/local/cuda-11.3 pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
cd ..
~~~

- sentencepiece
~~~
git clone https://github.com/google/sentencepiece.git 
cd sentencepiece
mkdir build
cd build
cmake ..
make -j $(nproc)
sudo make install
sudo ldconfig -v
cd ../..
~~~

- vcpkg
~~~
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh
./vcpkg integrate install
./vcpkg install sentencepiece
cd ..
~~~

- fairseq, sentencepiece のパスが通っているか確認
~~~
which fairseq-preprocess
which fairseq-train
which fairseq-generate

which spm_train
which spm_encode
~~~

- sentencepiece install
~~~
pip install sentencepiece
~~~

- sacrebleu install
~~~
pip install sacrebleu
~~~

## データの準備
- JParaCrawl-v3.0 のダウンロード、解凍
~~~
wget https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/bitext/en-ja.tar.gz
tar -zxvf en-ja.tar.gz
rm -r en-ja.tar.gz
~~~

- パラレルコーパス作成
~~~
import sys

args = sys.argv

def cut(fname):
    fin = open(fname, "r")
    f1 = open("train.en", "w")
    f2 = open("train.ja", "w")
    for line in fin:
        part = line.strip().split("\t")
        f1.write(part[3] + "\n")
        f2.write(part[4] + "\n")       
    fin.close()
    f1.close()
    f2.close()

if __name__ == '__main__':
    cut(args[1])
~~~

- 検証・評価用データ (WMT2020)入手
~~~
mkdir raw-corpus

sacrebleu -t wmt20/dev -l en-ja --echo src > raw-corpus/valid.en
sacrebleu -t wmt20/dev -l en-ja --echo ref > raw-corpus/valid.ja
sacrebleu -t wmt20 -l en-ja --echo src > raw-corpus/test.en
sacrebleu -t wmt20 -l en-ja --echo ref > raw-corpus/test.ja
~~~

- サブワード分割
~~~
#!/bin/bash

mkdir -p ../data/unigram-corpus
mkdir -p ../spm-model

# sentencepiece model の学習
spm_train --input=../data/raw-corpus/train.ja --model_prefix=../spm-model/spm.ja --vocab_size=32000 --character_coverage=0.9995 --model_type=unigram
spm_train --input=../data/raw-corpus/train.en --model_prefix=../spm-model/spm.en --vocab_size=32000 --character_coverage=1.0 --model_type=unigram --input_sentence_size=14500000 --shuffle_input_sentence=true

for lang in en ja
do
    for type in train valid test
    do
    spm_encode --model=../spm-model/spm.$lang.model --output_format=piece < ..data/raw-corpus/$type.$lang > ../data/unigram-corpus/$type.$lang
    done
done
~~~

- 文長が長いものを削除 (学習に悪影響、OOM対策)
~~~
def remove_250():
    with open("../unigram/train.ja", "r") as ja, open("../unigram/train.en", "r") as en, \
    open("../unigram/clean250.ja", "w") as f1, open("../unigram/clean250.en", "w") as f2:
        for ja_line,en_line in zip(ja, en):
            ja_words = ja_line.strip().split(" ")
            en_words = en_line.strip().split(" ")
            if len(ja_words) < 250 and len(en_words) < 250:
                f1.write(ja_line)
                f2.write(en_line)

def main():
    remove_250()


if __name__ == '__main__':
    main()
~~~

## Fairseq で学習
- fairseq-preprocess
~~~
#!/bin/bash

TRAIN_DIR=../../data/unigram
VALID_DIR=../../data/unigram
TEST_DIR=../../data/unigram

fairseq-preprocess \
    --destdir ../../data/clean250-bin \  # 保存先
    --source-lang en \  # ソース言語(ファイルの拡張子)
    --target-lang ja \  # ターゲット言語(ファイルの拡張子)
    --trainpref $TRAIN_DIR/clean250 \  # 訓練データのパス
    --validpref $VALID_DIR/valid \  # 検証データのパス
    --testpref $TEST_DIR/test \  # テストデータのパス
    --workers `nproc`
~~~

- fairseq-train (base)
~~~
#!/bin/bash

mkdir -p ../../results/base/

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
        ../../data/clean250-bin \
		--arch transformer \
		--optimizer adam --adam-betas '(0.9,0.98)' \
		--reset-optimizer --reset-dataloader --reset-meters \
		--lr 0.001 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-08 --warmup-updates 4000 \
		--dropout 0.3 --weight-decay 0.001 --clip-norm 1.0 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 4000 --update-freq 128 \
		--patience 20 \
		--fp16 \
		--save-interval-updates 100 --validate-interval-updates 100 \
		--keep-interval-updates 10 --no-epoch-checkpoints \
		--best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
		--eval-bleu \
		--eval-bleu-args '{"beam": 6, "lenpen": 1.0}' \
		--eval-bleu-detok space \
		--eval-bleu-print-samples \
		--eval-bleu-remove-bpe=sentencepiece \
		--tensorboard-logdir ../../results/base/tensorboard \
        --save-dir ../../results/base/checkpoints/ | tee -a ../../results/base/train.log
~~~

- fairseq-train (big)
~~~
#!/bin/bash

mkdir -p ../../results/big/

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
        ../../data/clean250-bin \
		--arch transformer \
		--encoder-layers 6 --encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 \
		--decoder-layers 6 --decoder-embed-dim 1024 --decoder-ffn-embed-dim 4096 \
        --encoder-attention-heads 16 --decoder-attention-heads 16 \
		--optimizer adam --adam-betas '(0.9,0.98)' \
		--reset-optimizer --reset-dataloader --reset-meters \
		--lr 0.001 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-08 --warmup-updates 4000 \
		--dropout 0.3 --weight-decay 0.001 --clip-norm 1.0 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 2000 --update-freq 256 \
		--max-update 36000 \
		--fp16 \
		--save-interval-updates 100 --validate-interval-updates 100 \
		--keep-interval-updates 10 --no-epoch-checkpoints \
		--tensorboard-logdir ../../results/big/tensorboard \
        --save-dir ../../results/big/checkpoints/ | tee -a ../../results/big/train.log
~~~

- fairseq-generate
~~~
#!/bin/bash

MODEL=base (big)

mkdir -p ../../results/$MODEL/generate

fairseq-generate ../../data/clean250-bin \
        --path ../../results/$MODEL/checkpoints/checkpoint_best.pt \
        --batch-size 128 \
        --beam 5  > ../../results/$MODEL/generate/result.txt

grep "^H-" ../../results/$MODEL/generate/result.txt | sort -V | cut -f3 > ../../results/$MODEL/generate/pred.ja
~~~

## 評価
~~~
cat ../../results/$MODEL/generate/pred.ja | sacrebleu -w2 -l en-ja -t wmt20
~~~
