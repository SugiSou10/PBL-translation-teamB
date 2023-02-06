# PBL-translation-teamB
2022年学部共通PBL(探求型)翻訳B班

## ディレクトリ構造図

~~~
home/  
  └sugihara/  
       ├Downloads/  
       |    └Anaconda3-2021.05-Linux-x86_64.sh  
       |
       ├utils/  
       |  ├apex/  
       |  ├fairseq/  
       |  ├sentencepiece/  
       |  └vcpkg/  
       |
       └workspace/  
            ├pbl/  
	    | ├data/  
	    | |  ├raw-corpus/  
	    | |  |    ├train.en(ja)  
	    | |  |    ├valid.en(ja)  
	    | |  |    └test.en(ja)  
	    | |  |
	    | |  ├scripts/  
	    | |  |   ├make_corpus.py  
	    | |  |   ├make_unigram.py  
	    | |  |   └remove_250.py  
	    | |  |
	    | |  └unigram-corpus/  
	    | |          ├train.en(ja)  
	    | |          ├valid.en(ja)  
	    | |          ├test.en(ja)  
	    | |          └clean250.en(ja)  
	    | |
	    | ├scripts/  
	    | |   ├preprocess/  
	    | |   |     └preprocess.sh  
	    | |   |
	    | |   ├train/  
	    | |   |  ├train_base.sh  
	    | |   |  ├train_big.sh  
	    | |   |  └train_efficient.sh  
	    | |   |
	    | |   └generate/  
	    | |   　   └generate.sh  
	    | |
	    | └spm-model/  
	    |      ├spm.en.model(vocab)  
	    |      └spm.ja.model(vocab)  
	    |
	    └PBL_APPLICATION/ 
	            └application/  
		           └app.py  
~~~
       
## 仮想環境構築
- Anaconda のインストール
~~~
mkdir Downloads
cd Downloads
wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
bash Anaconda3-2021.05-Linux-x86_64.sh
cd ..
~~~

- 仮想環境の作成、起動
~~~
conda create -n PBL_t python=3.8 -y
conda activate PBL_t
~~~

## ライブラリのインストール
~~~
mkdir utils
cd utils
~~~
- fairseq (ver.0.12.2)
~~~
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..
~~~

- apex (ver.0.1)
~~~
git clone https://github.com/NVIDIA/apex
cd apex
CUDA_HOME=/usr/local/cuda-11.3 pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
cd ..
~~~

- sentencepiece (0.1.97)
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

- sacrebleu install (ver.2.3.1)
~~~
pip install sacrebleu
~~~

## データの準備
~~~
mkdir data
cd data
~~~

- JParaCrawl-v3.0 のダウンロード、解凍
~~~
wget https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/release/3.0/bitext/en-ja.tar.gz
tar -zxvf en-ja.tar.gz
rm -r en-ja.tar.gz
~~~

- パラレルコーパス作成 [make_corpus.py]
~~~
import sys

args = sys.argv

def cut(fname, f_en, f_ja):
    fin = open(fname, "r")
    f1 = open(f_en, "w")
    f2 = open(f_ja, "w")
    for line in fin:
        part = line.strip().split("\t")
        f1.write(part[3] + "\n")
        f2.write(part[4] + "\n")       
    fin.close()
    f1.close()
    f2.close()

if __name__ == '__main__':
    cut(args[1], args[2], args[3])
~~~

- 検証・評価用データ (WMT2020)　入手
~~~
mkdir raw-corpus

sacrebleu -t wmt20/dev -l en-ja --echo src > raw-corpus/valid.en
sacrebleu -t wmt20/dev -l en-ja --echo ref > raw-corpus/valid.ja
sacrebleu -t wmt20 -l en-ja --echo src > raw-corpus/test.en
sacrebleu -t wmt20 -l en-ja --echo ref > raw-corpus/test.ja
~~~

- サブワード分割 [make_unigram.sh]
~~~
#!/bin/bash

mkdir -p ../unigram-corpus
mkdir -p ../../spm-model

# sentencepiece model の学習
spm_train --input=../raw-corpus/train.ja --model_prefix=../../spm-model/spm.ja --vocab_size=32000 --character_coverage=0.9995 --model_type=unigram
spm_train --input=../raw-corpus/train.en --model_prefix=../../spm-model/spm.en --vocab_size=32000 --character_coverage=1.0 --model_type=unigram --input_sentence_size=14500000 --shuffle_input_sentence=true

for lang in en
do
    for type in train valid test
    do
    spm_encode --model=../../spm-model/spm.$lang.model --output_format=piece < ../raw-corpus/$type.$lang > ../unigram-corpus/$type.$lang
    done
done
~~~

- 文長が長いものを削除 (学習に悪影響、OOM対策) [remove_250.py]
~~~
def remove_250():
    with open("../unigram-corpus/train.ja", "r") as ja, open("../unigram-corpus/train.en", "r") as en, \
    open("../unigram-corpus/clean250.ja", "w") as f1, open("../unigram-corpus/clean250.en", "w") as f2:
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
- preprocess.sh  
学習を行うための準備

- train_base.sh  
Transformer-Baseの学習

- train_big.sh  
Transformer-Bigの学習

- generate.sh  
テストデータを用いて学習した翻訳器を評価

