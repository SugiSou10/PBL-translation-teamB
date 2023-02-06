# PBL-translation-teamB
2022年学部共通PBL(探求型)翻訳B班

## プロジェクトの概要
- 英日機械翻訳のWebアプリを作成  
- Transformer-baseを基準として、モデルサイズ・翻訳にかかる時間に関して効率の良いモデルを作成

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
	    | |  |   ├remove_250.py  
	    | |  |   └create_data.sh
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
- fairseq (Ver.0.12.2)
~~~
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..
~~~

- apex (Ver.0.1)
~~~
git clone https://github.com/NVIDIA/apex
cd apex
CUDA_HOME=/usr/local/cuda-11.3 pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
cd ..
~~~

- sentencepiece (Ver.0.1.97)
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

- sacrebleu (Ver.2.3.1)
~~~
pip install sacrebleu
~~~

## データの準備
- create_data.sh  
	- JParaCrawl-v3.0 のダウンロード、解凍  
	- パラレルコーパス作成 (make_corpus.py の実行)  
	- 検証・評価用データ (WMT2020)　入手  
	- サブワード分割  
	- 学習に悪影響であることと、OOM対策のため文長が長いものを削除  (remove_250.py の実行)

## Fairseq で学習
- preprocess.sh  
学習を行うための準備

- train_base.sh  
Transformer-baseの学習

- train_big.sh  
Transformer-bigの学習

- generate.sh  
テストデータを用いて学習した翻訳器を評価

## Webアプリの使い方
- app.py を実行  
- flask run --host=0.0.0.0を実行  
- Running on http://133.71.101.87:5000と表示される
- http://133.71.101.87:5000に移動すると翻訳画面が表示される
- 左の枠内に翻訳したい英文を入力し、真ん中の翻訳ボタンを押す。

![スクリーンショット (26)](https://user-images.githubusercontent.com/115621597/216903794-7743bd8c-8f55-49d8-b36c-f5e461c2e4f7.png)

