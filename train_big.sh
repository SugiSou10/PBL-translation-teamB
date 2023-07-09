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
