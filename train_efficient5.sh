#!/bin/bash

mkdir -p ../../results/efficient_5/

CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train \
        ../../data/clean250-bin \
		--arch transformer \
		--encoder-layers 8 --encoder-embed-dim 512 --encoder-ffn-embed-dim 2048 \
		--decoder-layers 2 --decoder-embed-dim 512 --decoder-ffn-embed-dim 2048 \
        --encoder-attention-heads 8 --decoder-attention-heads 8 \
		--optimizer adam --adam-betas '(0.9,0.98)' \
		--reset-optimizer --reset-dataloader --reset-meters \
		--lr 0.001 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-08 --warmup-updates 4000 \
		--dropout 0.1 --weight-decay 0.001 --clip-norm 1.0 \
		--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
		--max-tokens 4000  --update-freq 128 \
		--max-update 18900 \
		--fp16 \
		--save-interval-updates 100 --validate-interval-updates 100 \
		--keep-interval-updates 10 --no-epoch-checkpoints \
		--tensorboard-logdir ../../results/efficient_5/tensorboard \
        --save-dir ../../results/efficient_5/checkpoints/ | tee -a ../../results/efficient_5/train.log