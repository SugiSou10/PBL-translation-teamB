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
