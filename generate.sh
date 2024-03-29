#!/bin/bash

MODEL=efficient_1

mkdir -p ../../results/$MODEL/generate

fairseq-generate ../../data/clean250-bin \
        --path ../../results/$MODEL/checkpoints/checkpoint_best.pt \
        --batch-size 128 \
        --beam 6  > ../../results/$MODEL/generate/result.txt

grep "^H-" ../../results/$MODEL/generate/result.txt | sort -V | cut -f3 > ../../results/$MODEL/generate/pred.ja